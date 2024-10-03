# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

import os
import re
import json
import queue
import argparse
import tempfile
import threading
import traceback
import subprocess
import pandas as pd
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt

from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import entropy
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from LLMInstruct.decontamination.similarity_filter import JaccardFilter
from LLMInstruct.utils import read_jsonl, write_jsonl

from utils import post_process_completion, read_problems
from LLMInstruct.executor.execution import check_correctness, clean_up_simulation


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=False, type=str, default="./code_repair_examples/output", help="error report output directory")
parser.add_argument("--temp", required=False, type=str, default="./code_repair_examples/tmp", help="Temperary directory for intermidiate output")
parser.add_argument("--num_samples", required=False, type=int, default=50, help="Number of samples per task for error report.")
parser.add_argument("--workers", required=False, type=int, default=64, help="Number of workers")
parser.add_argument("--benchmark_path", required=False, type=str, default="./dataset/benchmark", help="benchmark directory")
parser.add_argument("--exp_path", required=False, type=str, default="./code_repair_examples/exp", help="experiment directory")
parser.add_argument("--cache_path", required=False, type=str, default="./code_repair_examples/cache", help="cache directory")
parser.add_argument("--iters", required=False, type=int, default=1, help="Iterations to generate error report")
parser.add_argument('--self_consist', action='store_true', help='Perform self-consist check.')
args = parser.parse_args()



def read_benchmark(benchmark_path: str = args.benchmark_path):
    print(f'Load benchmark jsonl from {benchmark_path}')
    machine_df = pd.DataFrame(read_jsonl(os.path.join(benchmark_path, 'verilogeval-machine.jsonl.gz')))
    human_df = pd.DataFrame(read_jsonl(os.path.join(benchmark_path, 'verilogeval-manual.jsonl.gz')))
    rtllm_df = pd.DataFrame(read_jsonl(os.path.join(benchmark_path, 'rtllm.jsonl')))

    problem_df = pd.concat([machine_df, human_df, rtllm_df])
    problem_lookup = {}
    benchmark_lookup = {'rtllm': {}, 'machine': {}, 'human': {}}

    for idx, row in machine_df.iterrows():
        problem_lookup[(row.task_id, 'machine')] = row.canonical_solution
        benchmark_lookup['machine'][row.task_id] = row.to_dict()
    for idx, row in human_df.iterrows():
        problem_lookup[(row.task_id, 'human')] = row.canonical_solution
        benchmark_lookup['human'][row.task_id] = row.to_dict()
    for idx, row in rtllm_df.iterrows():
        problem_lookup[(row.task_id, 'rtllm')] = row.canonical_solution
        benchmark_lookup['rtllm'][row.task_id] = row.to_dict()

    prompt_lookup = {
        'human': {i['task_id']: i['detail_description'] for i in read_jsonl(os.path.join(benchmark_path, 'VerilogDescription_Human.jsonl'))},
        'machine': {i['task_id']: i['detail_description'] for i in read_jsonl(os.path.join(benchmark_path, 'VerilogDescription_Machine.jsonl'))},
        'rtllm': {i['task_id']: i['detail_description'] for i in read_jsonl(os.path.join(benchmark_path, 'rtllm.jsonl'))}
    }

    return problem_df, problem_lookup, benchmark_lookup, prompt_lookup


def read_checkpoint(exp_path: str):
    exp_dict = defaultdict(list)
    """
    exp_dict = {
        "exp_path": [
            "expname1",
            "expname2",
        ]
    }
    """
    for exp_name in os.listdir(exp_path):
        exp_dict[exp_path].append(exp_name)

    df_list = []
    for root, exp_list in exp_dict.items():
        for exp in exp_list:
            for ckpt in os.listdir(os.path.join(root, exp)):
                try:
                    df = pd.DataFrame(read_jsonl(os.path.join(root, exp, ckpt, "VerilogEval_Human_0_1.jsonl_results.jsonl")))
                    df['step'] = int(ckpt.split('step=')[1].split('-')[0])
                    df['benchmark'] = "human"
                    df['exp'] = exp
                    df_list.append(df)
                except Exception as ex:
                    print(ex)

                try:
                    df = pd.DataFrame(read_jsonl(os.path.join(root, exp, ckpt, "VerilogEval_Machine_0_1.jsonl_results.jsonl")))
                    df['step'] = int(ckpt.split('step=')[1].split('-')[0])
                    df['benchmark'] = "machine"
                    df['exp'] = exp
                    df_list.append(df)
                except Exception as ex:
                    print(ex)

                try:
                    df = pd.DataFrame(read_jsonl(os.path.join(root, exp, ckpt, "rtllm_0_1.jsonl_results.jsonl")))
                    df['step'] = int(ckpt.split('step=')[1].split('-')[0])
                    df['benchmark'] = "rtllm"
                    df['exp'] = exp
                    df_list.append(df)
                except Exception as ex:
                    print(ex)

    df = pd.concat(df_list)
    df = df.drop_duplicates(subset=['task_id', 'completion'])
    return postprocess_data(df)


def postprocess_data(df):
    completion_list = []
    prompt_list = []

    for idx, row in df.iterrows():
        completion = post_process_completion(row.completion, remove_header=True, rtllm=(row.benchmark=="rtllm"))
        prompt = benchmark_lookup[row.benchmark][row.task_id]['prompt']
        completion_list.append(completion)
        prompt_list.append(prompt)

    df['completion'] = completion_list
    df['prompt'] = prompt_list
    return df


def read_df(cache_file: str = "cache.csv"):
    cache_path = os.path.join(args.cache_path, cache_file)
    if os.path.exists(cache_path):
        print(f'read dataframe from cache {cache_path}')
        df = pd.read_csv(cache_path)
    else:
        df = read_checkpoint(args.exp_path)
        print(f'read dataframe from experiment directory {args.exp_path}')
        df.to_csv(cache_path, index=False)
        print(f'write dataframe to cache {cache_path}')
    return df


def split_difficulty(dff):
    easy = set()
    hard = set()
    soso = set()

    for task_id, sdf in dff.groupby('task_id'):
        passrate = sdf.passed.mean()
        if passrate > 0.9:
            easy.add(task_id)
        elif passrate < 0.2:
            hard.add(task_id)
        else:
            soso.add(task_id)

    return easy, soso, hard


def run_trajectory(df, queue, exp, benchmark, difficulty, task_id):
    report_list = []
    for step in sorted(df.step.unique()):
        sdf = df[df.step == step]
        passrate = sdf.passed.mean()
        if passrate < 0.2:
            prompt = sdf.prompt.values[0]
            error = sdf[sdf.passed == False].sample(1).completion.values[0]
            correct = problem_lookup[(task_id, benchmark)]
            report_list.append((prompt, error, correct, exp, benchmark, difficulty, task_id))
    return report_list

def parse(response_text: str):
    if response_text is None:
        return
    if "<CODE>"  in response_text and "</CODE>" in response_text:
        st = response_text.find("<CODE>") + len("<CODE>")
        ed = response_text.find("</CODE>")
        if st is None or ed is None:
            return None
        return response_text[st:ed].strip()
    
def self_consist(report: dict, retry: int = 3):

    prompt = report['problem']
    error = report['error']
    reason = report['reason']

    for _ in range(retry):
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url = "https://integrate.api.nvidia.com/v1",
                api_key = os.environ.get('API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC')
            )

            problem = f"""
            Here is an Verilog spec:
            ```
            {prompt}
            ```

            Here is an erroneous implementation:
            ```
            {error}
            ```

            Here is the error explanation:
            ```
            {reason}
            ```

            Now give me the correct code. Need to be complete different from the erroneous implementation.
            """

            completion = client.chat.completions.create(
                model="nvidia/nemotron-4-340b-instruct",
                messages=[
                    {"role": "system", "content": "You are expert in Verilog. Write the correct code and put it between <CODE> </CODE> tags."},
                    {"role": "user", "content": problem},
                ],
            )
            return parse(completion.choices[0].message.content)
        except:
            print(traceback.format_exc())
            raise


def generate_error_report(problem: str, error: str, correct: str, exp: str, benchmark: str, difficulty: str, task_id: str, retry: int = 3):

    for _ in range(retry):
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url = "https://integrate.api.nvidia.com/v1",
                api_key = os.environ.get('API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC')
            )

            prompt = f"""
            Here is an Verilog spec:
            ```
            {problem}
            ```

            Here is an erroneous implementation:
            ```
            {error}
            ```

            Here is an correct implementation:
            ```
            {correct}
            ```

            What error is made? Generate a detail error report.
            The error report should be describe the general error type made such that we can study more on this specific knowledge or on how to avoid the error.
            For example, errors in writing latches or arithmetic shifts.
            The error report should also be detailed enough to let beginners to repair the erroneous implementation
            """

            completion = client.chat.completions.create(
                model="nvidia/nemotron-4-340b-instruct",
                messages=[
                    {"role": "system", "content": "You are expert in Verilog."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
            )
            reason = completion.choices[0].message.content
            return {
                'problem': problem,
                'error': error,
                'correct': correct,
                'exp': exp,
                'benchmark': benchmark,
                'difficulty': difficulty,
                'task_id': task_id,
                'reason': reason
            }
        except Exception as ex:
            print(traceback.format_exc())
            raise


def queue_worker(task_queue, result_queue):
    while True:
        try:
            # Non-blocking get from the queue with timeout
            args_list = task_queue.get(timeout=1)
            if args_list is None:
                break
            report = generate_error_report(*args_list)
            if args.self_consist:
                report['completion'] =self_consist(report)
            result_queue.put(report)
            task_queue.task_done()
        except queue.Empty:
            continue
        except Exception as ex:
            result_queue.put(ex)
            task_queue.task_done()


def run_shell_cmd(cmd: str):
    job = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True
    )
    return job.stdout.strip()


def evaluate_functional_correctness(
    report_list: str,
    problem_file: str,
    k: list[int] = [1, 10],
    n_workers: int = 4,
    timeout: float = 30.0,
    unit_test: bool = False,
    clean_up: bool = True,
    post_process: bool = True,
    remove_header: bool = True,
    rtllm: bool = False,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    print(f"post_process: {post_process}")

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = []

        print("Reading samples...")
        for sample in tqdm(report_list):
            task_id = sample["task_id"]
            completion = sample["completion"]
            if post_process:
                completion = post_process_completion(completion, remove_header, rtllm)

            future = executor.submit(
                check_correctness, 
                problems[task_id],
                completion,
                timeout,
                completion_id[task_id],
                100 if unit_test else None,
                rtllm
            )
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append(result)
    
    if clean_up:
        clean_up_simulation()

    return results

def validate_self_consist(report_list):

    results = []
    results += evaluate_functional_correctness(
        [i for i in report_list if i['benchmark'] == 'human'],
        "./dataset/benchmark/VerilogEval_Human.jsonl"
    )
    results += evaluate_functional_correctness(
        [i for i in report_list if i['benchmark'] == 'machine'],
        "./dataset/benchmark/VerilogEval_Machine.jsonl"
    )
    results += evaluate_functional_correctness(
        [i for i in report_list if i['benchmark'] == 'rtllm'],
        "./dataset/benchmark/rtllm.jsonl",
        rtllm=True
    )
    return results


def process_dedup(task_id, trajectory):
    jaccard_filter = JaccardFilter(threshold=0.9)
    deduped_trajectory = []
    for row in trajectory:
        (prompt, error, correct, exp, benchmark, difficulty, task_id) = row
        data = f"{error}\n\n{correct}"
        if jaccard_filter.validate(data):
            jaccard_filter.add(data)
            deduped_trajectory.append(row)
    return task_id, deduped_trajectory


def parallel_deduplication(report_dict):
    deduped_report_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=156) as executor:
        futures = []
        before, after = 0, 0
        for task_id, trajectory in report_dict.items():
            futures.append(executor.submit(process_dedup, task_id, trajectory))
            before += len(trajectory)
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            task_id, deduped_trajectory = future.result()
            deduped_report_dict[task_id] = deduped_trajectory
            after += len(deduped_trajectory)

        print(f"Deduplication: before {before} after {after}")
    return deduped_report_dict


def run_exp_benchmark(df, task_queue, result_queue, target_number:int, iters: int = 10):

    temp_dir = os.path.join(args.temp, str(os.getpid()))
    os.makedirs(temp_dir)

    error_report = defaultdict(list)
    for _ in range(iters):

        print(f'{_} iterations: {sum([len(i) for i in error_report.values()])} reports')
        os.makedirs(os.path.join(temp_dir, str(_)))

        cnt = 0
        report_dict = defaultdict(list)
        for exp, expdf in tqdm(df.groupby('exp')):
            for benchmark, benchdf in expdf.groupby('benchmark'):
                easy, soso, hard = split_difficulty(benchdf)
                difficulty_levels = {**{tid: 'easy' for tid in easy}, **{tid: 'soso' for tid in soso}, **{tid: 'hard' for tid in hard}}
                for task_id in benchdf.task_id.values:
                    difficulty = difficulty_levels[task_id]
                    trajectory = run_trajectory(benchdf[benchdf.task_id == task_id], queue, exp, benchmark, difficulty, task_id)
                    report_dict[task_id] += trajectory
                    cnt += len(trajectory)

        print(f'{cnt} total trajectories')

        # deduplicate each task
        report_dict = parallel_deduplication(report_dict)
        with open(os.path.join(temp_dir, str(_), "report_dict.json"), "w") as f:
            print(f'output report dict at {os.path.join(temp_dir, str(_), "report_dict.json")}')
            json.dump(report_dict, f)

        # sample each task and control numbers
        sample_list = []
        for task_id, report_list in tqdm(report_dict.items()):
            number_still_need = target_number - len(error_report[task_id])
            if number_still_need <= 0 or len(report_list) <= 0:
                continue
            np.random.shuffle(report_list)
            number_still_need = min(number_still_need, 10)
            selected_report_list = report_list[:number_still_need]
            for i in selected_report_list:
                # task_queue.put(i)
                sample_list.append(i)
        print(f'{len(sample_list)} job ready')
        with open(os.path.join(temp_dir, str(_), "sample_list.json"), "w") as f:
            print(f'output sample list at {os.path.join(temp_dir, str(_), "sample_list.json")}')
            json.dump(sample_list, f)

        # submit jobs
        for i in tqdm(sample_list):
            task_queue.put(i)
        print(f'{len(sample_list)} job summitted')
        task_queue.join()

        # Collect results
        report_list = []
        with tqdm(total=len(sample_list)) as pbar:
            while not result_queue.empty():
                report = result_queue.get()
                pbar.update(1)
                if isinstance(report, dict):
                    report_list.append(report)
                else:
                    print(report)
        print(f'{len(report_list)} report recieved')
        with open(os.path.join(temp_dir, str(_), "report_list.json"), "w") as f:
            print(f'output report list at {os.path.join(temp_dir, str(_), "report_list.json")}')
            json.dump(report_list, f)

        # validate
        valid_list = []
        if args.self_consist:
            cnt = 0
            valid_list = validate_self_consist(report_list)
            for valid, report in zip(valid_list, report_list):
                if valid['passed']:
                    cnt += 1
                task_id = report['task_id']
                if len(error_report[task_id]) < target_number:
                    report['result'] = valid['result']
                    report['passed'] = valid['passed']
                    error_report[task_id].append(report)
            print(f"{cnt}/{len(valid_list)} passed")
            with open(os.path.join(temp_dir, str(_), "valid_list.json"), "w") as f:
                print(f'output valid list at {os.path.join(temp_dir, str(_), "valid_list.json")}')
                json.dump(valid_list, f)
        else:
            for report in report_list:
                task_id = report['task_id']
                report['passed'] = report['result'] = None
                error_report[task_id].append(report)

        print(f'Output error report to {os.path.join(temp_dir, str(_), f"error_report.jsonl")}')
        write_jsonl(os.path.join(temp_dir, str(_), f"error_report.jsonl"), [report for report in report_list])

    return error_report, report_list, valid_list



problem_df, problem_lookup, benchmark_lookup, prompt_lookup = read_benchmark()


def main():
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    num_workers = args.workers
    target_number = args.num_samples
    df = read_df()

    # Start worker threads
    workers = []
    for _ in range(num_workers):
        worker = threading.Thread(target=queue_worker, args=(task_queue, result_queue))
        worker.start()
        workers.append(worker)

    try:
        error_report, report_list, valid_list = run_exp_benchmark(df, task_queue, result_queue, target_number, iters=args.iters)

        tmp = []
        for task_id, report_list in error_report.items():
            for report in report_list:
                report = deepcopy(report)
                if not args.self_consist or report['passed']:
                    benchmark = report['benchmark']
                    task_id = report['task_id']
                    prompt = prompt_lookup[benchmark][task_id]
                    report['prompt'], report['problem'] = report['problem'], prompt
                    report['error'] = report['prompt'] + report['error']
                    report['correct'] = report['prompt'] + report['correct']
                    tmp.append(report)
        write_jsonl(os.path.join(args.output, 'error_report.jsonl'), tmp)
    finally:
        # Stop workers
        for _ in range(num_workers):
            task_queue.put(None)

        for worker in workers:
            worker.join()


if __name__ == "__main__":
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.temp, exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)
    main()