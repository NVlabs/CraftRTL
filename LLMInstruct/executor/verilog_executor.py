# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from typing import Optional, Callable, Dict
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile

import subprocess
import re
from threading import Timer
from LLMInstruct.executor.execution import (
    time_limit,
    swallow_io,
    create_tempdir,
    TimeoutException,
    WriteOnlyStringIO,
    reliability_guard,
)


def check_correctness(
    completion: str,
    timeout: float,
    test: str = "",
    completion_id: Optional[int] = None,
    unit_test_length: Optional[int] = None,
    compile_only: bool = False,
) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.
    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    problem = {
        "test": test,
        "task_id": 0,
    }

    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil

            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            reliability_guard()

            result["passed"] = True
            result["compiler_log"] = ""
            result["test_output"] = ""
            result["verilog_test"] = problem["test"] + "\n" + completion
            result["completion"] = completion
            if "wave.vcd" in result:
                result.pop("wave.vcd")

            compile_func = iverlog_compile
            if compile_only == "quartus":
                compile_func = quartus_compile
            elif compile_only == "vcs":
                compile_func = vcs_compile
            elif compile_only == "modelsim":
                compile_func = modelsim_compile

            out = compile_func(completion, problem["task_id"])
            result["compiler_log"] = out

            if not verilog_compile_is_correct(out):
                result["passed"] = False

            result["haha"] = compile_only
            if not compile_only and result["passed"]:

                iverlog_compile(completion, problem["task_id"], problem["test"])

                # simulate
                out, err = execute("vvp -n test.vvp", 20)
                result["test_output"] = f"{out}\n{err}"
                match = re.search(r"Mismatches: ([0-9]*) in ([0-9]*) samples", out)
                if match:
                    cor, tot = [int(i) for i in match.groups()]
                    if cor != 0:
                        result["passed"] = False
                else:
                    result["passed"] = False

                if os.path.exists("wave.vcd"):
                    result["wave.vcd"] = open("wave.vcd", "r").read()

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.dict()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result = {
            "passed": False,
            "compiler_log": "timed out",
            "test_result": "timed out",
        }

    return dict(
        task_id=problem["task_id"],
        passed=result["passed"],
        feedback=dict(result),
        completion_id=completion_id,
    )


def iverlog_compile(verilog_test: str, task_id: str, test: str = ""):

    extra_cmd = ""
    if test:
        verilog_test = f"{test}\n{verilog_test}"
        extra_cmd = "-s tb"

    with open(f"{task_id}.sv", "w") as f:
        f.write(verilog_test)
    out, err = execute(
        f"iverilog -Wall -Winfloop -Wno-timescale -g2012 {extra_cmd} -o test.vvp {task_id}.sv",
        10,
    )
    return err


def execute(cmd: str, timeout: int):
    try:
        with swallow_io():
            with time_limit(timeout):
                p = subprocess.Popen(
                    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                timer = Timer(timeout, p.kill)
                try:
                    timer.start()
                    out, err = p.communicate()
                finally:
                    timer.cancel()

                out, err = out.decode("utf-8"), err.decode("utf-8")
    except TimeoutException:
        out = err = "timed out"
    except BaseException as e:
        out = err = f"failed: {e}"
    finally:
        return out, err


def verilog_compile_is_correct(log: str):
    log = log.lower()

    if "error" in log or "give up" in log:
        return False

    return True


def quartus_compile(verilog_test: str, task_id: str):
    with open(f"{task_id}.sv", "w") as f:
        f.write(verilog_test)

    # grep '^Error|^Warning'
    with open("top_module.qsf", "w") as f:
        f.write(f"set_global_assignment -name SYSTEMVERILOG_FILE {task_id}.sv")
    out, err = execute(
        f"quartus_map --effort=fast --parallel=4 --read_settings_files=on --write_settings_files=off top_module -c top_module | grep '^Error'",
        30,
    ) # users should replace quartus_map to correct path of executable file

    tmp = []
    for i in out.strip().split("\n"):
        if "Error (" in i:
            trunc = i.find("Check for and fix")
            if trunc > 0:
                i = i[:trunc]
            tmp.append(i)
    out = "\n".join(tmp)

    return out


def vcs_compile(verilog_test: str, task_id: str):
    with open(f"{task_id}.sv", "w") as f:
        f.write(verilog_test)
    out, err = execute(f"vcs -j8 +v2k -sverilog -q -full64 {task_id}.sv", 20)
    return out


def modelsim_compile(verilog_test: str, task_id: str):
    with open(f"{task_id}.sv", "w") as f:
        f.write(verilog_test)
    out, err = execute(
        f"vlog -sv -quiet {task_id}.sv", 10
    ) # users should replace model_sim to correct path of executable file
    return out
