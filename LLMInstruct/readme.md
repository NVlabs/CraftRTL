# LLM-Instruct

The entrypoint scripts for experiments are under `scripts`. Example of a custom script is as follow:
```
#!/bin/bash

python LLMInstruct/main.py \
    --seed_code_start_index 0 \
    --max_new_data ${num_instructions_to_generate} \
    --data_dir ${data_dir} \
    --dataset_name ${dataset_name} \
    --engine ${engine} \
    --prompt ${prompt} \
    --task ${task}
```


The usage of the LLMInstruct package can be found under `LLMInstruct/task`. Example of a custom task demos the pipeline as follow:
```

class CustomTask(BaseTask):

    def __call__(self, example: dict):
        prompt = self.construct_prompt(example)
        raw_result = self.generate(prompt)
        result = self.parse(raw_result)
        filtered_result = self.decontaminate(result)
        
        return dict(
            input=example,
            output=filtered_result,
            raw=raw_result,
        )
```

Add you custom task to `LLMInstruct/task/factory.py`.
