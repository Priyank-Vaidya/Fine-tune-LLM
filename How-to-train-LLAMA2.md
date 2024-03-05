
# How to Train the model on LLAMA2

LLAMA2 is the one of the open Source collection of LLM which is built by Meta which is highly optimized for conversation and dialogue generation against ChatGPT and PaLM. 

To train the model on LLAMA2 means fine-tuning the model by updating the weights and parameters of pre-trained LLAMA2 on 7B to 70B Parameters. 

The concept revolves around fine-tuning the model by our own dataset and training the model resulting into coverting the model into a specific application.

Fine-tuning in machine learning is the process of adjusting the weights and parameters of a pre-trained model on new data to improve its performance on a specific task.

# How to Fine-Tune LLAMA2 

Before jumping directly to traiing the AI model, we need to first set the hardware configutration required to build it. 

There are the specific techniques which can be used to fine-tune the model. One of those techniques is QLoRA. 
QLoRA uses the 4-bit Quantization configutration to optimize VRAM usage. For that We will use the Hugging Face ecosystem of LLM Libraries listed below.


```bash
%%capture
%pip install accelerate peft bitsandbytes transformers trl
```


#### Load the necessary Libraries

```bash
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
```

### 4-Bit Quantization Technique

Technique via QLoRA allows to efficient fine-tune the large LLM Models on custom hardware. In this technique, the Model is quantized into 4-bits and it freezes the parameters. 

During the fine-tuning process, the gradients are then backpropoagres throught the frozen Quantization into Low-Rank Adapter Layer. 

This ensures that entire Pre-trained model is fixed withhin 4 buts while only changing the adapters. 

Therefore, 4-Bit does not compromise the Model performance.


```bash
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
```


![4-Bit Quantization](https://images.datacamp.com/image/upload/v1697713094/image7_3e12912d0d.png)


#### Loading LLAMA2 and tokenizer

```bash
model = AutoModelForCausalLM.from_pretrained(
    <Path to LLAMA2 Model>,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


```

Now the training parameters which includes epochs, 
batch_size/GPU instance for training and testing, gradient_accumulation_steps, gradient_check_points, 
learning_rate,
etc 

## Model Fine-tuning

Traditional parameter Fine-tuning requires the updating all of the model's parameter which is computationally expensive and requiest massive amounts of data.

PEFT (Parameter Efficient Fine-Tuning) works by updating only the subset of the Model's parameter, by making it much more efficient.

```bash
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
```

It uses the tensorboard Library. It uses the instance within the log directory and port number. 


### Method of Model Fine-tuning

Supervised Fine-tuning (SFT) is a key step ni reinforcement learning from Human feedback.  It comes with tools to train language models using reinforcement learning, starting with supervised fine-tuning, then reward modeling, and finally proximal policy optimization (PPO)

```bash
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
```

#### Evaludation

Wr van review the training results in the interactive session of tensorboard.

```bash
from tensorboard import notebook
log_dir = "results/runs"
notebook.start("--<same as log dir specified in PEFT> {} --port <Port>".format(log_dir))
```

### Testing with custom prompts

```bash
prompt = "Who wrote Dune Novel series?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```


## Authors

- [Priyank Vaidya](https://www.linkedin.com/in/priyank-vaidya/)


## Blog References


| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `Fine-Tuning` | `Blog` | https://www.datacamp.com/tutorial/fine-tuning-llama-2 |
| `QLoRA` | `Blog` | https://arxiv.org/abs/2305.14314 |
| `PEFT` | `Blog` | https://huggingface.co/docs/peft/index |
| `Using-model after training` | `Blog` |https://huggingface.co/docs/trl/main/en/use_model |




