---
library_name: peft
license: mit
base_model: microsoft/phi-1_5
tags:
- generated_from_trainer
datasets:
- /workspace/data/sentiment.jsonl
model-index:
- name: workspace/outputs/phi-sentiment-out
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.10.0.dev0`
```yaml
base_model: microsoft/phi-1_5
# optionally might have model_type or tokenizer_type
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
# Automatically upload checkpoint and final model to HF
# hub_model_id: username/custom_model_name

load_in_8bit: false
load_in_4bit: true

datasets:
  - #path: garage-bAInd/Open-Platypus
    path: /workspace/data/sentiment.jsonl
    type: alpaca

dataset_prepared_path:
val_set_size: 5
output_dir: /workspace/outputs/phi-sentiment-out

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

#axolotl own suggestion
eval_sample_packing: False

adapter: qlora
#lora_model_dir:
lora_r: 64
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true

wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 2
micro_batch_size: 4
num_epochs: 1
optimizer: adamw_torch_fused
adam_beta2: 0.95
adam_epsilon: 0.00001
max_grad_norm: 1.0
lr_scheduler: cosine
learning_rate: 0.0002

bf16: auto
#tf32: true

gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: True
resume_from_checkpoint:
logging_steps: 1
#flash_attention: true
flash_attention: false

warmup_steps: 100
evals_per_epoch: 4
saves_per_epoch: 1
weight_decay: 0.1
resize_token_embeddings_to_32x: true
special_tokens:
  pad_token: "<|endoftext|>"
```

</details><br>

# workspace/outputs/phi-sentiment-out

This model is a fine-tuned version of [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) on the /workspace/data/sentiment.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2148

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.95) and epsilon=1e-05 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 100
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 6.6611        | 0.0227 | 1    | 8.7855          |
| 6.2266        | 0.25   | 11   | 8.0873          |
| 2.2228        | 0.5    | 22   | 4.1190          |
| 0.3054        | 0.75   | 33   | 0.5615          |
| 0.2409        | 1.0    | 44   | 0.2148          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.5.1+cu124
- Datasets 3.5.0
- Tokenizers 0.21.1