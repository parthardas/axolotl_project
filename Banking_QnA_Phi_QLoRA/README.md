---
library_name: peft
license: mit
base_model: microsoft/phi-1_5
tags:
- generated_from_trainer
datasets:
- /workspace/data/alpaca_corrected_bankingqa.jsonl
model-index:
- name: workspace/outputs/phi-bankingqa-out5
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
    path: /workspace/data/alpaca_corrected_bankingqa.jsonl
    type: alpaca

dataset_prepared_path:
val_set_size: 0.1
output_dir: /workspace/outputs/phi-bankingqa-out5


sequence_len: 1024 #reduced to hasten training
sample_packing: true
pad_to_sequence_len: true

#axolotl own suggestion
eval_sample_packing: False

adapter: qlora
#lora_model_dir:
lora_r: 16
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

wandb_project: phi1.5-bankingqa-finetune
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 8 #increase to hasten training
micro_batch_size: 4
gradient_checkpointing: true #added to hasten training
num_epochs: 1
optimizer: adamw_torch_fused
adam_beta2: 0.95
adam_epsilon: 0.00001
max_grad_norm: 1.0
lr_scheduler: cosine
weight_decay: 0.01 # added to hasten training
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

# workspace/outputs/phi-bankingqa-out5

This model is a fine-tuned version of [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) on the /workspace/data/alpaca_corrected_bankingqa.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1071

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
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.95) and epsilon=1e-05 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 100
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 2.8635        | 0.0208 | 1    | 1.2920          |
| 2.8745        | 0.2494 | 12   | 1.2862          |
| 2.7446        | 0.4987 | 24   | 1.2616          |
| 2.4361        | 0.7481 | 36   | 1.1899          |
| 2.0611        | 0.9974 | 48   | 1.1071          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.51.3
- Pytorch 2.5.1+cu124
- Datasets 3.5.0
- Tokenizers 0.21.1