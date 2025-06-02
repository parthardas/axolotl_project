## Fine-tuning LLMs for industry use cases

This project contains the config listing for fine-tuning LLM foundation models for industry use cases.

The following pre-trained LLMs have been used for fine-tuning
  -  Microsoft Phi1.5

The following datasets have been used:
  -   alpaca format text classification (sentiment-analysis)
  -   alpaca format question and answer (text-generation)

The fine-tuning have used the following configs/hardware:
  - 4-bit QLoRA quantized downloading of model
  - alpaca dataset
  - adam variants for optimizer
  - hardware: NVIDIA T4 GPU
  - micro batch size and gradient accumulation steps optimized for speed
  - sequence length suitable for T4 GPU
  - adapter output as .safetensors

Output adapters merged with FM model

The following industry use cases have been addressed:
  - Airlines customer sentiment
  - Retail banking helpdesk question answering

Model testing code is included.
