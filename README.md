# Fine-Tuning GPT-2 with QLoRA on SQuAD

This notebook fine-tunes `openai-community/gpt2` for question answering using:
- 4-bit quantization (BitsAndBytes)
- LoRA adapters (PEFT)
- Hugging Face `Trainer`
- SQuAD dataset (`rajpurkar/squad`)

## What the notebook does
1. Installs required libraries (`bitsandbytes`, `accelerate`, `datasets`, `peft`, `transformers`, `evaluate`).
2. Loads SQuAD and uses subsets by default:
- Train: first 3000 examples
- Eval: first 200 examples
3. Loads GPT-2 in 4-bit with `device_map="auto"`.
4. Builds a custom chat-style prompt template for QA.
5. Applies QLoRA configuration:
- `r=16`
- `lora_alpha=32`
- target modules: `c_attn`, `c_proj`
- `lora_dropout=0.05`
6. Trains with `Trainer` for `max_steps=300` and evaluates every 10 steps.
7. Saves checkpoints under `./qa-gpt2-lora`.

## Environment
Use Python 3.10+ and a CUDA-enabled GPU (recommended).

Install dependencies:
```bash
pip install bitsandbytes accelerate datasets peft transformers evaluate torch
```

## Run
Open and run all cells in order:
```bash
jupyter notebook finetunefinal.ipynb
```

## Key training settings
- Model: `openai-community/gpt2`
- Quantization: 4-bit NF4
- Optimizer: `paged_adamw_8bit`
- Learning rate: `3e-4`
- Warmup steps: `50`
- FP16: enabled
- Gradient checkpointing: enabled
- Effective batch:
  - `per_device_train_batch_size=8`
  - `gradient_accumulation_steps=2`

## Outputs
- Training artifacts/checkpoints in: `qa-gpt2-lora/`
- Console logs include:
- training duration
- final training loss
- total global steps

## Notes
- `device_map="auto"` is important here for quantized loading and avoiding device mismatch issues.
- Current setup is optimized for faster experimentation with subset data; increase dataset size for stronger final performance.
