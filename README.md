# SE-MedVLM

SE-MedVLM is a training-free medical VQA framework that builds and reuses a self-evolving medical knowledge base during visual question answering.

The default VLM for the open-source version is `Qwen3-VL-235B-A22B`, accessed through an OpenAI-compatible API. LLM-as-judge evaluation uses `Qwen3-Max`, also through API calls.

## Repository Layout

- `train.py`: builds and updates the tag-based medical knowledge base.
- `evaluate.py`: evaluates a model with or without the generated knowledge base.
- `retrieval/`: tag taxonomy, knowledge-base storage, retrieval, reranking, classification, and answer evaluation logic.
- `utils/`: dataset loaders, prompts, inference wrappers, and reward parsing helpers.
- `scripts/`: minimal runnable shell entry point for API evaluation.
- `examples/`: small utilities for inspecting processed datasets.

Large local assets are intentionally not tracked: datasets, logs, checkpoints, generated knowledge bases, result folders, and PDFs.

## Setup

```bash
conda create -n se-medvlm python=3.10 -y
conda activate se-medvlm
pip install -r requirements.txt
```

## API Configuration

The code does not store API keys in source files. Set keys through environment variables:

```bash
export DASHSCOPE_API_KEY="your_api_key"
export OPENAI_API_KEY="your_api_key"
```

`--ak env:VAR_NAME` reads a key from any environment variable. The aliases `skd` and `dashscope` resolve to `DASHSCOPE_API_KEY`; `openai` resolves to `OPENAI_API_KEY`.

By default, `--model` and `--cm` use `Qwen3-VL-235B-A22B` for VQA and retrieval classification, while `--em` uses `Qwen3-Max` as the LLM-as-judge.

## Train Knowledge Base

```bash
python train.py \
  --infer_mode api \
  --en slake \
  --at closed \
  --ct all \
  --dt 300 \
  --model Qwen3-VL-235B-A22B \
  --em Qwen3-Max \
  --cm Qwen3-VL-235B-A22B \
  --ak env:DASHSCOPE_API_KEY \
  --bu https://dashscope.aliyuncs.com/compatible-mode/v1
```

Generated knowledge bases are written under `knowledges/`, with logs under `logs/`.

## Evaluate

```bash
bash scripts/run_api_eval.sh
```

or explicitly:

```bash
python evaluate.py \
  --infer_mode api \
  --en slake \
  --at closed \
  --ct all \
  --dt 1000 \
  --model Qwen3-VL-235B-A22B \
  --em Qwen3-Max \
  --cm Qwen3-VL-235B-A22B \
  --ak env:DASHSCOPE_API_KEY \
  --bu https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --kb knowledges/slake_tag_kb_Qwen3-VL-235B-A22B.pkl
```

Use `--gk false` to evaluate without the external knowledge base.

## Dataset Check

```bash
python examples/inspect_dataset.py --dataset slake --split test --truncate 5
```

Dataset loaders expect local datasets under `datasets/`. Downloading and preprocessing may require dataset-specific access permissions.

## Notes

- The reported `Qwen3-VL-235B-A22B` setting is API-based; this repository does not include local deployment scripts or model weights for that model.
- This repository does not include private review material, generated logs, or evaluation outputs.
