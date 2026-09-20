# ATHLLM

ATHLLM is an open research foundation for an Arabic-English, reasoning-first, coding-capable, tool-using language model.

## Spark-X2.5 foundation

This branch adds a Spark-X2.5-inspired 4B-class foundation. The official Spark-X2.5 documentation describes a hybrid attention design using three sliding-window layers followed by one full-attention layer, with native context up to 1M tokens. The implementation is isolated from checkpoint loading so ATHLLM can evolve independently.

Run the smoke test with: `python -m athllm.models.spark25`

Load an authorized upstream checkpoint with: `python -m athllm.tools.load_spark --model XHToken/Spark-X2.5-4B`

Weights are not committed to GitHub; the loader obtains them locally through Transformers.

## 5T-token semantic training pipeline

The target is a 5T-token training capacity, not a claim that 5T raw tokens are high quality. The corpus pipeline is designed around provenance/licensing, language and domain balancing, exact and near deduplication, semantic quality scoring, spam/boilerplate rejection, code execution checks, benchmark contamination exclusion, and immutable manifests.

Start with: `python -m athllm.data.build_manifest --input ./data/raw --output ./data/manifests/train.jsonl`

## Three RL tracks

1. Verifiable reasoning RL — math, logic, science and planning with executable or independently checkable rewards.
2. Coding/agent RL — repository repair, terminal tasks, tool calling and multi-step coding rewarded by task outcomes.
3. Self-improvement RL — verified task generation, rollout, independent verification, failure mining and curriculum updates.

The self-improvement track never promotes model-generated examples solely because the model generated them.

## Frontier evaluation

The objective is to continuously close measured gaps against strong frontier and open models. The repository does not make an unverified claim that ATHLLM beats frontier models. Every comparison should pin model/version, prompt/template, tool budget, context length, decoding parameters, benchmark version and evaluator version.

See `docs/TRAINING_PLAN.md` for the research loop.
