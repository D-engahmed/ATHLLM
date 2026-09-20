# ATHLLM Frontier Training Plan

## Foundation
Start with the Spark-X2.5 architectural hypothesis: three sliding-window layers followed by one full-attention layer. Validate tensor compatibility before any weight conversion.

## Corpus target: 5T tokens
5T is a target capacity. It is not evidence of quality. Every sample needs provenance, license status, language/domain labels, deduplication, semantic-quality scoring, contamination checks, and source-level mixture controls. Evaluation sets are immutable and excluded from training.

## RL-1: verifiable reasoning
Train math, science, logic and planning with rewards from exact solvers, symbolic checkers or independently reproducible verifiers where possible.

## RL-2: coding and agents
Use repositories, terminal environments, unit tests, tool calls and multi-step tasks. Reward successful execution and verified patches rather than textual preference alone.

## RL-3: self-improvement
Generate tasks, build environments, roll out solutions, independently verify them, mine failures and update the curriculum. Generated data must pass independent verification before entering training.

## Continuous improvement loop
Checkpoint → fixed evaluation → adversarial evaluation → failure mining → verified task generation → RL → regression evaluation → next checkpoint.

## Frontier comparison
Comparisons must pin model version, prompt, tool budget, context, decoding, benchmark version and evaluator version. No claim of superiority is accepted without reproducible evidence.

## Scaling
Validate the architecture at 4B first, then investigate 7–9B and MoE variants. Do not spend frontier-scale compute until the architecture and data pipeline survive ablation studies.


## Low-cost distillation path

For constrained hardware, do not run the 397B teacher on Kaggle. Use an expensive teacher offline to generate a verified hard-example dataset, then train ATHLLM-4B with distillation/SFT and verifiable RL. Start with a 50B-token student run, then 100B and 300B only if validation shows positive scaling. Quantize the validated student to int4 for Kaggle inference. See `docs/KAGGLE_DISTILLATION.md` and `configs/distillation_4b_kaggle.yaml`.