# Low-Cost ATHLLM Distillation Plan

## Objective

Turn expensive frontier-teacher capability into a small ATHLLM student that can be trained, quantized and evaluated on constrained GPU hardware.

## Do not run the 397B teacher on Kaggle

Use the large teacher **offline** to produce a small, high-value dataset. Store only approved examples and metadata.

## Teacher hierarchy

1. Use the strongest available teacher for the hardest 5–10% of tasks.
2. Use a cheaper strong teacher for routine tasks.
3. Use deterministic verifiers for filtering.

The teacher is a data generator, not the final deployed model.

## Dataset funnel

10M candidate tasks
→ cheap filtering
→ 1M hard candidates
→ teacher generation
→ independent verification
→ 200k–500k exceptional examples
→ student SFT/distillation
→ reasoning RL + coding RL
→ final int4 quantization.

The exact counts are starting budgets, not guarantees.

## Three training stages

### 1. Distillation

Distill verified teacher responses and, where supported, token/logit targets.

### 2. Verifiable RL

Reasoning receives reward from exact or reproducible verifiers.

Coding receives reward from actual test execution, patch correctness and task completion.

### 3. Self-improvement

Generate new tasks, execute them in isolated environments, independently verify outcomes, retain failures/successes for curriculum construction, and train only on verified data.

## Why this is cheaper

The expensive model generates data once. The inexpensive ATHLLM model receives repeated training iterations on the resulting dataset. Kaggle is used for the student experiments rather than hosting the frontier teacher.

## Quantization

Use QLoRA/int4 for constrained training and an int4 inference checkpoint after validation. Quantization must be evaluated against BF16/FP16 checkpoints to measure capability loss.

## Scaling

Start at 50B training tokens. Increase to 100B and 300B only when validation demonstrates positive scaling. Do not commit to 5T until smaller-scale scaling laws justify it.
