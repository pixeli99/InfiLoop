# InfiLoop on Sudoku-Extreme

Standalone data preparation, training, and evaluation. Python 3.11 or 3.12; PyTorch, NumPy, and Hugging Face Hub. Pretrained weights are not included.

## Setup

```bash
cd sudoku
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

```bash
python data.py --output data/sudoku --train-size 1000 --augmentations 1000
```

Downloads [Sudoku-Extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme), approximately 0.8 GB. Training uses 1,000 base puzzles, each stored with its original board and 1,000 augmentations. Evaluation uses all 422,786 test puzzles. Add `--source-dir /path/to/csv` to use local `train.csv` and `test.csv` files.

Preparation selects the training subset deterministically with seed 0. It follows the paper's data recipe; it does not reproduce the exact original subset or augmentation order. Previously prepared `all__inputs.npy`, `all__labels.npy`, and training `all__group_indices.npy` files are also supported.

## Train

```bash
python run.py train --data data/sudoku --output runs/sudoku \
  --batch-size 768 --micro-batch-size 32 --steps 78125
```

The 6.83M-parameter model uses width 512, two shared Transformer layers, `H=3`, `L=6`, and 12 supervision segments. Training uses AdamW, learning rate `1e-4`, weight decay 1.0, 2,000 warmup steps, cosine decay, and EMA 0.999. Microbatches accumulate to the specified batch size before each optimizer update; reduce `--micro-batch-size` to fit device memory.

CUDA and BF16 are selected when supported. Use `--device cpu` or `--dtype float32` to override. Each run writes `config.json`, `train.jsonl`, the latest EMA checkpoint `model.pt`, and numbered checkpoints every 6,510 steps. Use a new output directory for each run. Checkpoints contain inference weights; training resume is not implemented.

The original paper model was evaluated from the EMA checkpoint at step 71,610. This standalone training loop has not been retrained to the reported accuracy; matching the architecture and nominal budget alone does not guarantee the same result.

## Fixed-depth evaluation

```bash
python run.py evaluate --data data/sudoku --checkpoint runs/sudoku/model.pt \
  --h-cycles 6 --l-cycles 12 --segments 12 --output runs/sudoku/fixed.json
```

Reports exact-board and cell accuracy as fractions. Evaluation uses raw argmax predictions with no validity filtering. The executed Transformer layer count is `segments * H * (L + 1) * layers`: the command above uses 1,872 layers; `--segments 160` uses 24,960. Add `--limit 32` for a short check.

## Evaluation across depths

```bash
python verify.py --data data/sudoku --checkpoint runs/sudoku/step_71610.pt \
  --output runs/sudoku/verified.json
```

The paper's 97.9% Sudoku result combines predictions across the 39 depth settings in `depths.json`, with 12 segments per setting. Each setting restarts from the model's initial state. Accept the first prediction that preserves all clues and satisfies every row, column, and 3×3 box; ground-truth labels are used only for final scoring. Unaccepted puzzles use the last setting's prediction. The output records each setting's acceptance count and the mean layer count actually used.

This implements the multi-depth acceptance procedure. It has not been validated on the full test set with this standalone release, and 97.9% is not the expected output of the fixed-depth command above. The 39-setting sweep is substantially more expensive than one fixed-depth evaluation.

## CPU smoke check

```bash
python data.py --output data/smoke --train-size 16 --augmentations 2 --test-size 8
python run.py train --data data/smoke --output runs/smoke --device cpu \
  --hidden-size 64 --num-heads 4 --h-cycles 2 --l-cycles 2 \
  --batch-size 4 --micro-batch-size 2 --segments 2 --steps 3 --warmup 0
python run.py evaluate --data data/smoke --checkpoint runs/smoke/model.pt \
  --device cpu --batch-size 2 --segments 2 --limit 8
```

This checks execution and checkpoint loading, not model quality. Data download size is unchanged. Based on the project's TRM/FPRM implementation; see [LICENSE](LICENSE).
