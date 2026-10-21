# InfiLoop on Sudoku-Extreme

Data preparation, training, and evaluation for Sudoku-Extreme. Requires Python 3.11 or 3.12, PyTorch, NumPy, and Hugging Face Hub.

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

Downloads [Sudoku-Extreme](https://huggingface.co/datasets/sapientinc/sudoku-extreme), approximately 0.8 GB. Training uses 1,000 base puzzles, each stored with its original board and 1,000 augmentations (digit relabeling, band, stack, row, and column shuffles, and transposition). Evaluation uses all 422,786 test puzzles. Add `--source-dir /path/to/csv` to read local `train.csv` and `test.csv` files. Datasets built with the HRM or TRM builders (`all__inputs.npy`, `all__labels.npy`, `all__group_indices.npy`) load directly.

## Pretrained checkpoint

```bash
hf download pengxiang/InfiLoop-Sudoku model.pt --local-dir checkpoints
```

The EMA checkpoint at step 71,610 used for the paper's Sudoku-Extreme results. It loads directly into `run.py evaluate` and `verify.py`.

## Train

```bash
python run.py train --data data/sudoku --output runs/sudoku \
  --batch-size 768 --micro-batch-size 32 --steps 78125
```

The 6.83M-parameter model uses width 512, two shared Transformer layers, `H=3`, `L=6`, and 12 supervision segments. Training uses AdamW with learning rate `1e-4`, weight decay 1.0, 2,000 warmup steps, cosine decay, and EMA 0.999. Microbatches accumulate to `--batch-size` before each optimizer update; lower `--micro-batch-size` to fit device memory.

CUDA and BF16 are selected when available; use `--device cpu` or `--dtype float32` to override. Each run writes `config.json`, `train.jsonl`, the latest EMA checkpoint `model.pt`, and a numbered EMA checkpoint every 6,510 steps. The paper evaluates the checkpoint at step 71,610.

## Fixed-depth evaluation

```bash
python run.py evaluate --data data/sudoku --checkpoint checkpoints/model.pt \
  --h-cycles 6 --l-cycles 12 --segments 12 --output runs/sudoku/fixed.json
```

Reports exact-board and cell accuracy from the raw argmax predictions. A run executes `segments * H * (L + 1) * layers` Transformer layers per puzzle: 1,872 for the command above and 24,960 with `--segments 160`. Add `--limit 32` for a quick check.

## Evaluation across depths

```bash
python verify.py --data data/sudoku --checkpoint checkpoints/model.pt \
  --output runs/sudoku/verified.json
```

This is the procedure behind the paper's 97.9% Sudoku-Extreme result. The model runs the 39 depth settings in `depths.json` from shallow to deep, with 12 segments per setting, each starting from the initial state. A puzzle takes the first prediction that keeps all clues and satisfies every row, column, and 3×3 box. Ground-truth labels are used only for scoring. A puzzle that no setting accepts keeps the prediction of the deepest setting. The output records the acceptance count of each setting and the mean number of Transformer layers used per puzzle.

## CPU smoke check

```bash
python data.py --output data/smoke --train-size 16 --augmentations 2 --test-size 8
python run.py train --data data/smoke --output runs/smoke --device cpu \
  --hidden-size 64 --num-heads 4 --h-cycles 2 --l-cycles 2 \
  --batch-size 4 --micro-batch-size 2 --segments 2 --steps 3 --warmup 0
python run.py evaluate --data data/smoke --checkpoint runs/smoke/model.pt \
  --device cpu --batch-size 2 --segments 2 --limit 8
```

A three-step run of a small model that exercises data preparation, training, checkpointing, and evaluation.

## License

MIT. Built on the TRM and HRM codebases; see [LICENSE](LICENSE).
