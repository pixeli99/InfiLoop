import argparse
import csv
from pathlib import Path

import numpy as np


def read_csv(path, count, seed, random_subset):
    rng = np.random.default_rng(seed)
    rows = []
    with open(path, newline='') as handle:
        reader = csv.reader(handle)
        next(reader)
        for index, (_, puzzle, solution, _) in enumerate(reader):
            if count and not random_subset and index >= count:
                break
            if len(puzzle) != 81 or len(solution) != 81:
                raise ValueError(f'Invalid board on CSV row {index + 2}')
            row = (puzzle.replace('.', '0'), solution)
            if not count or len(rows) < count:
                rows.append(row)
            else:
                position = rng.integers(index + 1)
                if position < count:
                    rows[position] = row
    if not rows or (count and len(rows) < count):
        raise ValueError(f'{path} has fewer than {max(count, 1)} puzzles')
    values = np.frombuffer(''.join(x + y for x, y in rows).encode('ascii'), dtype=np.uint8).reshape(-1, 2, 81) - ord('0')
    if values.max() > 9 or values[:, 1].min() < 1:
        raise ValueError('Sudoku digits must be 0..9; solutions must be 1..9')
    return values[:, 0].reshape(-1, 9, 9), values[:, 1].reshape(-1, 9, 9)


def augment(puzzle, solution, rng):
    digits = np.concatenate(([0], rng.permutation(np.arange(1, 10))))
    rows = np.concatenate([3 * band + rng.permutation(3) for band in rng.permutation(3)])
    columns = np.concatenate([3 * stack + rng.permutation(3) for stack in rng.permutation(3)])
    if rng.random() < 0.5:
        puzzle, solution = puzzle.T, solution.T
    return digits[puzzle[np.ix_(rows, columns)]], digits[solution[np.ix_(rows, columns)]]


def prepare(args):
    output = Path(args.output)
    if (output / 'train').exists() or (output / 'test').exists():
        raise FileExistsError(f'{output} already contains a split; choose another output directory')
    rng = np.random.default_rng(args.seed)
    for split in ('train', 'test'):
        if args.source_dir:
            path = Path(args.source_dir) / f'{split}.csv'
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download('sapientinc/sudoku-extreme', f'{split}.csv', repo_type='dataset')
        count = args.train_size if split == 'train' else args.test_size
        inputs, labels = read_csv(path, count, args.seed, split == 'train')
        copies = 1 + args.augmentations if split == 'train' else 1
        directory = output / split
        directory.mkdir(parents=True)
        shape = (len(inputs) * copies, 81)
        x = np.lib.format.open_memmap(directory / 'all__inputs.npy', mode='w+', dtype=np.uint8, shape=shape)
        y = np.lib.format.open_memmap(directory / 'all__labels.npy', mode='w+', dtype=np.uint8, shape=shape)
        for i, (puzzle, solution) in enumerate(zip(inputs, labels)):
            for j in range(copies):
                px, py = (puzzle, solution) if j == 0 else augment(puzzle, solution, rng)
                x[i * copies + j], y[i * copies + j] = px.reshape(-1) + 1, py.reshape(-1) + 1
        x.flush()
        y.flush()
        if split == 'train':
            np.save(directory / 'all__group_indices.npy', np.arange(0, len(x) + 1, copies))
        print(f'{split}: {len(inputs)} base puzzles, {len(x)} stored examples', flush=True)


def load_split(path, split):
    directory = Path(path) / split
    x = np.load(directory / 'all__inputs.npy', mmap_mode='r')
    y = np.load(directory / 'all__labels.npy', mmap_mode='r')
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 81 or len(x) == 0:
        raise ValueError(f'Expected nonempty paired arrays of shape (N, 81) in {directory}')
    return x, y


def train_batches(path, batch_size, seed):
    x, y = load_split(path, 'train')
    boundaries = np.load(Path(path) / 'train' / 'all__group_indices.npy')
    if boundaries[0] != 0 or boundaries[-1] != len(x) or np.any(np.diff(boundaries) <= 0):
        raise ValueError('Invalid training group boundaries')
    rng = np.random.default_rng(seed)
    order = np.empty(0, dtype=np.int64)
    while True:
        while len(order) < batch_size:
            order = np.concatenate((order, rng.permutation(len(boundaries) - 1)))
        groups, order = order[:batch_size], order[batch_size:]
        indices = rng.integers(boundaries[groups], boundaries[groups + 1])
        yield x[indices], y[indices]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/sudoku')
    parser.add_argument('--source-dir')
    parser.add_argument('--train-size', type=int, default=1000)
    parser.add_argument('--augmentations', type=int, default=1000)
    parser.add_argument('--test-size', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.train_size < 1 or args.augmentations < 0 or args.test_size < 0:
        parser.error('train-size must be positive; augmentations and test-size must be nonnegative')
    prepare(args)
