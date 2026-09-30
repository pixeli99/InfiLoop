import argparse
import json
from pathlib import Path

import torch

from data import load_split
from model import Config, InfiLoop
from run import environment, precision, tensors


def valid_completion(predictions, inputs):
    board = predictions.reshape(-1, 9, 9)
    digits = torch.arange(2, 11, device=board.device)
    rows = board.sort(-1).values.eq(digits).all(-1).all(-1)
    columns = board.transpose(1, 2).sort(-1).values.eq(digits).all(-1).all(-1)
    boxes = board.reshape(-1, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    boxes = boxes.sort(-1).values.eq(digits).all(-1).all(-1)
    clues = ((inputs == 1) | predictions.eq(inputs)).all(-1)
    return rows & columns & boxes & clues


@torch.no_grad()
def evaluate(args):
    device, dtype = environment(args)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = InfiLoop(Config(**checkpoint['config']), dtype).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    depths = json.loads(Path(args.depths).read_text())
    if not isinstance(depths, list) or not depths:
        raise ValueError('depths must be a nonempty list')
    for setting in depths:
        if set(setting) != {'h_cycles', 'l_cycles', 'segments'} or any(type(v) is not int or v < 1 for v in setting.values()):
            raise ValueError('Each depth must contain positive integer h_cycles, l_cycles, and segments')
    x, y = load_split(args.data, 'test')
    count = min(args.limit, len(x)) if args.limit else len(x)
    solved = cells = accepted_total = total_layers = 0
    reports = [{**depth, 'evaluated': 0, 'accepted': 0} for depth in depths]
    for start in range(0, count, args.batch_size):
        inputs, labels = tensors(x[start:min(start + args.batch_size, count)], y[start:min(start + args.batch_size, count)], device)
        selected = torch.ones_like(inputs)
        remaining = torch.arange(len(inputs), device=device)
        for index, depth in enumerate(depths):
            if not len(remaining):
                break
            model.config.h_cycles, model.config.l_cycles = depth['h_cycles'], depth['l_cycles']
            state = None
            with precision(device, dtype):
                for _ in range(depth['segments']):
                    logits, state = model(inputs[remaining], state)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite evaluation logits')
            predictions = logits.argmax(-1)
            selected[remaining] = predictions
            accepted = valid_completion(predictions, inputs[remaining])
            n_accepted = accepted.sum().item()
            reports[index]['evaluated'] += len(remaining)
            reports[index]['accepted'] += n_accepted
            accepted_total += n_accepted
            total_layers += len(remaining) * depth['segments'] * depth['h_cycles'] * (depth['l_cycles'] + 1) * model.config.layers
            remaining = remaining[~accepted]
        correct = selected.eq(labels)
        solved += correct.all(-1).sum().item()
        cells += correct.sum().item()
        print(f'{min(start + args.batch_size, count)}/{count}', flush=True)
    result = dict(examples=count, exact_accuracy=solved / count, cell_accuracy=cells / (81 * count),
                  accepted=accepted_total, acceptance_rate=accepted_total / count,
                  mean_transformer_layer_evaluations=total_layers / count, depths=reports,
                  checkpoint_step=checkpoint['step'], dtype=str(dtype), device=str(device))
    print(json.dumps(result, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/sudoku')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--depths', default=str(Path(__file__).with_name('depths.json')))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=('auto', 'float32', 'bfloat16'), default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output')
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.limit < 0:
        parser.error('batch-size and threads must be positive; limit must be nonnegative')
    evaluate(args)
