import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data import load_split, train_batches
from model import Config, InfiLoop, loss_function


def environment(args):
    torch.set_num_threads(args.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    dtype = args.dtype
    if dtype == 'auto':
        dtype = 'bfloat16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'float32'
    return device, getattr(torch, dtype)


def precision(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype == torch.bfloat16 else nullcontext()


def tensors(x, y, device):
    return (torch.as_tensor(np.array(x, dtype=np.int64), device=device),
            torch.as_tensor(np.array(y, dtype=np.int64), device=device))


def save_model(path, model, ema, step):
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    state.update({name: value.detach().cpu().clone() for name, value in ema.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(dict(config=asdict(model.config), model=state, step=step), temporary)
    temporary.replace(path)


def train(args):
    device, dtype = environment(args)
    torch.manual_seed(args.seed)
    config = Config(args.hidden_size, args.num_heads, args.layers, args.h_cycles, args.l_cycles)
    model = InfiLoop(config, dtype).to(device).train()
    parameters = {name: value for name, value in model.named_parameters() if name != 'prefix'}
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1.0)
    ema = {name: value.detach().clone() for name, value in parameters.items()}
    batches = train_batches(args.data, args.batch_size, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'model.pt').exists() or (output / 'train.jsonl').exists():
        raise FileExistsError(f'{output} already contains a run; choose another output directory')
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    train_x, _ = load_split(args.data, 'train')
    groups = np.load(Path(args.data) / 'train/all__group_indices.npy')
    print(json.dumps(dict(device=str(device), dtype=str(dtype), parameters=sum(p.numel() for p in model.parameters()),
                          train_examples=len(train_x), train_base_puzzles=len(groups) - 1,
                          batch_size=args.batch_size, micro_batch_size=min(args.micro_batch_size, args.batch_size),
                          steps=args.steps, segments=args.segments)), flush=True)
    states = []
    with (output / 'train.jsonl').open('w') as log:
        for step in range(1, args.steps + 1):
            if (step - 1) % args.segments == 0:
                x, y = next(batches)
                states = [None] * math.ceil(len(x) / args.micro_batch_size)
            optimizer.zero_grad(set_to_none=True)
            model.prefix.grad = None
            loss_value = solved = cells = 0
            for index, start in enumerate(range(0, len(x), args.micro_batch_size)):
                inputs, labels = tensors(x[start:start + args.micro_batch_size], y[start:start + args.micro_batch_size], device)
                with precision(device, dtype):
                    logits, states[index] = model(inputs, states[index])
                    loss = loss_function(logits, labels) * (len(inputs) / len(x))
                loss.backward()
                loss_value += loss.item()
                correct = logits.detach().argmax(-1).eq(labels)
                solved += correct.all(-1).sum().item()
                cells += correct.sum().item()
            if not math.isfinite(loss_value) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Non-finite loss or gradient at step {step}')
            if step < args.warmup:
                multiplier = step / max(1, args.warmup)
            else:
                progress = (step - args.warmup) / max(1, args.steps - args.warmup)
                multiplier = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = args.lr * multiplier
            with torch.no_grad():
                prefix_lr = args.prefix_lr * multiplier
                model.prefix.mul_(1 - prefix_lr).add_(model.prefix.grad.sign(), alpha=-prefix_lr)
            optimizer.step()
            with torch.no_grad():
                for name, value in parameters.items():
                    ema[name].mul_(0.999).add_(value, alpha=0.001)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                record = dict(step=step, loss=loss_value, exact_accuracy=solved / len(x),
                              cell_accuracy=cells / (81 * len(x)), lr=args.lr * multiplier)
                text = json.dumps(record)
                print(text, flush=True)
                log.write(text + '\n')
                log.flush()
            if step % args.save_every == 0 or step == args.steps:
                save_model(output / 'model.pt', model, ema, step)
                save_model(output / f'step_{step}.pt', model, ema, step)


@torch.no_grad()
def evaluate(args):
    device, dtype = environment(args)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    settings = checkpoint['config'].copy()
    for name in ('h_cycles', 'l_cycles'):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    model = InfiLoop(Config(**settings), dtype).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    x, y = load_split(args.data, 'test')
    count = min(args.limit, len(x)) if args.limit else len(x)
    solved = cells = 0
    for start in range(0, count, args.batch_size):
        inputs, labels = tensors(x[start:min(start + args.batch_size, count)], y[start:min(start + args.batch_size, count)], device)
        state = None
        with precision(device, dtype):
            for _ in range(args.segments):
                logits, state = model(inputs, state)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite evaluation logits')
        correct = logits.argmax(-1).eq(labels)
        solved += correct.all(-1).sum().item()
        cells += correct.sum().item()
        print(f'{min(start + args.batch_size, count)}/{count}', flush=True)
    result = dict(examples=count, exact_accuracy=solved / count, cell_accuracy=cells / (81 * count),
                  segments=args.segments, h_cycles=model.config.h_cycles, l_cycles=model.config.l_cycles,
                  transformer_layer_evaluations=args.segments * model.config.h_cycles * (model.config.l_cycles + 1) * model.config.layers,
                  checkpoint_step=checkpoint['step'], dtype=str(dtype), device=str(device))
    print(json.dumps(result, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    evaluation = commands.add_parser('evaluate')
    for command in (training, evaluation):
        command.add_argument('--data', default='data/sudoku')
        command.add_argument('--batch-size', type=int, default=32)
        command.add_argument('--segments', type=int, default=12)
        command.add_argument('--device', default='auto')
        command.add_argument('--dtype', choices=('auto', 'float32', 'bfloat16'), default='auto')
        command.add_argument('--threads', type=int, default=4)
    training.add_argument('--output', default='runs/sudoku')
    training.set_defaults(batch_size=768)
    training.add_argument('--steps', type=int, default=78125)
    training.add_argument('--micro-batch-size', type=int, default=32)
    training.add_argument('--hidden-size', type=int, default=512)
    training.add_argument('--num-heads', type=int, default=8)
    training.add_argument('--layers', type=int, default=2)
    training.add_argument('--h-cycles', type=int, default=3)
    training.add_argument('--l-cycles', type=int, default=6)
    training.add_argument('--lr', type=float, default=1e-4)
    training.add_argument('--prefix-lr', type=float, default=1e-3)
    training.add_argument('--warmup', type=int, default=2000)
    training.add_argument('--seed', type=int, default=991231)
    training.add_argument('--log-every', type=int, default=10)
    training.add_argument('--save-every', type=int, default=6510)
    evaluation.add_argument('--checkpoint', required=True)
    evaluation.add_argument('--h-cycles', type=int)
    evaluation.add_argument('--l-cycles', type=int)
    evaluation.add_argument('--limit', type=int, default=0)
    evaluation.add_argument('--output')
    args = parser.parse_args()
    for key in ('batch_size', 'micro_batch_size', 'segments', 'threads', 'steps', 'log_every', 'save_every', 'h_cycles', 'l_cycles'):
        value = getattr(args, key, None)
        if value is not None and value < 1:
            parser.error(f'{key} must be positive')
    if args.command == 'train' and (args.warmup < 0 or args.lr <= 0 or args.prefix_lr <= 0):
        parser.error('Learning rates must be positive and warmup must be nonnegative')
    if args.command == 'evaluate' and args.limit < 0:
        parser.error('limit must be nonnegative')
    train(args) if args.command == 'train' else evaluate(args)


if __name__ == '__main__':
    main()
