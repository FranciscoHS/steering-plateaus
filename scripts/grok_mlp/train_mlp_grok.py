#!/usr/bin/env python3
"""
Faithful replication of the MNIST-MLP grokking recipe from
AhmedImtiazPrio/grok-adversarial (train_mlp_mnist.py), stripped of the
local-complexity diagnostic logging (which is orthogonal to training and slow).

Recipe (matches config_base_mnist_mlp): depth-4 width-200 GeLU MLP, float64,
seed 0, 1000 random MNIST train points, MSE loss on one-hot targets, AdamW
lr=1e-3 weight_decay=0.01, batch_size=200, 1e6 optimization steps, init scale 1.0.

Saves periodic checkpoints (state_dict) so a grokked checkpoint can be picked
later, plus a metrics file with train/test acc/loss curves to verify grokking.
"""
import argparse
import math
import os
import random
import time
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torchvision


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


ACT = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid, 'GeLU': nn.GELU}


def build_mlp(depth, width, in_dims=(28, 28), n_classes=10, act='GeLU', init_scale=1.0):
    """Exactly the construction in train_mlp_mnist.py."""
    layers = [nn.Flatten()]
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(int(np.prod(in_dims)), width))
            layers.append(ACT[act]())
        elif i == depth - 1:
            layers.append(nn.Linear(width, n_classes))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(ACT[act]())
    mlp = nn.Sequential(*layers)
    with torch.no_grad():
        for p in mlp.parameters():
            p.data = init_scale * p.data
    return mlp


@torch.no_grad()
def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    N = min(len(dataset), N)
    batch_size = min(batch_size, N)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    correct, total = 0, 0
    for x, labels in islice(loader, N // batch_size):
        logits = network(x.to(device))
        correct += torch.sum(torch.argmax(logits, dim=1) == labels.to(device)).item()
        total += x.size(0)
    return correct / total


@torch.no_grad()
def compute_loss(network, dataset, device, one_hots, N=2000, batch_size=50):
    N = min(len(dataset), N)
    batch_size = min(batch_size, N)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss(reduction='sum')
    total, points = 0.0, 0
    for x, labels in islice(loader, N // batch_size):
        y = network(x.to(device))
        total += loss_fn(y, one_hots[labels]).item()
        points += len(labels)
    return total / points


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--width', type=int, default=200)
    p.add_argument('--train_points', type=int, default=1000)
    p.add_argument('--optimization_steps', type=int, default=1_000_000)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--init_scale', type=float, default=1.0)
    p.add_argument('--activation', type=str, default='GeLU')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', type=str, default='results/grok_mlp')
    p.add_argument('--data_root', type=str, default='./data')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- data (matches original sampling: randint with replacement after seeding) ---
    train_full = torchvision.datasets.MNIST(root=args.data_root, train=True,
        transform=torchvision.transforms.ToTensor(), download=True)
    train_idx = torch.randint(0, len(train_full), (args.train_points,))
    train = torch.utils.data.Subset(train_full, train_idx)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test = torchvision.datasets.MNIST(root=args.data_root, train=False,
        transform=torchvision.transforms.ToTensor(), download=True)

    mlp = build_mlp(args.depth, args.width, act=args.activation,
                    init_scale=args.init_scale).to(device)
    print(f"Model:\n{mlp}", flush=True)

    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    one_hots = torch.eye(10, 10).to(device)

    # checkpoint/log schedule mirrors the original's log-spacing
    log_freq = math.ceil(args.optimization_steps / 150)

    def is_log_step(s):
        return (s < 30 or (s < 150 and s % 10 == 0) or (s < 1000 and s % 100 == 0)
                or (s < 10000 and s % 1000 == 0) or s % log_freq == 0)

    metrics = {'log_steps': [], 'train_loss': [], 'test_loss': [],
               'train_acc': [], 'test_acc': []}

    # save train_idx so the perturbation analysis can reuse the exact train set
    torch.save({'train_idx': train_idx, 'args': vars(args)},
               os.path.join(args.out_dir, 'meta.pt'))

    t0 = time.time()
    steps = 0
    for x, labels in islice(cycle(train_loader), args.optimization_steps):
        if is_log_step(steps):
            mlp.eval()
            tr_l = compute_loss(mlp, train, device, one_hots, N=len(train))
            te_l = compute_loss(mlp, test, device, one_hots, N=len(test))
            tr_a = compute_accuracy(mlp, train, device, N=len(train))
            te_a = compute_accuracy(mlp, test, device, N=len(test))
            metrics['log_steps'].append(steps)
            metrics['train_loss'].append(tr_l)
            metrics['test_loss'].append(te_l)
            metrics['train_acc'].append(tr_a)
            metrics['test_acc'].append(te_a)
            torch.save({'model_state_dict': mlp.state_dict(), 'step': steps,
                        'depth': args.depth, 'width': args.width},
                       os.path.join(ckpt_dir, f'ckpt_s{steps}.pt'))
            torch.save(metrics, os.path.join(args.out_dir, 'metrics.pt'))
            print(f"step {steps:>7} | train L {tr_l:.2e} acc {tr_a*100:5.1f}% | "
                  f"test L {te_l:.2e} acc {te_a*100:5.1f}% | {time.time()-t0:.0f}s",
                  flush=True)
            mlp.train()

        opt.zero_grad()
        y = mlp(x.to(device))
        loss = loss_fn(y, one_hots[labels])
        loss.backward()
        opt.step()
        steps += 1

    # final checkpoint + metrics
    torch.save({'model_state_dict': mlp.state_dict(), 'step': steps,
                'depth': args.depth, 'width': args.width},
               os.path.join(ckpt_dir, f'ckpt_s{steps}_final.pt'))
    torch.save(metrics, os.path.join(args.out_dir, 'metrics.pt'))
    print(f"Done in {time.time()-t0:.0f}s. Final test acc "
          f"{metrics['test_acc'][-1]*100:.1f}%", flush=True)


if __name__ == '__main__':
    main()
