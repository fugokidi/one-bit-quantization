import os

import torch
import torch.nn as nn
from sacred import Ingredient
from tqdm import tqdm

from ingredients.quantize import batch_dither, quantize

test_ingredient = Ingredient('test')
test_ingredient.add_config('config.json')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@test_ingredient.capture
def test(model, loader, bitdepth=8, dither=False):
    total_loss, total_err = 0.0, 0.0
    iterator = tqdm(loader)
    with torch.no_grad():
        for X, y in iterator:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if bitdepth != 8 and dither is False:
                X = quantize(X, bitdepth)
            elif bitdepth != 8 and dither is True:
                X = batch_dither(X, bitdepth)

            X = X.to(device, non_blocking=True)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            total_err += (yp.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * X.shape[0]
            iterator.set_description(
                "{}bit Current Loss: {}".format(bitdepth, total_loss))
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


@test_ingredient.capture
def adv_test(model, loader, attack):
    total_err = 0.0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        X = attack(model=model, X=X, y=y)
        yp = model(X)
        total_err += (yp.max(dim=1)[1] == y).sum().item()
    return total_err / len(loader.dataset)


@test_ingredient.capture
def adv_quantize_test(model, loader, attack, bitdepth=8, dither=False):
    total_err = 0.0
    eps = 8 / 255
    step_size = 2 / 255
    iterator = tqdm(loader)
    for X, y in iterator:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if bitdepth != 8 and dither is True:
            X = batch_dither(X, bitdepth)
        elif bitdepth != 8 and dither is False:
            X = quantize(X, bitdepth)

        X = X.to(device, non_blocking=True)

        X = attack(model=model, inputs=X, targets=y, eps=eps,
                   step_size=step_size)

        # X = 2 * (torch.rand_like(X) - 0.5) * (8/255)
        # X = torch.clamp(X, 0, 1)

        if bitdepth != 8 and dither is True:
            X = batch_dither(X, bitdepth)
        elif bitdepth != 8 and dither is False:
            X = quantize(X, bitdepth)

        X = X.to(device, non_blocking=True)

        yp = model(X)
        total_err += (yp.max(dim=1)[1] == y).sum().item()
        iterator.set_description("{}bit Adv Test:".format(bitdepth))
    return total_err / len(loader.dataset)


@test_ingredient.capture
def adv_different_epsilon(model, loader, attack, bitdepth=8, dither=False):
    epsilons = [2, 4, 8, 16, 24, 32]
    step_size = 2 / 255
    accuracy = []
    for eps in epsilons:
        total_err = 0.0
        eps = eps / 255
        iterator = tqdm(loader)
        for X, y in iterator:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if bitdepth != 8 and dither is True:
                X = batch_dither(X, bitdepth)
            elif bitdepth != 8 and dither is False:
                X = quantize(X, bitdepth)

            X = X.to(device, non_blocking=True)

            X = attack(model=model, inputs=X, targets=y, eps=eps,
                       step_size=step_size)

            if bitdepth != 8 and dither is True:
                X = batch_dither(X, bitdepth)
            elif bitdepth != 8 and dither is False:
                X = quantize(X, bitdepth)

            X = X.to(device, non_blocking=True)

            yp = model(X)
            total_err += (yp.max(dim=1)[1] == y).sum().item()
            iterator.set_description(
                "{}bit: {} epsilon:".format(bitdepth, eps))
        accuracy.append(total_err / len(loader.dataset))
    return accuracy
