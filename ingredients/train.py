import os

import torch
import torch.nn as nn
import torch.optim as optim
from sacred import Ingredient

from ingredients.quantize import batch_dither, quantize

train_ingredient = Ingredient('train')
train_ingredient.add_config('config.json')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@train_ingredient.capture
def train(trained_models_dir, epochs, lr_steps, model, loader, _run,
          momentum, weight_decay, opt, model_name, bitdepth=8, dither=False):

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_steps, gamma=0.1)
    n_batches = len(loader)
    for epoch in range(epochs):
        total_loss, total_err = 0.0, 0.0
        for i, (X, y) in enumerate(loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if bitdepth != 8 and dither is False:
                X = quantize(X, bitdepth)
            elif bitdepth != 8 and dither is True:
                X = batch_dither(X, bitdepth)

            X = X.to(device, non_blocking=True)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            _run.log_scalar('loss', float(loss.data))
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.6f}"
                      .format(epoch+1, epochs, i+1, n_batches, loss.item()))

        scheduler.step()

    if dither:
        filename = f'{model_name}-{bitdepth}bit-dither.pt'
    else:
        filename = f'{model_name}-{bitdepth}bit.pt'

    path = os.path.join(trained_models_dir, filename)
    torch.save(model.state_dict(), path)
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


@train_ingredient.capture
def adv_train(trained_models_dir, epochs, lr_steps, model, loader, _run,
              momentum, weight_decay, opt, model_name, attack):

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_steps, gamma=0.1)
    n_batches = len(loader)
    for epoch in range(epochs):
        total_loss, total_err = 0.0, 0.0
        for i, (X, y) in enumerate(loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            X = attack(model=model, X=X, y=y)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            _run.log_scalar('loss', float(loss.data))
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.6f}"
                      .format(epoch+1, epochs, i+1, n_batches, loss.item()))

        scheduler.step()

    filename = f'adv-trained-{model_name}.pt'
    path = os.path.join(trained_models_dir, filename)
    torch.save(model.state_dict(), path)
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)
