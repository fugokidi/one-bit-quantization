import json
import os

import torch
import torch.optim as optim
from sacred import Experiment
from sacred.observers import MongoObserver

from ingredients.attacks import attacks_ingredient, pgd_linf
from ingredients.data_loader import data_ingredient, load_cifar10
from ingredients.test import adv_test, test, test_ingredient
from ingredients.train import train, train_ingredient
from model.resnet import resnet20

ex = Experiment('bitdepth-train', ingredients=[data_ingredient,
                                               attacks_ingredient,
                                               train_ingredient,
                                               test_ingredient])
ex.add_config('config.json')
ex.observers.append(MongoObserver.create())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@ex.automain
def run(data_dir, trained_models_dir, momentum, weight_decay, results_dir):
    # check data and trained-models dir
    if not os.path.exists(data_dir):
        os.makedir(data_dir)
    if not os.path.exists(trained_models_dir):
        os.makedir(trained_models_dir)
    if not os.path.exists(results_dir):
        os.makedir(results_dir)

    train_loader, test_loader = load_cifar10()

    # train 1-8bit quantization and with dithering
    train_errs, train_losses = [], []
    train_dither_errs, train_dither_losses = [], []
    for i in range(1, 9):
        model = resnet20().to(device)
        opt = optim.SGD(model.parameters(), lr=1e-1, momentum=momentum,
                        weight_decay=weight_decay)
        train_err, train_loss = train(model=model, loader=train_loader,
                                      bitdepth=i, opt=opt,
                                      model_name='resnet20')
        train_errs.append(train_err)
        train_losses.append(train_loss)

    for i in range(1, 8):
        model = resnet20().to(device)
        opt = optim.SGD(model.parameters(), lr=1e-1, momentum=momentum,
                        weight_decay=weight_decay)
        train_err, train_loss = train(model=model, loader=train_loader,
                                      bitdepth=i, dither=True,
                                      opt=opt, model_name='resnet20')
        train_dither_errs.append(train_err)
        train_dither_losses.append(train_loss)

    # save the results in json
    results = {}
    results['train_errors'] = train_errs
    results['train_losses'] = train_losses
    results['train_dither_errors'] = train_dither_errs
    results['train_dither_losses'] = train_dither_losses

    path = os.path.join(results_dir, 'train-results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
