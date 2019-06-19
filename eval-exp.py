import json
import os

import torch
import torch.optim as optim
from sacred import Experiment
from sacred.observers import MongoObserver

from ingredients.attacks import attacks_ingredient, pgd_linf
from ingredients.data_loader import data_ingredient, load_cifar10
from ingredients.test import adv_quantize_test, test, test_ingredient
from ingredients.train import train, train_ingredient
from model.resnet import resnet20

ex = Experiment('bitdepth-eval', ingredients=[data_ingredient,
                                              attacks_ingredient,
                                              train_ingredient,
                                              test_ingredient])
ex.add_config('config.json')
# ex.observers.append(MongoObserver.create())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@ex.automain
def run(data_dir, trained_models_dir, momentum, weight_decay, results_dir,
        bit, dithering):
    # check data and trained-models dir
    if not os.path.exists(data_dir):
        os.makedir(data_dir)
    if not os.path.exists(trained_models_dir):
        os.makedir(trained_models_dir)
    if not os.path.exists(results_dir):
        os.makedir(results_dir)

    train_loader, test_loader = load_cifar10()

    test_errs, test_dither_errs = [], []
    adv_test_errs = []
    adv_test_dither_errs = []

    if dithering:
        filename = f'resnet20-{bit}bit-dither.pt'
    else:
        filename = f'resnet20-{bit}bit.pt'
    path = os.path.join(trained_models_dir, filename)
    model = resnet20().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    for i in range(1, 9):
        test_err, _ = test(model=model, loader=test_loader,
                           bitdepth=i, dither=False)
        adv_err = adv_quantize_test(model=model, loader=test_loader,
                                    attack=pgd_linf, bitdepth=i)
        test_errs.append(test_err)
        adv_test_errs.append(adv_err)

    for i in range(1, 8):
        test_err, _ = test(model=model, loader=test_loader,
                           bitdepth=i, dither=True)
        adv_err = adv_quantize_test(model=model, loader=test_loader,
                                    attack=pgd_linf, bitdepth=i, dither=True)
        test_dither_errs.append(test_err)
        adv_test_dither_errs.append(adv_err)

    # save the results in json
    results = {}
    results['test_errors'] = test_errs
    results['test_dither_errors'] = test_dither_errs
    results['adv_test_errors'] = adv_test_errs
    results['adv_test_dither_errors'] = adv_test_dither_errs

    if dithering:
        filename = f'bit{bit}-dither.json'
    else:
        filename = f'bit{bit}.json'
    path = os.path.join(results_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
