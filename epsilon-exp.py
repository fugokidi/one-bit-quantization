import json
import os

import torch
import torch.optim as optim
from sacred import Experiment
from sacred.observers import MongoObserver

from ingredients.attacks import attacks_ingredient, pgd_linf
from ingredients.data_loader import data_ingredient, load_cifar10
from ingredients.test import adv_different_epsilon, test_ingredient
from model.resnet import resnet20

ex = Experiment('bitdepth-eval', ingredients=[data_ingredient,
                                              attacks_ingredient,
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

    filename = 'resnet20-8bit.pt'
    path = os.path.join(trained_models_dir, filename)
    model = resnet20().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    acc = adv_different_epsilon(model=model, loader=test_loader,
                                attack=pgd_linf, bitdepth=8, dither=False)

    results = {}
    results['acc'] = acc
    filename = 'epsilon-8.json'
    path = os.path.join(results_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
