"""
Attack code is extracted from Madry Lab and NIPS 2018 tutorial
https://github.com/MadryLab/robustness_lib
https://adversarial-ml-tutorial.org/
"""
import torch
import torch.nn as nn
from sacred import Ingredient

attacks_ingredient = Ingredient('attacks')
attacks_ingredient.add_config('config.json')


@attacks_ingredient.capture
def pgd_linf2(model, X, y, epsilon, alpha):
    epsilon = epsilon / 255
    alpha = 2 / 255
    delta = torch.zeros_like(X, requires_grad=True)

    X_prime = X + delta
    for t in range(20):
        loss = nn.CrossEntropyLoss()(model(X_prime), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(
            -epsilon, epsilon
        )
        delta.grad.zero_()
        X_prime = torch.clamp(X+delta, 0, 1)
    return X_prime.detach()


@attacks_ingredient.capture
def pgd_linf(model, inputs, targets, eps, step_size):
    ori = inputs.clone()
    for _ in range(20):
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs = model(inputs)
        losses = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        loss = torch.mean(losses)
        grad, = torch.autograd.grad(loss, [inputs])
        with torch.no_grad():
            inputs = torch.sign(grad) * step_size + inputs
            inputs = torch.clamp(inputs, 0, 1)
            diff = inputs - ori
            diff = torch.clamp(diff, -eps, eps)
            inputs = diff + ori
    return inputs.clone().detach()
