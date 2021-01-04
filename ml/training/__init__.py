import torch
from torch import nn
import torch.optim as optim

from common.terminal_utils import printProgressBar, clearProgressBar
from common.timer import Timer


def train(*, learn_params, model, optimizer: optim.Optimizer, training_dataloader, policy_weight=None, value_weight=None, report_hook=None, azure_run=None):
    if policy_weight is not None:
        policy_weight = torch.as_tensor(policy_weight, device=model.device)
    if value_weight is not None:
        value_weight = torch.as_tensor(value_weight, device=model.device)

    policy_loss_function = nn.NLLLoss(reduction='mean', weight=policy_weight, ignore_index=-1)
    value_loss_function = nn.NLLLoss(reduction='mean', weight=value_weight)
    timer = Timer(f'Trained {len(training_dataloader.dataset)} samples. Training per sample:')
    device = model.device
    for epoch in range(learn_params.num_epochs):
        epoch_loss = 0
        model.zero_grad()
        # We have either s (= index map) or o (= positional_encoding)
        for x, s, y, p, v in training_dataloader:
            x = x.to(device)
            s = s.to(device)
            y = y.to(device)
            p = p.to(device)
            v = v.to(device)
            optimizer.zero_grad()
            py, pv = model(x, s, p)
            # batch x tags
            policy_loss = policy_loss_function(py, y)
            v = v.squeeze()
            # print(f'pv: {pv.shape}  v: {v.shape}')
            value_loss = value_loss_function(pv, v)
            loss = policy_loss + value_loss*learn_params.value_loss_weight
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), learn_params.gradient_clipping)
            optimizer.step()
            epoch_loss += loss
        if report_hook:
            timer.pause()
            report_hook(epoch, epoch_loss)
            timer.resume()
        printProgressBar(epoch, learn_params.num_epochs, prefix='train')

    clearProgressBar()
    duration_per_sample = timer.stop_and_log_average(learn_params.num_epochs*len(training_dataloader.dataset))
    if azure_run is not None:
        azure_run.log('duration_per_sample', duration_per_sample)
