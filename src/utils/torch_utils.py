# define Mean Absolute Relative Error for loss function

import torch

def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x).reshape(-1,)
    loss = criterion(output.float(),y.float())
    loss.backward()
    optimizer.step()

    return loss,output

def MARELoss(preds_, targets_):
    size_ = len(targets_)
    loss_ = sum(abs(preds_-targets_)/targets_)/size_
    return torch.sum(loss_)
