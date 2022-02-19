from tqdm import tqdm
import torch


def training_loop(dataloader, model, optimizer, criterion):
    device = next(model.parameters()).device
    losses = []
    for inputs, labels, specs in tqdm(dataloader):
        optimizer.zero_grad()
        bi, bl = inputs.to(device), labels.to(device)
        y_pred = model(bi)
        loss = criterion(y_pred, bl)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses
