import torch


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        total_loss += loss.item()

    return total_loss / len(loader), correct / total
