import torch


def evaluate(model, loss_fn, dataloader):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item()
            total_correct += (preds.argmax(1) == yb).sum().item()
            total_samples += len(yb)
    return total_loss / len(dataloader), total_correct / total_samples


def train_model(model, loss_fn, optimizer, train_dl, valid_dl, epochs=20):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_loss, val_acc = evaluate(model, loss_fn, valid_dl)
        print(
            f"[Epoch {epoch+1}/{epochs}] Train Loss: {total_loss/len(train_dl):.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )


def test_model(model, loss_fn, test_dl):
    test_loss, test_acc = evaluate(model, loss_fn, test_dl)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")
