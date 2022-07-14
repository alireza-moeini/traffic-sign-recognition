import torch
from torch.utils.data import DataLoader


class DeviceDataLoader:
    """Wraps DataLoader to move batches to device automatically."""

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for xb, yb in self.dataloader:
            yield xb.to(self.device), yb.to(self.device)

    def __len__(self):
        return len(self.dataloader)


def create_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, device):
    from torch.utils.data import DataLoader

    train_dl = DeviceDataLoader(DataLoader(train_dataset, batch_size=batch_size, shuffle=True), device)
    valid_dl = DeviceDataLoader(DataLoader(valid_dataset, batch_size=batch_size), device)
    test_dl = DeviceDataLoader(DataLoader(test_dataset, batch_size=batch_size), device)
    return train_dl, valid_dl, test_dl
