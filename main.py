import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split
from src.dataset import TrafficSignDataset
from src.dataloader import create_dataloaders
from src.models import MLP, LeNet, count_parameters
from src.train import train_model, test_model
from src.visualize import show_sample_images, plot_confusion_matrix

# Configurations
dataset_dir = "dataset"
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform & Datasets
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

train_dataset = TrafficSignDataset(dataset_dir, "Train.csv", transform)
test_dataset = TrafficSignDataset(dataset_dir, "Test.csv", transform)

# Split training into train + validation
valid_size = int(len(train_dataset) * 0.2)
train_size = len(train_dataset) - valid_size
train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

# Load class names
with open(f"{dataset_dir}/Classes.csv") as f:
    reader = csv.reader(f)
    next(reader)
    class_names = [row[1] for row in reader]
num_classes = len(class_names)

# DataLoaders
train_dl, valid_dl, test_dl = create_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, device)

# Model Selection
model = LeNet(num_classes).to(device)
print(f"Model has {count_parameters(model)} trainable parameters")

# Training
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, loss_fn, optimizer, train_dl, valid_dl, epochs=20)
test_model(model, loss_fn, test_dl)

torch.save(model.state_dict(), "models/lenet.pth")

# Visualization
show_sample_images(test_dataset, class_names)
# Generate predictions for confusion matrix
predictions = []
with torch.no_grad():
    for xb, _ in test_dl:
        preds = model(xb)
        predictions.extend(preds.argmax(1).cpu().numpy())

plot_confusion_matrix(test_dataset.labels, predictions, class_names)
