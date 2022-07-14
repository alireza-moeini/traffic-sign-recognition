import os
import csv
from PIL import Image
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):
    """Custom dataset for traffic sign images."""

    def __init__(self, root_dir, csv_file, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.features, self.labels = [], []
        self._read_csv(os.path.join(root_dir, csv_file))

    def _read_csv(self, csv_path):
        with open(csv_path, newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                image_path = row[7]
                label = int(row[6])
                self.features.append(self._load_image(image_path))
                self.labels.append(label)

    def _load_image(self, image_path):
        image = Image.open(os.path.join(self.root_dir, image_path))
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
