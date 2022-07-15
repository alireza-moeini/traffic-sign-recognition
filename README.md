# Traffic Sign Classification with MLP and CNN

This project implements a traffic sign classification system using two types of neural networks:

1. **Multilayer Perceptron (MLP)** – a fully connected network.
2. **Convolutional Neural Network (LeNet)** – a CNN architecture.

The dataset used is the [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data) from Kaggle.


## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results Analysis](#results-analysis)


## Project Structure

```
.
├── LICENSE
├── main.py
├── README.md
├── recommit.sh
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── visualize.py
└── weights
    ├── lenet.pth
    └── mlp.pth
```

- `src/` contains all the source code for datasets, models, training, evaluation, and visualization.
- `weights/` stores trained model weights (`.pth` files).
- `dataset/` contains CSV files and image data for training and testing (after adding the dataset to the project).


## Installation

1. Clone the repository:

```bash
git clone https://github.com/alireza-moeini/traffic-sign-recognition.git
cd traffic_sign_classification
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Dataset: the dataset can be downloaded from Kaggle GTSRB. Place the extracted dataset in the dataset/ folder with the CSV files (Train.csv, Test.csv, Classes.csv) and images.

## Usage
Train and test the models:

```bash
python main.py
```

The script will:

- Train both MLP and LeNet models.
- Save trained weights in weights/.
- Display sample images from the test set.
- Generate a normalized confusion matrix.


## Results Analysis
Historically, MLPs were used for computer vision tasks, but modern problems typically require CNNs. In this project:

- Accuracy: CNN achieved about 10% higher accuracy than MLP on both validation and test sets.

- Parameters: MLP has ~400,000 parameters, while CNN (LeNet) has only ~65,000.

- Test Accuracy:

    - MLP: >80%
    - CNN: ~90%

These results demonstrate that CNNs are generally more effective for computer vision tasks.