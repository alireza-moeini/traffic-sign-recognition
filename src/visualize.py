import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def show_sample_images(dataset, class_names, rows=8, cols=8):
    labels = np.array(dataset.labels)
    plt.figure(figsize=(24, 18))
    for idx, class_name in enumerate(class_names):
        sample_idx = np.random.choice(np.where(labels == idx)[0])
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(np.transpose(dataset[sample_idx][0], (1, 2, 0)))
        plt.title(class_name)
        plt.axis("off")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / cf_matrix.sum(axis=1)[:, None], index=class_names, columns=class_names).round(2)
    plt.figure(figsize=(24, 18))
    sns.heatmap(df_cm, annot=True, cmap="Blues")
    plt.show()
