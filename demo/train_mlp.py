import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model.mlp import MLP
from common.activation import *
from common.tensor import *
from optimizer.optim import Momentum, SGD
from transform.normalization import MeanNorm, MinMaxNorm
from transform.augmentation import augment_image
from data import *
from loss.losses import cross_entropy_loss
import util
from evaluate.classification import *
import tqdm
import viz
import logging
import os
import time


save_dir = os.path.join("logs", time.strftime("%Y-%m-%d %H_%M_%S", time.localtime()))
util.setup_logging(save_dir)
np.random.seed(0)

# Params
batch_size = 64

train_df = pd.read_csv('dataset/optdigits.tra', header=None)
train_labels = pd.get_dummies(train_df.pop(64)).to_numpy().astype(int)
train_features = train_df.to_numpy().astype(float)
logging.info(f"Size of training set: {train_features.shape}")

test_df = pd.read_csv('dataset/optdigits.tes', header=None)
test_labels = pd.get_dummies(test_df.pop(64)).to_numpy().astype(int)
test_features = test_df.to_numpy().astype(float)
logging.info(f"Size of test set: {test_features.shape}")

# Preprocessing
mu = np.mean(train_features, axis=0)
sigma = np.std(train_features, axis=0)
mean_norm = MeanNorm(mu,sigma)

train_ds = dataset.Dataset(tensor(train_features), tensor(train_labels))
test_ds = dataset.Dataset(tensor(test_features), tensor(test_labels))
train_ds.transform = mean_norm
test_ds.transform = mean_norm

def train_classifier(model, train_dl, test_ds, epochs=2000, patience=50, threshold=1e-3, visualize=True):
    epoch_not_improved = 0
    best_loss = np.inf
    optimizer = Momentum(model.params, lr=1e-1)
    Loss = []
    Loss_base = []
    Acc = []
    if visualize:
        iter = tqdm.tqdm(range(epochs))
    else:
        iter = range(epochs)
    for epoch in iter:
        epoch_loss = []
        epoch_base = []
        for x, y in train_dl:
            y_pred = model(x)
            loss_base = cross_entropy_loss(y_pred, y)
            optimizer.zero_grad()
            loss = loss_base
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)
            epoch_base.append(loss_base)
        avg_loss = np.mean(epoch_loss)
        avg_base = np.mean(epoch_base)
        Loss.append(avg_loss)
        Loss_base.append(avg_base)
        confusion_matrix, acc = evaluate_multi_class(model, test_ds)
        Acc.append(acc)
        if avg_loss < best_loss - threshold:
            best_loss = avg_loss
            epoch_not_improved = 0
        else:
            epoch_not_improved += 1
        if epoch_not_improved >= patience:
            if visualize:
                logging.info(f"Early stopping at epoch {epoch}")
            break
        if visualize:
            iter.set_description(f"Loss: {avg_loss:.6f}")
    if visualize:
        viz.plot_loss_curve([Loss], ['Loss'])
        plt.savefig(os.path.join(save_dir, "loss_curve.jpg"), dpi=300)
        # plt.show()
        viz.plot_metrics(['Accuracy'], [Acc])
        plt.savefig(os.path.join(save_dir, "metrics.jpg"), dpi=300)
        # plt.show()
        logging.info("Confusion Matrix")
        logging.info(confusion_matrix)
        viz.plot_confusion_matrix(confusion_matrix)
        plt.savefig(os.path.join(save_dir, "confusion_matrix.jpg"), dpi=300)
        # plt.show()
        logging.info(f"Final metric: Accuracy: {Acc[-1]:.4f}")
    return Loss, Acc


input_size = train_ds.data_raw.shape[1]
mlp_layer = [input_size, 100, 10]
act = sigmoid
logging.info(f"MLP Layer: {mlp_layer}, activation: {act}")
model = MLP(mlp_layer, [act]*(len(mlp_layer)-2) + [lambda x:x])

logging.info(f"Training with batch size {batch_size}")
train_dl = dataloader.Dataloader(train_ds, batch_size=batch_size, shuffle=True)  # Create a DataLoader for the training set
train_classifier(model, train_dl, test_ds, visualize=True)
_, _, failure_cases = evaluate_multi_class(model, test_ds, return_failure_case=True)

plt.figure(figsize=(10, 5))
for i, (idx, true_label, pred_label) in enumerate(failure_cases[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_features[idx].reshape(8, 8), cmap='gray')
    plt.title(f"True: {true_label}, Pred: {pred_label}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "failure_cases.jpg"), dpi=300)
# plt.show()

# plt.figure()
# plt.imshow(augment_image(test_features[0]).reshape(8,8), cmap='gray')
# plt.show()