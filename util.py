from data.dataset import Dataset
from data.dataloader import Dataloader
import numpy as np
from itertools import product
import tqdm
import logging
import os
import traceback
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from common.tensor import tensor

def equalize_dataset(dataset: Dataset, noise_variance=0.01, multiplier=1.0):
    '''Equalize the number of samples in each class by oversampling minority classes and adding noise'''
    labels = dataset.labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = int(np.max(counts))
    new_data = []
    new_labels = []

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dataset.data_raw)

    for label, count in zip(unique_labels, counts):
        idx = np.where(labels == label)[0]
        label_data = data_scaled[idx]

        if count < max_count:
            # Oversample the indices
            oversample_count = int(count * (multiplier - 1))
            oversampled_idx = np.random.choice(idx, size=oversample_count, replace=True)
            oversampled_data = data_scaled[oversampled_idx]
            # Add noise to the oversampled data
            noise = np.random.normal(0, noise_variance, oversampled_data.shape)
            oversampled_data += noise
            new_data.append(oversampled_data)
            new_labels.extend([label] * oversample_count)

        new_data.append(label_data)
        new_labels.extend([label] * count)

    new_data = np.vstack(new_data)
    new_labels = np.array(new_labels)

    # Inverse transform to original scale
    new_data = scaler.inverse_transform(new_data)

    # Shuffle the data
    indices = np.arange(new_data.shape[0])
    np.random.shuffle(indices)
    new_data = new_data[indices]
    new_labels = new_labels[indices]

    return Dataset(tensor(new_data), tensor(new_labels), transform=dataset.transform)


def equalize_dataset_downsample(dataset: Dataset):
    '''Equalize the number of samples in each class by oversampling minority classes'''
    labels = dataset.labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = np.min(counts)
    indices = []
    for label, count in zip(unique_labels, counts):
        idx = np.where(labels == label)[0]
        if count > min_count:
            # Oversample the indices
            oversampled_idx = np.random.choice(idx, size=min_count, replace=False)
            indices.append(oversampled_idx)
        else:
            indices.append(idx)
    indices = np.concatenate(indices)
    np.random.shuffle(indices)
    data = dataset.data_raw[indices]
    labels = dataset.labels[indices]
    return Dataset(data, labels)

def split_dataset(data, labels, split=[0.7, 0.3]):
    '''split data into train and test set'''
    n = len(labels)
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_size = int(n*split[0])
    print(f"Train size: {train_size}\nTest size: {n-train_size}")
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]
    train_data = data[train_idx]
    train_labels = labels[train_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]
    train_ds = Dataset(train_data, train_labels)
    test_ds = Dataset(test_data,test_labels)
    return train_ds, test_ds

def k_fold_split(k, train_ds:Dataset, val_idx):
    '''for cross validation'''
    n = len(train_ds)
    fold_size = int(n/k)
    val_start = val_idx * fold_size
    val_end = val_start + fold_size if val_idx < k-1 else n
    val_indices = list(range(val_start, val_end))
    train_indices = list(range(0, val_start)) + list(range(val_end, n))
    val_labels = train_ds.labels[val_indices]
    val_data = train_ds.data_raw[val_indices]
    train_labels = train_ds.labels[train_indices]
    train_data = train_ds.data_raw[train_indices]
    val_ds = Dataset(val_data, val_labels)
    train_ds = Dataset(train_data, train_labels)
    return train_ds, val_ds



def k_fold_val(K, model_func, param_grid, trainer, train_ds, batch_size=50, metric_name=None):
    '''cross validation'''
    param_combinattions = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]
    best_param_dict = None
    best_metric = np.inf # minimize it
    logging.info(f"K Fold Cross Validation with {K} folds")
    for param_dict in param_combinattions:
        logging.info(f"Trying: {param_dict}")
        metric_sum = 0
        for val_idx in tqdm.tqdm(range(K)):
            train_sub_ds, val_sub_ds = k_fold_split(K, train_ds, val_idx) # avoid using train_ds, it will modified the later loop
            model = model_func(**param_dict)
            train_sub_dl = Dataloader(train_sub_ds, batch_size=batch_size, shuffle=True)
            metric = trainer(model, train_sub_dl, val_sub_ds, visualize=False, return_metric=True)
            metric_sum += metric
        avg_metric = metric_sum / K
        if metric is not None:
            logging.info(f"{metric_name}: {avg_metric}")
        if avg_metric < best_metric:
            best_metric = avg_metric
            best_param_dict = param_dict
    return best_param_dict

def setup_logging(save_dir, console="info",
                  info_filename="info.log", debug_filename="debug.log"):
    """Set up logging files and console output.
    Creates one file for INFO logs and one for DEBUG logs.
    Args:
        save_dir (str): creates the folder where to save the files.
        debug (str):
            if == "debug" prints on console debug messages and higher
            if == "info"  prints on console info messages and higher
            if == None does not use console (useful when a logger has already been set)
        info_filename (str): the name of the info file. if None, don't create info file
        debug_filename (str): the name of the debug file. if None, don't create debug file
    """
    if os.path.exists(save_dir):
        raise FileExistsError(f"{save_dir} already exists!")
    os.makedirs(save_dir, exist_ok=True)
    # logging.Logger.manager.loggerDict.keys() to check which loggers are in use
    # base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    
    if info_filename is not None:
        info_file_handler = logging.FileHandler(os.path.join(save_dir, info_filename))
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)
    
    if debug_filename is not None:
        debug_file_handler = logging.FileHandler(os.path.join(save_dir, debug_filename))
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)
    
    if console is not None:
        console_handler = logging.StreamHandler()
        if console == "debug":
            console_handler.setLevel(logging.DEBUG)
        if console == "info":
            console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)
    
    def exception_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = exception_handler