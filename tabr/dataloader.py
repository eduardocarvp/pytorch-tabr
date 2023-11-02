from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy
import json
from sklearn.utils import check_array
import pandas as pd
import warnings


class TorchDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return index, x, y


class SparseTorchDataset(Dataset):
    """
    Format for csr_matrix

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        y = self.y[index]
        return index, x, y


class PredictDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        return x


class SparsePredictDataset(Dataset):
    """
    Format for csr_matrix

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        return x


def create_sampler(weights, y_train):
    """
    This creates a sampler from the given weights

    Parameters
    ----------
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    y_train : np.array
        Training targets
    """
    if isinstance(weights, int):
        if weights == 0:
            need_shuffle = True
            sampler = None
        elif weights == 1:
            need_shuffle = False
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )

            weights = 1.0 / class_sample_count

            samples_weight = np.array([weights[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            raise ValueError("Weights should be either 0, 1, dictionnary or list.")
    elif isinstance(weights, dict):
        # custom weights per class
        need_shuffle = False
        samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        # custom weights
        if len(weights) != len(y_train):
            raise ValueError("Custom weights should match number of train samples.")
        need_shuffle = False
        samples_weight = np.array(weights)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return need_shuffle, sampler


def create_dataloaders(
    X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory
):
    """
    Create dataloaders with or without subsampling depending on weights and balanced.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    y_train : np.array
        Mapped Training targets
    eval_set : list of tuple
        List of eval tuple set (X, y)
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    batch_size : int
        how many samples per batch to load
    num_workers : int
        how many subprocesses to use for data loading. 0 means that the data
        will be loaded in the main process
    drop_last : bool
        set to True to drop the last incomplete batch, if the dataset size is not
        divisible by the batch size. If False and the size of dataset is not
        divisible by the batch size, then the last batch will be smaller
    pin_memory : bool
        Whether to pin GPU memory during training

    Returns
    -------
    train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
        Training and validation dataloaders
    """
    need_shuffle, sampler = create_sampler(weights, y_train)

    if scipy.sparse.issparse(X_train):
        train_dataloader = DataLoader(
            SparseTorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
    else:
        train_dataloader = DataLoader(
            TorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    valid_dataloaders = []
    for X, y in eval_set:
        if scipy.sparse.issparse(X):
            valid_dataloaders.append(
                DataLoader(
                    SparseTorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )
        else:
            valid_dataloaders.append(
                DataLoader(
                    TorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )

    return train_dataloader, valid_dataloaders


def filter_weights(weights):
    """
    This function makes sure that weights are in correct format for
    regression and multitask TabNet

    Parameters
    ----------
    weights : int, dict or list
        Initial weights parameters given by user

    Returns
    -------
    None : This function will only throw an error if format is wrong
    """
    err_msg = """Please provide a list or np.array of weights for """
    err_msg += """regression, multitask or pretraining: """
    if isinstance(weights, int):
        if weights == 1:
            raise ValueError(err_msg + "1 given.")
    if isinstance(weights, dict):
        raise ValueError(err_msg + "Dict given.")
    return


def validate_eval_set(eval_set, eval_name, X_train, y_train):
    """Check if the shapes of eval_set are compatible with (X_train, y_train).

    Parameters
    ----------
    eval_set : list of tuple
        List of eval tuple set (X, y).
        The last one is used for early stopping
    eval_name : list of str
        List of eval set names.
    X_train : np.ndarray
        Train owned products
    y_train : np.array
        Train targeted products

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.
    eval_set : list of tuple
        Validated list of eval_set.

    """
    eval_name = eval_name or [f"val_{i}" for i in range(len(eval_set))]

    assert len(eval_set) == len(
        eval_name
    ), "eval_set and eval_name have not the same length"
    if len(eval_set) > 0:
        assert all(
            len(elem) == 2 for elem in eval_set
        ), "Each tuple of eval_set need to have two elements"
    for name, (X, y) in zip(eval_name, eval_set):
        check_input(X)
        msg = (
            f"Dimension mismatch between X_{name} "
            + f"{X.shape} and X_train {X_train.shape}"
        )
        assert len(X.shape) == len(X_train.shape), msg

        msg = (
            f"Dimension mismatch between y_{name} "
            + f"{y.shape} and y_train {y_train.shape}"
        )
        assert len(y.shape) == len(y_train.shape), msg

        msg = (
            f"Number of columns is different between X_{name} "
            + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        )
        assert X.shape[1] == X_train.shape[1], msg

        if len(y_train.shape) == 2:
            msg = (
                f"Number of columns is different between y_{name} "
                + f"({y.shape[1]}) and y_train ({y_train.shape[1]})"
            )
            assert y.shape[1] == y_train.shape[1], msg
        msg = (
            f"You need the same number of rows between X_{name} "
            + f"({X.shape[0]}) and y_{name} ({y.shape[0]})"
        )
        assert X.shape[0] == y.shape[0], msg

    return eval_name, eval_set


def define_device(device_name):
    """
    Define the device to use during training and inference.
    If auto it will detect automatically whether to use cuda or cpu

    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda"

    Returns
    -------
    str
        Either "cpu" or "cuda"
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    else:
        return device_name


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def check_input(X):
    """
    Raise a clear error if X is a pandas dataframe
    and check array according to scikit rules
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        err_message = (
            "Pandas DataFrame are not supported: apply X.values when calling fit"
        )
        raise TypeError(err_message)
    check_array(X, accept_sparse=True)


def check_warm_start(warm_start, from_unsupervised):
    """
    Gives a warning about ambiguous usage of the two parameters.
    """
    if warm_start and from_unsupervised is not None:
        warn_msg = "warm_start=True and from_unsupervised != None: "
        warn_msg = (
            "warm_start will be ignore, training will start from unsupervised weights"
        )
        warnings.warn(warn_msg)
    return


def check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim):
    """
    Check parameters related to embeddings and rearrange them in a unique manner.
    """
    if (cat_dims == []) ^ (cat_idxs == []):
        if cat_dims == []:
            msg = """If cat_idxs is non-empty, cat_dims must
                be defined as a list of same length."""
        else:
            msg = """If cat_dims is non-empty, cat_idxs must
                be defined as a list of same length."""
        raise ValueError(msg)
    elif len(cat_dims) != len(cat_idxs):
        msg = "The lists cat_dims and cat_idxs must have the same length."
        raise ValueError(msg)

    if isinstance(cat_emb_dim, int):
        cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
    else:
        cat_emb_dims = cat_emb_dim

    # check that all embeddings are provided
    if len(cat_emb_dims) != len(cat_dims):
        msg = f"""cat_emb_dim and cat_dims must be lists of same length, 
            got {len(cat_emb_dims)} and {len(cat_dims)}"""
        raise ValueError(msg)

    # Rearrange to get reproducible seeds with different ordering
    if len(cat_idxs) > 0:
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        cat_emb_dims = [cat_emb_dims[i] for i in sorted_idxs]

    return cat_dims, cat_idxs, cat_emb_dims
