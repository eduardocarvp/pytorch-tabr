# The model.

from dataclasses import dataclass, field
from typing import Any

import copy
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import torch
import warnings
import numpy as np
from tqdm.notebook import tqdm
from pytorch_tabr.arch import TabR
from pytorch_tabr.dataloader import (
    validate_eval_set,
    create_dataloaders,
    define_device,
)
from pytorch_tabr.callbacks import (
    CallbackContainer,
    History,
    EarlyStopping,
    LRSchedulerCallback,
)
from pytorch_tabr.metrics import MetricContainer, check_metrics


from sklearn.base import BaseEstimator
from typing import Callable, Literal, Optional, Union

KWArgs = dict[str, Any]
JSONDict = dict[str, Any]  # must be JSON-serializable


@dataclass
class TabRBase(BaseEstimator):
    """
    Base class for the TabR model, providing foundational methods and attributes
    used in both TabRClassifier and TabRRegressor. It includes methods for setting
    up the network, optimizer, metrics, callbacks, and conducting training and prediction epochs.

    Attributes
    ----------
    cat_indices : list[int]
        List of indices for categorical features.
    cat_cardinalities : list[int]
        List of cardinalities for each categorical feature.
    bin_indices : list[int]
        List of indices for binary features.
    num_embeddings : Optional[dict]
        Specifications for embedding numerical features.
    type_embeddings : str
        Type of embeddings to use for categorical features.
    cat_emb_dims : int
        Embedding dimension for categorical features.
    d_main : int
        Main dimension size used in network layers.
    d_multiplier : float
        Multiplier for scaling dimensions in the network.
    encoder_n_blocks : int
        Number of blocks in the encoder.
    predictor_n_blocks : int
        Number of blocks in the predictor.
    mixer_normalization : Union[bool, Literal["auto"]]
        Flag for mixer normalization, can be a boolean or 'auto'.
    context_dropout : float
        Dropout rate for the context layers.
    dropout0 : float
        Initial dropout rate.
    dropout1 : Union[float, Literal["dropout0"]]
        Secondary dropout rate, can match dropout0.
    normalization : str
        Normalization technique used in the network.
    activation : str
        Activation function used in the network.
    device_name : str
        Name of the device to run computations on.
    optimizer_fn : Any
        Function to create the optimizer.
    optimizer_params : dict
        Parameters for the optimizer.
    scheduler_fn : Any
        Function to create the learning rate scheduler.
    scheduler_params : dict
        Parameters for the scheduler.
    context_size : int
        Size of the context.
    context_sample_size : int
        Sample size for the context.
    memory_efficient : bool
        Flag to enable memory efficiency.
    candidate_encoding_batch_size : Optional[int]
        Batch size for candidate encoding.
    seed : int
        Random seed for reproducibility.
    verbose : int
        Verbosity level.

    Methods
    -------
    __post_init__(self):
        Initializes the model, sets the device, and creates copies of mutable parameters.
    _set_optimizer(self):
        Sets up the optimizer for the model.
    _set_network(self):
        Configures and initializes the neural network model.
    _set_metrics(self, metrics, eval_names):
        Sets the metrics for model evaluation.
    _set_callbacks(self, custom_callbacks):
        Initializes callback functions for training.
    fit(self, X_train, y_train, ...):
        Fits the model to the training data.
    _predict_epoch(self, name, loader):
        Conducts predictions for an epoch.
    _predict_batch(self, X):
        Predicts a single batch of data.
    _train_epoch(self, train_loader):
        Trains the model for one epoch.
    _train_batch(self, X, y, candidate_x, candidate_y, context_size):
        Trains the model on a single batch.
    predict(self, X):
        Generates predictions for the given input data.
    """

    cat_indices: list[int] = field(default_factory=list)
    cat_cardinalities: list[int] = field(default_factory=list)
    bin_indices: list[int] = field(default_factory=list)
    num_embeddings: Optional[dict] = None  # lib.deep.ModuleSpec
    type_embeddings: str = None
    cat_emb_dims: int = 2
    d_main: int = 96
    d_multiplier: float = 2.0
    encoder_n_blocks: int = 2
    predictor_n_blocks: int = 2
    mixer_normalization: Union[bool, Literal["auto"]] = "auto"
    context_dropout: float = 0
    dropout0: float = 0.5
    dropout1: Union[float, Literal["dropout0"]] = 0.5
    normalization: str = "LayerNorm"
    activation: str = "ReLU"
    device_name: str = "auto"
    optimizer_fn: Any = torch.optim.Adam
    optimizer_params: dict = field(default_factory=lambda: dict(lr=2e-4))
    scheduler_fn: Any = None
    scheduler_params: dict = field(default_factory=dict)
    context_size: int = 96
    context_sample_size: int = None
    memory_efficient: bool = False
    candidate_encoding_batch_size: Optional[int] = None
    device_name: str = "cpu"
    seed: int = 0
    verbose: int = 0

    def __post_init__(self):
        # These are default values needed for saving model
        torch.manual_seed(self.seed)
        # Defining device
        self.device = torch.device(define_device(self.device_name))
        if self.verbose != 0:
            warnings.warn(f"Device used : {self.device}")

        # create deep copies of mutable parameters
        self.optimizer_fn = copy.deepcopy(self.optimizer_fn)
        self.scheduler_fn = copy.deepcopy(self.scheduler_fn)

    def _set_optimizer(self):
        """Setup optimizer."""
        self._optimizer = self.optimizer_fn(
            self.network.parameters(), **self.optimizer_params
        )

    def _set_network(self):
        self.network = TabR(
            dim_input=self.input_dim,
            cat_indices=self.cat_indices,
            cat_cardinalities=self.cat_cardinalities,
            bin_indices=self.bin_indices,
            output_dim=self.output_dim,
            embed_target=self.embed_target,
            type_embeddings=self.type_embeddings,
            cat_emb_dims=self.cat_emb_dims,
            num_embeddings=self.num_embeddings,  # lib.deep.ModuleSpec
            d_main=self.d_main,
            d_multiplier=self.d_multiplier,
            encoder_n_blocks=self.encoder_n_blocks,
            predictor_n_blocks=self.predictor_n_blocks,
            mixer_normalization=self.mixer_normalization,
            context_dropout=self.context_dropout,
            dropout0=self.dropout0,
            dropout1=self.dropout1,
            normalization=self.normalization,
            activation=self.activation,
            context_sample_size=self.context_sample_size,
            memory_efficient=self.memory_efficient,
            candidate_encoding_batch_size=self.candidate_encoding_batch_size,
        ).to(self.device)
        return

    def _set_metrics(self, metrics, eval_names):
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = metrics or [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update(
                {name: MetricContainer(metrics, prefix=f"{name}_")}
            )

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        self.early_stopping_metric = (
            self._metrics_names[-1] if len(self._metrics_names) > 0 else None
        )

    def _set_callbacks(self, custom_callbacks):
        """Setup the callbacks functions.

        Parameters
        ----------
        custom_callbacks : list of func
            List of callback functions.

        """
        # Setup default callbacks history, early stopping and scheduler
        callbacks = []
        self.history = History(self, verbose=self.verbose)
        callbacks.append(self.history)
        if (self.early_stopping_metric is not None) and (self.patience > 0):
            early_stopping = EarlyStopping(
                early_stopping_metric=self.early_stopping_metric,
                is_maximize=(
                    self._metrics[-1]._maximize if len(self._metrics) > 0 else None
                ),
                patience=self.patience,
            )
            callbacks.append(early_stopping)
        else:
            wrn_msg = "No early stopping will be performed, last training weights will be used."
            warnings.warn(wrn_msg)

        if self.scheduler_fn is not None:
            # Add LR Scheduler call_back
            is_batch_level = self.scheduler_params.pop("is_batch_level", False)
            scheduler = LRSchedulerCallback(
                scheduler_fn=self.scheduler_fn,
                scheduler_params=self.scheduler_params,
                optimizer=self._optimizer,
                early_stopping_metric=self.early_stopping_metric,
                is_batch_level=is_batch_level,
            )
            callbacks.append(scheduler)

        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        self._callback_container = CallbackContainer(callbacks)
        self._callback_container.set_trainer(self)

        return

    def fit(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        eval_set: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
        eval_name: Optional[list[str]] = None,
        eval_metric: Optional[list[str]] = None,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        num_workers: int = 0,
        callbacks: Optional[list[Callable]] = None,
        drop_last: bool = True,
        pin_memory: bool = True,
        warm_start: bool = False,
    ) -> None:
        """
        Fits the model to the training data.

        Parameters
        ----------
        X_train : Union[np.ndarray, torch.Tensor]
            Training data features.
        y_train : Union[np.ndarray, torch.Tensor]
            Training data targets or labels.
        eval_set : Optional[list[tuple[np.ndarray, np.ndarray]]], optional
            A list of validation datasets with their corresponding labels.
        eval_name : Optional[list[str]], optional
            Names for each of the validation datasets, used for logging purposes.
        eval_metric : Optional[list[str]], optional
            List of evaluation metrics to be used.
        max_epochs : int, optional
            Maximum number of epochs for training. Defaults to 100.
        patience : int, optional
            Number of epochs with no improvement after which training will be stopped. Defaults to 10.
        batch_size : int, optional
            Batch size for training. Defaults to 256.
        num_workers : int, optional
            Number of workers for data loading. Defaults to 0.
        callbacks : Optional[list[Callable]], optional
            List of callback functions for various stages of training. Defaults to None.
        drop_last : bool, optional
            Whether to drop the last incomplete batch in each epoch. Defaults to True.
        pin_memory : bool, optional
            If True, the data loader will copy tensors into CUDA pinned memory before returning them. Defaults to True.
        warm_start : bool, optional
            If True, training will start from the most recent state. Defaults to False.

        Notes
        -----
        This method handles the complete process of fitting the model to the training data.
        It includes setting up the data loaders, updating fit parameters, initializing the network,
        and iterating through epochs of training and validation. Callbacks are invoked at the appropriate stages.
        """
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.pin_memory = pin_memory and (self.device.type != "cpu")

        eval_set = eval_set if eval_set else []

        self.X_train = torch.Tensor(X_train).to(self.device)
        self.y_train = torch.Tensor(y_train).to(self.device)
        self.y_train = self.convert_targets(self.y_train)[0]

        # Validate and reformat eval set depending on training data
        eval_names, eval_set = validate_eval_set(eval_set, eval_name, X_train, y_train)

        self.train_indices = torch.Tensor(range(X_train.shape[0]))

        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            y_train,
            eval_set,
            0,  # weights
            batch_size,
            num_workers,
            drop_last,
            pin_memory,
        )

        self.update_fit_params(X_train, y_train, eval_set)

        if not hasattr(self, "network") or not warm_start:
            # model has never been fitted before of warm_start is False
            self._set_network()

        self._set_metrics(eval_metric, eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        self._callback_container.on_train_begin()

        try:
            for epoch in tqdm(range(max_epochs), desc=" epochs", position=0):
                self._callback_container.on_epoch_begin(epoch)

                self._train_epoch(train_dataloader)

                # Apply predict epoch to all eval sets
                for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                    self._predict_epoch(eval_name, valid_dataloader)

                # Call method on_epoch_end for all callbacks
                self._callback_container.on_epoch_end(
                    epoch, logs=self.history.epoch_metrics
                )

                if self._stop_training:
                    break
        except KeyboardInterrupt:
            print("Training stopped. Calling callbacks...")
            # self._callback_container.on_train_end()
            # self.network.eval()

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self.network.eval()

        return

    def _predict_epoch(self, name, loader):
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        self.network.eval()

        list_y_true = []
        list_y_score = []

        # Main loop
        for batch_idx, (_, X, y) in enumerate(loader):
            scores = self._predict_batch(X)
            list_y_true.append(y)
            list_y_score.append(scores)

        y_true, scores = self.stack_batches(list_y_true, list_y_score)

        metrics_logs = self._metric_container_dict[name](y_true, scores)
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return

    def _predict_batch(self, X):
        """
        Predict one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Owned products

        Returns
        -------
        np.array
            model scores
        """
        X = X.to(self.device).float()

        # compute model output
        scores = self.network(
            x=X,
            y=None,
            candidate_x=self.X_train,
            candidate_y=self.y_train,
            context_size=self.context_size,
        )

        if isinstance(scores, list):
            scores = [x.cpu().detach().numpy() for x in scores]
        else:
            scores = scores.cpu().detach().numpy()

        return scores

    def _train_epoch(self, train_loader):
        """
        Conducts the training process for one epoch using the given DataLoader.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader providing batches of training data.

        Notes
        -----
        This method iterates over the batches in the DataLoader, performing training on each batch.
        It also handles the callback for the end of each batch.
        """

        self.network.train()

        for batch_idx, (indices, X, y) in enumerate(
            tqdm(train_loader, desc=" batches", position=1, leave=False)
        ):
            candidate_indices = ~torch.isin(self.train_indices, indices)
            candidate_x = self.X_train[candidate_indices]
            candidate_y = self.y_train[candidate_indices]
            loss = self._train_batch(X, y, candidate_x, candidate_y, self.context_size)

            self._callback_container.on_batch_end(batch_idx, loss)

        return

    def _train_batch(self, X, y, candidate_x, candidate_y, context_size):
        """
        Performs the training operation on a single batch.

        Parameters
        ----------
        X : torch.Tensor
            Input features for the batch.
        y : torch.Tensor
            Target labels for the batch.
        candidate_x : torch.Tensor
            Input features for the candidate set.
        candidate_y : torch.Tensor
            Target labels for the candidate set.
        context_size : int
            Size of the context to be considered during training.

        Returns
        -------
        dict
            A dictionary containing the batch size and loss for the batch.

        Notes
        -----
        This method conducts forward pass, loss calculation, and backpropagation for a single batch.
        """
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device)

        candidate_x = candidate_x.to(self.device).float()
        candidate_y = candidate_y.to(self.device)
        y, candidate_y = self.convert_targets(y, candidate_y)

        for param in self.network.parameters():
            param.grad = None

        output = self.network(
            x=X,
            y=y,
            candidate_x=candidate_x,
            candidate_y=candidate_y,
            context_size=context_size,
        )

        loss = self.compute_loss(output, y)

        # Perform backward pass and optimization
        loss.backward()
        # if self.clip_value:
        #     clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs

    def predict(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        predictions : np.array
            Predictions of the regression problem
        """
        preds = self.predict_proba(X)
        return self.predict_func(preds)
