# The model.

from dataclasses import dataclass, field
from typing import Any

import copy
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import warnings
from torch.utils.data import DataLoader
from tabr.arch import TabR
from tabr.utils import infer_output_dim, check_output_dim
from tabr.dataloader import (
    SparsePredictDataset,
    PredictDataset,
    validate_eval_set,
    create_dataloaders,
    define_device,
    ComplexEncoder,
    check_input,
    check_warm_start,
    check_embedding_parameters,
)

from sklearn.base import BaseEstimator
from typing import Literal, Optional, Union

KWArgs = dict[str, Any]
JSONDict = dict[str, Any]  # must be JSON-serializable


@dataclass
class TabRClassifier(BaseEstimator):
    cat_indices: list[int] = field(default_factory=list)
    cat_cardinalities: list[int] = field(default_factory=list)
    bin_indices: list[int] = field(default_factory=list)
    num_embeddings: Optional[dict] = None  # lib.deep.ModuleSpec
    d_main: int = 96
    d_multiplier: float = 2.0
    encoder_n_blocks: int = 0
    predictor_n_blocks: int = 1
    mixer_normalization: Union[bool, Literal["auto"]] = "auto"
    context_dropout: float = 0
    dropout0: float = 0
    dropout1: Union[float, Literal["dropout0"]] = 0
    normalization: str = "LayerNorm"
    activation: str = "ReLU"
    device_name: str = "auto"
    optimizer_fn: Any = torch.optim.Adam
    optimizer_params: dict = field(default_factory=lambda: dict(lr=2e-2))
    scheduler_fn: Any = None
    scheduler_params: dict = field(default_factory=dict)
    context_size: int = 96
    device_name: str = "cpu"
    seed: int = 0
    verbose: int = 0

    def __post_init__(self):
        # These are default values needed for saving model
        self.batch_size = 1024
        self.virtual_batch_size = 128
        self.loss_fn = F.binary_cross_entropy_with_logits

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
            n_classes=self.output_dim,
        )
        return

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
    ):
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = "auc" if self.output_dim == 2 else "accuracy"
        self.classes_ = train_labels
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        max_epochs=100,
        patience=10,
        batch_size=16,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        warm_start=False,
    ):
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.pin_memory = pin_memory and (self.device.type != "cpu")

        eval_set = eval_set if eval_set else []

        self.X_train = torch.Tensor(X_train)
        self.y_train = torch.Tensor(y_train).long()

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

        self._set_optimizer()

        for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
            self._predict_epoch(eval_name, valid_dataloader)

        for epoch in range(max_epochs):
            print(epoch)
            self._train_epoch(train_dataloader)

    def _train_epoch(self, train_loader):
        self.network.train()

        for batch_idx, (indices, X, y) in enumerate(train_loader):
            candidate_indices = ~torch.isin(self.train_indices, indices)
            candidate_x = self.X_train[candidate_indices]
            candidate_y = self.y_train[candidate_indices]
            loss = self._train_batch(X, y, candidate_x, candidate_y, self.context_size)

        return

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.float())

    def _train_batch(self, X, y, candidate_x, candidate_y, context_size):
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device).long()

        for param in self.network.parameters():
            param.grad = None

        output = self.network(
            x=X,
            y=y,
            candidate_x=candidate_x,
            candidate_y=candidate_y,
            context_size=context_size,
        )

        loss = self.compute_loss(output.squeeze(-1), y)

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
        self.network.eval()

        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        results = []
        for batch_nb, data in enumerate(dataloader):
            data = torch.Tensor(data).to(self.device).float()

            output = self.network(
                x=data,
                y=None,
                candidate_x=self.X_train,
                candidate_y=self.y_train,
                context_size=self.context_size,
            )
            predictions = output.cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res
