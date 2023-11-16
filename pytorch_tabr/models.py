import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import DataLoader
from pytorch_tabr.base_model import TabRBase
from pytorch_tabr.utils import infer_output_dim, check_output_dim
from pytorch_tabr.dataloader import (
    SparsePredictDataset,
    PredictDataset
)


class TabRClassifier(TabRBase):
    def __post_init__(self):
        super().__post_init__()
        self.loss_fn = F.cross_entropy
        self._default_metric = "accuracy"

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.long())

    def convert_targets(self, *target_tensors):
        response = []
        for ten in target_tensors:
            response.append(ten.long())
        return tuple(response)

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
        self.embed_target = True
        self._default_metric = "auc" if self.output_dim == 2 else "accuracy"
        self.classes_ = train_labels

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs):
        return np.argmax(outputs, axis=1)

    def predict_proba(self, X):
        # Make predictions on a batch (valid)

        """
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
            predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
            results.append(predictions)
        res = np.vstack(results)
        return res


class TabRRegressor(TabRBase):
    def __post_init__(self):
        super().__post_init__()
        self._task = "regression"
        self.loss_fn = torch.nn.functional.mse_loss
        self._default_metric = "mse"

    def convert_targets(self, *target_tensors):
        response = []
        for ten in target_tensors:
            response.append(ten.float())
        return tuple(response)

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
    ):
        if len(y_train.shape) != 2:
            msg = (
                "Targets should be 2D : (n_samples, n_regression) "
                + f"but y_train.shape={y_train.shape} given.\n"
                + "Use reshape(-1, 1) for single regression."
            )
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.embed_target = False

    def predict_func(self, outputs):
        return outputs

    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score

    def predict_proba(self, X):
        # Make predictions on a batch (valid)

        """
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
