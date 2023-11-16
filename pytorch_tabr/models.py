import numpy as np
import scipy
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import DataLoader
from pytorch_tabr.base_model import TabRBase
from pytorch_tabr.utils import infer_output_dim, check_output_dim
from pytorch_tabr.dataloader import SparsePredictDataset, PredictDataset


class TabRClassifier(TabRBase):
    """
    A classifier based on TabRBase for tabular data. This class handles classification
    tasks including computing loss, converting targets, updating fit parameters,
    stacking batches, and making predictions.

    Inherits from TabRBase.
    """

    def __post_init__(self) -> None:
        """Initialize the classifier with default loss function and metric."""
        super().__post_init__()
        self.loss_fn = F.cross_entropy
        self._default_metric = "accuracy"

    def compute_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute the loss using the defined loss function.

        Parameters
        ----------
        y_pred : Tensor
            The predictions made by the model.
        y_true : Tensor
            The true labels.

        Returns
        -------
        Tensor
            The computed loss value.
        """
        return self.loss_fn(y_pred, y_true.long())

    def convert_targets(self, *target_tensors: Tensor) -> tuple[Tensor, ...]:
        """
        Convert target tensors to long data type.

        Parameters
        ----------
        target_tensors : Tensor
            One or more tensor objects.

        Returns
        -------
        Tuple[Tensor, ...]
            Tensors converted to long data type.
        """
        response = [ten.long() for ten in target_tensors]
        return tuple(response)

    def update_fit_params(
        self,
        X_train: np.ndarray,
        y_train: Tensor,
        eval_set: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """
        Update fitting parameters based on training and evaluation data.

        Parameters
        ----------
        X_train : np.ndarray
            Training data.
        y_train : np.ndarray
            Training labels.
        eval_set : list[tuple[np.ndarray, np.ndarray]]
            A list of tuples containing evaluation data and labels.
        """
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self.embed_target = True
        self._default_metric = "auc" if self.output_dim == 2 else "accuracy"
        self.classes_ = train_labels

    def stack_batches(
        self, list_y_true: list[np.ndarray], list_y_score: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stack batches of true labels and scores.

        Parameters
        ----------
        list_y_true : List[Tensor]
            A list of true label tensors.
        list_y_score : List[Tensor]
            A list of score tensors.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Stacked true labels and scores.
        """
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)
        return y_true, y_score

    def predict_func(self, outputs: Tensor) -> np.ndarray:
        """
        Determine the predicted class based on model outputs.

        Parameters
        ----------
        outputs : Tensor
            Model outputs.

        Returns
        -------
        np.ndarray
            Predicted classes.
        """
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
    """
    A regressor based on TabRBase for tabular data regression tasks. This class handles
    regression tasks including converting targets, computing loss, updating fit parameters,
    stacking batches, and making predictions.

    Inherits from TabRBase.
    """

    def __post_init__(self):
        """
        Initialize the regressor with default task type, loss function, and metric.
        """
        super().__post_init__()
        self._task = "regression"
        self.loss_fn = torch.nn.functional.mse_loss
        self._default_metric = "mse"

    def convert_targets(self, *target_tensors: Tensor) -> tuple[Tensor, ...]:
        """
        Convert target tensors to float data type.

        Parameters
        ----------
        target_tensors : Tensor
            One or more tensor objects.

        Returns
        -------
        Tuple[Tensor, ...]
            Tensors converted to float data type.
        """
        response = [ten.float() for ten in target_tensors]
        return tuple(response)

    def compute_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute the loss using the defined loss function for regression.

        Parameters
        ----------
        y_pred : Tensor
            The predictions made by the model.
        y_true : Tensor
            The true labels.

        Returns
        -------
        Tensor
            The computed loss value.
        """
        return self.loss_fn(y_pred, y_true)

    def update_fit_params(
        self,
        X_train: np.ndarray,
        y_train: Tensor,
        eval_set: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """
        Update fitting parameters based on training and evaluation data.

        Parameters
        ----------
        X_train : np.ndarray
            Training data.
        y_train : np.ndarray
            Training labels.
        eval_set : list[tuple[np.ndarray, np.ndarray]]
            A list of tuples containing evaluation data and labels.
        """
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
        """
        Return the outputs directly for regression problems.

        Parameters
        ----------
        outputs : Tensor
            Model outputs.

        Returns
        -------
        Tensor
            The outputs as is, suitable for regression tasks.
        """
        return outputs

    def stack_batches(
        self, list_y_true: list[np.ndarray], list_y_score: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stack batches of true labels and scores for regression.

        Parameters
        ----------
        list_y_true : List[Tensor]
            A list of true label tensors.
        list_y_score : List[Tensor]
            A list of score tensors.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Stacked true labels and scores.
        """
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
