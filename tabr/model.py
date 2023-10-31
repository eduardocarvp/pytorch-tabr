# The model.

# >>>
if __name__ == "__main__":
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ["PROJECT_DIR"] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from arch import TabR

from sklearn.base import BaseEstimator

KWArgs = dict[str, Any]
JSONDict = dict[str, Any]  # must be JSON-serializable


class TabRClassifier(BaseEstimator):
    def __init__(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        optimizer_fn: Any = torch.optim.Adam,
        optimizer_params: dict = field(default_factory=lambda: dict(lr=2e-2)),
        scheduler_fn: Any = None,
        scheduler_params: dict = field(default_factory=dict),
        context_size: int,
        device="cpu",
    ):
        self.network = TabR()
        self.network.to(device)

        self.optimizer_fn = optimizer_fn

        if torch.cuda.device_count() > 1:
            self.network = nn.DataParallel(self.network)  # type: ignore[code]
    
    def _set_optimizer(self):
        """Setup optimizer."""
        self._optimizer = self.optimizer_fn(
            self.network.parameters(), **self.optimizer_params
        )

    def fit(
        self,
        X_train,
        y_train,
        eval_set=None,
        eval_name=None,
        eval_metric=None,
        loss_fn=F.binary_cross_entropy_with_logits,
        weights=0,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=True,
        callbacks=None,
        pin_memory=True,
        from_unsupervised=None,
        warm_start=False,
        augmentations=None,
        compute_importance=True
    ):
        pass

    def _train_epoch(self, train_loader):
        self.network.train()

        for batch_idx, (X, y) in enumerate(train_loader):
            _ = self._train_batch(X, y)

        return

    def compute_loss(self, y_true, y_pred):
        return self.loss_fn(y_pred, y_true.long())

    def _train_batch(self, X, y):
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()
        y = y.to(self.device).float()

        for param in self.network.parameters():
            param.grad = None

        output = self.network(X)

        loss = self.compute_loss(output, y)

        # Perform backward pass and optimization
        loss.backward()
        # if self.clip_value:
        #     clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs

    def predict_proba():
        pass

    def _train_batch():
        pass

    def _predict_epoch():
        pass

    def _predict_batch():
        pass






@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # Model
    context_size: int
    optimizer: KWArgs  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]


def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    device = lib.get_device()

    # >>> data
    dataset = (
        C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    ).to_torch(device)
    if dataset.is_regression:
        dataset.data["Y"] = {k: v.float() for k, v in dataset.Y.items()}
    Y_train = dataset.Y["train"].to(
        torch.long if dataset.is_multiclass else torch.float
    )

    # >>> model
    model = Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
        **C.model,
    )
    report["n_parameters"] = lib.get_n_parameters(model)
    logger.info(f'n_parameters = {report["n_parameters"]}')
    report["prediction_type"] = None if dataset.is_regression else "logits"
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # type: ignore[code]

    # >>> training
    def zero_wd_condition(
        module_name: str,
        module: nn.Module,
        parameter_name: str,
        parameter: nn.parameter.Parameter,
    ):
        return (
            "label_encoder" in module_name
            or "label_encoder" in parameter_name
            or lib.default_zero_weight_decay_condition(
                module_name, module, parameter_name, parameter
            )
        )

    optimizer = lib.make_optimizer(
        model, **C.optimizer, zero_weight_decay_condition=zero_wd_condition
    )
    loss_fn = lib.get_loss_fn(dataset.task_type)

    train_size = dataset.size("train")
    train_indices = torch.arange(train_size, device=device)

    epoch = 0
    eval_batch_size = 32768
    chunk_size = None
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    def get_Xy(part: str, idx) -> tuple[dict[str, Tensor], Tensor]:
        batch = (
            {
                key[2:]: dataset.data[key][part]
                for key in dataset.data
                if key.startswith("X_")
            },
            dataset.Y[part],
        )
        return (
            batch
            if idx is None
            else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
        )

    def apply_model(part: str, idx: Tensor, training: bool):
        x, y = get_Xy(part, idx)

        candidate_indices = train_indices
        is_train = part == "train"
        if is_train:
            # NOTE: here, the training batch is removed from the candidates.
            # It will be added back inside the model's forward pass.
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]
        candidate_x, candidate_y = get_Xy(
            "train",
            # This condition is here for historical reasons, it could be just
            # the unconditional `candidate_indices`.
            None if candidate_indices is train_indices else candidate_indices,
        )

        return model(
            x_=x,
            y=y if is_train else None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=C.context_size,
            is_train=is_train,
        ).squeeze(-1)

    @torch.inference_mode()
    def evaluate(parts: list[str], eval_batch_size: int):
        model.eval()
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx, False)
                                for idx in torch.arange(
                                    dataset.size(part), device=device
                                ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f"eval_batch_size = {eval_batch_size}")
                else:
                    break
            if not eval_batch_size:
                RuntimeError("Not enough memory even for eval_batch_size=1")
        metrics = (
            dataset.calculate_metrics(predictions, report["prediction_type"])
            if lib.are_valid_predictions(predictions)
            else {x: {"score": -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def save_checkpoint():
        lib.dump_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "random_state": delu.random.get_state(),
                "progress": progress,
                "report": report,
                "timer": timer,
                "training_log": training_log,
            },
            output,
        )
        lib.dump_report(report, output)
        lib.backup_output(output)

    print()
    timer = lib.run_timer()
    while epoch < C.n_epochs:
        print(f"[...] {lib.try_get_relative_path(output)} | {timer}")

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(train_size, C.batch_size, device),
            desc=f"Epoch {epoch}",
        ):
            loss, new_chunk_size = lib.train_step(
                optimizer,
                lambda idx: loss_fn(apply_model("train", idx, True), Y_train[idx]),
                batch_idx,
                chunk_size or C.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f"chunk_size = {chunk_size}")

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ["val", "test"], eval_batch_size
        )
        lib.print_metrics(mean_loss, metrics)
        training_log.append(
            {"epoch-losses": epoch_losses, "metrics": metrics, "time": timer()}
        )
        writer.add_scalars("loss", {"train": mean_loss}, epoch, timer())
        for part in metrics:
            writer.add_scalars("score", {part: metrics[part]["score"]}, epoch, timer())

        progress.update(metrics["val"]["score"])
        if progress.success:
            lib.celebrate()
            report["best_epoch"] = epoch
            report["metrics"] = metrics
            save_checkpoint()
            lib.dump_predictions(predictions, output)

        elif progress.fail or not lib.are_valid_predictions(predictions):
            break

        epoch += 1
        print()
    report["time"] = str(timer)

    # >>> finish
    model.load_state_dict(lib.load_checkpoint(output)["model"])
    report["metrics"], predictions, _ = evaluate(
        ["train", "val", "test"], eval_batch_size
    )
    report["chunk_size"] = chunk_size
    report["eval_batch_size"] = eval_batch_size
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    save_checkpoint()
    lib.finish(output, report)
    return report
