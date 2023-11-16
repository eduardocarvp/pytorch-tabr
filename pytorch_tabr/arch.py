import math
from typing import Literal, Optional, Union

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch_tabr.embeddings import OneHotEncoder, make_module, EmbeddingGenerator


class TabR(nn.Module):
    def __init__(
        self,
        *,
        #
        dim_input: int,
        cat_indices: list[int],
        cat_cardinalities: list[int],
        bin_indices: list[int],
        output_dim: Optional[int],
        embed_target: bool = True,
        #
        type_embeddings: str = None,
        cat_emb_dims: int = 2,
        num_embeddings: Optional[dict] = None,  # lib.deep.ModuleSpec
        d_main: int = 96,
        d_multiplier: float = 2.0,
        encoder_n_blocks: int = 2,
        predictor_n_blocks: int = 2,
        mixer_normalization: Union[bool, Literal["auto"]] = "auto",
        context_dropout: float = 0,
        dropout0: float = 0.5,
        dropout1: Union[float, Literal["dropout0"]] = 0.5,
        normalization: str = "LayerNorm",
        activation: str = "ReLU",
        context_sample_size: int = None,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == "auto":
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == "dropout0":
            dropout1 = dropout0

        self.context_sample_size = context_sample_size

        # numerical features
        self.n_bin_features = len(bin_indices)
        self.bin_indices = bin_indices

        self.n_cat_features = len(cat_indices)
        self.cat_indices = cat_indices

        self.n_num_features = dim_input - self.n_bin_features - self.n_cat_features
        self.num_indices = [
            i
            for i in range(dim_input)
            if i not in self.bin_indices and i not in self.cat_indices
        ]

        if self.n_cat_features == 0:
            self.cat_encoder = None
            self.cat_post_embedding_size = 0
        else:
            if type_embeddings is None or type_embeddings == "one-hot":
                self.cat_encoder = (
                    OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
                )
                self.cat_post_embedding_size = sum(cat_cardinalities)
            elif type_embeddings == "embeddings":
                if isinstance(cat_emb_dims, int):
                    cat_emb_dims = [cat_emb_dims for _ in range(len(cat_cardinalities))]
                else:
                    assert len(cat_emb_dims) == len(cat_cardinalities)
                self.cat_encoder = EmbeddingGenerator(
                    dim_input, cat_cardinalities, cat_indices, cat_emb_dims
                )
                self.cat_post_embedding_size = sum(cat_emb_dims)
            else:
                raise ValueError("Embedding type not recognized.")

        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=self.n_num_features)
        )

        # >>> E
        d_in = (
            self.n_num_features
            * (1 if num_embeddings is None else num_embeddings["d_embedding"])
            + self.n_bin_features
            + self.cat_post_embedding_size
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(output_dim, d_main)
            if embed_target is False
            else nn.Sequential(
                nn.Embedding(
                    output_dim, d_main
                ),  # delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )

        # out_dim = 1 if output_dim is None or output_dim == 2 else output_dim
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, output_dim),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_num = x[:, self.num_indices]
        x_bin = x[:, self.bin_indices]
        x_cat = x[:, self.cat_indices]
        del x

        x = []
        if x_num.shape[1] == 0:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin.shape[1] != 0:
            x.append(x_bin)
        if x_cat.shape[1] == 0:
            assert self.cat_encoder is None
        else:
            assert self.cat_encoder is not None
            x.append(self.cat_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    # candidate_x = train - current_batch in training mode
    # candidate_x = train in prediction mode
    # we should make a function of dataset that removes the current batch in
    # training time
    def forward(
        self,
        *,
        x: Tensor,
        y: Optional[Tensor],
        candidate_x: Tensor,
        candidate_y: Tensor,
        context_size: int,
    ) -> Tensor:
        # >>>
        if self.context_sample_size is not None and self.training:
            subset = np.random.permutation(candidate_x.shape[0])[
                : self.context_sample_size
            ]
            subset = torch.Tensor(subset).long()
            candidate_x = candidate_x[subset]

        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            # NOTE: during evaluation, candidate keys can be computed just once, which
            # looks like an easy opportunity for optimization. However:
            # - if your dataset is small or/and the encoder is just a linear layer
            #   (no embeddings and encoder_n_blocks=0), then encoding candidates
            #   is not a bottleneck.
            # - implementing this optimization makes the code complex and/or unobvious,
            #   because there are many things that should be taken into account:
            #     - is the input coming from the "train" part?
            #     - is self.training True or False?
            #     - is PyTorch autograd enabled?
            #     - is saving and loading checkpoints handled correctly?
            # This is why we do not implement this optimization.

            # When memory_efficient is True, this potentially heavy computation is
            # performed without gradients.
            # Later, it is recomputed with gradients only for the context objects.
            candidate_k = (
                self._encode(candidate_x)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(batch)[1]
                        for batch in delu.iter_batches(
                            candidate_x, self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        x, k = self._encode(x)
        if self.training:
            # NOTE: here, we add the training batch back to the candidates after the
            # function `apply_model` removed them. The further code relies
            # on the fact that the first batch_size candidates come from the
            # training batch.
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        # >>>
        # The search below is optimized for larger datasets and is significantly faster
        # than the naive solution (keep autograd on + manually compute all pairwise
        # squared L2 distances + torch.topk).
        # For smaller datasets, however, the naive solution can actually be faster.
        batch_size, d_main = k.shape
        device = k.device

        with torch.no_grad():
            if self.search_index is None:
                if device.type == "cuda":
                    # build a flat (CPU) index
                    index_flat = faiss.IndexFlatL2(d_main)
                    # make it into a gpu index
                    self.search_index = faiss.index_cpu_to_gpu(
                        faiss.StandardGpuResources(), 0, index_flat
                    )
                else:
                    self.search_index = faiss.IndexFlatL2(d_main)
                # self.search_index = (
                #     faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                #     if device.type == "cuda"
                #     else faiss.IndexFlatL2(d_main)
                # )
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            self.search_index.add(candidate_k)  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k, context_size + (1 if self.training else 0)
            )
            if self.training:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert self.training
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                torch.cat([x, candidate_x])[context_idx].flatten(0, 1)
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb.squeeze(-2) + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x
