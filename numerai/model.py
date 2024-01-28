from typing import Callable, Mapping, Optional, Union
from torch import nn
import torch


class LowRankMixtureCrossNet(torch.nn.Module):
    r"""
    Low Rank Mixture Cross Net is a DCN V2 implementation from the `paper
    <https://arxiv.org/pdf/2008.13535.pdf>`_:

    `LowRankMixtureCrossNet` defines the learnable crossing parameter per layer as a
    low-rank matrix :math:`(N*r)` together with mixture of experts. Compared to
    `LowRankCrossNet`, instead of relying on one single expert to learn feature crosses,
    this module leverages such :math:`K` experts; each learning feature interactions in
    different subspaces, and adaptively combining the learned crosses using a gating
    mechanism that depends on input :math:`x`..

    On each layer l, the tensor is transformed into:

    .. math::    x_{l+1} = MoE({expert_i : i \in K_{experts}}) + x_l

    and each :math:`expert_i` is defined as:

    .. math::   expert_i = x_0 * (U_{li} \cdot g(C_{li} \cdot g(V_{li} \cdot x_l)) + b_l)

    where :math:`U_{li} (N, r)`, :math:`C_{li} (r, r)` and :math:`V_{li} (r, N)` are
    low-rank matrices, :math:`*` means element-wise multiplication, :math:`x` means
    matrix multiplication, and :math:`g()` is the non-linear activation function.

    When num_expert is 1, the gate evaluation and MOE will be skipped to save
    computation.

    Args:
        in_features (int): the dimension of the input.
        num_layers (int): the number of layers in the module.
        low_rank (int): the rank setup of the cross matrix (default = 1).
            Value must be always >= 1
        activation (Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the non-linear activation function, used in defining experts.
            Default is relu.

    Example::

        batch_size = 3
        num_layers = 2
        in_features = 10
        input = torch.randn(batch_size, in_features)
        dcn = LowRankCrossNet(num_layers=num_layers, num_experts=5, low_rank=3)
        output = dcn(input)
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        num_experts: int = 1,
        low_rank: int = 1,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
    ) -> None:
        super().__init__()
        assert num_experts >= 1, "num_experts must be larger or equal to 1"
        assert low_rank >= 1, "Low rank must be larger or equal to 1"

        self._num_layers = num_layers
        self._num_experts = num_experts
        self._low_rank = low_rank
        self._in_features = in_features
        self.U_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(
                            self._num_experts, self._in_features, self._low_rank
                        )
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.V_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(
                            self._num_experts, self._low_rank, self._in_features
                        )
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.zeros_(torch.empty(self._in_features, 1))
                )
                for i in range(self._num_layers)
            ]
        )
        self.gates: Optional[torch.nn.Module] = (
            torch.nn.ModuleList(
                [
                    torch.nn.Linear(self._in_features, 1, bias=False)
                    for i in range(self._num_experts)
                ]
            )
            if self._num_experts > 1
            else None
        )

        self._activation = activation
        self.C_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(self._num_experts, self._low_rank, self._low_rank)
                    )
                )
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): tensor with shape [batch_size, in_features].

        Returns:
            torch.Tensor: tensor with shape [batch_size, in_features].
        """

        x_0 = input.unsqueeze(2)  # (B, N, 1)
        x_l = x_0

        for layer in range(self._num_layers):
            # set up gating:
            if self._num_experts > 1:
                gating = []
                for i in range(self._num_experts):
                    # pyre-ignore[16]: `Optional` has no attribute `__getitem__`.
                    gating.append(self.gates[i](x_l.squeeze(2)))
                gating = torch.stack(gating, 1)  # (B, K, 1)

            # set up experts
            experts = []
            for i in range(self._num_experts):
                expert = torch.matmul(
                    self.V_kernels[layer][i],
                    x_l,
                )  # (B, r, 1)
                expert = torch.matmul(
                    self.C_kernels[layer][i],
                    self._activation(expert),
                )  # (B, r, 1)
                expert = torch.matmul(
                    self.U_kernels[layer][i],
                    self._activation(expert),
                )  # (B, N, 1)
                expert = x_0 * (expert + self.bias[layer])  # (B, N, 1)
                experts.append(expert.squeeze(2))  # (B, N)
            experts = torch.stack(experts, 2)  # (B, N, K)

            if self._num_experts > 1:
                # MOE update
                moe = torch.matmul(
                    experts,
                    # pyre-ignore[61]: `gating` may not be initialized here.
                    torch.nn.functional.softmax(gating, 1),
                )  # (B, N, 1)
                x_l = moe + x_l  # (B, N, 1)
            else:
                x_l = experts + x_l  # (B, N, 1)

        return torch.squeeze(x_l, dim=2)  # (B, N)


class DeepNetwork(torch.nn.Module):
    def __init__(self, num_layers: int, in_features: int):
        super().__init__()
        layers = (
            [
                torch.nn.BatchNorm1d(num_features=in_features),
                torch.nn.Dropout1d(p=0.1),
                nn.Linear(
                    in_features=in_features, out_features=2 * in_features, bias=False
                ),
                nn.ReLU(),
            ]
            + [
                torch.nn.BatchNorm1d(num_features=2 * in_features),
                torch.nn.Dropout1d(p=0.1),
                nn.Linear(
                    in_features=2 * in_features,
                    out_features=2 * in_features,
                    bias=False,
                ),
                nn.ReLU(),
            ]
            * (num_layers - 2)
            + [
                torch.nn.BatchNorm1d(num_features=2 * in_features),
                torch.nn.Dropout1d(p=0.1),
                nn.Linear(
                    in_features=2 * in_features, out_features=in_features, bias=False
                ),
            ]
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DCNv2(nn.Module):
    def __init__(self, num_layers: int, in_features: int):
        super().__init__()
        self.cross_network = LowRankMixtureCrossNet(
            in_features=in_features,
            num_layers=num_layers,
            num_experts=5,
            low_rank=in_features // 4,
        )
        self.deep_network = DeepNetwork(num_layers, in_features)

    def forward(self, x):
        return self.deep_network(self.cross_network(x))


class EmbeddingsLayer(torch.nn.Module):
    def __init__(self, features: set[str]):
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                feature_name: torch.nn.Embedding(num_embeddings=5, embedding_dim=3)
                for feature_name in features
            }
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for fn in self.layers:
            embeddings.append(self.layers[fn](inputs[fn]))
        return torch.concatenate(embeddings, dim=1)


class NumeraiModel(nn.Module):
    def __init__(self, features: set[str]):
        super().__init__()

        num_features = len(features)

        self.embeddings_layer = EmbeddingsLayer(features)
        self.shared_bottom = DCNv2(3, 3 * num_features)
        self.head_target = nn.Sequential(
            torch.nn.BatchNorm1d(num_features=3 * num_features),
            torch.nn.Dropout1d(p=0.1),
            nn.Linear(3 * num_features, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, input: Mapping[str, torch.Tensor]) -> torch.Tensor:
        embeddings = self.embeddings_layer(input)
        x = self.shared_bottom(embeddings)
        target = self.head_target(x)
        return target


def create_loss_fn(weights, device: str = "cpu"):
    target_loss = nn.CrossEntropyLoss(weight=weights["target"]).to(device)

    def loss_fn(inputs, targets):
        return target_loss(input=inputs, target=targets["target"])

    return loss_fn
