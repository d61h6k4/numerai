# from numerai.model import loss_fn
import torch
import unittest

from numerai.model import EmbeddingsLayer

# def test_loss_fn():
#     x = torch.rand(4, 5)
#     print(x, torch.argmax(x, dim=-1))
#     print(loss_fn(x, {"target": torch.argmax(x, dim=-1)}))
#     print(loss_fn(x, {"target": torch.tensor([0, 1, 2, 3])}))


class TestEmbeddingsLayer(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(0)

        inputs = {
            "feature_0": torch.tensor([0, 1, 2, 3, 4]),
            "feature_1": torch.tensor([4, 3, 2, 1, 0]),
        }

        m = EmbeddingsLayer({"feature_0", "feature_1"})
        result = m(inputs)

        self.assertEqual(result.shape, (5, 3 * 2))


class TestLowRankMixtureCrossNet(unittest.TestCase):
    def test_gating(self):
        torch.manual_seed(0)

        in_features = 10
        num_experts = 5
        batch_size = 4
        x = torch.rand((batch_size, in_features))
        gates = torch.nn.Linear(in_features, num_experts, bias=False)
        g = gates(x)

        self.assertEqual(g.shape, (batch_size, num_experts))

    def test_expert_matmul(self):
        torch.manual_seed(0)

        in_features = 10
        low_rank = 3
        num_experts = 5

        batch_size = 4
        x = torch.rand((batch_size, in_features, 1))

        V_kernel = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(
                torch.empty(num_experts, low_rank, in_features)
            )
        )
        C_kernel = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty(num_experts, low_rank, low_rank))
        )

        experts = []
        for i in range(num_experts):
            expert = torch.matmul(V_kernel[i], x)
            expert = torch.matmul(C_kernel[i], torch.relu(expert))
            experts.append(expert.squeeze(2))
        experts = torch.stack(experts, 2)

        self.assertEqual(experts.shape, (batch_size, low_rank, num_experts))
