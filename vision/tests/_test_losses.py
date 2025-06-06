import torch
import torch.nn.functional as F

from src.metrics import (
    logistic_bregman_multiclass,
    logistic_bregman_binary,
    binary_cross_entropy,
    entropy,
    binary_entropy,
    cross_entropy_with_logits,
    kl_divergence,
    binary_kl_divergence,
)


def test_logistic_bregman_multiclass():
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    label_logits = torch.tensor([[5.0, 0.0, -10.0], [1.0, 2.0, 3.0]])

    result = logistic_bregman_multiclass(logits, label_logits)

    assert result.shape == (2,)
    assert torch.all(result >= 0)  # The divergence should be non-negative

    expected = torch.tensor([3.4076 - 5.0067 - (-3.9598), 0.0])
    kl_out = kl_divergence(
        torch.softmax(logits, dim=1), torch.softmax(label_logits, dim=1)
    )

    xe = cross_entropy_with_logits(logits, torch.softmax(label_logits, dim=1))
    ent = entropy(label_logits)

    assert torch.allclose(result, expected, atol=1e-4)
    assert torch.allclose(result, kl_out, atol=1e-4)
    assert torch.allclose(result, xe - ent, atol=1e-4)


def test_logistic_bregman_binary():
    logits = torch.tensor(
        [
            [
                1.0,
            ],
            [-4.0],
        ]
    )
    label_logits = torch.tensor([[-4.0], [-4.0]])

    result = logistic_bregman_binary(logits, label_logits)

    assert result.shape == (2,)
    assert torch.all(result >= 0)  # The divergence should be non-negative

    expected = torch.tensor(
        [
            1.3132616875182228
            - 0.01814992791780978
            - (0.017986209962091555 * (1.0 + 4.0)),
            0.0,
        ]
    )
    kl_out = binary_kl_divergence(torch.sigmoid(logits), torch.sigmoid(label_logits))

    xe = binary_cross_entropy(logits, torch.sigmoid(label_logits))
    ent = binary_entropy(label_logits)

    assert torch.allclose(result, expected, atol=1e-4)
    assert torch.allclose(result, kl_out, atol=1e-4)
    assert torch.allclose(result, xe - ent, atol=1e-4)


# if __name__ == "__main__":
#     pytest.main()
