from torch import nn
import torch
import torch.nn.functional as F
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    target: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        target: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    target = target.float()
    print(target)
    p = torch.softmax(inputs,dim = -1)
    print(-torch.log(p))
    ce_loss = -torch.log(p) * target 
    ce_loss2 = -torch.log((1 - p)) * (1 - target)
    print(ce_loss)
    print(ce_loss2)

    # print(p * target)
    print(F.cross_entropy(inputs, target,reduction="none"))

    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


batch_size = 16
inputs = torch.randn(batch_size, 5, requires_grad=True)
target = torch.randint(0, 5, (batch_size,))
one_hot_target = torch.zeros_like(inputs)
one_hot_target.scatter_(1, target.unsqueeze(1), 1.0)
one_hot_target
loss = sigmoid_focal_loss(inputs, one_hot_target, alpha=0.25, gamma=2, reduction="mean")
print(loss)