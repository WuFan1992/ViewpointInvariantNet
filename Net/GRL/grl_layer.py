
import torch
from torch.autograd import Function


# -------- GRL (Gradient Reversal Layer) --------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        # no reshape assumptions; just return x
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grl(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)