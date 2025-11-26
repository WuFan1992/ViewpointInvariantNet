
import torch
from torch.autograd import Function


# -------- GRL (Gradient Reversal Layer) --------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        # forward is identity
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # reverse the gradient (multiply by -lambda)
        return -ctx.lambda_ * grad_output, None

def grl(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)