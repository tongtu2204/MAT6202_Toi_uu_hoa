"""One file per algorithm. Every optimizer returns an `OptResult` whose history
lets `benchmark`/`plots` draw the log(f - f*) convergence curves.

    gd     -> gradient descent (fixed step / optional backtracking)
    agd    -> Nesterov accelerated gradient
    newton -> damped / line-search Newton (= IRLS)
    sgd    -> minibatch stochastic gradient descent
    ista   -> ISTA / FISTA / coordinate descent / subgradient (the L1 bonus)
    adaptive -> AdaGrad / RMSprop / Adam (+AdamW, AMSGrad): diagonal preconditioners
"""
from .base import OptResult
from .gd import gradient_descent
from .agd import accelerated_gd
from .newton import newton
from .sgd import sgd
from .ista import ista, fista, coordinate_descent, subgradient
from .adaptive import adaptive, adagrad, rmsprop, adam

__all__ = [
    "OptResult",
    "gradient_descent", "accelerated_gd", "newton", "sgd", "ista", "fista",
    "coordinate_descent", "subgradient", "adaptive", "adagrad", "rmsprop", "adam",
]
