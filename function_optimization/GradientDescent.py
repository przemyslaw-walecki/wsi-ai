'''
Gradient Descent Function Optimization Algortihm
'''
from numdifftools import Gradient
import time
import numpy as np


class GradientResults:
    def __init__(self, generations) -> None:
        self.values = np.zeros(generations)
        self.times = np.zeros(generations)


class GradientParams:
    def __init__(self) -> None:
        self.generations: int
        self.step: float
        self.function: function
        self.x0 = None


def L2norm(vector: np.array):
    return np.sqrt(sum(vector**2))


def gradient_descent(grad_params: GradientParams):
    f = grad_params.function
    gradient = Gradient(f)

    currentX = prevX = grad_params.x0
    step = grad_params.step

    results = GradientResults(grad_params.generations)
    results.values[0] = f(currentX)

    start = time.time_ns()
    currentX = currentX - step * gradient(grad_params.x0)
    generation_count = 0

    while generation_count < grad_params.generations - 1:
        generation_count += 1
        while f(currentX) >= f(prevX):
            step *= 0.9
            currentX = prevX - step * gradient(prevX)

        results.values[generation_count] = f(currentX)
        results.times[generation_count] = (time.time_ns() - start) / 1000000

        prevX = currentX
        currentX = currentX - step * gradient(currentX)

    results.values[generation_count] = f(currentX)
    results.times[generation_count] = (time.time_ns() - start) / 1000000

    return results
