'''
Gradient Descent and Evolutionary Algorithms
Implemented to Minimize cec2017 benchmark functions
'''

import matplotlib
from GradientDescent import gradient_descent, GradientParams
from Evolutionary import evolutionaryAlgorithm, EvoParams
from cec2017.functions import f1, f9
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt


def main():

    generations = 100
    algorithm_samples = 10
    dimensions = 10
    fun = f9

    grad_params = GradientParams()
    grad_params.function = fun
    grad_params.generations = generations
    grad_params.step = 0.5

    grad_desc_results = np.zeros((generations))
    for _ in range(algorithm_samples):
        grad_params.x0 = np.random.uniform(-100, 100, dimensions)
        grad_desc_results += gradient_descent(grad_params).values
    grad_desc_results /= algorithm_samples

    evo_params = EvoParams()
    evo_params.function = fun
    evo_params.fun_dims = 10
    evo_params.generations = generations
    evo_params.mutation_rate = 0.5
    evo_params.crossbreed_rate = 0.2
    evo_params.population_size = 1024
    evo_params.sample_size = 50

    evo_alg_results = np.zeros((generations, evo_params.sample_size))
    for _ in range(algorithm_samples):
        evo_alg_results += evolutionaryAlgorithm(evo_params).samples

    evo_alg_results /= algorithm_samples

    plt.figure()
    plt.plot(grad_desc_results, label=f'Gradient Descent, step={grad_params.step}')

    generations = np.arange(generations)
    for i in range(evo_params.sample_size):
        plt.scatter(generations, evo_alg_results[:, i], marker="o", s=0.5)

    plt.xlabel("Generation")
    plt.yscale('log')
    plt.ylabel("Function Value (log scale)")
    plt.title("Function f9 minimization results")
    plt.legend()
    plt.grid(True)

    plt.savefig(fname=f"Opimization_f1v2.pdf")


if __name__ == "__main__":
    main()
