'''Evolutionary Function Optimization Algorithm'''

import numpy as np


class EvoResults:
    def __init__(self) -> None:
        self.samples = []


class EvoParams:
    def __init__(self) -> None:
        self.population_size: int
        self.sample_size: int
        self.crossbreed_rate: float
        self.mutation_rate: float
        self.function: function
        self.fun_dims: int
        self.generations: int

def mutation(population, mutation_rate):
    for x in population:
        x += mutation_rate * np.random.normal(0, 1, size=len(x))
    return population


def crossing(parents, crossbreed_rate, population_size):
    population = []
    for _ in range(population_size):
        pIDS = np.random.choice(population_size, size=2, replace=False)
        if np.random.rand() < crossbreed_rate:
            a = np.random.random()
            population.append(parents[pIDS[0]] * a + parents[pIDS[1]] * (1 - a))
            a = np.random.random()
            population.append(parents[pIDS[0]] * a + parents[pIDS[1]] * (1 - a))
        else:
            population.append(parents[pIDS[0]])
            population.append(parents[pIDS[1]])

    return np.array(population)


def tournament_reproduction(population, evo_params: EvoParams, results: EvoResults):
    parents = []
    fun = evo_params.function
    psize = evo_params.population_size
    sample_size = evo_params.sample_size
    eval = [fun(x) for x in population]

    for _ in range(psize):
        cIDS = np.random.choice(psize, size=2, replace=False)
        if eval[cIDS[0]] < eval[cIDS[1]]:
            parents.append(population[cIDS[0]])
        else:
            parents.append(population[cIDS[1]])
    parents_eval = [fun(x) for x in parents]
    results.samples.append((sorted(np.random.choice(parents_eval, size=sample_size))))
    return np.array(parents)


def evolutionaryAlgorithm(evo_params: EvoParams):
    psize = evo_params.population_size
    results = EvoResults()
    population = np.random.uniform(-100, 100, (psize, evo_params.fun_dims))
    for __ in range(evo_params.generations):
        population = mutation(
            crossing(
                tournament_reproduction(population, evo_params, results),
                evo_params.crossbreed_rate,
                psize,
            ),
            evo_params.mutation_rate,
        )

    return results
