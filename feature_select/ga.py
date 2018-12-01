"""
A binary genetic algorithm implementation
"""
from random import randint, choices, random
import numpy as np
from functools import total_ordering

@total_ordering
class GAIndividual(object):

    def __init__(self):
        self.feature_vector = []
        self.feature_importance = None
        self.fitness = None

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __repr__(self):
        return f'<{self.fitness}, {self.feature_vector}>'

    def __len__(self):
        return len(self.feature_vector)

    def __getitem__(self, item):
        return self.feature_vector[item]

    def __setitem__(self, key, value):
        self.feature_vector[key] = value

    def mutate(self, mutation_rate):
        """Mutate features w/ probability rate"""
        # """For each feature in feature_vector, mutate it with a probability inversely proportional to that feature's
        # importance"""

        if not self.feature_vector or not self.feature_importance:
            return self

        self.feature_vector = [x if random() > mutation_rate else int(not x) for x in self.feature_vector]
        self.feature_importance = None
        self.fitness = None

        return self


    @classmethod
    def random(cls, size):
        ind = GAIndividual()
        while not ind.feature_vector or max(ind.feature_vector) == 0:
            ind.feature_vector = [randint(0, 1) for i in range(size)]
        return ind

    @classmethod
    def crossover(cls, ind1, ind2):
        """Implement a 1-point crossover"""
        child1 = GAIndividual()
        child2 = GAIndividual()

        # [cp1, cp2] = choices(range(len(ind1)), k=2)
        cp = randint(0, len(ind1) - 1)

        child1[:cp] = ind1[:cp]
        child1[cp:] = ind2[cp:]

        child2[:cp] = ind2[:cp]
        child2[cp:] = ind1[cp:]

        return [child1, child2]

class GeneticAlgorithmFeatureSelection(object):

    def __init__(self, fitness_func, data, target, mu=10, lambda_=50, fitness_args=[], mutation_rate=0.2):
        self.data = data
        self.target = target
        self.fitness_func = fitness_func
        self.fitness_args = fitness_args
        self.mu = mu
        self.lambda_ = lambda_
        self.individual_size = len(data.columns) - 1
        self.mutation_rate = mutation_rate

        self.current_generation = []

        self.best_candidate = None
        self.best_fitness = None

        self.iterations = 0

        self.iterate()


    def iterate(self):
        # Generate current generation if needed
        if not self.current_generation:
            self.current_generation = self.generate_population(self.lambda_)
            self.evaluate_fitness()


        # Generate next generation w/ crossover and mutation
        # Select MU parents
        # Calc relative weights


        X = np.array([x.fitness for x in self.current_generation])
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        next_generation = []
        while len(next_generation) < self.lambda_:
            [p1, p2] = choices(self.current_generation, weights=X_scaled, k=2)

            next_generation.extend(GAIndividual.crossover(p1, p2))

        self.current_generation = [x.mutate(self.mutation_rate) for x in next_generation]
        self.evaluate_fitness()
        self.current_generation = sorted(self.current_generation, reverse=True)

        self.iterations += 1

        print(f'iteration {self.iterations} complete- best fitness {self.best_candidate}, generation best {self.current_generation[0]}')


    # Generate population
    def generate_population(self, num_individuals):
        return [GAIndividual.random(self.individual_size)
                    for j in range(num_individuals)]

    # Evaluate fitness
    def evaluate_fitness(self):

        for x in self.current_generation:
            if x.fitness:
                continue
            mse, feat_importance = self.fitness_func(x.feature_vector, self.data, self.target, *self.fitness_args)

            x.fitness = -mse
            x.feature_importance = feat_importance

            # Update best candidate
            if not self.best_candidate or x > self.best_candidate:
                self.best_candidate = x


    def run_iterations(self, num_iterations):
        for x in range(num_iterations):
            self.iterate()
