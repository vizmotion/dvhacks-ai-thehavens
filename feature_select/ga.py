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
        return f'<{self.fitness}, {sum(self.feature_vector)}/{len(self.feature_vector)} features>'

    def __len__(self):
        return len(self.feature_vector)

    def __getitem__(self, item):
        return self.feature_vector[item]

    def __setitem__(self, key, value):
        self.feature_vector[key] = value

    def __hash__(self):
        return hash(frozenset(self.feature_vector))

    def mutate(self, mutation_feat_imp):
        """Mutate features w/ probability rate based on feature importance"""
        # """For each feature in feature_vector, mutate it with a probability inversely proportional to that feature's
        # importance"""

        if not self.feature_vector:
            return self

        features = np.array(self.feature_vector)

        rand_vec = np.random.rand(len(self))
        mutation_rate_vec = 1 - mutation_feat_imp

        mutated = rand_vec < mutation_rate_vec

        features[mutated] = 1 - features[mutated]

        self.feature_vector = features.tolist()
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
        """Implement a 2-point crossover"""
        child1 = GAIndividual()
        child2 = GAIndividual()

        cp1 = randint(0, len(ind1) - 2)
        cp2 = randint(cp1+1, len(ind1) - 1)

        child1[:cp1] = ind1[:cp1]
        child1[cp1:cp2] = ind2[cp1:cp2]
        child1[cp2:] = ind1[cp2:]

        child2[:cp1] = ind2[:cp1]
        child2[cp1:cp2] = ind1[cp1:cp2]
        child2[cp2:] = ind2[cp2:]

        return [child1, child2]

    @classmethod
    def full_rank(cls, individual_size):
        ind = GAIndividual()
        ind.feature_vector = [1 for x in range(individual_size)]
        return ind


class GeneticAlgorithmFeatureSelection(object):

    def __init__(self, fitness_func, data, target, mu=5, lambda_=20, fitness_args=[],
                 mutation_rate=0.1, sample_size=1000):
        self.data = data
        self.target = target
        self.fitness_func = fitness_func
        self.fitness_args = fitness_args
        self.mu = mu
        self.lambda_ = lambda_
        self.individual_size = len(data.columns) - 1

        self.feature_importance_sum = np.zeros(self.individual_size)
        self.feature_importance = np.zeros(self.individual_size)
        self.feature_importance_iter = 0

        self.mutation_rate = mutation_rate
        self.sample_size = sample_size

        self.current_generation = []

        self.best_candidate = None

        self.iterations = 0
        self.evaluated_members = {}

        self.iterate()


    def iterate(self):
        # Generate current generation if needed
        if not self.current_generation:
            self.current_generation = self.generate_population(self.lambda_)
            self.evaluate_fitness()


        # Generate next generation w/ crossover and mutation
        # Select MU parents
        # Calc relative weights
        elite = self.current_generation[:self.mu]

        X = np.array([x.fitness for x in self.current_generation])
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

        next_generation = []
        while len(next_generation) < self.lambda_:
            [p1, p2] = choices(self.current_generation, weights=X_scaled, k=2)

            next_generation.extend(GAIndividual.crossover(p1, p2))

        self.current_generation = elite + [x.mutate(self.feature_importance) for x in next_generation]
        self.evaluate_fitness()

        self.iterations += 1

        print(f'iteration {self.iterations} complete- generation best {self.current_generation[0]}, best fitness {self.best_candidate}')


    # Generate population
    def generate_population(self, num_individuals):
        return [GAIndividual.random(self.individual_size)
                    for j in range(num_individuals-1)] + [GAIndividual.full_rank(self.individual_size)]  # Always have a full-rank

    # Evaluate fitness
    def evaluate_fitness(self):
        fitness_data = self.data if self.data.shape[0] < self.sample_size else self.data.sample(self.sample_size)

        for x in self.current_generation:
            if x.fitness:
                continue

            if x in self.evaluated_members:
                x.fitness = self.evaluated_members[x]
                continue

            mse, feat_importance = self.fitness_func(x.feature_vector, fitness_data, self.target, *self.fitness_args)

            x.fitness = -mse
            x.feature_importance = feat_importance

            self.feature_importance_sum += feat_importance
            self.feature_importance_iter += 1

            self.evaluated_members[x] = x.fitness

            # Update best candidate
            if not self.best_candidate or x > self.best_candidate:
                self.best_candidate = x

        self.feature_importance = self.feature_importance_sum / self.feature_importance_iter

        self.feature_importance = (1 - self.mutation_rate*2) / (self.feature_importance.max() - self.feature_importance.min()) * \
                                  (self.feature_importance - self.feature_importance.min()) + self.mutation_rate

        self.current_generation = sorted(self.current_generation, reverse=True)

    def run_iterations(self, num_iterations):
        for x in range(num_iterations):
            self.iterate()
