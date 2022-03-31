
from functools import reduce

import numpy as np
from numpy.random import rand, randint
from src.utils import eig

class GeneticAlgorithm:
    """
    This a class that implements the genetic algorithm 
    to find the best pairs out of all the possible pairs.
    """
    def __init__(self, n_pairs, n_iter, n_pop, r_cross, r_mut, global_pcm) -> None:
        self.n_pairs = n_pairs
        self.n_iter = n_iter
        self.n_pop = n_pop
        self.r_cross = r_cross
        self.r_mut = r_mut
        self.global_pcm = global_pcm

    def fitness_func(self, solution):
        [mu, mu_cova] = eig.run_modeling_Bradley_Terry(self.global_pcm)
        EIG_mtx = eig.EIG(mu, mu_cova)
        solution = np.array(solution)

        # keeping index of the selected pairs
        result = np.where(solution == 1)

        # Fill the temp list with the EIG of the corresponding indexex 
        temp = []
        temp = list(map(lambda a : EIG_mtx[a], result[0]))
        
        # normalize the data 
        temp = [float(i)/sum(temp) for i in temp]

        sum_eig = 0.0
        if temp:
            sum_eig = reduce(lambda x, y:float(x)+float(y), temp)
            out = sum_eig/len(temp)
        else:
            out = 0.0
        # print(f"eig is {out}, count is {(sum(solution)/len(solution))}")
        # return  sum(solution)
        return  ((10*out) - (sum(solution)/len(solution)))

    def selection(self, pop, scores, k=3):
        selection_ix = randint(len(pop))
        for ix in randint(0, len(pop), k-1):
            if scores[ix] > scores[selection_ix]:
                selection_ix = ix
        return pop[selection_ix]

    def crossover(self, p1, p2):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        if rand() < self.r_cross:
            pt = randint(1, len(p1)-2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def mutation(self, bitstring):
        for i in range(len(bitstring)):
            if rand() < self.r_mut:
                bitstring[i] = 1 - bitstring[i]

    
    def run(self):
        pop = [randint(0, 2, self.n_pairs).tolist() for _ in range(self.n_pop)]
        
        best, best_eval = 0, self.fitness_func(pop[0])
        for gen in range(self.n_iter):
            # print(gen)
            scores = [self.fitness_func(c) for c in pop]
            for i in range(self.n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    # print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))

            # select parents
            selected = [self.selection(pop, scores) for _ in range(self.n_pop)]
            children = list()
            for i in range(0, self.n_pop, 2):
                p1, p2 = selected[i], selected[i+1]
                for c in self.crossover(p1, p2):
                    self.mutation(c)
                    children.append(c)
                pop = children
        return [best, best_eval]
    