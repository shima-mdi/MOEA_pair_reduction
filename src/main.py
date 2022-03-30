import datetime
import json
import math
import random
import sys
from collections import Counter
from functools import reduce
from pathlib import Path
from random import Random
from statistics import mean
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.hermite as herm
from numpy.random import rand, randint
from scipy import linalg
from sklearn import preprocessing

from src.data import DATA_DIR
from src.utils import evaluation, genetic, simulation


def Initial_learning(pcm):
    pvs_num,pvs_num_ = np.shape(pcm) 
    pcm = np.ones((pvs_num,pvs_num_))
    pcm[range(pvs_num),range(pvs_num)]=0
    return pcm


def main():
   
    n_compared_pairs = []
    performance = []
    
    n_iter = 40

    # degraded images per a reference image
    n_images = 5

    # full design comparison
    full_design = (n_images*(n_images - 1))/2

    # simulate dataset
    error_rate = 0.1
    score, std = simulation.generate_dataset(n_images)

    n_pairs = 10

    # define the population size
    n_pop = 10

    r_cross = 0.9
    r_mut = 1.0 / float(n_pairs)

    pcm = np.ones([n_images, n_images])
    global_pcm = Initial_learning(pcm)

    GA = genetic.GeneticAlgorithm(n_pairs, n_iter, n_pop, r_cross, r_mut, global_pcm)

    best, score = GA.run()
    print('Done!')
    print('f(%s) = %f' % (best, score))

    # while (sum(global_pcm) < full_design):
        
    #     GA = genetic.GeneticAlgorithm(n_pairs, n_iter, n_pop, r_cross, r_mut, global_pcm)

    #     best, score = GA.run()
    
    #     print('Done!')
    #     print('f(%s) = %f' % (best, score))

    #     # simulate subjective test and update global pcm
    #     global_pcm = simualtion.simulate_decision(score, std, error_rate, best, global_pcm, n_images)

    #     # learn performance
    #     p = evaluation.learning_performance_evaluation(global_pcm, score)

    #     # save the performance evaluation and number of pairs
    #     n_compared_pairs.append(sum(global_pcm))
    #     performance.append(p)

    # loop over the MOEA untill  maximuim number of pairs is reach
        

            
if __name__ == '__main__':
    main()    

