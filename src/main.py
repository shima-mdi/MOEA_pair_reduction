import numpy as np
from src.data import DATA_DIR
from src.utils import evaluation, genetic, simulation


def Initial_learning(pcm):
    pvs_num, pvs_num_ = np.shape(pcm)
    pcm = np.ones((pvs_num, pvs_num_))
    pcm[range(pvs_num), range(pvs_num)] = 0
    return pcm


def main():

    n_compared_pairs = []
    pearson = []
    spearman = []
    mae = []

    n_iter = 100

    # degraded images per a reference image
    n_images = 16

    # full design comparison
    full_design = (n_images*(n_images - 1))/2

    # simulate dataset
    error_rate = 0.1
    mu, std = simulation.generate_dataset(n_images)

    # number of genes in one chromosome
    n_pairs = int(full_design)

    # define the population size
    n_pop = 50

    r_cross = 0.9
    r_mut = 1.0 / float(n_pairs)

    pcm = np.ones([n_images, n_images])
    global_pcm = Initial_learning(pcm)
    n_subjects = 15

    # loop over the MOEA untill  maximuim number of pairs is reach
    count = (sum(sum(global_pcm))-full_design)/2
    while (count < (n_subjects * full_design)):
     
        GA = genetic.GeneticAlgorithm(n_pairs, n_iter, n_pop, r_cross, r_mut, global_pcm)
        best, score = GA.run()
        print('Done!')
        print('f(%s) = %f' % (best, score))

        # simulate subjective test and update global pcm
        global_pcm = simulation.simulate_subjective_test(mu, std, error_rate, best, global_pcm, n_images)
        # print(global_pcm)
        # learn performance
        [p, s, m] = evaluation.learning_performance_evaluation(global_pcm, mu)

        # save the performance evaluation and number of pairs
        n_compared_pairs.append(sum(sum(global_pcm))-full_design)
        pearson.append(p)
        spearman.append(s)
        mae.append(m)

        count = (sum(sum(global_pcm))-full_design)/2
        print(f"performance plc={p} and count {count}")
    print(global_pcm)


if __name__ == '__main__':
    main()