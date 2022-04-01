import numpy as np
from src.data import DATA_DIR
from src.utils import evaluation, genetic, simulation
import matplotlib.pyplot as plt

def Initial_learning(pcm):
    pvs_num, pvs_num_ = np.shape(pcm)
    pcm = np.ones((pvs_num, pvs_num_))
    pcm[range(pvs_num), range(pvs_num)] = 0
    return pcm

def plot(x:list, y:list):
    plt.plot(x, y)
    plt.xlabel('Number of pairs')
    plt.ylabel('PLCC')
    plt.title("PLCC vs Number of pairs")
    plt.show()
    plt.legend('MOEA    ')
    plt.savefig('filename.png')


def main():

    n_compared_pairs = []
    pearson = []
    spearman = []
    mae = []

    n_iter = 10

    # degraded images per a reference image
    n_images = 16

    # full design comparison
    full_design = (n_images * (n_images - 1)) / 2

    # simulate dataset
    error_rate = 0.1
    mu, std = simulation.generate_dataset(n_images)

    # number of genes in one chromosome
    n_pairs = int(full_design)

    # define the population size(should be an even number)
     
    n_pop = 120

    r_cross = 0.9
    r_mut = 1.0 / float(n_pairs)

    pcm = np.ones([n_images, n_images])
    global_pcm = Initial_learning(pcm)
    n_subjects = 15

    # loop over the MOEA untill  maximuim number of pairs is reach
    count = 0
    while (count < (n_subjects * full_design)):
     
        GA = genetic.GeneticAlgorithm(n_pairs, n_iter, n_pop, r_cross, r_mut, global_pcm)
        best, score = GA.run()
        print(f"best solution is {best} and score is {score}")
        
        # simulate subjective test and update global pcm
        global_pcm = simulation.simulate_subjective_test(mu, std, error_rate, best, global_pcm, n_images)
        
        # learn performance
        [p, s, m] = evaluation.learning_performance_evaluation(global_pcm, mu)

        # save the performance evaluation and number of pairs
        n_compared_pairs.append(sum(sum(global_pcm))-full_design)
        pearson.append(p)
        spearman.append(s)
        mae.append(m)

        count += sum(best)
        print(f"performance plc={p} and count {count}")

    print(global_pcm)
    plot(n_compared_pairs, pearson)


if __name__ == '__main__':
    main()