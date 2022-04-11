import matlab.engine
import numpy as np

from src.data import DATA_DIR
from src.utils import evaluation, genetic, loading_matlab, plotting, simulation
from src.utils.eig import run_modeling_Bradley_Terry

def Initial_learning(pcm):
    pvs_num, pvs_num_ = np.shape(pcm)
    pcm = np.ones((pvs_num, pvs_num_))
    pcm[range(pvs_num), range(pvs_num)] = 0
    return pcm


def gt_pcm(n, data):
    z = np.zeros((n, n))
    for row in data['data_ref']:
         z[row[0]-1,row[1]-1] = z[row[0]-1,row[1]-1]+1
    return z


def main():

    # Choices for mode are simultion and IQA
    mode = 'simulation'
    # mode = 'IQA'

    n_compared_pairs = []
    pearson = []
    spearman = []
    mae = []

    n_iter = 30

    # degraded images per a reference image
    n_images = 16

    # full design comparison
    full_design = (n_images * (n_images - 1)) / 2

    # simulate dataset
    error_rate = 0.1

    if mode == 'simulation':
        mu, std = simulation.generate_dataset(n_images)

    elif mode == 'IQA':
    # Read IQA dataset
        eng = matlab.engine.start_matlab()
        # Load first refernce of IQA dataset. You can change the pathe in utils/matlab.py to load onother refernce
        IQA_data = loading_matlab.load_mat_file()
        z = gt_pcm(n_images, IQA_data)
        gt_score_IQA, _ = run_modeling_Bradley_Terry(z)

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
        # global_pcm = simulation.simulate_subjective_test(mode, mu, std, error_rate, best, global_pcm, n_images)
        
        # learn performance
        if mode == 'simulation':
            global_pcm = simulation.simulate_subjective_test(mode, mu, error_rate, best, global_pcm, n_images, std)
            [p, s, m] = evaluation.learning_performance_evaluation(mode, global_pcm, mu)

        elif mode == 'IQA':
            global_pcm = simulation.simulate_subjective_test(mode, IQA_data, error_rate, best, global_pcm, n_images)
            [p, s, m] = evaluation.learning_performance_evaluation(mode, global_pcm, gt_score_IQA)

        # save the performance evaluation and number of pairs
        pearson.append(p)
        spearman.append(s)
        mae.append(m)

        count += sum(best)
        n_compared_pairs.append(count)
        print(f"performance plc={p} and count {count}")

    print(global_pcm)
    plotting.state_of_the_art(n_compared_pairs, pearson)
    # plot(n_compared_pairs, pearson)


if __name__ == '__main__':
    main()
