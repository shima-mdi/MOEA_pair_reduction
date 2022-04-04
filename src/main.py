import numpy as np
from src.data import DATA_DIR
from src.utils import evaluation, genetic, simulation
import matplotlib.pyplot as plt
import scipy.io


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


def state_of_the_art(n_compared_pairs, pearson):
    """A function to compare the state of the art performance
    """
    x = np.arange(1, 1801).reshape(1,-1)

    crowdBT = scipy.io.loadmat('crowdBT.plcc.mat')
    hrrg_active = scipy.io.loadmat('hrrg.active.plcc.mat')
    hrrg_random = scipy.io.loadmat('hrrg.random.plcc.mat')
    Hyper_mst = scipy.io.loadmat('hpermst.plcc.mat')
    trial = scipy.io.loadmat('hypermst.trail.mat')

    crowdBT = crowdBT['plcc_result']
    hrrg_active = hrrg_active['plcc_result']
    hrrg_random = hrrg_random['plcc_result']
    Hyper_mst = Hyper_mst['plcc_result']
    trial = trial['trial']

    pairs = []
    crowdBT_plcc = []
    hrrg_active_plcc = []
    hrrg_random_plcc = []
    for i in range (1800):
        pairs.append(x[0][i])
        crowdBT_plcc.append(crowdBT[0][i])
        hrrg_active_plcc.append(hrrg_active[0][i])
        hrrg_random_plcc.append(hrrg_random[0][i])

    y = []
    Hyper_mst_plcc =[]
    for i in range(233):
        y.append(trial[0,i])
        Hyper_mst_plcc.append(Hyper_mst[0][i]) 
    # plt.plot(pairs, np.arctanh(crowdBT_plcc), color='r', label='Crowd-BT')
    # plt.plot(pairs, np.arctanh(hrrg_active_plcc), color='g', label='HRRG-active')
    # plt.plot(pairs, np.arctanh(hrrg_random_plcc), color='b', label='HRRG-random')
    # plt.plot(y, np.arctanh(Hyper_mst_plcc), color='c', label='Hyper-mst')
    # plt.plot(n_compared_pairs, np.arctanh(pearson),  color='m', label='Genetic-based')

    plt.plot(pairs, crowdBT_plcc, color='r', label='Crowd-BT')
    plt.plot(pairs, hrrg_active_plcc, color='g', label='HRRG-active')
    plt.plot(pairs, hrrg_random_plcc, color='b', label='HRRG-random')
    plt.plot(y, Hyper_mst_plcc, color='c', label='Hyper-mst')
    plt.plot(n_compared_pairs, pearson,  color='m', label='Genetic-based')


    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("number of pairs")
    plt.ylabel("PLCC")
    plt.title("State of the art comparison")
    
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    
    # To load the display window
    plt.show()
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

    # Read IQA dataset

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
       
        pearson.append(p)
        spearman.append(s)
        mae.append(m)

        count += sum(best)
        n_compared_pairs.append(count)
        print(f"performance plc={p} and count {count}")

    print(global_pcm)
    state_of_the_art(n_compared_pairs, pearson)
    # plot(n_compared_pairs, pearson)


if __name__ == '__main__':
    main()