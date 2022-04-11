import matplotlib.pyplot as plt
import scipy.io
import numpy as np





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
