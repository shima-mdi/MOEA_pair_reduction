import numpy as np



def simulate_subjective_test(score, std, error_rate, next_pair, pre_pcm, n_images):

    pcm = np.zeros([n_images, n_images])
    ind = np.triu_indices(n_images, 1)
    next_pair_mtx = np.zeros((n_images,n_images)).astype(int)
    next_pair_mtx[ind] = next_pair
    for pair_idx in range(len(ind[0])):
        i = ind[0][pair_idx]
        j = ind[1][pair_idx]

        if(next_pair_mtx[i][j] != 1): 
            continue

    
        DATA_i = np.random.normal(score[i], std[i])
        DATA_j = np.random.normal(score[j], std[j])
        rand = np.random.uniform(0, 1)
        if DATA_i > DATA_j:
            if rand > error_rate:
                pcm[i,j] = pcm[i,j]+1
            else:
                pcm[j,i] = pcm[j,i]+1
    
        elif DATA_i < DATA_j:
            if rand > error_rate:
                pcm[j,i] = pcm[j,i]+1
            else:
                pcm[i,j] = pcm[i,j]+1
        else:
            if rand > 0.5:
                pcm[i,j] =  pcm[i,j]+1
            else:
                pcm[j,i] = pcm[j,i]+1
            
    pcm = pcm + pre_pcm
    return pcm


def generate_dataset(n_images):
    iteration_num = 100
    score = []
    std = []
    score = 5*np.random.uniform(low=0.0, high=1.0, size=(iteration_num, n_images))
    std = 0.7*np.random.uniform(low=0.0, high=1.0, size=(iteration_num, n_images))
    return score, std