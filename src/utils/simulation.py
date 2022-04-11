import numpy as np
import random



def simulate_subjective_test(mode, score, error_rate, next_pair, pre_pcm, n_images, *args): 

    
    pcm = np.zeros([n_images, n_images])
    ind = np.triu_indices(n_images, 1)
    next_pair_mtx = np.zeros((n_images,n_images)).astype(int)
    next_pair_mtx[ind] = next_pair
    for pair_idx in range(len(ind[0])):
        i = ind[0][pair_idx]
        j = ind[1][pair_idx]

        if(next_pair_mtx[i][j] != 1): 
            continue

        if mode == "simulation":
            std = args[0]
            DATA_i = np.random.normal(score[0, i], std[0, i])
            DATA_j = np.random.normal(score[0, j], std[0, j])
            rand = np.random.uniform(0, 1)
            if DATA_i > DATA_j:
                if rand > error_rate:
                    pcm[i,j] = pcm[i,j] + 1
                else:
                    pcm[j,i] = pcm[j,i] + 1
        
            elif DATA_i < DATA_j:
                if rand > error_rate:
                    pcm[j,i] = pcm[j,i] + 1
                else:
                    pcm[i,j] = pcm[i,j] + 1
            else:
                if rand > 0.5:
                    pcm[i,j] =  pcm[i,j] + 1
                else:
                    pcm[j,i] = pcm[j,i] + 1
                            

        elif mode == "IQA":
            res = simulate_decision_real_dataset(i, j, score)
            rand = np.random.uniform(0, 1)
            x = res[0]
            y = res[1]
            if rand > error_rate:
                pcm[x, y] = pcm[x, y] + 1
            else:
                 pcm[y, x] = pcm[y, x] + 1
            

        elif mode == "VQA":
            pass
        
    pcm = pcm + pre_pcm
    return pcm
        

def generate_dataset(n_images):
    iteration_num = 100
    score = []
    std = []
    score = 5 * np.random.uniform(low=0.0, high=1.0, size=(iteration_num, n_images))
    std = 0.7 * np.random.uniform(low=0.0, high=1.0, size=(iteration_num, n_images))
    return score, std


def simulate_decision_real_dataset(idx_1, idx_2, data):
    lst = []
    for row in data['data_ref']:
   
        if ((row[0]-1 == idx_1 and row[1]-1 == idx_2) or (row[0]-1 == idx_2 and row[1]-1 == idx_1)):
            lst.append(row-1)
    return random.choice(lst)