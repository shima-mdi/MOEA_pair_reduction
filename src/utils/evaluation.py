from src.utils import eig
import sklearn.metrics as metrics
from scipy import stats
from scipy.stats.mstats import spearmanr, pearsonr

def learning_performance_evaluation(mode, global_pcm, gt_score):
    [mu, mu_cova] = eig.run_modeling_Bradley_Terry(global_pcm)

    if mode == 'simulation':
        pearson, _ = pearsonr(mu, gt_score[0])
        spearman, _ = spearmanr(mu, gt_score[0])
        mse = metrics.mean_squared_error(mu, gt_score[0])
    elif mode == 'IQA':
        pearson, _ = pearsonr(mu, gt_score)
        spearman, _ = spearmanr(mu, gt_score)
        mse = metrics.mean_squared_error(mu, gt_score)
   
    return pearson,spearman, mse