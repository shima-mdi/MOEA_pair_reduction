from src.utils import eig
import sklearn.metrics as metrics
from scipy import stats
from scipy.stats.mstats import spearmanr, pearsonr

def learning_performance_evaluation(global_pcm, gt_score):
    [mu, mu_cova] = eig.run_modeling_Bradley_Terry(global_pcm)
    pearson, _ = pearsonr(mu, gt_score[0])
    spearman, _ = spearmanr(mu, gt_score[0])
    mse = metrics.mean_squared_error(mu, gt_score[0])
    return pearson,spearman, mse