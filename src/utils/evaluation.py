from src.utils import eig
import sklearn.metrics as metrics
from scipy import stats


def learning_performance_evaluation(global_pcm, gt_score):
    [mu,mu_cova,stdv] = eig.run_modeling_Bradley_Terry(global_pcm)
    pearson, _ = stats.pearsonr(mu, gt_score)
    spearman, _ = stats.spearmanr(mu, gt_score)
    mse = metrics.mean_squared_error(mu, gt_score)
    return pearson,spearman, mse