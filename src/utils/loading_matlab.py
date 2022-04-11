from scipy.io import loadmat
from src.data import DATA_DIR


def load_mat_file():
    ref_1 = loadmat("src/data/data1.mat")
    return ref_1