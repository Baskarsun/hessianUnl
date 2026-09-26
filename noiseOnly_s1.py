import numpy as np
import data_structure
from datasets import load_dataset

forget_dataset =  load_dataset("locuslab/TOFU", "forget01")["train"]
retain_dataset =  load_dataset("locuslab/TOFU", "retain99")["train"]
alpha = 1.0
gamma = 0.01
eta = 1e-3
expectation_simulation = 10
training_iter = 400

def sample_batch(dataset, batch_size):
    indices = np.random.choice(len(dataset), size=batch_size, replace=False)
    return dataset[indices]

#  Initialize A0 (e.g. small random or scaled identity factor) : From paper; I took small random values
A = np.random.randn(data_structure.embeding_dim, data_structure.noise_rank) * 0.01
