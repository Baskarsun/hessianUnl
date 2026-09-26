import numpy as np
sequence_len = 10;
embeding_dim = 5;
noise_rank = 3;
Retain_embedings = np.array((sequence_len, embeding_dim))
forget__embedings = np.array((sequence_len, embeding_dim))
Transform_forget = np.array((embeding_dim, embeding_dim))
Transform_retain = np.array((embeding_dim, embeding_dim))
noise_distribution = np.array((embeding_dim,noise_rank))