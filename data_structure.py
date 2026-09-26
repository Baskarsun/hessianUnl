import numpy as np
sequence_len = 10;
embeding_dim = 5;
noise_rank = 3;
Retain_embedings = np.array((sequence_len, embeding_dim))
forget__embedings = np.array((sequence_len, embeding_dim))
Transform_forget = np.array((embeding_dim, embeding_dim))
Transform_retain = np.array((embeding_dim, embeding_dim))
noise_distribution = np.array((embeding_dim,noise_rank))

def getembedding_oftext(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]        
    pooled = last_hidden.mean(dim=1).squeeze(0)     
    return pooled   