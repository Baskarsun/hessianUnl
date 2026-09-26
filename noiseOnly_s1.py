import numpy as np
import torch
import data_structure
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

forget_dataset = load_dataset("locuslab/TOFU", "forget01")["train"]
retain_dataset = load_dataset("locuslab/TOFU", "retain99")["train"]

alpha = 1.0
gamma = 0.01
eta = 1e-3
expectation_simulation = 10
training_iter = 400
noise_rank = 32
device = 'cpu'

totalForget = len(forget_dataset)
totalRetain = len(retain_dataset)

def make_addnoise(zeta):
    def addnoise(module, layer_input, layer_output):
        if isinstance(layer_output, tuple):
            hidden = layer_output[0]
            return (hidden + zeta,) + layer_output[1:]
        return layer_output + zeta
    return addnoise

def sample_batch(dataset, batch_size):
    indices = np.random.choice(len(dataset), size=batch_size, replace=False)
    return dataset.select(indices.tolist())   

# The paper has not prescribed how to infuse the noise; this is one way.
# It may have issues if the tokens have padding.

def scorewithnoise(model,tokenizer,text,zeta,real_embed,forget_batch):

    input_prompt = forget_batch[sample]["question"]
    output_answer = forget_batch[sample]["answer"] 
    input_promptIds=tokenizer(input_prompt, return_tensors="pt", truncation=True).to(device) 
    prompt_len =  input_promptIds["input_ids"].shape[1]
    labels  = real_embed["input_ids"].clone();
    labels[:, :prompt_len] = -100 
    target_layer = model.model.layers[-1]
    handle = target_layer.register_forward_hook(make_addnoise(zeta)) 
    try:
        outputs = model(**real_embed, labels=labels)
        score = outputs.loss               
    finally:
        handle.remove()                      

    return score



model_name = "locuslab/tofu_ft_llama2-7b"   
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

for param in model.parameters():
    param.requires_grad = False              

embed_dim = model.config.hidden_size          




    
A = torch.randn(embed_dim, noise_rank, device=device, requires_grad=True) * 0.01


optimizer = torch.optim.SGD([A], lr=eta)   

for iteration in range(1, training_iter):
    forget_batch = sample_batch(forget_dataset, totalForget)   
    retain_batch = sample_batch(retain_dataset, totalRetain)

    Frgetstage1_estimate = torch.tensor(0.0, device=device)
    Retainstage1_estimate = torch.tensor(0.0, device=device)

    for sample in range(len(forget_batch)):                    
        text = forget_batch[sample]["question"] + " " + forget_batch[sample]["answer"]
        real_embed = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        scores = []
        for simulation in range(expectation_simulation):        
            epsilon = torch.randn(noise_rank, device=device)     
            zeta = A @ epsilon                                    
            score = scorewithnoise(model,tokenizer,text,zeta,real_embed,forget_batch)
            scores.append(score)
        expectations = torch.stack(scores).mean()
        Frgetstage1_estimate = Frgetstage1_estimate + expectations

    Frgetstage1_estimate = Frgetstage1_estimate / len(forget_batch)

    for sample in range(len(retain_batch)):                    
            text = retain_batch[sample]["question"] + " " + retain_batch[sample]["answer"]
            real_embed = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    
            scores = []
            for simulation in range(expectation_simulation):        
                epsilon = torch.randn(noise_rank, device=device)     
                zeta = A @ epsilon                                    
                score = scorewithnoise(model,tokenizer,text,zeta,real_embed,retain_batch)
                scores.append(score)
            expectations = torch.stack(scores).mean()
            Retainstage1_estimate = Retainstage1_estimate + expectations

    Retainstage1_estimate = Retainstage1_estimate / len(forget_batch)        
    
            





            
            

  