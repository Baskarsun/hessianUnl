import numpy as np
import torch
import data_structure
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

forget_dataset = load_dataset("locuslab/TOFU", "forget01")["train"]
retain_dataset = load_dataset("locuslab/TOFU", "retain99")["train"]

alpha = 1.0
gamma = 0.01          
lam_da = 0.01         
eta = 1e-3
expectation_simulation = 1
training_iter = 2
noise_rank = 32
device = 'cpu'



batch_size_forget = 1
batch_size_retain = 1

totalForget = len(forget_dataset)
totalRetain = len(retain_dataset)


def make_NoiseandDTransform(phi,zeta):
    def NoiseandDTransform(module, layer_input, layer_output):
        phi_f16 = phi.to(torch.float16)
        zeta_f16 = zeta.to(torch.float16)
        if isinstance(layer_output, tuple):
            hidden = layer_output[0]
            return (hidden @ phi_f16 + zeta_f16,) + layer_output[1:]
        return layer_output @ phi_f16 + zeta_f16
    return NoiseandDTransform

def make_addnoise(zeta):
    def addnoise(module, layer_input, layer_output):
        if isinstance(layer_output, tuple):
            hidden = layer_output[0]
            return (hidden + zeta,) + layer_output[1:]
        return layer_output + zeta
    return addnoise

def scorewithnoise(model, real_embed, labels, zeta, target_layer):
    handle = target_layer.register_forward_hook(make_addnoise(zeta))
    try:
        outputs = model(**real_embed, labels=labels)
        score = outputs.loss
    finally:
        handle.remove()
    return score    

def sample_batch(dataset, batch_size):
    indices = np.random.choice(len(dataset), size=batch_size, replace=False)
    return dataset.select(indices.tolist())   

# The paper has not prescribed how to infuse the noise; this is one way.
# It may have issues if the tokens have padding.

def scorewithnoiseandDTransform(model, real_embed, labels, phi, zeta, target_layer):
    handle = target_layer.register_forward_hook(make_NoiseandDTransform(phi,zeta))
    try:
        outputs = model(**real_embed, labels=labels)
        score = outputs.loss
    finally:
        handle.remove()
    return score


local_path = "/Users/baskar/.cache/huggingface/hub/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd"

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)


#model_name = "locuslab/tofu_ft_llama2-7b"   
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()
target_layer = model.model.layers[-1]

for param in model.parameters():
    param.requires_grad = False              

embed_dim = model.config.hidden_size          




    
A = torch.randn(embed_dim, noise_rank, device=device,dtype=torch.float32) * 0.01
A.requires_grad = True
phi = torch.eye(embed_dim, device=device, dtype=torch.float32, requires_grad=True)


#optimizer = torch.optim.SGD([A], lr=eta)   

for iteration in range(1, training_iter):
    print(f"Iteration {iteration}/{training_iter}")
    forget_batch = sample_batch(forget_dataset, batch_size_forget)   
    retain_batch = sample_batch(retain_dataset, batch_size_retain)

    Forgetstage3_estimate = torch.tensor(0.0, device=device)
    Retainstage3_estimate = torch.tensor(0.0, device=device)

    for sample in range(len(forget_batch)):                    
        print(f"Iteration inside forget batch {sample}/{len(forget_batch)}")
        prompt = forget_batch[sample]["question"]
        text = prompt + " " + forget_batch[sample]["answer"]

        real_embed = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        prompt_len = prompt_ids["input_ids"].shape[1]

        labels = real_embed["input_ids"].clone()
        labels[:, :prompt_len] = -100
        #real_embed = tokenizer(text, return_tensors="pt", truncation=True).to(device)
        

        scores = []
        for simulation in range(expectation_simulation):        
            epsilon = torch.randn(noise_rank, device=device,dtype=torch.float32)     
            #zeta = A @ epsilon  
            zeta = (A @ epsilon).to(torch.float16)                                  
            #score = scorewithnoise(model,tokenizer,text,zeta,real_embed,forget_batch)
            score = scorewithnoiseandDTransform(model, real_embed, labels, phi, zeta, target_layer)
            scores.append(score)
        expectations = torch.stack(scores).mean()
        Forgetstage3_estimate = Forgetstage3_estimate + expectations

    Forgetstage3_estimate = Forgetstage3_estimate / len(forget_batch)

    for sample in range(len(retain_batch)):                    
            print(f"Iteration inside retain batch {sample}/{len(retain_batch)}")
            prompt = retain_batch[sample]["question"]
            text = prompt + " " + retain_batch[sample]["answer"]

            real_embed = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            prompt_len = prompt_ids["input_ids"].shape[1]

            labels = real_embed["input_ids"].clone()
            labels[:, :prompt_len] = -100

            
            scores = []
            for simulation in range(expectation_simulation):        
                epsilon = torch.randn(noise_rank, device=device,dtype=torch.float32)     
                #zeta = A @ epsilon  
                zeta = (A @ epsilon).to(torch.float16)                                    
                score = scorewithnoise(model, real_embed, labels, zeta, target_layer)
                scores.append(score)
            expectations = torch.stack(scores).mean()
            Retainstage3_estimate = Retainstage3_estimate + expectations

    Retainstage3_estimate = Retainstage3_estimate / len(retain_batch) 
    sigma_current = A @ A.T
    regterm_phi = lam_da * (phi - torch.eye(embed_dim, device=device, dtype=torch.float32)).square().sum()
    regterm_sigma = gamma * sigma_current.square().sum()
   
    j3value = Forgetstage3_estimate - alpha * Retainstage3_estimate - regterm_phi - regterm_sigma
    gradient_phi, gradient_A = torch.autograd.grad(j3value, [phi, A])

    with torch.no_grad():
        phi.add_(eta * gradient_phi)
        A.add_(eta * gradient_A)  
    
    # Detach A from the autograd graph and save it to your hard drive

final_phi = phi.detach().cpu()
final_A = A.detach().cpu()
torch.save({"phi": final_phi, "A": final_A}, "stage3_optimized.pt")

print("Stage 3 complete! The optimized noise matrix has been saved as 'stage3_optimized.pt'")        




            
            

  