import numpy as np
import torch
import data_structure
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

forget_dataset = load_dataset("locuslab/TOFU", "forget01")["train"]
retain_dataset = load_dataset("locuslab/TOFU", "retain99")["train"]

alpha = 1.0
lam_da = 0.01
eta = 1e-3
expectation_simulation = 1
training_iter = 2
#noise_rank = 32
device = 'cpu'

batch_size_forget = 1
batch_size_retain = 1



def make_dTransform(phi):
    def dTransform(module, layer_input, layer_output):
        phi_f16 = phi.to(torch.float16)
        if isinstance(layer_output, tuple):
            hidden = layer_output[0]
            return (hidden @ phi_f16,) + layer_output[1:]
        return layer_output @ phi_f16
    return dTransform

def scoreretain(model, real_embed, labels):
    outputs = model(**real_embed, labels=labels)
    score = outputs.loss
    return score

def sample_batch(dataset, batch_size):
    indices = np.random.choice(len(dataset), size=batch_size, replace=False)
    return dataset.select(indices.tolist())   

# The paper has not prescribed how to infuse the noise; this is one way.
# It may have issues if the tokens have padding.

def scorewithtransform(model, real_embed, labels,phi, target_layer):
    handle = target_layer.register_forward_hook(make_dTransform(phi))
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
phi = torch.eye(embed_dim,  device=device,dtype=torch.float32, requires_grad=True)



for iteration in range(1, training_iter):
    print(f"Iteration {iteration}/{training_iter}")
    forget_batch = sample_batch(forget_dataset, batch_size_forget)   
    retain_batch = sample_batch(retain_dataset, batch_size_retain)

    Frgetstage1_estimate = torch.tensor(0.0, device=device)
    Retainstage1_estimate = torch.tensor(0.0, device=device)
    Forgetscores = []
    Retainscores = []

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
        score = scorewithtransform(model, real_embed, labels, phi, target_layer)
        Forgetscores.append(score)
    Forgetstage2_estimate = torch.stack(Forgetscores).mean()   

    for sample in range(len(retain_batch)):                    
            print(f"Iteration inside retain batch {sample}/{len(retain_batch)}")
            prompt = retain_batch[sample]["question"]
            text = prompt + " " + retain_batch[sample]["answer"]

            real_embed = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            prompt_len = prompt_ids["input_ids"].shape[1]

            labels = real_embed["input_ids"].clone()
            labels[:, :prompt_len] = -100
            score = scoreretain(model, real_embed, labels)
            Retainscores.append(score)

    Retainstage2_estimate = torch.stack(Retainscores).mean()
    regterm = lam_da * (phi - torch.eye(embed_dim, device=device, dtype=torch.float32)).square().sum()
    j2value = Forgetstage2_estimate - alpha * Retainstage2_estimate - regterm
    gradient_Phi = torch.autograd.grad(j2value, phi)[0]
    with torch.no_grad():
        phi += eta * gradient_Phi  
    
    # Detach A from the autograd graph and save it to your hard drive
final_phi = phi.detach().cpu()
torch.save(final_phi, "stage2_optimized_phi.pt")

print("Stage 2 complete! The optimized transformation matrix has been saved as 'stage2_optimized_phi.pt'")        





            
            

  