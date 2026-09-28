import numpy as np
import torch
import data_structure
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

forget_dataset = load_dataset("locuslab/TOFU", "forget01")["train"]
retain_dataset = load_dataset("locuslab/TOFU", "retain99")["train"]

alphainner= 1.0
gamma = 0.01          
lam_da = 0.01         
etainner= 1e-3
expectation_simulation = 1
training_iter = 2
noise_rank = 32
device = 'cpu'

def make_NoiseandDTransform(phi,zeta):
    def NoiseandDTransform(module, layer_input, layer_output):
        phi_f16 = phi.to(torch.float16)
        zeta_f16 = zeta.to(torch.float16)
        if isinstance(layer_output, tuple):
            hidden = layer_output[0]
            return (hidden @ phi_f16 + zeta_f16,) + layer_output[1:]
        return layer_output @ phi_f16 + zeta_f16
    return NoiseandDTransform

def scorewithnoiseandDTransform(model, real_embed, labels, phi, zeta, target_layer):
    handle = target_layer.register_forward_hook(make_NoiseandDTransform(phi,zeta))
    try:
        outputs = model(**real_embed, labels=labels)
        score = outputs.loss
    finally:
        handle.remove()
    return score

def outer_calculate():
    Retainstage4_estimate = torch.tensor(0.0, device=device)
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
            score = scorewithnoiseandDTransform(model, real_embed, labels,phi_r, zeta, target_layer)
            scores.append(score)
            
            
        expectations = torch.stack(scores).mean()
        Retainstage4_estimate = Retainstage4_estimate + expectations
    Retainstage4_estimate = Retainstage4_estimate / len(retain_batch)
    regterm_r = lam_daR * (phi_r - torch.eye(embed_dim, device=device, dtype=torch.float32)).square().sum()   
    gvalue = Retainstage4_estimate + regterm_r
    return gvalue



def inner_loop():
    for iteration in range(inner_loop_iterations):
        forget_batch = sample_batch(forget_dataset, batch_size_forget)   
        retain_batch = sample_batch(retain_dataset, batch_size_retain)

        Forgetstage4_estimate = torch.tensor(0.0, device=device)

        Retainstage4_estimate = torch.tensor(0.0, device=device)

        for sample in range(len(forget_batch)):                    
            print(f"Iteration inside forget batch {sample}/{len(forget_batch)}")
            prompt = forget_batch[sample]["question"]
            text = prompt + " " + forget_batch[sample]["answer"]

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
                #score = scorewithnoise(model,tokenizer,text,zeta,real_embed,forget_batch)
                score = scorewithnoiseandDTransform(model, real_embed, labels,phi_f, zeta, target_layer)
                scores.append(score)
            expectations = torch.stack(scores).mean()
            Forgetstage4_estimate = Forgetstage4_estimate + expectations
        Forgetstage4_estimate = Forgetstage4_estimate / len(forget_batch)

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
                score = scorewithnoiseandDTransform(model, real_embed, labels,phi_r, zeta, target_layer)
                scores.append(score)
            expectations = torch.stack(scores).mean()
            Retainstage4_estimate = Retainstage4_estimate + expectations
        Retainstage4_estimate = Retainstage4_estimate / len(retain_batch)

        regterm_phi = lam_da * (phi_f - torch.eye(embed_dim, device=device, dtype=torch.float32)).square().sum()   
        regterm_sigma = gamma * sigma_current.square().sum()
        qvalue = Forgetstage4_estimate - alphainner * Retainstage4_estimate - regterm_phi - regterm_sigma
        gradient_phi_f, gradient_A   = torch.autograd.grad(qvalue, [phi_f, A]) 
        with torch.no_grad():   
            phi_f.add_(etainner * gradient_phi_f)
            A.add_(etainner * gradient_A)  
        print(f"Inner loop iteration {iteration}/{inner_loop_iterations} complete. Updated phi_f and A.")
    return phi_f, A     


def computeQvalue():
    Forgetstage4_estimate = torch.tensor(0.0, device=device)
        
    Retainstage4_estimate = torch.tensor(0.0, device=device)
    for sample in range(len(forget_batch)):       

        print(f"Iteration inside forget batch {sample}/{len(forget_batch)}")
        prompt = forget_batch[sample]["question"]
        text = prompt + " " + forget_batch[sample]["answer"]

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
            #score = scorewithnoise(model,tokenizer,text,zeta,real_embed,forget_batch)
            score = scorewithnoiseandDTransform(model, real_embed, labels,phi_f, zeta, target_layer)
            scores.append(score)
        expectations = torch.stack(scores).mean()
        Forgetstage4_estimate = Forgetstage4_estimate + expectations
    Forgetstage4_estimate = Forgetstage4_estimate / len(forget_batch)
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
            score = scorewithnoiseandDTransform(model, real_embed, labels,phi_r, zeta, target_layer)    
            scores.append(score)
        expectations = torch.stack(scores).mean()
        Retainstage4_estimate = Retainstage4_estimate + expectations   

    Retainstage4_estimate = Retainstage4_estimate / len(retain_batch)

    regterm_phi = lam_da * (phi_f - torch.eye(embed_dim, device=device, dtype=torch.float32)).square().sum()   
    regterm_sigma = gamma * sigma_current.square().sum()
    qvalue = Forgetstage4_estimate - alphainner * Retainstage4_estimate - regterm_phi - regterm_sigma
    return qvalue
        
        

def hvp_fn(direction):
    return hvp_vv(qvalue,vparams,direction)

def flatten(tensors):
    return torch.cat([t,shape(-1) for t in tensors])

def hvp_vv(qvalue,vparams,inverse_hvp):
    grad_v = torch.autograd.grad(qvalue,vparams,create_graph=True)
    flat_grad = flatten(grad_v)
    grad_dot = torch.dot(flat_grad ,inverse_hvp)
    hessian = torch.autograd.grad(grad_dot,vparams,retain_graph=True)
    return -flatten(hessian)

def conjugate_gradient(outer_grd_wrt_inner,n_iters,tol=1e-6):
    z = torch.zeros_like(outer_grd_wrt_inner)
    r = outer_grd_wrt_inner.clone()
    direction = r.clone()
    rs_old = torch.dot(r,r)
    for i in range(n_iters):
        hessian = hvp_fn(direction)
        alpha_cg = rs_old / (torch.dot(direction,hessian) + 1e-10)
        z = z + alpha_cg * direction
        r = r - alpha_cg * hessian
        rs_new = torch.dot(r,r)
        if rs_new.sqrt() < tol :
            break
        direction = r + (rs_new/rs_old) * direction
        rs_old = rs_new
    return z

def hvp_uv(qvalue,phi_r,vparams,inverse_hvp_vector) :
    grad_v = torch.autograd.grad(qvalue,vparams,create_graph=True)
    flat_grad = flatten(grad_v)
    grad_dot_inversehvp = tortch.dot(flat_grad,inverse_hvp_vector)
    indirect_effect_grad = touch.autograd.grad(grad_dot_inversehvp,phi_r,retain_graph=True)[0]
    return indirect_effect_grad


def outer_loop():
    phi_f, A = inner_loop()
    forget_batch = sample_batch(forget_dataset, batch_size_forget)   
    retain_batch = sample_batch(retain_dataset, batch_size_retain)
    qvalue = computeQvalue()

    retain_batch_G = sample_batch(retain_batch,batch_size_retain)
    gvalue = outercalculate()
    vparams = [phi_f, A]
    
    direct_effect_grad = torch.autograd.grad(gvalue,phi_r,retain_graph=True)[0]
    outer_grd_wrt_inner = flatten(torch.autograd.grad(gvale,vparams,retain_graph=True))
    
    inverse_hvp_vector = conjugate_gradient(outer_grd_wrt_inner, n_iters=conGr_iters)

    indirect_effect_grad = hvp_uv(qvalue,phi_r,vparams,inverse_hvp_vector)

    hyper_grad = direct_effect_grad + indirect_effect_grad
    with torch.nograd():
        phi_r.add(-eta_outer * hyper_grad)
    
    print(f"outer : G = {gvalue.item():.4f}")

    return phi_f,phi_r,A

 

