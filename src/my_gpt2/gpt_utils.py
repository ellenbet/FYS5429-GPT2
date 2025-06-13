import torch 
from .text_generation import generate_text_simple, generate
from .text_processing import token_ids_to_text, text_to_token_ids
from .gpt2 import GPTModel
import tiktoken
import re

tokenizer = tiktoken.get_encoding("gpt2")

"""
Base setup of models below

To select medium, use: 
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
gpt = GPTModel(BASE_CONFIG)


"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def load_gpt2_assistant_with_weights(model_dir = "gpt2_params/fine-tuned_1206_gpt2-medium355M-sft.pth", CHOOSE_MODEL = "gpt2-medium (355M)", gpt = None):
    # choose and set up our model
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    if gpt is None:
        gpt = GPTModel(BASE_CONFIG)
    gpt.load_state_dict(torch.load(model_dir))
    return gpt
    

def gpt2_assistant(input, gpt, device = "cpu", tokenizer = None):
    if tokenizer is None: 
        tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model = gpt, 
        idx = text_to_token_ids(input,  tokenizer).to(device),
        max_new_tokens = 150, 
        context_size = BASE_CONFIG["context_length"],
        top_k = 70,
        temperature = 1.5
    )
    ans = token_ids_to_text(token_ids, tokenizer)
    
    # include only first three sentences
    target = {'.', '!', '?'}
    count = 0
    for i, char in enumerate(input):
        if char in target:
            count += 1
            if count == 3:
                ans = input[:i+1]

    if "<|endoftext|>" in ans:
        print("output:\n", ans[: ans.index("<|endoftext|>")])
    else:
        print(ans)

def gradio_gpt2_assistant(input, gpt, num_sentences, device = "cpu", tokenizer = tokenizer):
    token_ids = generate(
        model = gpt, 
        idx = text_to_token_ids(input,  tokenizer).to(device),
        max_new_tokens = max(100, num_sentences * 10), 
        context_size = BASE_CONFIG["context_length"],
        top_k = 70,
        temperature = 1.5
    )
    ans = token_ids_to_text(token_ids, tokenizer)
        # include only first three sentences
    target = {'.', '!', '?'}
    count = 0
    for i, char in enumerate(input):
        if char in target:
            count += 1
            if count == num_sentences:
                ans = input[:i+1]

    if "<|endoftext|>" in ans:
        return ans[: ans.index("<|endoftext|>")]
    else:
        return ans

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model, idx = encoded, 
            max_new_tokens = 50, context_size = context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))
    model.train()

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None: 
        num_batches = len(data_loader)
    else: 
        # in case of non-consistenct between loader and batch size
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train() # set to training model

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() #reset pr. batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() #get loss gradient
            optimizer.step() #update model weights using loss gradients
            tokens_seen += input_batch.numel() #numel = number of elements
            global_step += 1
        
            # evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch: {epoch+1}\nStep: {global_step:06d}\nTrain loss: {train_loss:.3f}\nValidation loss: {val_loss:.3f}")
            
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, val_loss


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: left : {left.shape} while right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
    
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def print_gradients(model, x):
    #Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    #Find loss
    loss = nn.MSELoss()
    loss = loss(output, target)

    #Backward pass
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            #mean abs grad 
            print(name, "has gradient mean of", param.grad.abs().mean().item())