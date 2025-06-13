import torch 


"""
generate(): 

generates text with probabilistic sampling using temperature tuning and
top k samling.

"""
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits/temperature # this is just temperature scaling - if temp > 1 -> more uniform dist, else sharper dist
    return torch.softmax(scaled_logits, dim = 0)

def generate(model, idx, max_new_tokens, context_size, temperature = 0.0, top_k = None, eos_id = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, - context_size: ]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]

            # put all non-topk logits to -inf
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits/temperature
            probs = torch.softmax(logits, dim = -1)
            # probabilistic sampling
            idx_next = torch.multinomial(probs, num_samples = 1)
        else: 
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim = 1)
    return idx

"""
generate_text_simple() 

generates text based on the most probable token.
"""

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size : ]

        with torch.no_grad():
            logits = model(idx_cond)
        
        #( batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]
        #probas = torch.softmax(logits, dim = -1)
        idx_next = torch.argmax(logits, dim = -1, keepdim= True)#probas, dim = -1, keepdim = True)
        idx = torch.cat((idx, idx_next), dim = 1)
    return idx

