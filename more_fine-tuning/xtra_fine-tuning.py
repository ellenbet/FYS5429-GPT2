# Python packages
import torch
import csv
import json
import time
from functools import partial
import re

# my GPT2 packages
from my_gpt2.gpt_utils import load_gpt2_assistant_with_weights, tokenizer, train_model_simple, BASE_CONFIG
from my_gpt2.text_processing import custom_collate_fn, format_input, token_ids_to_text, text_to_token_ids
from my_gpt2.text_generation import generate
from my_gpt2.Datasets import InstructionDataset, DataLoader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
Due to unimpressive function of fine-tuned medium llm on 1100 data entries, 
I increased finetuning to 52 000 entries

"""

# setting some plotting params up
#set_plt_params()
CHOOSE_MODEL = "gpt2-medium (355M)"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\on device: {device}")
# loading in model
model = load_gpt2_assistant_with_weights()
model.to(device)

customized_collate_fn = partial(
    custom_collate_fn,
    device = device,
    allowed_max_length = 512
)

file_path = "training_data/alpaca_data.json"
with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

num_workers = 0
batch_size = 2
torch.manual_seed(123)

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]


train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle = True,
    drop_last = True,
    num_workers = num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)


model.eval()

# Load settings and params
model.to(device)
print("done loading!\ntesting...")
model.eval()

# testing that all params are on the cuda
for name, param in model.named_parameters():
    if "cpu" in name:
        print(f"WARNING: {name} is on {param.device}")


# on to some actual finetuning training!! 

start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00005, weight_decay = 0.1)

num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs = num_epochs, eval_freq=5, eval_iter=5,
    start_context = format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# visualizing loss: 

"""train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_eval(epochs_tensor, tokens_seen, train_losses, val_losses, train_label = "Train loss", val_label = "Validation loss", y_label = "Loss", save_as = "fine-tuning_loss.pdf")
plot_eval(epochs_tensor, tokens_seen, np.exp(train_losses), np.exp(val_losses), train_label = "Train perplexity", val_label = "Validation perplexity", y_label = "Perplexity", save_as = "fine-tuning_perplexity.pdf")
"""
# consider https://www.gradio.app/

file_name = f"gpt2_params/alpaca_fine-tuned_1306_{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

torch.manual_seed(123)


for entry in test_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
)

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")


rows = zip(train_losses, val_losses, tokens_seen, num_epochs)

# Write to CSV
with open('52k_training_output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Optional: write header
    writer.writerow(['train_losses', 'val_losses', 'tokens_seen', 'num_epochs'])
    
    # Write data rows
    writer.writerows(rows)