# Python packages
import torch
from tqdm import tqdm
import json

# my GPT2 packages
from my_gpt2.gpt_utils import load_gpt2_assistant_with_weights
from my_gpt2.plotting import set_plt_params
from my_gpt2.gpt2_eval_utils import generate_model_scores, query_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# setting some plotting params up
set_plt_params()
CHOOSE_MODEL = "gpt2-medium (355M)"

file_path = "training_data/instruction-data.json"
with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

torch.manual_seed(123)

train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\on device: {device}")
# loading in model
model = load_gpt2_assistant_with_weights()
model.to(device)


model_tester = "llama3"
result = query_model("What do Llamas eat?", model_tester)
print(result)

scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")