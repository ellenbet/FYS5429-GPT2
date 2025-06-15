# homemade gpt imports
from my_gpt2.gpt_utils import gpt2_assistant, load_gpt2_assistant_with_weights

# other packages
import torch
import os
import shutil
from huggingface_hub import hf_hub_download

# ignore future warning in loading of model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#local version
xtra_finetuned_path = "gpt2_params/fine-tuned_1206_gpt2-medium355M-sft.pth"
os.makedirs(os.path.dirname(xtra_finetuned_path), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(xtra_finetuned_path):
    print(f"Loading local weights from: {xtra_finetuned_path}")
    gpt = load_gpt2_assistant_with_weights(device=device, model_dir=xtra_finetuned_path)
else:
    # Download the file from Hugging Face
    downloaded_path = hf_hub_download(
        repo_id="ellenbet/gpt2-raschka-finetuning",
        filename="fine-tuned_1206_gpt2-medium355M-sft.pth",
        repo_type="model"
    )
    
    print(f"Downloaded file to temporary path: {downloaded_path}")
    
    # move params to xtra_finetuned_path
    shutil.copy(downloaded_path, xtra_finetuned_path)
    print(f"Saved to desired path: {xtra_finetuned_path}")
    gpt = load_gpt2_assistant_with_weights(device=device, model_dir=xtra_finetuned_path)


print("Welcome to your own, personal assistant! To end the chat, press bye or x!")


print("What can I help you with?\n")
chatting = True
while chatting: 
    prompt = input("\n\nInput request below: \n")
    if prompt.lower() == "x" or "bye" in prompt.lower().split():
        print("Okay, bye!")
        chatting = False
    else: 
        print(("\n I'm thinking... \n"))
        gpt2_assistant("###Instruction: " + prompt, gpt)
