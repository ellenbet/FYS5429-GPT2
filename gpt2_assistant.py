# homemade gpt imports
from my_gpt2.gpt_utils import gpt2_assistant, load_gpt2_assistant_with_weights

# other packages
import torch
import os
from huggingface_hub import hf_hub_download

# ignore future warning in loading of model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
xtra_finetuned_path = "gpt2_params/alpaca_fine-tuned_1306_gpt2-medium355M-sft.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(xtra_finetuned_path):
    # Load model from local path
    print(f"Loading local weights from: {xtra_finetuned_path}")
    
    # Replace with your actual function to load the model
    gpt = load_gpt2_assistant_with_weights(device = device, model_dir = xtra_finetuned_path)

else:
    # Download from Hugging Face dataset repo
    file_path = hf_hub_download(
        repo_id="ellenbet/gpt2-alpaca-finetuned",
        filename= "alpaca_fine-tuned_1306_gpt2-medium355M-sft.pth",
        repo_type = "dataset" #this is actually model weights and not a dataset...
    )

    print(f"Downloaded file to: {file_path}")

   # load using downloaded weights
    gpt = load_gpt2_assistant_with_weights(device = device, model_dir=file_path)

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
