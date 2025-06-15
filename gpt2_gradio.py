# import fine-tuned weights from huggingface
import os
from huggingface_hub import hf_hub_download

# other neccessary packages
import gradio as gr
import torch

"""
Gradio pastes a link in the terminal that can be pased into a browser for a prettier GUI. 

This script downloads 1,7 GB model params from huggingface
-> If user would rather test model running from my machine, we can agree on a time-slot where the model
is accessible through a url while running on my mac
"""

# homemade gpt imports
from my_gpt2.gpt_utils import gradio_gpt2_assistant, load_gpt2_assistant_with_weights

# ignore future warning in loading of model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Local path to check
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


def assist(input, return_sentences, model = gpt):
    input = "###Instruction:" + input
    return gradio_gpt2_assistant(input, gpt = model, max_num_sentences = return_sentences)

demo = gr.Interface(
    fn = assist,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(share = True)