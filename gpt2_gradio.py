# import fine-tuned weights from huggingface
import os
from huggingface_hub import hf_hub_download

# other neccessary packages
import gradio as gr
import torch
import shutil

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
        repo_type="dataset"
    )
    
    print(f"Downloaded file to temporary path: {downloaded_path}")
    
    # move params to xtra_finetuned_path
    shutil.copy(downloaded_path, xtra_finetuned_path)
    print(f"Saved to desired path: {xtra_finetuned_path}")
    gpt = load_gpt2_assistant_with_weights(device=device, model_dir=xtra_finetuned_path)


def assist(input, return_sentences, model = gpt):
    return gradio_gpt2_assistant(input, gpt = model, max_num_sentences = return_sentences)

demo = gr.Interface(
    fn = assist,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(share = True)