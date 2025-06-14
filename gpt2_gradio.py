import gradio as gr

"""
Gradio pastes a link in the terminal that can be pased into a browser for a prettier GUI. 
"""

# homemade gpt imports
from my_gpt2.gpt_utils import gradio_gpt2_assistant, load_gpt2_assistant_with_weights

# ignore future warning in loading of model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

xtra_finetuned_path = "gpt2_params/fine-tuned_1206_gpt2-medium355M-sft.pth"
gpt = load_gpt2_assistant_with_weights(model_dir = xtra_finetuned_path)


def assist(input, return_sentences, model = gpt):
    input = "###Instruction:" + input
    return gradio_gpt2_assistant(input, gpt = model, num_sentences = return_sentences)

demo = gr.Interface(
    fn = assist,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(share=True)