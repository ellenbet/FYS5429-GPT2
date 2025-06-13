import gradio as gr
# homemade gpt imports
from my_gpt2.gpt_utils import gradio_gpt2_assistant, load_gpt2_assistant_with_weights

# ignore future warning in loading of model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
gpt = load_gpt2_assistant_with_weights()


def assist(input, return_sentences, model = gpt):
    return gradio_gpt2_assistant(input, gpt = model, num_sentences = return_sentences)

demo = gr.Interface(
    fn = assist,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch(share=True)