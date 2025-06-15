# LLM
FYS5429 Project - Build an LLM from scratch, based on 'free coding path'.

Large language models (LLMs) are gradually becoming more integrated in everyday life, for students and employees alike. They can be used to increase productivity and facilitate the learning of new concepts, and is rapidly being integrated into professional tools. Workplaces and universities now host courses on "How to become an AI expert", a course essentially meant to teach the average person how to use a large language model to progress in their tasks. As we become increasingly dependent on LLMs, it is important to understand their very basics. How are they made, how can we train them and how do they generate the text that we put so much faith in? To answer these questions, I have built a Generative Pretrained Transformer, a GPT, using the recipie from Rascka's "How to build an LLM from scratch". I tested two different sizes of the GPT2 architecture, and found that while pretraining for proof of function was only neccessary on the small version - the weights retrieved from OpenAI's GPT2 model demonstrated that the medium version outperformed the small. I then fine-tuned the medium GPT2 using data supplied from the same author, and used my fine-tuned LLM as a personal assistant.

## How to use: 
1. Clone this repository into your local computer or code space. 
2. Follow instructions on making a venv and installation of project.
3. Launch script of choice

The repository contains two main scripts: 
- gpt2_assistant.py
- gpt2_gradio.py

Both of which launchs a pretrained and fine-tuned GPT2 by either using local model params in a gpt2_params directory, or download the model params from huggingface. The params take up 1,7GB of space. 

### gpt2_assistant.py 
Launches an in-terminal assistant that tries to answer whatever questions you might have.

### gpt2_gradio.py
Lauches a url or local domain with a prettier GUI to answer whatever questions you might have. 

PS - if downloading 1,7GB params is not an option to test the model, send a message to @ellenbet to set up a time-slot for using the url and test the model while it's lauched from a personal computer. 

## Make a viritual enviroment

Using python module venv or uv venv: 

Python module venv:
```sh
python -m venv gpt2-venv
```

or UV venv: 
```sh
uv venv gpt2-venv
```

Then enter the venv using 
```sh
source gpt2-venv/bin/activate
```

Finally, install the requirements using
```sh
pip install -r requirements.txt
```

Optionally in UV with
```sh
uv pip install -r requirements.txt
```

## Installation
To install this project using pip, run the following command while inside appropriate venv:
```sh
python3 -m pip install -e .
```

To install it using UV, run
```sh
uv pip install -e .
```

## Data
The data used in this project includes "The Verdict" by Edith Warton for simple pretrianing, then the full set of weights and biases
are uploaded and put into the model from OpenAI. This model is then fine-tuned using a 1100 set of instruction-response pairs supplied from Rascka's "How to build an LLM from scratch". 

## Future applications
Distillation to create an automatic epigenomic annotator? RAG?

## Authors
- [Ellen-Beate Tysvær]
