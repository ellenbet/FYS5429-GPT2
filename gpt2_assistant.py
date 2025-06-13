# homemade gpt imports
from my_gpt2.gpt_utils import gpt2_assistant, load_gpt2_assistant_with_weights

# ignore future warning in loading of model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

gpt = load_gpt2_assistant_with_weights()
print("Welcome to your own, personal assistant! To end the chat, press bye or x!")


print("What can I help you with?\n")
chatting = True
while chatting: 
    prompt = input("\n\nInput request below: \n")
    if prompt.lower() == "x" or "bye" in prompt.lower().split():
        print("Okay, bye!")
        chatting = False
    else: 
        gpt2_assistant(prompt, gpt)
