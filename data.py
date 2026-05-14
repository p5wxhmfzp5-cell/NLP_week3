#data

from huggingface_hub import login
from datasets import load_dataset
import os


# Dein Hugging Face Token (besser über Umgebungsvariable laden!)
HF_TOKEN = os.getenv("HFMYTOKEN")

# Login
login(token=HF_TOKEN)

from huggingface_hub import whoami

user = whoami(token=HF_TOKEN)
print(f"Logged in as: {user['name']}")



def load_triviaqa():
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")
    return ds["validation"]

def load_nq():
    ds = load_dataset("sentence-transformers/natural-questions")
    return ds["train"].select(range(5000))  # subset for speed