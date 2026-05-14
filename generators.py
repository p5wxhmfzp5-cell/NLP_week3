from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

class Generator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if "t5" in model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_seq2seq = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.is_seq2seq = False

    def generate(self, question, contexts):
        if len(contexts) == 0:
            prompt = f"Question: {question}\nAnswer:"
        else:
            context = "\n".join(contexts)
            prompt = f"Answer using context.\nContext:\n{context}\nQuestion: {question}\nAnswer:"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        output = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)