import json
import gradio as gr
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

def get_response(input_data):
    input_data = json.loads(input_data)
    input_text = input_data["query"]
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    response = model.generate(**input_ids, max_new_tokens=512)
    output = tokenizer.decode(response[0], skip_special_tokens=True)
    return {"output": output}

demo = gr.Interface(
    fn=get_response,
    inputs="json",
    outputs="json"
)

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True, show_error=True)
