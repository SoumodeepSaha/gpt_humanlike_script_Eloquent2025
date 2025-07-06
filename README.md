# GPT-2 Humanlike Text Generator 🎤🧠

This repo contains a Python script using the base GPT-2 model with post-processing to make text more humanlike.

## Features
- Uses GPT-2 from Hugging Face
- Adds fillers, emotion, self-doubt, casual tone
- Processes JSON input & saves results
- Zips all outputs for download

## Example
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from inference_script import generate_text_gpt2_humanlike, post_process_text

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = generate_text_gpt2_humanlike('What is AI?', model, tokenizer)
final = post_process_text(text)
print(final)
```
