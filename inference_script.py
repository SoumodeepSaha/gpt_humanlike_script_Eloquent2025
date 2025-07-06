import os
import json
import zipfile
import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['voight-kampfftesttopics']['topics']
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def generate_text_gpt2_humanlike(prompt, model, tokenizer, max_length=500, temperature=0.7, top_p=0.9, top_k=50):
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, device=inputs.device)
        outputs = model.generate(inputs, attention_mask=attention_mask, max_length=max_length, 
                                 temperature=temperature, top_p=top_p, top_k=top_k,
                                 num_return_sequences=1, no_repeat_ngram_size=2, 
                                 do_sample=True, pad_token_id=50256)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated Text: {generated_text[:200]}...")
        return generated_text
    except Exception as e:
        print(f"Error in text generation: {e}")
        return ""

def post_process_text(generated_text):
    try:
        fillers = ["you know", "like", "um", "well", "actually", "I mean", "so", "basically", "honestly", "seriously"]
        interruptions = ["Anyway, ", "Well, you see...", "Uh, yeah...", "I mean, wait, let me think...", "Oh, actually..."]
        self_doubt = ["I'm not sure, but...", "I think, maybe...", "Could be wrong, but...", "Honestly, I'm not so sure."]
        inconsistent_flow = ["I don't know, it's just... it's just how it is.", "What I'm saying is... well, it's complicated.", 
                             "Actually, that’s a good point, but…", "You know what? Never mind that."]
        emotions = ["Wow, that’s crazy!", "Can you believe it?", "I mean, who would’ve thought!", "You won’t believe this..."]

        text_lines = generated_text.split(". ")
        new_text = []

        for line in text_lines:
            if random.random() > 0.85:
                line = random.choice([random.choice(fillers), random.choice(interruptions), random.choice(self_doubt)]) + " " + line
            if random.random() > 0.6:
                line = random.choice(emotions) + " " + line
            if random.random() > 0.5:
                line = line + " " + random.choice(inconsistent_flow)
            new_text.append(line)

        final_text = ". ".join(new_text) + random.choice([".", "!", "?"])
        final_text = final_text.replace("... .", "...")
        return final_text
    except Exception as e:
        print(f"Error in post-processing: {e}")
        return generated_text

def generate_texts_humanlike(data, team_name):
    try:
        os.makedirs(team_name, exist_ok=True)
        model_name = "gpt2"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        for i, example in enumerate(data):
            prompt = example['Content']
            genre_style = example['Genre and Style']
            prompt_id = example['id']
            print(f"Generating text for prompt {i + 1} (Genre: {genre_style}, ID: {prompt_id})...")
            try:
                generated_text = generate_text_gpt2_humanlike(prompt, model, tokenizer)
                if not generated_text:
                    continue
                final_text = post_process_text(generated_text)
                if len(final_text.split()) > 500:
                    final_text = " ".join(final_text.split()[:500])
                filename = os.path.join(team_name, f"{prompt_id}.txt")
                with open(filename, 'w') as f:
                    f.write(final_text)
                print(f"Generated text saved as {filename}")
            except Exception as e:
                print(f"Error generating or saving text for prompt {i + 1} (ID: {prompt_id}): {e}")
    except Exception as e:
        print(f"Error in generating texts: {e}")

def zip_generated_texts(team_name):
    try:
        zip_filename = f"{team_name}.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, _, files in os.walk(team_name):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), team_name))
        print(f"Generated texts have been zipped into {zip_filename}")
    except Exception as e:
        print(f"Error in zipping files: {e}")

def main():
    json_path = "/kaggle/input/eloquent2025/task-vk-test-2025 (1).json"
    data = load_data(json_path)
    if not data:
        print("No data found.")
        return
    team_name = "JUNLP_SS"
    generate_texts_humanlike(data, team_name)
    zip_generated_texts(team_name)
    print("Task completed! Your generated texts have been saved and zipped.")

if __name__ == "__main__":
    main()
