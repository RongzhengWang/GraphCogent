import os
import json
import re
import argparse
from tqdm import tqdm
from openai import OpenAI

parser = argparse.ArgumentParser(description='Process JSON files in a folder and generate translations.')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing JSON files')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed files')
parser.add_argument('--prompt_file', type=str, required=True, help='Path to the prompt file')
parser.add_argument('--model', type=str, required=True, help='GPT model name to use')
parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
parser.add_argument('--base_url', type=str, required=True, help='OpenAI API base URL')
args = parser.parse_args()
client = OpenAI(
    api_key=args.api_key,
    base_url=args.base_url
)

def chat_completion_request(messages, model=args.model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

with open(args.prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

os.makedirs(args.output_dir, exist_ok=True)

for filename in tqdm(os.listdir(args.input_dir), desc="Processing Files"):
    if not filename.endswith('.json'):
        continue
        
    file_path = os.path.join(args.input_dir, filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    output_data = []
    
    for entry in tqdm(data, desc=f"Processing {filename}"):
        answer = entry.get("answer", "")
        clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        processed_answer = clean_answer.split("Question:")[-1].strip()
        label = entry.get('label', '')
        
        messages_temp = [{"role": "user", "content": prompt + processed_answer}]
        try:
            think_answer = chat_completion_request(messages_temp).choices[0].message.content
        except Exception as e:
            print(f"Error in chat_completion_request: {e}")
            think_answer = "Translation failed"

        filtered_question = {
            'instruction': processed_answer,
            'input': processed_answer,
            'output': think_answer
        }
        
        output_data.append(filtered_question)

    output_filename = f"{os.path.splitext(filename)[0]}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

print("All files processed successfully.")
