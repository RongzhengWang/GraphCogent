import os
import json
import re
import argparse
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams

parser = argparse.ArgumentParser(description='Process JSON files and generate DPO data.')
parser.add_argument('--input_file', type=str, required=True, help='Input JSON file path')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed files')
parser.add_argument('--prompt_file', type=str, required=True, help='Path to the prompt file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the LLM model')
parser.add_argument('--gpu_devices', type=str, default="3,4", help='CUDA visible devices')
args = parser.parse_args()

MAX_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 1.0
NUM_TRIALS = 10

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

llm = LLM(
    model=args.model_path,
    tokenizer_mode="auto",
    tensor_parallel_size=len(args.gpu_devices.split(',')),
    dtype="half",
    max_model_len=MAX_TOKENS
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P
)

with open(args.prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

os.makedirs(args.output_dir, exist_ok=True)

with open(args.input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for entry in tqdm(data, desc="Processing Entries"):
    answer = entry.get("answer", "")
    clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    processed_answer = clean_answer.split("Question:")[-1].strip()
    label = entry['label']
    
    chosen = None
    rejected = None
    
    for i in range(NUM_TRIALS):
        mess = []
        mess.append({"role": "system", "content": prompt})
        mess.append({"role": "user", "content": processed_answer})
        
        outputs = llm.chat(mess, sampling_params=sampling_params)
        first = outputs[0].outputs[0].text
        clean_first = re.sub(r'.*\n\n', '', first, flags=re.DOTALL)
        
        if ("tool_name: " + 'NULL').lower() in clean_first.lower() or ("tool_name: " + 'None').lower() in clean_first.lower():
            if chosen is None:
                chosen = first
        else:
            if rejected is None:
                rejected = first
        
        if chosen is not None and rejected is not None:
            break
    
    if chosen is None:
        chosen = 'no found'
    if rejected is None:
        rejected = 'no found'
    
    result = {
        "conversations": [
            {
                "from": "system",
                "value": prompt
            },
            {
                "from": "human",
                "value": processed_answer
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": chosen
        },
        "rejected": {
            "from": "gpt",
            "value": rejected
        }
    }
    
    output_file_path = os.path.join(args.output_dir, f"{label}_filtered_v2_grpo+dpo.json")
    
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
    
    with open(output_file_path, 'r+', encoding='utf-8') as f:
        existing_data = json.load(f)
        existing_data.append(result)
        f.seek(0)
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

print("Processing completed successfully.")
