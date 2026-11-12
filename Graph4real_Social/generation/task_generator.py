import json
from openai import OpenAI
from tqdm import tqdm
import os
import argparse

GPT_MODEL = "deepseek-r1"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL")
)

def chat_completion_request(messages, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8192,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process text files and generate JSON outputs.')
    parser.add_argument('--input_folder', required=True, help='Path to the input folder containing txt files')
    parser.add_argument('--output_folder', required=True, help='Path to the output folder for JSON files')
    parser.add_argument('--num_examples', type=int, required=True, help='Total number of examples to generate')
    
    args = parser.parse_args()

    # Use the arguments
    input_folder = args.input_folder
    output_folder = args.output_folder
    num_examples = args.num_examples

    # If output folder doesn't exist, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all txt files
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    for txt_file in tqdm(txt_files, desc="Processing Files"):
        # Read file content and title
        file_path = os.path.join(input_folder, txt_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            mess = f.read().strip()
        title = os.path.splitext(txt_file)[0]

        # Output JSON file path (based on txt filename)
        output_file_path = os.path.join(output_folder, f"{title}.json")

        # Initialize data storage
        data = []
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = []

        original_string = mess
        # Define replacement content list
        contents = [
            "Travel route planning",
            "Logistics delivery optimization",
            "Urban emergency response planning",
            "Public transit network scheduling",
            "Shared mobility platforms"
        ]

        # Number of examples per content
        count_per_content = num_examples // len(contents)

        # Replace and print the entire string
        for content in tqdm(contents, desc=f"Calling API for {txt_file}"):
            for _ in tqdm(range(count_per_content), desc=f"content {content}"):
                # Replace {{}} in the string with current content
                modified_string = original_string.replace("{{}}", content)
                print(modified_string)

                messages = [{"role": "user", "content": modified_string}]
                chat_response = chat_completion_request(messages)
                
                if isinstance(chat_response, Exception):
                    continue
                
                # Get API response content
                answer = chat_response.choices[0].message.content
                print(answer)
                
                # Record data
                data.append({
                    'prompt': modified_string,
                    'label': title,
                    'type': content,
                    'answer': answer
                })
                
                # Save results in real time
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

    print("All JSON files have been updated.")

if __name__ == "__main__":
    main()


# python script.py --input_folder zzzQuesGen/Trans/p1 --output_folder zzzQuesGen/Trans/json --num_examples 400

#python Graph4real/generation/task_generator.py --input_folder Graph4real/prompt_template/trans --output_folder Graph4real/ques/trans --num_examples 400