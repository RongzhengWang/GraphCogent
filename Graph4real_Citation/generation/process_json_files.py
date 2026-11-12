import os
import json

def process_json_files(directory):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            
            # Read the JSON file
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {filename}")
                    continue
            
            # Process each object in the JSON file
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Remove the 'prompt' key if it exists
                        if 'prompt' in item:
                            del item['prompt']
                        
                        # Clean the 'answer' field if it exists
                        if 'answer' in item:
                            answer = item['answer']
                            if isinstance(answer, str):
                                # Remove the '\n\n问题：' prefix
                                item['answer'] = answer.replace('\n\n问题：', '')
            
            # Write the modified data back to the file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"Processed file: {filename}")


process_json_files('Graph4real__Trans/ques')
