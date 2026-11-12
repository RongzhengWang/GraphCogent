import json
import re
import os
from pathlib import Path

def format_edges(edges):
    edge_strs = []
    for edge in edges:
        edge_strs.append(f"{edge}")
    return ", ".join(edge_strs)

def extract_number_from_question(question):
    match = re.search(r'\d+', question)
    return match.group(0) if match else None

def replace_pattern_in_question(question, pattern, replacement):
    return re.sub(pattern, replacement, question)

def process_single_pair(json1_path, json2_path, output_path):
    with open(json1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    
    with open(json2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)
    
    if len(data1) < len(data2):
        raise ValueError(f"{json1_path} must be longer than or equal to {json2_path}")
    
    merged_data = []
    
    for i, item2 in enumerate(data2):
        item1 = data1[i]
        
        num_from_json1 = extract_number_from_question(item1.get("question", ""))
        
        original_question = item2.get("answer", "")
        if num_from_json1:
            modified_question = replace_pattern_in_question(
                original_question, 
                r'123',
                num_from_json1
            )
        else:
            modified_question = original_question
        
        edges_str = format_edges(item1.get("Edges", []))
        full_edges = f"The edges are: {edges_str}."
        
        full_ques = f"{modified_question} {full_edges}".strip()
        
        merged_item = {
            "origin_question": item2.get("answer", ""),
            "question": full_ques,
            "type": item2.get("label", ""),
            "background_type": item2.get("type", ""),
            "Node_List": item1.get("Node_List", []),
            "Edges": item1.get("Edges", []),
            "Edge_Count": item1.get("Edge_Count", 0),
            "Description": item1.get("Description", ""),
            "answer": item1.get("answer", ""),
        }
        
        merged_data.append(merged_item)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    
    print(f"Merged data saved to {output_path}")

def merge_json_folders(folder1_path, folder2_path, output_folder_path, task_name):
    task_filename = f"{task_name}.json"
    json1_path = os.path.join(folder1_path, task_filename)
    json2_path = os.path.join(folder2_path, task_filename)
    output_path = os.path.join(output_folder_path, task_filename)
    
    if not os.path.exists(json1_path):
        print(f"Warning: {task_filename} not found in {folder1_path}, skipping")
        return
        
    if not os.path.exists(json2_path):
        print(f"Warning: {task_filename} not found in {folder2_path}, skipping")
        return
    
    try:
        process_single_pair(json1_path, json2_path, output_path)
    except Exception as e:
        print(f"Error processing {task_filename}: {str(e)}")
        raise