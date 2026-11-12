# from utils import format_edges, load_json_file, save_json_file
# import os

# def merge_json_folders(folder1_path, folder2_path, output_folder_path):
#     os.makedirs(output_folder_path, exist_ok=True)
    
#     for filename in os.listdir(folder1_path):
#         if filename.endswith('.json'):
#             json1_path = os.path.join(folder1_path, filename)
#             json2_path = os.path.join(folder2_path, filename)
#             output_path = os.path.join(output_folder_path, filename)
            
#             if os.path.exists(json2_path):
#                 data1 = load_json_file(json1_path)
#                 data2 = load_json_file(json2_path)
                
#                 if len(data1) < len(data2):
#                     raise ValueError(f"{json1_path} length must be ≥ {json2_path}")
                
#                 merged_data = []
#                 for i, item2 in enumerate(data2):
#                     item1 = data1[i]
#                     edges_str = format_edges(item1.get("Edges", []))
#                     full_edges = f"The edges are: {edges_str}."
#                     question = item2.get("answer", "")
#                     full_ques = f"{question} {full_edges}".strip()
                    
#                     merged_item = {
#                         "origin_question": item2.get("answer", ""),
#                         "question": full_ques,
#                         "type": item2.get("label", ""),
#                         "background_type": item2.get("type", ""),
#                         "Node_List": item1.get("Node_List", []),
#                         "Edges": item1.get("Edges", []),
#                         "Edge_Count": item1.get("Edge_Count", 0),
#                         "Description": item1.get("Description", ""),
#                         "answer": item1.get("answer", ""),
#                     }
#                     merged_data.append(merged_item)
                
#                 save_json_file(merged_data, output_path)
#                 print(f"Merged successfully: {output_path}")
#             else:
#                 print(f"Skipped: {filename} not found in {folder2_path}")


from utils import format_edges, load_json_file, save_json_file
import os

def merge_json_folders(folder1_path, folder2_path, output_folder_path, task_name):
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 构建任务相关的文件名模式
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
    
    # 执行合并逻辑
    try:
        data1 = load_json_file(json1_path)
        data2 = load_json_file(json2_path)
        
        if len(data1) < len(data2):
            raise ValueError(f"{json1_path} length must be ≥ {json2_path}")
        
        merged_data = []
        for i, item2 in enumerate(data2):
            item1 = data1[i]
            edges_str = format_edges(item1.get("Edges", []))
            full_edges = f"The edges are: {edges_str}."
            question = item2.get("answer", "")
            full_ques = f"{question} {full_edges}".strip()
            
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
        
        save_json_file(merged_data, output_path)
        print(f"Merged successfully: {output_path}")
    except Exception as e:
        print(f"Failed to merge {task_filename}: {str(e)}")
        raise
