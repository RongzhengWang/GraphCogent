import importlib
import json
import os
import sys
import traceback 

# Add the merge_tools directory to Python path
merge_tools_dir = os.path.join(os.path.dirname(__file__), 'merge_tools')
sys.path.insert(0, merge_tools_dir)

TASK_PARAM_MAP = {
    # 0-parameter tasks
    "Edge_Count": 0,
    "Node_Count": 0,
    "Cycle_Detection": 0,
    "Triangle_Count": 0,
    
    # 1-parameter tasks
    "Degree_Count": 1,
    "Node_Existence": 1,
    
    # 2-parameter tasks
    "Edge_Existence": 2,
    "Path_Existence": 2,
    "Shortest_Path": 2,
}

def get_param_count(task_name):
    return TASK_PARAM_MAP.get(task_name, -1)  # Default -1 for unknown tasks

def merge_json_folders(folder1_path, folder2_path, output_folder_path, task_name):
    param_count = get_param_count(task_name)
    if param_count == -1:
        raise ValueError(f"Unknown task name: {task_name}")
    
    # Import the correct merge module
    merge_module = importlib.import_module(f"merge{param_count}")
    

    os.makedirs(output_folder_path, exist_ok=True)

    merge_module.merge_json_folders(folder1_path, folder2_path, output_folder_path, task_name)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge JSON folders for all tasks")
    parser.add_argument("folder1", help="Path to first JSON folder")
    parser.add_argument("folder2", help="Path to second JSON folder")
    parser.add_argument("output_folder", help="Path to output folder")
    
    args = parser.parse_args()
    
    # Process all tasks in TASK_PARAM_MAP
    for task_name in TASK_PARAM_MAP.keys():
        try:
            print(f"Processing task: {task_name}")
            merge_json_folders(args.folder1, args.folder2, args.output_folder, task_name)
            print(f"Merge completed for {task_name}!")
        except Exception as e:
            print(f"Merge failed for {task_name}: {str(e)}")
            traceback.print_exc()  # Print full error stack
    
    print("All tasks processed!")

