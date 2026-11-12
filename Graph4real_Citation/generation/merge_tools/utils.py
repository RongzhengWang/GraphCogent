import json

TASK_PARAM_MAP = {
    # 0-parameter tasks
    "Edge_Count": 0,
    "Node_Count": 0,
    "Cycle_Detection": 0,
    # 1-parameter tasks
    "Degree_Count": 1,
    "Node_Existence": 1,
    "Triangle Count": 1,
    # 2-parameter tasks
    "Edge_Existence": 2,
    "Path_Existence": 2,
    "Shortest_Path": 2,
}

def get_param_count(task_name):
    return TASK_PARAM_MAP.get(task_name, -1)

def format_edges(edges):
    return ", ".join([f"{edge}" for edge in edges])

def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
