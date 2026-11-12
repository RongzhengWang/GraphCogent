import networkx as nx
import random
import json
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate graph datasets')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input graph file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output JSON files')
    parser.add_argument('--scale', type=int, required=True, help='Number of nodes to sample in each subgraph')
    parser.add_argument('--num_examples', type=int, required=True, help='Number of subgraphs to generate')
    parser.add_argument('--granularity', type=int, required=True, help='Number of edges per description chunk')
    return parser.parse_args()

def read_graph_from_file(file_path):
    G = nx.Graph()  
    
    with open(file_path, 'r') as file:
        for line in file:
            nodes = line.strip().split()
            if len(nodes) == 3:  
                node1 = int(nodes[0])
                node2 = int(nodes[1])
                weight = random.randint(1, 50)
                G.add_edge(node1, node2, weight=weight)  
    
    return G

def has_cycle(graph):
    return not nx.is_forest(graph)  

def number_of_edges(graph):
    return graph.number_of_edges()

def edge_exists(graph, node1, node2):
    return graph.has_edge(node1, node2)

def node_degree(graph, node):
    return graph.degree(node)

def count_triangles(graph):
    triangle_counts = nx.triangles(graph)
    total_triangles = sum(triangle_counts.values()) // 3
    return total_triangles

def shortest_path(graph, source, target):
    try:
        path = nx.shortest_path(graph, source=source, target=target)
        weight_sum = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])) 
        return weight_sum
    except nx.NetworkXNoPath:
        return 0

def is_reachable(graph, source, target):
    return nx.has_path(graph, source, target)

def node_exists(graph, node):
    return node in graph.nodes()

def number_of_nodes(graph):
    return graph.number_of_nodes()

def max_flow(graph, source, target):
    try:
        flow_value = nx.maximum_flow_value(graph, source, target, capacity='weight')
        return flow_value
    except:
        return 0 

def has_eulerian_circuit(graph):
    if not nx.is_connected(graph):
        return False
    
    for node in graph.nodes():
        if graph.degree(node) % 2 != 0:
            return False
    
    return True

def random_walk_sampling(G, num_nodes_to_sample):
    while True:  
        sampled_nodes = set()  
        visited_nodes = set()  
        start_node = random.choice(list(G.nodes())) 
        current_node = start_node
        sampled_nodes.add(current_node)  
        visited_nodes.add(current_node)  

        H = nx.Graph()

        loop_counter = 0

        while len(sampled_nodes) < num_nodes_to_sample:
            loop_counter += 1  
            
            if loop_counter > 10000:
                print("Exceeded maximum iterations, restarting sampling...")
                break  

            neighbors = list(G.neighbors(current_node))  
            
            if not neighbors: 
                unvisited_nodes = list(set(G.nodes()) - visited_nodes)
                if not unvisited_nodes: 
                    break
                current_node = random.choice(unvisited_nodes)
                sampled_nodes.add(current_node)  
                visited_nodes.add(current_node)  
                continue
            
            next_node = random.choice(neighbors)

            weight = G[current_node][next_node]['weight']  
            H.add_edge(current_node, next_node, weight=weight) 

            sampled_nodes.add(next_node)  
            visited_nodes.add(next_node) 
            
            current_node = next_node  

        print(f"Sampled nodes count: {len(sampled_nodes)}")
        if len(sampled_nodes) >= num_nodes_to_sample:
            return H 

def main():
    args = parse_arguments()
    
    G = read_graph_from_file(args.input_path)

    datasets = {
        "Cycle_Detection": [],
        "Edge_Count": [],
        "Edge_Existence": [],
        "Degree_Count": [],
        "Triangle_Count": [],
        "Shortest_Path": [],
        "Path_Existence": [],
        "Node_Existence": [],
        "Node_Count": [],
        "Maxflow": [], 
        "Euler_Path": [] 
    }

    for i in tqdm(range(args.num_examples), desc="Generating subgraphs"):
        H = nx.Graph()
        H = random_walk_sampling(G, args.scale)
        node_list = list(H.nodes())
        output_descriptions = []
        edge_infos = []
        current_description = []
        current_edges = []

        for u, v, weight in H.edges(data='weight'):
            current_description.append(f"Node {u} is connected to Node {v} with weight {weight}")
            current_edges.append((u, v, weight))

            if len(current_description) >= args.granularity:
                edge_infos.append(current_edges)
                output_descriptions.append([", ".join(current_description)])
                current_description = []
                current_edges = []

        if current_description:
            edge_infos.append(current_edges)
            output_descriptions.append([", ".join(current_description)])
    

        if len(H.nodes) >= 2:
            source_node, target_node = random.sample(list(H.nodes()), 2)
        else:
            raise ValueError("Insufficient nodes in subgraph for random source-target selection.")

        subgraph_info = {
            "Node_List": node_list,
            "Edges": edge_infos,
            "Edge_Count": H.number_of_edges(),
            "Description": output_descriptions,
            "question": "",
            "answer": ""
        }

        subgraph_info["question"] = "Does the undirected graph have a cycle?"
        subgraph_info["answer"] = has_cycle(H)
        print(subgraph_info["question"], "->", subgraph_info["answer"]) 
        datasets["Cycle_Detection"].append(subgraph_info)

        subgraph_info = subgraph_info.copy()  
        subgraph_info["question"] = "What is the number of edges in the undirected graph?"
        subgraph_info["answer"] = number_of_edges(H)
        print(subgraph_info["question"], "->", subgraph_info["answer"]) 
        datasets["Edge_Count"].append(subgraph_info)

        subgraph_info = subgraph_info.copy() 
        subgraph_info["question"] = f"Does the edge between {source_node} and {target_node} exist?"
        subgraph_info["answer"] = edge_exists(H, source_node, target_node)
        print(subgraph_info["question"], "->", subgraph_info["answer"]) 
        datasets["Edge_Existence"].append(subgraph_info)

        subgraph_info = subgraph_info.copy()  
        subgraph_info["question"] = f"What is the degree of node {source_node}?"
        subgraph_info["answer"] = node_degree(H, source_node)
        print(subgraph_info["question"], "->", subgraph_info["answer"])  
        datasets["Degree_Count"].append(subgraph_info)

        subgraph_info = subgraph_info.copy() 
        subgraph_info["question"] = "How many triangles are in the undirected graph?"
        subgraph_info["answer"] = count_triangles(H)
        print(subgraph_info["question"], "->", subgraph_info["answer"])  
        datasets["Triangle_Count"].append(subgraph_info)

        subgraph_info = subgraph_info.copy()  
        subgraph_info["question"] = f"What is the shortest path from {source_node} to {target_node} in the undirected graph?"
        subgraph_info["answer"] = shortest_path(H, source_node, target_node)
        print(subgraph_info["question"], "->", subgraph_info["answer"]) 
        datasets["Shortest_Path"].append(subgraph_info)

        subgraph_info = subgraph_info.copy()
        subgraph_info["question"] = f"Is there a path between {source_node} and {target_node}?"
        subgraph_info["answer"] = is_reachable(H, source_node, target_node)
        print(subgraph_info["question"], "->", subgraph_info["answer"]) 
        datasets["Path_Existence"].append(subgraph_info)

        subgraph_info = subgraph_info.copy() 
        subgraph_info["question"] = f"Does node {source_node} exist in the undirected graph?"
        subgraph_info["answer"] = node_exists(H, source_node)
        print(subgraph_info["question"], "->", subgraph_info["answer"]) 
        datasets["Node_Existence"].append(subgraph_info)

        subgraph_info = subgraph_info.copy() 
        subgraph_info["question"] = "What is the number of nodes in the undirected graph?"
        subgraph_info["answer"] = number_of_nodes(H)
        print(subgraph_info["question"], "->", subgraph_info["answer"]) 
        datasets["Node_Count"].append(subgraph_info)

        subgraph_info = subgraph_info.copy()  
        subgraph_info["question"] = f"What is the maximum flow between {source_node} and {target_node} in the undirected graph?"
        subgraph_info["answer"] = max_flow(H, source_node, target_node)
        print(subgraph_info["question"], "->", subgraph_info["answer"])
        datasets["Maxflow"].append(subgraph_info)

        subgraph_info = subgraph_info.copy()
        subgraph_info["question"] = "Does the undirected graph have an Eulerian circuit?"
        subgraph_info["answer"] = has_eulerian_circuit(H)
        print(subgraph_info["question"], "->", subgraph_info["answer"]) 
        datasets["Euler_Path"].append(subgraph_info)

    for task_name, data in datasets.items():
        output_file_path = f'{args.output_dir}/{task_name}.json'  
        with open(output_file_path, 'w') as output_file:
            json.dump(data, output_file)

    print("Transportation graph datasets saved to JSON files.")

if __name__ == "__main__":
    main()

# python Graph4real__Trans/generation/sampling.py --input_path Graph4real__Trans/data/adj_matrix_node.txt --output_dir Graph4real__Trans/task --scale 100 --num_examples 1000 --granularity 50
