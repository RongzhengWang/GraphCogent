import re
import networkx as nx
from vllm import LLM
from vllm.sampling_params import SamplingParams
from typing import Dict, Any, List, Optional

class GraphExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.initialize_model()
        self.Full_G = nx.Graph()  
        self.subgraph_results = [] 
        self.total_ed = 0  
        self.total_retry = 0  
    
    def initialize_model(self):
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
        
        self.llm = LLM(
            model=self.config["graph_extractor"]["model_path"],
            tokenizer_mode="auto",
            tensor_parallel_size=1,
            dtype="half",
            max_model_len=4096
        )
        
        self.sampling_params = SamplingParams(
            max_tokens=4096, 
            temperature=0.7, 
            top_p=1
        )
    
    
    def draw_graph(self, edges, descriptions=None):

        G = nx.Graph()
        for edge in edges:
            if len(edge) == 3:  
                node1, node2, weight = edge
                G.add_edge(node1, node2, weight=weight)
            else:
                raise ValueError("Each edge must be a three-element tuple (node1, node2, weight)")
        return G
    
    def parse_graph(self, input_data: str):
        import re
        
        pattern = r"\(\s*('(\d+)'|(\d+))\s*,\s*('(\d+)'|(\d+))\s*,\s*('[\d.]+'|[\d.]+)\s*\)"
        edges = re.findall(pattern, input_data)
        
        G = nx.Graph()
        for edge in edges:
            node1 = edge[0]
            node2 = edge[3]
            weight = edge[6]
            
            node1 = int(node1.strip("'")) if node1 else int(edge[1])
            node2 = int(node2.strip("'")) if node2 else int(edge[4])
            weight = float(weight.strip("'")) if weight.startswith("'") else float(weight)
            
            G.add_edge(node1, node2, weight=weight)
        
        return G
    
    def edge_edit_distance(self, G, H):

        edges_G = set(G.edges())
        edges_H = set(H.edges())
        return len(edges_H - edges_G) + len(edges_G - edges_H)
    
    def process_subgraph(self, edges, descriptions):
        description = "G: " + ", ".join(descriptions) + "."
        messages = [
            {"role": "system", "content": self.config["graph_extractor"]["prompt"]},
            {"role": "user", "content": description}
        ]
        
        H = self.draw_graph(edges, descriptions)
        
        G = None
        tolerance = 0
        retry = 0
        
        while True:

            outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
            generated_text = outputs[0].outputs[0].text
            print(generated_text)
            
            G = self.parse_graph(generated_text)
            
            allowed_min = len(H.edges) - tolerance
            allowed_max = len(H.edges) + tolerance
            
            if allowed_min <= len(G.edges) <= allowed_max:
                break
            else:
                retry += 1
                if retry % 5 == 0:
                    tolerance += 1
                    print(f"Increasing tolerance to ±{tolerance} after {retry} retries")
                print(f"Generated graph G has {len(G.edges)} edges, not in [{allowed_min}, {allowed_max}]")
        
        self.Full_G = nx.compose(self.Full_G, G)
        
        ed = self.edge_edit_distance(G, H)
        self.total_ed += ed
        self.total_retry += retry
        
        return {
            "description": description,
            "G_edges": list(G.edges(data=True)),
            "H_edges": list(H.edges(data=True)),
            "edit_distance": ed,
            "retry": retry,
            "tolerance": tolerance
        }
    
    def process(self, input_data: Dict) -> nx.Graph:
        for edges, descriptions in zip(input_data['Edges'], input_data['Description']):
            subgraph_result = self.process_subgraph(edges, descriptions)
            self.subgraph_results.append(subgraph_result)
        
        print(f'Full_G edges: {self.Full_G.edges(data=True)}')
        print(f'Full_G number of edges: {len(self.Full_G.edges())}')
        print(f'Full_G number of nodes: {len(self.Full_G.nodes())}')
        print(f'Total edit distance: {self.total_ed}')
        print(f'Total retry times: {self.total_retry}')
        
        return self.Full_G
