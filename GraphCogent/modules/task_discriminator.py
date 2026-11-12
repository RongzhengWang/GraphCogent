import re
import networkx as nx
from vllm import LLM
from vllm.sampling_params import SamplingParams
from typing import Dict, Any, List, Optional


class TaskDiscriminator:
    def __init__(self, config: Dict):
        self.config = config
        self.api_name_list = [
            'cycle_detection', 'degree_count', 'edge_count', 'edge_existence',
            'node_count', 'node_existence', 'path_existence', 'shortest_path', 
            'triangle_count'
        ]
        self.initialize_model()
    
    def initialize_model(self):
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
        
        self.llm = LLM(
            model=self.config["task_discriminator"]["model_path"],
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
    
    
    def process(self, graph_data: Dict) -> Dict:
        question = graph_data["translated_question"]
        
        messages = [
            {"role": "system", "content": self.config["task_discriminator"]["prompt"]},
            {"role": "user", "content": question}
        ]
        
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
        response_text = outputs[0].outputs[0].text
        
        clean_response = re.sub(r'.*\n\n', '', response_text, flags=re.DOTALL)
        
        detected_task = self.analyze_task_response(clean_response)
        
        return {
            "question": question,
            "response": clean_response,
            "detected_task": detected_task
        }
    
    def analyze_task_response(self, response: str) -> str:
        response_lower = response.lower()
        
        for task in self.api_name_list:
            if f"tool_name: {task}".lower() in response_lower or f"tool_name:{task}".lower() in response_lower:
                return task
        
        if "tool_name: null" in response_lower or "tool_name:null" in response_lower or "tool_name:none" in response_lower or "tool_name: none" in response_lower:
            return "unknown"
        
        return "unknown"

