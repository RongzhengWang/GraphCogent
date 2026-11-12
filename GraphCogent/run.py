import os
import json
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
import networkx as nx
from multiprocessing import Process, Queue
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


class GraphReasoningPipeline:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.stage_results = {} 
        self.final_results = []  
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def graph_to_dict(graph: nx.Graph) -> Dict:
        return {
            "nodes": list(graph.nodes()),
            "edges": [{
                "source": u,
                "target": v,
                "weight": data.get("weight", 1.0)
            } for u, v, data in graph.edges(data=True)]
        }

    @staticmethod
    def dict_to_graph(graph_dict: Dict) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(graph_dict["nodes"])
        for edge in graph_dict["edges"]:
            G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
        return G

    def save_stage_results(self, stage_name: str, results: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{stage_name}_results.json")
        
        serializable_results = []
        for result in results:
            serialized = {}
            for key, value in result.items():
                if isinstance(value, nx.Graph):
                    serialized[key] = self.graph_to_dict(value)
                else:
                    serialized[key] = value
            serializable_results.append(serialized)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=4)
        
        print(f"Saved {stage_name} results to: {output_path}")
        return output_path

    def load_stage_results(self, stage_name: str, input_dir: str) -> Optional[List[Dict]]:
        input_path = os.path.join(input_dir, f"{stage_name}_results.json")
        
        if not os.path.exists(input_path):
            return None
        
        with open(input_path, 'r', encoding='utf-8') as f:
            loaded_results = json.load(f)
        
        results = []
        for loaded in loaded_results:
            result = {}
            for key, value in loaded.items():
                if isinstance(value, dict) and "nodes" in value and "edges" in value:
                    result[key] = self.dict_to_graph(value)
                else:
                    result[key] = value
            results.append(result)
        
        return results

    def run_stage_in_process(self, stage_func, input_data, output_dir, stage_name, gpu_id):
        def worker(input_queue, output_queue):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.empty_cache()
            
            try:
                print(f"\n[{stage_name}] Starting process on GPU {gpu_id}")
                print(f"[{stage_name}] Input data type: {type(input_data)}")
                if isinstance(input_data, (list, dict)):
                    print(f"[{stage_name}] Input data length: {len(input_data)}")
                
                results = stage_func(input_data, output_dir)
                output_queue.put(("success", results))
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
                output_queue.put(("error", error_msg))

        input_queue = Queue()
        output_queue = Queue()
        

        p = Process(target=worker, args=(input_queue, output_queue))
        p.start()
        
        status, result = output_queue.get()
        p.join()
        
        if status == "error":
            raise RuntimeError(f"{stage_name} stage failed: {result}")
        
        return result

    def execute_pipeline(self, input_data: List[Dict], output_path: str, resume_from: str = None):
        try:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            intermediate_dir = os.path.join(output_dir, "intermediate")
            os.makedirs(intermediate_dir, exist_ok=True)
            
            if resume_from in [None, "graph_extraction"]:
                print("\n" + "="*50)
                print("Starting Graph Extraction Stage on GPU 1")
                print("="*50)
                
                graph_results = self.run_stage_in_process(
                    self.run_graph_extraction_stage,
                    input_data,
                    intermediate_dir,
                    "graph_extraction",
                    self.config.get("graph_extractor", {}).get("cuda_device", "0")
                )
            else:
                graph_results = self.load_stage_results("graph_extraction", intermediate_dir)
                if graph_results is None:
                    raise ValueError(f"Cannot resume from {resume_from}: missing graph_extraction results")
            
            torch.cuda.empty_cache()
            
            if resume_from in [None, "graph_extraction", "task_discrimination"]:
                print("\n" + "="*50)
                print("Starting Task Discrimination Stage on GPU 2")
                print("="*50)
                
                task_results = self.run_stage_in_process(
                    self.run_task_discrimination_stage,
                    graph_results,
                    intermediate_dir,
                    "task_discrimination",
                    self.config.get("task_discriminator", {}).get("cuda_device", "0")
                )
            else:
                task_results = self.load_stage_results("task_discrimination", intermediate_dir)
                if task_results is None:
                    raise ValueError(f"Cannot resume from {resume_from}: missing task_discrimination results")
            
            # 确保显存释放
            torch.cuda.empty_cache()
            
            # 3. 任务推理阶段 (GPU 3)
            if resume_from in [None, "graph_extraction", "task_discrimination", "task_reasoning"]:
                print("\n" + "="*50)
                print("Starting Task Reasoning Stage on GPU 3")
                print("="*50)

                task_discrimination_results = self.load_stage_results("task_discrimination", intermediate_dir)
                if task_discrimination_results is None:
                    raise ValueError("Cannot load task_discrimination_results.json")
                
                final_results = self.run_stage_in_process(
                    self.run_task_reasoning_stage,
                    task_discrimination_results, 
                    intermediate_dir,
                    "task_reasoning",
                    self.config.get("task_reasoner", {}).get("cuda_device", "0")
                )
            else:
                final_results = self.load_stage_results("task_reasoning", intermediate_dir)
                if final_results is None:
                    raise ValueError(f"Cannot resume from {resume_from}: missing task_reasoning results")
            

            serializable_results = []
            for result in final_results:
                serialized = {}
                for key, value in result.items():
                    if isinstance(value, nx.Graph):
                        serialized[key] = self.graph_to_dict(value)
                    elif isinstance(value, (list, dict)):
                        serialized[key] = self._recursive_serialize(value)
                    else:
                        serialized[key] = value
                serializable_results.append(serialized)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "results": serializable_results
                }, f, ensure_ascii=False, indent=4)
            
            print(f"\nFinal results saved to: {output_path}")
            return final_results
        
        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            raise

    def _recursive_serialize(self, data):
        if isinstance(data, nx.Graph):
            return self.graph_to_dict(data)
        elif isinstance(data, list):
            return [self._recursive_serialize(item) for item in data]
        elif isinstance(data, dict):
            return {key: self._recursive_serialize(value) for key, value in data.items()}
        else:
            return data

    def run_graph_extraction_stage(self, input_data: List[Dict], output_dir: str) -> List[Dict]:
        from modules.graph_extractor import GraphExtractor
        
        extractor = GraphExtractor(self.config)
        results = []
        
        for idx, unit in enumerate(input_data):
            print(f"\nProcessing unit {idx + 1}/{len(input_data)}")
            graph_result = extractor.process(unit)
            
            result = {
                "original_data": unit,
                "graph_extraction": graph_result
            }
            results.append(result)
        
        self.save_stage_results("graph_extraction", results, output_dir)
        return results
    
    def run_task_discrimination_stage(self, graph_results: List[Dict], output_dir: str) -> List[Dict]:
        from modules.task_discriminator import TaskDiscriminator
        
        discriminator = TaskDiscriminator(self.config)
        results = []
        
        for idx, unit_result in enumerate(graph_results):
            print(f"\nProcessing unit {idx + 1}/{len(graph_results)}")
            task_result = discriminator.process(unit_result["original_data"])
            
            result = {
                **unit_result,
                "task_discrimination": task_result
            }
            results.append(result)
        
        self.save_stage_results("task_discrimination", results, output_dir)
        return results
    

    def run_task_reasoning_stage(self, task_results: List[Dict], output_dir: str) -> List[Dict]:
        from modules.task_reasoner import TaskReasoner
        
        reasoner = TaskReasoner(self.config)
        results = []
        
        for idx, unit_result in enumerate(task_results):
            print(f"\nProcessing unit {idx + 1}/{len(task_results)}")
            reasoning_result = reasoner.process(unit_result)
            
            result = {
                **unit_result,
                "task_reasoning": reasoning_result
            }
            results.append(result)
        
        self.save_stage_results("task_reasoning", results, output_dir)
        return results




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Reasoning Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--resume-from", type=str, choices=["graph_extraction", "task_discrimination", "task_reasoning"],
                       help="Resume pipeline from specified stage")
    args = parser.parse_args()
    
    with open(args.input, 'r', encoding='utf-8') as f:
        input_units = json.load(f)
    
    pipeline = GraphReasoningPipeline(args.config)
    
    pipeline.execute_pipeline(input_units, args.output, args.resume_from)



#python Evaluation/run.py --config Evaluation/config.json --input .Graph4Real/Trans/Json/Small/Trans/Degree_Count.json --output Evaluation/temp/results.json