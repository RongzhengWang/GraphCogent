import re
import networkx as nx
from vllm import LLM
from vllm.sampling_params import SamplingParams
from typing import Dict, Any, List, Optional
import subprocess
import tempfile
import sys
import os
import json
import torch
import transformers

class TaskReasoner:
    def __init__(self, config: Dict):
        self.config = config
        self.api_name_list = {
            "node_existence": 
            "The answer is correct, keep the original answer and the parameters' order, since I have get the API_Name so I only need you to provide API input strictly following my template definition.\nThe example is as follow:\n###\nAPI_Input: (graph=G, node=1)\n###",
            "path_existence": 
            "The answer is correct, keep the original answer and the parameters' order, since I have get the API_Name so I only need you to provide API input strictly following my template definition.\nThe example is as follow:\n###\nAPI_Input: (graph=G, path_source=0, path_target=1)\n###",
            "edge_existence": 
            "The answer is correct, keep the original answer and the parameters' order, since I have get the API_Name so I only need you to provide API input strictly following my template definition.\nThe example is as follow:\n###\nAPI_Input: (graph= G, edge_source=0, edge_target=1)\n###",
            "cycle_detection": 
            "Template content for cycle_detection",
            "edge_count": 
            "Template content for edge_count",
            "degree_count": 
            "The answer is correct, keep the original answer and the parameters' order, since I have get the API_Name so I only need you to provide API input strictly following my template definition.\nThe example is as follow:\n###\nAPI_Input: (graph=G, node=1)\n###",
            "node_count": 
            "Template content for node_count",
            "shortest_path": 
            "The answer is correct, keep the original answer and the parameters' order, since I have get the API_Name so I only need you to provide API input strictly following my template definition.\nThe example is as follow:\n###\nAPI_Input: (graph=G, path_source=0, path_target=1)\n###",
            "triangle_count": 
            "Template content for triangle_count"
        }
        self.load_prompt_mappings()
        self.initialize_model()
        self.initialize_pipeline()
    
    def load_prompt_mappings(self):
        """Load label to prompt mappings from JSON file"""
        try:
            with open(self.config["prompt_mappings_file"], 'r') as f:
                prompt_data = json.load(f)
                self.prompt_mappings = prompt_data["prompt_mappings"]
                
                # Validate that all required fields exist
                for label, config in self.prompt_mappings.items():
                    if "sys_prompt" not in config:
                        raise ValueError(f"Missing sys_prompt for label: {label}")
                    if "code_template" not in config:
                        raise ValueError(f"Missing code_template for label: {label}")
                    if "action" not in config:
                        raise ValueError(f"Missing action for label: {label}")
                        
        except Exception as e:
            raise ValueError(f"Failed to load prompt mappings: {str(e)}")
    
    def get_prompt_config_for_label(self, label: str) -> Dict:
        """Get the complete prompt configuration for a given label"""
        return self.prompt_mappings.get(label, self.prompt_mappings["default"])
    
    def initialize_model(self):
        """Initialize the LLM model"""
        self.llm = LLM(
            model=self.config["task_reasoner"]["model_path"],
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
    
    def initialize_pipeline(self):
        """Initialize the text generation pipeline"""
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.config["task_reasoner"]["model_path"],
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    def execute_combined_code(self, code_part1: str, llm_output: str, conda_env_name: str = "GLM") -> tuple:
        """Execute the combined code from LLM output"""
        code_blocks = re.findall(r'```python(.*?)```', llm_output, re.DOTALL)
        code_part2 = code_blocks[0].strip() if len(code_blocks) > 0 else ""
        code_part3 = code_blocks[1].strip() if len(code_blocks) > 1 else ""

        full_code = f"""
            import sys
            {code_part1}
            # Add required dependencies
            try:
                import torch
                import networkx as nx
                import random
                import torch.nn.functional as F
                from torch_geometric.nn import GCNConv
                from torch_geometric.data import Data
            except ImportError as e:
                print("Missing dependencies:", e)
                sys.exit(1)

            {code_part2}

            # Execute the function
            if __name__ == "__main__":
                try:
                    result = {code_part3.split('(')[0].strip()}(data)
                    print("Execution Result:", result)
                except Exception as e:
                    print("Execution Error:", str(e))
            """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            tmp_path = f.name

        conda_sh_path = "/home/data2t1/tempuser/miniconda3/etc/profile.d/conda.sh"
        
        if sys.platform.startswith('win'):
            python_command = fr"conda activate {conda_env_name} && python"
        else:
            python_command = (
                f"bash -c '"
                f"source {conda_sh_path} && "
                f"conda activate {conda_env_name} && "
                f"python {tmp_path}'"
            )

        try:
            result = subprocess.run(
                python_command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            output = result.stdout
        except subprocess.CalledProcessError as e:
            output = f"ERROR: {e.stderr}"
        except Exception as e:
            output = f"UNEXPECTED ERROR: {str(e)}"

        os.unlink(tmp_path)
        
        return full_code, output
    
    def handle_unknown_task(self, input_data: Dict) -> Dict:
        """Handle unknown tasks using prompt mappings"""
        label = input_data.get("label", "")
        prompt_config = self.get_prompt_config_for_label(label)
        
        if prompt_config["action"] == "pass":
            return {"status": "passed", "reason": f"Label {label} requires special handling"}
        
        processed_answer = input_data["original_data"]["translated_question"]
        sys_prompt = prompt_config["sys_prompt"]
        code_part1 = prompt_config["code_template"]
        
        messages = [
            {"role": "system", "content": processed_answer + '\n' + sys_prompt},
            {"role": "user", "content": ''}
        ]
        
        try:
            outputs = self.pipeline(
                messages,
                max_new_tokens=8192,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=0.7,
                top_p=1,
            )
            llm_output = outputs[0]["generated_text"][-1]['content']
        except Exception as e:
            return {"error": f"Generation failed: {str(e)}"}
        
        try:
            full_code, output = self.execute_combined_code(code_part1, llm_output)
            return {
                "label": label,
                "llm_output": llm_output,
                "full_code": full_code,
                "execution_output": output
            }
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}
    
    def process(self, input_data: Dict) -> Dict:
        task = input_data["task_discrimination"]["detected_task"]
        
        if task == 'unknown':
            return self.handle_unknown_task(input_data)
        else:
            question = self.api_name_list.get(task)
            messages = [
                {"role": "system", "content": input_data["original_data"]["translated_question"]},
                {"role": "user", "content": question}
            ]
            
            outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
            response_text = outputs[0].outputs[0].text
        
            return {
                "question": question,
                "final_answer": response_text
            }