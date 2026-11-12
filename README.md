# <center> GraphCogent: Mitigating LLMs’ Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding</center>


This repository contains the code, data, and models for "GraphCogent: Mitigating LLMs’ Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding."

🔗 **GitHub**: [https://anonymous.4open.science/r/GraphCogent](https://anonymous.4open.science/r/GraphCogent)  
📜 **Paper**: [Added later]() | 📊 **Benchmark**: [Graph4real](https://anonymous.4open.science/r/GraphCogent) | 🤖 **Agent**: [Added later]() 


**📢 Notice: Ongoing Maintenance**: 
This repository is currently under active development. Core code, execution scripts, and related materials will be uploaded soon.

## Key Features

### 1. GraphCogent Framework
We propose **GraphCogent**, a **collaborative agentic framework** that addresses LLMs' limitations in graph reasoning through:
- **Sensory Module**: Standardizes diverse graph text representations (adjacency lists, symbolic notations, linguistic descriptions) via subgraph sampling.
- **Buffer Module**: Integrates and indexes graph data for efficient retrieval across formats (NetworkX, PyG, NumPy).
- **Execution Module**: Combines **tool calling** (for in-toolset tasks) and **tool generation** (for out-toolset tasks) for robust reasoning.

### 2. Graph4real Benchmark
We introduce **Graph4real**, a **real-world graph reasoning benchmark** with:
- **Large-scale graphs** (40–1000+ nodes) from **4 domains** (Web, Social, Transportation, Citation).
- **21 diverse tasks** (Structural Querying, Algorithmic Reasoning, Predictive Modeling).
- **Intent-driven queries** mirroring real-world reasoning scenarios.

### 3. Performance Improvements
GraphCogent achieves:

✅ **98.5% accuracy** on in-toolset tasks (20% improvement over baselines).  
✅ **97.6% accuracy** on **1000-node graphs**, where prior methods fail.  
✅ **30–80% reduction in token consumption** compared to agent-based methods.  
✅ **Robust generalization** across public benchmarks (>90% accuracy).

---



Our entire work flow can be summarized as follows:

<div align="center">
<img src="pics\method.jpg" width="800px">
</div>

**Overview of GraphCogent:** Sensory Module (left) standardizes various graph text representations through subgraph sampling and conversion; Buffer Module (center) establishes cross-format data (e.g., NetworkX) integrating and indexing transformations; Execution Module (right) enables two reasoning modes: Execution Agent is employed for tool discrimination and implements tool calling for in-toolset tasks, Tool Creator handles out-toolset tasks based on tool creation.


## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Environment Preparation'>1. Environment Preparation </a>
* <a href='#Graph4real Construction Workflow'>2. Graph4real Construction Workflow </a>
  * <a href='#Dataset Preparation Rules'>2.1 Dataset Preparation Rules
  * <a href='#Pipeline Implementation'>2.2 Pipeline Implementation
  * <a href='#Task Scenarios'>2.3 Task Scenarios
  * <a href='#Benchmark Details'>2.4 Benchmark Details
  * <a href='#Execution Example'>2.5 Execution Example

* <a href='#Evaluating GraphCogent'>3. Evaluating GraphCogent</a>
  * <a href='#Task Execution'>3.1 Task Execution</a>
  * <a href='#Task Evaluation'>3.2 Task Evaluation</a>
* <a href='#GraphCogent Training Process'>4. GraphCogent Training Process </a>
  * <a href='#LLaMA-Factory Installation'>4.1. LLaMA-Factory Installation</a>
  * <a href='#Thinking Path Collection'>4.2. Thinking Path Collection</a>
  * <a href='#Thinking Policy Initialization for Tool Invocation'>4.3. Thinking Policy Initialization for Tool Invocation</a>
  * <a href='# Capability-Margin Preference Optimization'>4.4  Capability-Margin Preference Optimization</a>
  * <a href='#Export Model'>4.5 Export Model</a>



****


<span id='Environment Preparation'/>


### 1. Environment Preparation  <a href='#all_catelogue'>[Back to Top]</a>
Please first clone the repo and install the required environment, which can be done by running the following commands:
```shell
conda create -n GraphCogent python=3.12

conda activate GraphCogent

# Torch with CUDA 12.3
# Please clone our GraphCogent first, and switch to the directory.
cd GraphCogent
# Install required libraries
pip install -r requirements.txt
```

<span id='Graph4real Construction Workflow'/>

The Graph4real task generators and `Baseline_Test/gpt4o_fc.py` read `OPENAI_API_KEY` from the environment. Set `OPENAI_BASE_URL` if you use a custom OpenAI-compatible endpoint.

### 2. Graph4real Construction Workflow  <a href='#all_catelogue'>[Back to Top]</a>


<span id='Dataset Preparation Rules'/>

#### 2.1 Dataset Preparation Rules <a href='#all_catelogue'>[Back to Top]</a>

##### Real-World Graph Scaling
- **Domains**: 
  - Web (Google Web Graph)
  - Social (SNAP Social Circles) 
  - Transportation (PeMS)
  - Citation (Cora)
- **Scales**:
  - Small: 40 nodes
  - Medium: 100 nodes
  - Large: 1000 nodes
- **Sampling Method**: Biased random walks preserving topological properties

##### Text Representations
| Format | Example | Use Case |
|--------|---------|----------|
| Adjacency List | `[0,1],[0,2]` | Algorithmic tasks |
| Symbolic Notation | `0→1` | Structural queries |
| Linguistic Descriptions | "Station A connects to Station B" | Domain-specific tasks |

<span id='Pipeline Implementation'/>

#### 2.2 Pipeline Implementation <a href='#all_catelogue'>[Back to Top]</a>

##### Directory Structure

Although the task generation scripts for the four scenarios have some differences, their execution logic is largely similar. Here, we use **Transportation** as an example to introduce the construction process of our Graph4real, and we provide a Middle Scale task example.

For task construction, we employ a two-phase methodology: (1) Graph Sampling and Generation, where we extract real-world data (e.g., PeMS) to generate graphs of varying scales and derive reasoning questions from them, and (2) Task Generation, where we enhance LLM intent understanding evaluation by moving beyond rigid templates—instead leveraging prompt_template to dynamically construct scenario-question pairs, using DeepSeek-R1 to synthesize reasoning questions that mimic authentic queries, ultimately merging these with the sampled graphs to form the Graph4real benchmark.

```
Graph4real_Trans
├── ques
│   ├── Edge_Count.json
│   └── ...
├── sampling                   
│   ├── Degree_Count.json
│   └── ...
├── task_middle                 # Final Task
│   ├── Shortest_Path.json
│   └── ...
├── prompt_template             # Task generation template
│   ├── Node_Count.txt
│   └── ...
├── data
│   └── adj_matrix_node.txt     # Original graph data
└── generation
    ├── bash.sh
    ├── sampling.py             # Graph sampling and generation
    ├── task_generator.py       # Task generation
    ├── merge_tools             
    │   ├── merge0.py
    │   ├── merge1.py
    │   ├── merge2.py
    │   └── utils.py
    ├── process_json_files.py
    └── merge.py                # Graph and Task merge
```


##### Key Scripts

##### Graph Sampling and Generation
```bash
python Graph4real__Trans/generation/sampling.py \
  --input_path Graph4real__Trans/data/adj_matrix_node.txt \
  --output_dir Graph4real__Trans/task \
  --scale 100 \
  --num_examples 1000 \
  --granularity 50

```

##### Task Generation
```bash
python Graph4real/generation/task_generator.py \
  --input_folder Graph4real/prompt_template/trans \
  --output_folder Graph4real/ques/trans \
  --num_examples 400

```

<span id='Task Scenarios'/>

#### 2.3 Task Scenarios <a href='#all_catelogue'>[Back to Top]</a>

| **Domain**     | **Scenarios** |
|----------------|---------------|
| **Social**     | - Information diffusion analysis <br> - Community detection and recommendation systems <br> - Fraudulent account detection <br> - Influence maximization algorithms <br> - Social network dynamics analysis |
| **Web**        | - Web crawler efficiency optimization <br> - Search engine ranking optimization <br> - Web structural integrity diagnosis <br> - Topical community discovery <br> - DDoS attack mitigation |
| **Transportation** | - Travel route planning <br> - Logistics delivery optimization <br> - Urban emergency response planning <br> - Public transit network scheduling <br> - Shared mobility platforms |
| **Citation**   | - Scholarly influence tracking <br> - Interdisciplinary research identification <br> - Seminal paper discovery <br> - Literature retrieval ranking optimization <br> - Research frontier identification |

<span id='Benchmark Details'/>

#### 2.4 Benchmark Details <a href='#all_catelogue'>[Back to Top]</a>

| Task              | Description                                                                 | Tool Algorithm     | Time Complexity       |
|-------------------|-----------------------------------------------------------------------------|--------------------|-----------------------|
| Edge Count        | Count the total number of edges in a given graph.                          | Direct Lookup      | $O(1)$               |
| Node Count        | Count the total number of nodes in a given graph.                          | Direct Lookup      | $O(1)$               |
| Degree Count      | Count the number of edges connected to a specific node in a given graph.   | Direct Lookup      | $O(1)$               |
| Edge Existence    | Determine whether a specific edge exists between two nodes in a given graph. | Direct Lookup      | $O(1)$               |
| Node Existence    | Determine whether a specific node exists in a given graph.                 | Direct Lookup      | $O(1)$               |
| Cycle Detection   | Determine whether there exists any cycle in a given graph.                 | Depth-First Search | $O(\|V\|+\|E\|)$     |
| Triangle Count    | Count the total number of triangles in a given graph.                      | Node Iterator      | $O(\|V\| \cdot d_{\text{max}}^2)$ |
| Path Existence    | Determine whether a specific path exists between two nodes in a given graph. | Depth-First Search | $O(\|V\|+\|E\|)$     |
| Shortest Path     | Determine the minimum distance between two nodes in a given graph.        | Dijkstra           | $O({\|V\|}^2+\|E\|)$ |

> **Total**: 4,200 questions across 21 tasks

<span id='Execution Example'/>

#### 2.5 Execution Example <a href='#all_catelogue'>[Back to Top]</a>

```bash
# Generate full Transportation benchmark
bash Graph4real_Trans/generation/bash.sh
```



<span id='Evaluating GraphCogent'/>

## 3. Evaluating GraphCogent <a href='#all_catelogue'>[Back to Top]</a>


**Notice:** Due to the fact that we use 8*4090 GPUs running in parallel during the inference process, our execution script is not suitable for most users. Therefore, we recommend assigning specific tasks to specific GPUs and running each specific task separately. To simplify execution, we have designed a stage-by-stage pipeline that processes all input data per stage before moving to the next, minimizing GPU memory usage and allowing flexible resumption from any intermediate step. 

### **GraphCogent: Unified Execution Framework**  
**Repository Structure**:  
```
GraphCogent/
├── run.py                  # Unified execution script
├── eval.py                 # Count Results
├── config.json             # Configuration json
├── modules/                # Reasoning Components
│   ├── __init__.py         # Package initialization
│   ├── graph_extractor.py  # Graph extraction module
│   ├── task_discriminator.py  # Task classification module 
│   └── task_reasoner.py    # Core reasoning module
```

---
<span id='Task Execution'/>

#### **3.1 Task Execution**  <a href='#all_catelogue'>[Back to Top]</a>
Run any task (Sensory → Buffer → Execution) with a single command:  
```bash
python Evaluation/run.py \
    --config Evaluation/config.json \     # Path to configuration file
    --input .Graph4Real/Trans/Json/Small/Trans/Degree_Count.json \  # Input graph data
    --output Evaluation/outputs/results.json \  # Output results path
    # Optional: "resume-from" if needed choices=["graph_extraction", "task_discrimination", "task_reasoning"], our code supports resuming pipeline from specified stage"
    # --resume-from

```

---
<span id='Task Evaluation'/>

#### **3.2 Task Evaluation**  <a href='#all_catelogue'>[Back to Top]</a>
Evaluate results against ground truth:  
```bash
python eval.py \
    --result_dir ./outputs \             # Path to GraphCogent's outputs
    --report_file ./outputs/summary.json
```

**Output Format (`summary.json`)**:  
```json
{
  "accuracy": 98.5,
  "tasks": {
    "shortest_path": {"type": "transportation","correct": 98, "total": 100}
  }
}
```

---


<span id='GraphCogent Training Process'/>

## 4. GraphCogent Training Process <a href='#all_catelogue'>[Back to Top]</a>

For fine-tuning both the Reasoning Agent and Model Agent, we utilize the Llama-Factory framework. We will provide the relevant files for key training steps along with the corresponding execution commands.

### Training Pipeline 
Reasoning Agent tuning paradigm consists of four stages: (1) LLaMA-Factory Installation; (2) Thinking Path Collection; (3) Thinking Policy Initialization for Tool Invocation; (4)  Capability-Margin Preference Optimization; (5) Export Model.

---

<span id='LLaMA-Factory Installation'/>

#### 4.1 LLaMA-Factory Installation <a href='#all_catelogue'>[Back to Top]</a>

```shell
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

**Prerequisites**:  
- Download **Llama3.1-8B-Instruct** weights: [HuggingFace Hub](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)  
- Place under `./models/llama3.1-8b-instruct/`  

---

<span id='Thinking Path Collection'/>

#### 4.2 Thinking Path Collection <a href='#all_catelogue'>[Back to Top]</a>

Generate reasoning trajectories for SFT using Graph4real tasks:
```python
python GraphCogent_train/Think_generation.py \
    --input_dir "/path/to/Graph4real/folder" \
    --output_dir "/path/to/LLaMA-Factory/folder" \
    --prompt_file "GraphCogent_train/reasoning_prompt.txt" \
    --model "deepseek-r1" \
    --api_key "your_api_key_here" \
    --base_url "your_base_url_here"
```

* **Output Format** (Alpaca-style):
```json
{
  "instruction": "You are a Graph expert, you should use one most suitable tool to solve the following task...",
  "input": "When planning a scenic driving route through the mountain towns, a traveler wonders...",
  "output": "<Think>Okay, so l need to figure out...</Think> ... Tool_name: Shortest_Path"
}
```


* **Prepare data:** To use LLaMA-Factory for model fine-tuning, we need to add the following three formats to the LLaMA-Factory/data/dataset_info.json path:


```shell
  "Reasoning": {
    "file_name": "${GraphCogent_train/Train/Thinking.json}",
    "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
    }
  }
```


---

<span id='Thinking Policy Initialization for Tool Invocation'/>

#### 4.3 Thinking Policy Initialization for Tool Invocation <a href='#all_catelogue'>[Back to Top]</a>

Use the following command to run LoRA **fine-tuning** of the Llama3.1-8B-Instruct model under the path: examples/train_lora/llama3_lora_sft.yaml. 

```shell
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```


```shell
# Our fine-tuning parameter settings are as follows.
model_path= Llama3.1/model
output_model= saves/llama3.1-8b/lora

### model
model_name_or_path: ${model_path}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: Reasoning
template: llama3
cutoff_len: 4096
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ${output_model}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500

```

---


<span id=' Capability-Margin Preference Optimization'/>

#### 4.4  Capability-Margin Preference Optimization <a href='#all_catelogue'>[Back to Top]</a>
#### Prepare Preference Data:
```python
python GraphCogent_train/DPO_construction.py \
    --input_file "/path/to/Graph4real/folder" \
    --output_dir "/path/to/LLaMA-Factory/folder" \
    --prompt_file "GraphCogent_train/reasoning_prompt.txt" \
    --model_path "SFT-based/Llama3.1/path" 
```

* **Prepare data:** To use LLaMA-Factory for model dpo-tuning, we need to add the following three formats to the LLaMA-Factory/data/dataset_info.json path:


```shell
  "dpo_Graph4real": {
    "file_name": "${GraphCogent_train/Train/dpo_Graph4real.json}",
    "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
    }
  }
```


#### Config: `examples/train_dpo/graphcogent_dpo.yaml`
```shell
model_path= SFT-based/Llama3.1/path
output_model= saves/llama3.1-8b/dpo

### model
model_name_or_path: ${model_path}
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid  

### dataset
dataset: dpo_Graph4real
template: llama3
cutoff_len: 4096
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: {output_model}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500

```

**Run DPO**:
```shell
llamafactory-cli train examples/train_dpo/graphcogent_dpo.yaml 
```

---

<span id='Export Model'/>

#### 4.5 Export Model <a href='#all_catelogue'>[Back to Top]</a>
Merge LoRA adapters into deployable model:
```shell
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

```shell
# Our settings are as follows.
### model
model_name_or_path: DPO-based/Llama3.1/path
adapter_name_or_path: saves/llama3.1-8b/dpo
template: llama3
finetuning_type: lora

### export
export_dir: GraphCogent/model/Reasoning_Agent
export_size: 4
export_device: cpu
export_legacy_format: false
```



---



