#!/bin/bash

# Execute four Python commands sequentially
echo "Executing the first command: sampling.py"
python Graph4real__Trans/generation/sampling.py \
    --input_path Graph4real__Trans/data/adj_matrix_node.txt \
    --output_dir Graph4real__Trans/task \
    --scale 100 \
    --num_examples 200 \
    --granularity 50

# Check if the first command executed successfully
if [ $? -ne 0 ]; then
    echo "First command failed, exiting script"
    exit 1
fi

echo "Executing the second command: task_generator.py"
python Graph4real/generation/task_generator.py \
    --input_folder Graph4real/prompt_template/trans \
    --output_folder Graph4real/ques/trans \
    --num_examples 200

# Check if the second command executed successfully
if [ $? -ne 0 ]; then
    echo "Second command failed, exiting script"
    exit 1
fi

echo "Executing the third command: process_json_files.py"
python Graph4real__Trans/generation/process_json_files.py

# Check if the third command executed successfully
if [ $? -ne 0 ]; then
    echo "Third command failed, exiting script"
    exit 1
fi

echo "Executing the fourth command: merge.py"
python Graph4real__Trans/generation/merge.py \
    Graph4real__Trans/task \
    Graph4real__Trans/ques \
    Graph4real__Trans/task_middle

# Check if the fourth command executed successfully
if [ $? -ne 0 ]; then
    echo "Fourth command failed, exiting script"
    exit 1
fi

echo "All commands executed successfully"
