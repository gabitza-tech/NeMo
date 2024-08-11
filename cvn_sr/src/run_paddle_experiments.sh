#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <enroll_embs> <test_embs> <enroll_embs2> <test_embs2> <n_ways> <out_file> <out_file2>"
    exit 1
fi

# Extract arguments
enroll_embs=$1
test_embs=$2
enroll_embs2=$3
test_embs2=$4
n_way_values=($5)
out_file=$6
out_file2=$7

# Define combinations for n-way, k-shot, and n-tasks
k_shot_values=(5 10)  # Example values for k-shot


# Loop over combinations of parameters
for n_way_val in "${n_way_values[@]}"; do
    for k_shot_val in "${k_shot_values[@]}"; do
        if [ "$k_shot_val" -le 5 ]; then
            no_tasks_val=5
        else
            no_tasks_val=5
        fi
        
        echo "Running few_shot.py with n-way=${n_way_val}, k-shot=${k_shot_val}, n-tasks=${no_tasks_val}, enroll_embs=${enroll_embs}, test_embs=${test_embs}"
        python src/paddle_identification.py n_way=$n_way_val k_shot=$k_shot_val n_tasks=$no_tasks_val data.enrollment_embs=$enroll_embs data.test_embs=$test_embs data.out_file=$out_file
        echo "-----------------------------------------"
        
    done
done

# Loop over combinations of parameters
for n_way_val in "${n_way_values[@]}"; do
    for k_shot_val in "${k_shot_values[@]}"; do
        if [ "$k_shot_val" -le 5 ]; then
            no_tasks_val=200
        else
            no_tasks_val=100
        fi
        
        echo "Running few_shot.py with n-way=${n_way_val}, k-shot=${k_shot_val}, n-tasks=${no_tasks_val}, enroll_embs=${enroll_embs}, test_embs=${test_embs}"
        python src/paddle_identification.py n_way=$n_way_val k_shot=$k_shot_val n_tasks=$no_tasks_val data.enrollment_embs=$enroll_embs2 data.test_embs=$test_embs2 data.out_file=$out_file2
        echo "-----------------------------------------"
        
    done
done
