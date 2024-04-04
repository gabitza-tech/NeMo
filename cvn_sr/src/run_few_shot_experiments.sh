#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <enroll_embs> <test_embs> <n_ways> <out_file> "
    exit 1
fi

# Extract arguments
enroll_embs=$1
test_embs=$2
n_way_values=($3)
out_file=$4


# Define combinations for n-way, k-shot, and n-tasks
k_shot_values=(30 36 44 60 70 80 90 97)  # Example values for k-shot

# Loop over combinations of parameters
for n_way_val in "${n_way_values[@]}"; do
    for k_shot_val in "${k_shot_values[@]}"; do
        #if [ "$k_shot_val" -le 5 ]; then
        #    no_tasks_val=100
        #else
        #    no_tasks_val=10
        #fi
        no_tasks_val=100 
        echo "Running few_shot.py with n-way=${n_way_val}, k-shot=${k_shot_val}, n-tasks=${no_tasks_val}, enroll_embs=${enroll_embs}, test_embs=${test_embs}"
        python src/few_shot_identification.py n_way=$n_way_val k_shot=$k_shot_val n_tasks=$no_tasks_val data.enrollment_embs=$enroll_embs data.test_embs=$test_embs data.out_file=$out_file
        echo "-----------------------------------------"
        
    done
done