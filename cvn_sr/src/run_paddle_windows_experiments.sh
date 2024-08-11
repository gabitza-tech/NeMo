#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <out_file> <out_file2>"
    exit 1
fi

# Extract arguments
n_way_values=(1251)
out_file=$1
#out_file2=$2

# Define combinations for n-way, k-shot, and n-tasks
k_shot_values=(5 3 1)  # Example values for k-shot


# Loop over combinations of parameters
for n_way_val in "${n_way_values[@]}"; do
    for k_shot_val in "${k_shot_values[@]}"; do
        #if [ "$k_shot_val" -le 5 ]; then
        #    no_tasks_val=200
        #else
        #    no_tasks_val=100
        #fi
        no_tasks_val=5

        batch_size=16
        test_embs="saved_embs/voxceleb1_5s_dur_3_ovl_0.9/voxceleb1_5s_dur_3_ovl_0.9_query_ecapa_embs.pkl"
        enroll_embs="saved_embs/voxceleb1_5s_dur_3_ovl_0.9/voxceleb1_5s_dur_3_ovl_0.9_support_ecapa_embs.pkl"
        suffix="_dur_3_ovl_0.9"
        out_file_comb="$out_file$suffix"
        echo "Running few_shot.py with n-way=${n_way_val}, k-shot=${k_shot_val}, n-tasks=${no_tasks_val}, enroll_embs=${enroll_embs}, test_embs=${test_embs}"
        python src/paddle_windows_identification.py batch_size=$batch_size n_way=$n_way_val k_shot=$k_shot_val n_tasks=$no_tasks_val data.enrollment_embs=$enroll_embs data.test_embs=$test_embs data.out_file=$out_file_comb
        echo "-----------------------------------------"

        
        #batch_size=8
        #test_embs="saved_embs/voxceleb1_5s_dur_1_ovl_0.5/voxceleb1_5s_dur_1_ovl_0.5_query_ecapa_embs.pkl"
        #enroll_embs="saved_embs/voxceleb1_5s_dur_1_ovl_0.5/voxceleb1_5s_dur_1_ovl_0.5_support_ecapa_embs.pkl"
        #suffix="_dur_1_ovl_0.5"
        #out_file_comb="$out_file2$suffix"
        #echo "Running few_shot.py with n-way=${n_way_val}, k-shot=${k_shot_val}, n-tasks=${no_tasks_val}, enroll_embs=${enroll_embs}, test_embs=${test_embs}"
        #python src/paddle_windows_identification.py batch_size=$batch_size n_way=$n_way_val k_shot=$k_shot_val n_tasks=$no_tasks_val data.enrollment_embs=$enroll_embs data.test_embs=$test_embs data.out_file=$out_file_comb
        #echo "-----------------------------------------"
        
    done
done