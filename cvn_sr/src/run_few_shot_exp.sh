#!/bin/bash

methods=('paddle' 'latex' 'simpleshot'  )
k_shots=(1 3 5)
log_dir='logs_metasr'
n_tasks=20
batch_size=8
backbone='meta_sr'

pairs_3s_list=(
    'saved_embs/voxceleb1_3s_metasr/voxceleb1_3s_support_metasr_embs.pkl,saved_embs/voxceleb1_3s_metasr/voxceleb1_3s_query_metasr_embs.pkl'
    'saved_embs/voxceleb1_3s_dur_1_ovl_0.5_metasr/voxceleb1_3s_dur_1_ovl_0.5_support_metasr_embs.pkl,saved_embs/voxceleb1_3s_dur_1_ovl_0.5_metasr/voxceleb1_3s_dur_1_ovl_0.5_query_metasr_embs.pkl'
    'saved_embs/voxceleb1_3s_dur_1.5_ovl_0.75_metasr/voxceleb1_3s_dur_1.5_ovl_0.75_support_metasr_embs.pkl,saved_embs/voxceleb1_3s_dur_1.5_ovl_0.75_metasr/voxceleb1_3s_dur_1.5_ovl_0.75_query_metasr_embs.pkl'
                )

for method in "${methods[@]}"; do 
    for k_shot in "${k_shots[@]}"; do
        for pairs_3s in "${pairs_3s_list[@]}"; do
            IFS=',' read -r enroll_embs test_embs <<< "$pairs_3s"

            filename=$(basename "$enroll_embs")
            out_file="${filename%_support*}"

            python3 src/few_shot.py data.enrollment_embs=$enroll_embs data.test_embs=$test_embs k_shot=$k_shot data.out_file=$out_file n_tasks=$n_tasks method=$method batch_size=$batch_size backbone=$backbone log_dir=$log_dir
        done
    done
done

pairs_5s_list=(
    'saved_embs/voxceleb1_5s_metasr/voxceleb1_5s_support_metasr_embs.pkl,saved_embs/voxceleb1_5s_metasr/voxceleb1_5s_query_metasr_embs.pkl'
    'saved_embs/voxceleb1_5s_dur_1_ovl_0.5_metasr/voxceleb1_5s_dur_1_ovl_0.5_support_metasr_embs.pkl,saved_embs/voxceleb1_5s_dur_1_ovl_0.5_metasr/voxceleb1_5s_dur_1_ovl_0.5_query_metasr_embs.pkl'
    'saved_embs/voxceleb1_5s_dur_1.5_ovl_0.75_metasr/voxceleb1_5s_dur_1.5_ovl_0.75_support_metasr_embs.pkl,saved_embs/voxceleb1_5s_dur_1.5_ovl_0.75_metasr/voxceleb1_5s_dur_1.5_ovl_0.75_query_metasr_embs.pkl'
    'saved_embs/voxceleb1_5s_dur_3_ovl_0.9_metasr/voxceleb1_5s_dur_3_ovl_0.9_support_metasr_embs.pkl,saved_embs/voxceleb1_5s_dur_3_ovl_0.9_metasr/voxceleb1_5s_dur_3_ovl_0.9_query_metasr_embs.pkl'                        
                )

for method in "${methods[@]}"; do 
    for k_shot in "${k_shots[@]}"; do
        for pairs_5s in "${pairs_5s_list[@]}"; do
            IFS=',' read -r enroll_embs test_embs <<< "$pairs_5s"
            
            filename=$(basename "$enroll_embs")
            out_file="${filename%_support*}"

            python3 src/few_shot.py data.enrollment_embs=$enroll_embs data.test_embs=$test_embs k_shot=$k_shot data.out_file=$out_file n_tasks=$n_tasks method=$method batch_size=$batch_size backbone=$backbone log_dir=$log_dir
        done
    done
done


#---------------------- THIS ONE WILL BE REMOVED AFTER 

#!/bin/bash

methods=('paddle' 'latex' 'simpleshot'  )
k_shots=(1 3 5)
log_dir='logs_ecapa'
n_tasks=20
batch_size=8
backbone='ecapa'

pairs_3s_list=(
    'saved_embs/voxceleb1_3s/voxceleb1_3s_support_embs.pkl,saved_embs/voxceleb1_3s/voxceleb1_3s_query_embs.pkl'
    'saved_embs/voxceleb1_3s_dur_1_ovl_0.5/voxceleb1_3s_dur_1_ovl_0.5_support_embs.pkl,saved_embs/voxceleb1_3s_dur_1_ovl_0.5/voxceleb1_3s_dur_1_ovl_0.5_query_embs.pkl'
    'saved_embs/voxceleb1_3s_dur_1.5_ovl_0.75/voxceleb1_3s_dur_1.5_ovl_0.75_support_embs.pkl,saved_embs/voxceleb1_3s_dur_1.5_ovl_0.75/voxceleb1_3s_dur_1.5_ovl_0.75_query_embs.pkl'            
                )

for method in "${methods[@]}"; do 
    for k_shot in "${k_shots[@]}"; do
        for pairs_3s in "${pairs_3s_list[@]}"; do
            IFS=',' read -r enroll_embs test_embs <<< "$pairs_3s"
            
            filename=$(basename "$enroll_embs")
            out_file="${filename%_support*}"

            python3 src/few_shot.py data.enrollment_embs=$enroll_embs data.test_embs=$test_embs k_shot=$k_shot data.out_file=$out_file n_tasks=$n_tasks method=$method batch_size=$batch_size backbone=$backbone log_dir=$log_dir
        done
    done
done

pairs_5s_list=(
    'saved_embs/voxceleb1_5s/voxceleb1_5s_support_embs.pkl,saved_embs/voxceleb1_5s/voxceleb1_5s_query_embs.pkl'
    'saved_embs/voxceleb1_5s_dur_1_ovl_0.5/voxceleb1_5s_dur_1_ovl_0.5_support_embs.pkl,saved_embs/voxceleb1_5s_dur_1_ovl_0.5/voxceleb1_5s_dur_1_ovl_0.5_query_embs.pkl'
    'saved_embs/voxceleb1_5s_dur_1.5_ovl_0.75/voxceleb1_5s_dur_1.5_ovl_0.75_support_embs.pkl,saved_embs/voxceleb1_5s_dur_1.5_ovl_0.75/voxceleb1_5s_dur_1.5_ovl_0.75_query_embs.pkl'
    'saved_embs/voxceleb1_5s_dur_3_ovl_0.9/voxceleb1_5s_dur_3_ovl_0.9_support_ecapa_embs.pkl,saved_embs/voxceleb1_5s_dur_3_ovl_0.9/voxceleb1_5s_dur_3_ovl_0.9_query_ecapa_embs.pkl'            
                )

for method in "${methods[@]}"; do 
    for k_shot in "${k_shots[@]}"; do
        for pairs_5s in "${pairs_5s_list[@]}"; do
            IFS=',' read -r enroll_embs test_embs <<< "$pairs_5s"
            
            filename=$(basename "$enroll_embs")
            out_file="${filename%_support*}"

            python3 src/few_shot.py data.enrollment_embs=$enroll_embs data.test_embs=$test_embs k_shot=$k_shot data.out_file=$out_file n_tasks=$n_tasks method=$method batch_size=$batch_size backbone=$backbone log_dir=$log_dir
        done
    done
done