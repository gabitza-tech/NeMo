#!/bin/bash

enroll_3s='saved_embs/voxceleb1_3s/voxceleb1_3s_support_ecapa_embs.pkl'
enroll_5s='saved_embs/voxceleb1_5s/voxceleb1_5s_support_ecapa_embs.pkl'
test_3s='saved_embs/voxceleb1_3s/voxceleb1_3s_query_ecapa_embs.pkl'
test_5s='saved_embs/voxceleb1_5s/voxceleb1_5s_query_ecapa_embs.pkl'

methods=('simpleshot' 'paddle' 'latex')

for method in "${methods[@]}"; do 

    python3 src/few_shot.py data.enrollment_embs=$enroll_3s data.test_embs=$test_3s k_shot=5 data.out_file=$method n_tasks=20 method=$method batch_size=8 backbone=ecapa
    python3 src/few_shot.py data.enrollment_embs=$enroll_5s data.test_embs=$test_3s k_shot=3 data.out_file=$method n_tasks=20 method=$method batch_size=8 backbone=ecapa
    python3 src/few_shot.py data.enrollment_embs=$enroll_3s data.test_embs=$test_3s k_shot=10 data.out_file=$method n_tasks=20 method=$method batch_size=8 backbone=ecapa
    python3 src/few_shot.py data.enrollment_embs=$enroll_5s data.test_embs=$test_3s k_shot=6 data.out_file=$method n_tasks=20 method=$method batch_size=8 backbone=ecapa

done

enroll_3s='saved_embs/voxceleb1_3s_metasr/voxceleb1_3s_support_metasr_embs.pkl'
enroll_5s='saved_embs/voxceleb1_5s_metasr/voxceleb1_5s_support_metasr_embs.pkl'
test_3s='saved_embs/voxceleb1_3s_metasr/voxceleb1_3s_query_metasr_embs.pkl'
test_5s='saved_embs/voxceleb1_5s_metasr/voxceleb1_5s_query_metasr_embs.pkl'

methods=('simpleshot' 'paddle' 'latex')

for method in "${methods[@]}"; do 

    python3 src/few_shot.py data.enrollment_embs=$enroll_3s data.test_embs=$test_3s k_shot=5 data.out_file=$method n_tasks=20 method=$method batch_size=8 backbone=metasr
    python3 src/few_shot.py data.enrollment_embs=$enroll_5s data.test_embs=$test_3s k_shot=3 data.out_file=$method n_tasks=20 method=$method batch_size=8 backbone=metasr
    python3 src/few_shot.py data.enrollment_embs=$enroll_3s data.test_embs=$test_3s k_shot=10 data.out_file=$method n_tasks=20 method=$method batch_size=8 backbone=metasr
    python3 src/few_shot.py data.enrollment_embs=$enroll_5s data.test_embs=$test_3s k_shot=6 data.out_file=$method n_tasks=20 method=$method batch_size=8 backbone=metasr

done

