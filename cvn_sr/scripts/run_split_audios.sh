#!/bin/bash

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_5s_query.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_5s -om manifests/voxceleb1_5s_dur_3_ovl_0.9_query.json --split_duration 3 -ovl 0.9

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_5s_support.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_5s -om manifests/voxceleb1_5s_dur_3_ovl_0.9_support.json --split_duration 3 -ovl 0.9
