#!/bin/bash

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_3s_support.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_3s -om manifests/voxceleb1_3s_dur_0.5_ovl_0_support.json --split_duration 0.5 -ovl 0

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_5s_query.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_5s -om manifests/voxceleb1_5s_dur_0.5_ovl_0_query.json --split_duration 0.5 -ovl 0

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_5s_support.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_5s -om manifests/voxceleb1_5s_dur_0.5_ovl_0_support.json --split_duration 0.5 -ovl 0

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_3s_query.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_3s -om manifests/voxceleb1_3s_dur_1_ovl_0.5_query.json --split_duration 1 -ovl 0.5

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_3s_support.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_3s -om manifests/voxceleb1_3s_dur_1_ovl_0.5_support.json --split_duration 1 -ovl 0.5

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_5s_query.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_5s -om manifests/voxceleb1_5s_dur_1_ovl_0.5_query.json --split_duration 1 -ovl 0.5

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_5s_support.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_5s -om manifests/voxceleb1_5s_dur_1_ovl_0.5_support.json --split_duration 1 -ovl 0.5

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_3s_query.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_3s -om manifests/voxceleb1_3s_dur_0.5_ovl_0.5_query.json --split_duration 0.5 -ovl 0.5

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_3s_support.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_3s -om manifests/voxceleb1_3s_dur_0.5_ovl_0.5_support.json --split_duration 0.5 -ovl 0.5

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_5s_query.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_5s -om manifests/voxceleb1_5s_dur_0.5_ovl_0.5_query.json --split_duration 0.5 -ovl 0.5

python3 scripts/split_audios_from_manifest_mp.py --input_manifest manifests/voxceleb1_5s_support.json --output_dir /media/gabi/gabi_data/datasets_processed/voxceleb1_5s -om manifests/voxceleb1_5s_dur_0.5_ovl_0.5_support.json --split_duration 0.5 -ovl 0.5
