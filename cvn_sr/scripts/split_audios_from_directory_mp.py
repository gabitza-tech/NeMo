import os
import json
import librosa
import argparse
import logging
import soundfile as sf
import pickle
import time
from multiprocessing import Pool, cpu_count
"""
-i/--input_dir must have the following structure:

DATASET
--class
----fileid_1.wav
----fileid_2.wav
...

Output directory will have the following structure:
DATASET_dur_{x}_ovl_{y}_min_{z}
--class
----fileid_1_start_{s1}.wav
----fileid_1_start_{s2}.wav
...
"""
def parse_dataset(root_dir):
    dataset = []
    
    # Walk through the root directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.wav')):  # Add other audio file extensions if needed
                # Extracting the class and subdir from the root path
                path_parts = root.split(os.sep)
                class_label = path_parts[-2]  # Assuming class is two levels up from the file
                subdir = path_parts[-1]      # Assuming subdir is one level up from the file
                file_id = f"{subdir}/{file}"
                filepath = os.path.join(root, file)
                
                # Create dictionary and append to the dataset list
                data_entry = {
                    "label": class_label,
                    "file_id": file_id,
                    "audio_filepath": filepath
                }
                dataset.append(data_entry)
    
    return dataset

def setup_logger(log_file):
    # Create a logger
    logger = logging.getLogger("short_audios_logger")
    logger.setLevel(logging.INFO)

    # Create a file handler and set the level to INFO
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set the format for the handler
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger

def parse_arguments():
    parser = argparse.ArgumentParser(description="Segment audio files")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Input directory containing audio files")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Output directory to save segmented audio files")
    parser.add_argument("-om", "--output_manifest", type=str, default=None,
                        help="Output directory to save segmented audio files")
    parser.add_argument("-dur", "--split_duration", type=float, default =0.5,
                        help="Duration of each segment in seconds")
    parser.add_argument("-ovl", "--overlap", type=float, default=0,
                        help="Overlap between segments in seconds (default: 0)")
    
    return parser.parse_args()

def process_file(row,out_dir,split_dur,overlap,logger):
    out_dicts = []
    data = json.loads(row.strip())
    input_filepath = data['audio_filepath']
    out_file_root = os.path.join(out_dir, data['label'], data['file_id'])

    final_dir = os.path.join(out_dir, data['label'], data['file_id'].split('/')[0])

    #if not os.path.exists(final_dir):
    try:
        os.makedirs(final_dir)
    except:
        pass
    y, sr = librosa.load(input_filepath, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration < 3:
        logger.info(f"File {input_filepath} is smaller than 3 seconds, having a duration of {duration} seconds. We gonna ignore it")
        return None

    frame_len = int(sr * split_dur)
    frame_overlap = int(frame_len * overlap)
    num_segments = (len(y) - frame_len) // (frame_len - frame_overlap) + 1
    patch_segs = []

    for i in range(num_segments):
        out = {}
        if i == 0:
            start_idx = i * frame_len
        else:
            start_idx = i * (frame_len - frame_overlap)

        end_idx = start_idx + frame_len
        if end_idx > len(y):
            continue

        segment = y[start_idx:end_idx]
        part = str(i).zfill(len(str(abs(num_segments))))
        duration_start = start_idx / sr
        segment_file_path = out_file_root + f"_window_{part}.wav"
        sf.write(segment_file_path, segment, sr)

        out['audio_filepath'] = segment_file_path
        out['label'] = data['label']
        out['duration'] = data['duration']
        out['file_id'] = data['file_id']
        out['patch'] = data['file_id'] + f"_window_{part}"
        
        out_dicts.append(out)

    logger.info(f"Finished spliting {data['audio_filepath']} audio.")
    return out_dicts

def generate_splits_voxceleb1(input_dir, output_manifest, output_dir, split_dur, overlap, logger):
    out_dir = output_dir + "_dur_" + str(split_dur) + "_ovl_" + str(overlap)

    dataset = parse_dataset(input_dir)

    pool = Pool(cpu_count())
    #results = [pool.apply_async(process_file, args=(row, out_dir, split_dur,overlap,logger)) for row in lines]
    #results = pool.map(process_file, [(row, out_dir) for row in lines])
    results=[]
    processed_files = set()  # Track processed files
    for row in dataset:
        if row not in processed_files:
            result = pool.apply_async(process_file, args=((row, out_dir,split_dur, overlap, logger)))
            results.append(result)
            processed_files.add(row)  # Mark as processed

    pool.close()
    pool.join()

    with open(output_manifest, 'a') as fp:
        for result in results:
            for dict_out in result.get():
                json.dump(dict_out, fp)
                fp.write('\n')


if __name__ == "__main__":
    args = parse_arguments()

    input_dir = args.input_dir
    output_manifest = args.output_manifest
    output_dir = args.output_dir
    split_dur = args.split_duration
    overlap = args.overlap

    log_name = os.path.basename(input_dir)+"_dur_" + str(split_dur) + "_ovl_" + str(overlap)
    logger = setup_logger(f"{log_name}.log")

    generate_splits_voxceleb1(input_dir, output_manifest, output_dir,split_dur, overlap, logger)

