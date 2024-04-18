import os
import json
import librosa
import argparse
import logging
import soundfile as sf
import pickle
import time
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
    parser.add_argument("-i", "--input_manifest", type=str, required=True,
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
            
            
def generate_splits_voxceleb1(input_manifest, output_manifest, output_dir, split_dur, overlap, logger):
    
    # name the output dir depending on the size of the split, ovl between windows and minimum duration of the audios processed
    out_dir = output_dir + "_dur_" + str(split_dur) + "_ovl_" + str(overlap)
        
    with open(input_manifest,'r') as f:
        lines = f.readlines()
    out = {}
    out['audio_filepath'] = []
    out['label'] = []
    out['duration'] = []
    out['file_id'] = []
    out['patch'] = []
    start = time.time()
    for (index,row) in enumerate(lines):
        if (index % 100) == 0:
            dur = time.time() - start
            logger.info(f"Audios split: {index}/{len(lines)} {dur}.")
            print(f"Audios split: {index}/{len(lines)} {dur}.")
            start = time.time()
            
        data = json.loads(row.strip())
        input_filepath = data['audio_filepath']
        out_file_root = os.path.join(out_dir,data['label'],data['file_id'])
        
        final_dir = os.path.join(out_dir,data['label'],data['file_id'].split('/')[0])

        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        
        y , sr = librosa.load(input_filepath, sr=16000)
        # Calculate the duration in seconds
        duration = librosa.get_duration(y=y, sr=sr)

        # we ignore files that are smaller than a minimum duration
        # example: 2 seconds
        if duration < 3:
            logger.info(f"File {input_filepath} is smaller than 3 seconds, having a duration of {duration} seconds. We gonna ignore it")
            continue
        else:
            # Calculate number of frames for each segment
            frame_len = int(sr * split_dur)
            frame_overlap = int(frame_len * overlap)

            # Calculate the number of segments, 
            # in order to use zfill to have formats like 01,02,03 to easily sort them
            num_segments = (len(y) - frame_len) // (frame_len - frame_overlap) +1
            patch_segs = []
            for i in range(num_segments):
                
                if i == 0:
                    start_idx = i * frame_len 
                else:
                    start_idx = i * (frame_len - frame_overlap)

                end_idx = start_idx + frame_len
                if end_idx > len(y):
                    continue

                # Extract segment
                segment = y[start_idx:end_idx]

                # Save segment
                part = str(i).zfill(len(str(abs(num_segments))))
                duration_start = start_idx/sr

                # save files with the root being fileid and the rest being the window part
                segment_file_path = out_file_root + f"_window_{part}.wav"
                sf.write(segment_file_path, segment, sr)
                #logger.info(f"Saved segment from {data['file_id']} starting at second {duration_start} to {segment_file_path}.")

                out['audio_filepath'].append(segment_file_path)
                out['label'].append(data['label'])
                out['duration'].append(data['duration'])
                out['file_id'].append(data['file_id'])
                patch_segs.append(data['file_id'] + f"_window_{part}")
                #with open(output_manifest,'a') as f:
                #    json.dump(out,f)
                #    f.write('\n')

            out['patch'].append(patch_segs)
            
    with open(output_manifest, 'wb') as fp:
        pickle.dump(out, fp)




if __name__ == "__main__":
    args = parse_arguments()

    input_manifest = args.input_manifest
    output_manifest = args.output_manifest
    output_dir = args.output_dir
    split_dur = args.split_duration
    overlap = args.overlap

    log_name = os.path.basename(input_manifest)+"_dur_" + str(split_dur) + "_ovl_" + str(overlap)
    logger = setup_logger(f"{log_name}.log")

    generate_splits_voxceleb1(input_manifest, output_manifest, output_dir,split_dur, overlap, logger)

