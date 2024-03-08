import os
import json
import librosa
import argparse

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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Segment audio files")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Input directory containing audio files")
    #parser.add_argument("-o", "--output_dir", type=str, required=True,
    #                    help="Output directory to save segmented audio files")
    parser.add_argument("-dur", "--split_duration", type=float, default =0.5,
                        help="Duration of each segment in seconds")
    parser.add_argument("-ovl", "--overlap", type=float, default=0,
                        help="Overlap between segments in seconds (default: 0)")
    parser.add_argument("-min", "--minimum_length", type=float, default=3,
                        help="Minimum duration of an utterance from the database, we ignore/discard utterances that are less than that length. (we do not split them)")
    
    return parser.parse_args()

def generate_splits(input_dir, split_dur, overlap, min_utter):
    out_dir = input_dir + "_dur_" + str(split_dur) + "_ovl_" + str(overlap) + "_min_" + str(min_utter)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    speaker_classes = os.listdir()
    speaker_dirs = []

    for speaker in speaker_classes:
        speaker_dirs.append(os.path.join(input_dir,speaker))

    for speaker_dir in speaker_dirs:
        
        out_speaker_dir = os.path.join(out_dir,speaker_dir)
        if not os.path.exists(out_speaker_dir):
            os.mkdir(out_speaker_dir)
        
        for file in speaker_dir:
            input_filepath = os.path.join(speaker_dir,file)

            
        


if __name__ == "__main__":
    args = parse_arguments()

    input_dir = args.input_dir
    #output_dir = args.output_dir
    split_dur = args.segment_duration
    overlap = args.overlap
    min_utter = args.minimum_length

    generate_splits(input_dir, split_dur, overlap, min_utter)

