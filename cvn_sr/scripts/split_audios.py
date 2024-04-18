import os
import json
import librosa
import argparse
import logging
import soundfile as sf

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
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Input directory containing audio files")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Output directory to save segmented audio files")
    parser.add_argument("-dur", "--split_duration", type=float, default =0.5,
                        help="Duration of each segment in seconds")
    parser.add_argument("-ovl", "--overlap", type=float, default=0,
                        help="Overlap between segments in seconds (default: 0)")
    parser.add_argument("-min", "--minimum_length", type=float, default=2,
                        help="Minimum duration of an utterance from the database, we ignore/discard utterances that are less than that length. (we do not split them)")
    
    return parser.parse_args()

def generate_splits(input_dir, output_dir, split_dur, overlap, min_utter, logger):
    
    # name the output dir depending on the size of the split, ovl between windows and minimum duration of the audios processed
    if output_dir is None:
        out_dir = input_dir + "_dur_" + str(split_dur) + "_ovl_" + str(overlap) + "_min_" + str(min_utter)
    else:
        out_dir = output_dir + "_dur_" + str(split_dur) + "_ovl_" + str(overlap) + "_min_" + str(min_utter)
        
    print(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    speaker_classes = os.listdir(input_dir)
    print(speaker_classes)

    """
    the datasets should have the following structure:
    class
    -->file
    -->file2
    -->file3
    ...
    class2
    -->file
    -->file2
    ...


    At output, we save the files in the same structure.
    """
    for speaker in speaker_classes:

        speaker_dir = os.path.join(input_dir,speaker)
        out_speaker_dir = os.path.join(out_dir,speaker)

        if not os.path.exists(out_speaker_dir):
            os.mkdir(out_speaker_dir)
        
        for file in sorted(os.listdir(speaker_dir)):
            input_filepath = os.path.join(speaker_dir,file)
            speaker_class = speaker_dir.split("/")[-1]
            print(input_filepath)
            y , sr = librosa.load(input_filepath, sr=16000)
            # Calculate the duration in seconds
            duration = librosa.get_duration(y=y, sr=sr)

            # we ignore files that are smaller than a minimum duration
            # example: 2 seconds
            if duration < min_utter:
                logger.info(f"File {file} from speaker {speaker_class} is smaller than {min_utter} seconds, having a duration of {duration} seconds. We gonna ignore it")
                continue
            else:
                # Calculate number of frames for each segment
                frame_len = int(sr * split_dur)
                frame_overlap = int(frame_len * overlap)

                # Calculate the number of segments, 
                # in order to use zfill to have formats like 01,02,03 to easily sort them
                num_segments = (len(y) - frame_len) // (frame_len - frame_overlap) +1

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
                    fileid = os.path.splitext(file)[0]
                    part = str(i).zfill(len(str(abs(num_segments))))
                    duration_start = start_idx/sr

                    # save files with the root being fileid and the rest being the window part
                    segment_file_path = os.path.join(out_speaker_dir, f"{fileid}_window_{part}.wav")
                    sf.write(segment_file_path, segment, sr)
                    logger.info(f"Saved segment from {file} starting at second {duration_start} to {segment_file_path}.")
            
            
def generate_splits_voxceleb1(input_dir, output_dir, split_dur, overlap, min_utter, logger):
    
    # name the output dir depending on the size of the split, ovl between windows and minimum duration of the audios processed
    if output_dir is None:
        out_dir = input_dir + "_dur_" + str(split_dur) + "_ovl_" + str(overlap) + "_min_" + str(min_utter)
    else:
        out_dir = output_dir + "_dur_" + str(split_dur) + "_ovl_" + str(overlap) + "_min_" + str(min_utter)
        
    print(out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for root,dirs,files in sorted(os.walk(args.input_dir, topdown=True)):
        if root != args.input_dir:
            out_root = out_dir + root.split(args.input_dir)[1]
        else:
            out_root = out_dir
        if len(dirs) != 0:
            for dir in dirs:
                print(os.path.join(out_root,dir))
                if not os.path.exists(os.path.join(out_root,dir)):
                    os.mkdir(os.path.join(out_root,dir))
        if len(files) != 0:
            for file in files:
                input_filepath = os.path.join(root,file)
                y , sr = librosa.load(input_filepath, sr=16000)
                # Calculate the duration in seconds
                duration = librosa.get_duration(y=y, sr=sr)

                # we ignore files that are smaller than a minimum duration
                # example: 2 seconds
                if duration < min_utter:
                    logger.info(f"File {input_filepath} is smaller than {min_utter} seconds, having a duration of {duration} seconds. We gonna ignore it")
                    continue
                else:
                    # Calculate number of frames for each segment
                    frame_len = int(sr * split_dur)
                    frame_overlap = int(frame_len * overlap)

                    # Calculate the number of segments, 
                    # in order to use zfill to have formats like 01,02,03 to easily sort them
                    num_segments = (len(y) - frame_len) // (frame_len - frame_overlap) +1

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
                        fileid = os.path.splitext(file)[0]
                        part = str(i).zfill(len(str(abs(num_segments))))
                        duration_start = start_idx/sr

                        # save files with the root being fileid and the rest being the window part
                        segment_file_path = os.path.join(out_root, f"{fileid}_window_{part}.wav")
                        sf.write(segment_file_path, segment, sr)
                        logger.info(f"Saved segment from {file} starting at second {duration_start} to {segment_file_path}.")


if __name__ == "__main__":
    args = parse_arguments()

    input_dir = args.input_dir
    output_dir = args.output_dir
    split_dur = args.split_duration
    overlap = args.overlap
    min_utter = args.minimum_length

    log_name = os.path.basename(input_dir)
    logger = setup_logger(f"{log_name}.log")

    generate_splits_voxceleb1(input_dir, output_dir,split_dur, overlap, min_utter, logger)

