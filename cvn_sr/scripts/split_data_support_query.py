import os
import json
import argparse
import pickle
import random
import librosa


"""
dataset
-->class
----> file1
----> file2
--> class2
...
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i","--input_dir", type=str, required=True, help="Path to input dir",
    )
    parser.add_argument(
        "-o","--output_dir", type=str, required=False, default="./", help="Path to out dir",
    )
    parser.add_argument(
        "-min","--minimum_dur", type=float, required=False, default=0, help="Under this audio duration, we ignore the file and not add it to the manifests or jsons",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    min_dur = args.minimum_dur
    out_dir = args.output_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # we first create a dictionary with all classes and their file ids, 
    # as well as the number of files per each file_id, useful for windows
    speakers_dict = {}
    for speaker_class in sorted(os.listdir(input_dir)):

        speakers_dict[speaker_class] = {}
        subdir = os.path.join(input_dir, speaker_class)
        for audio in sorted(os.listdir(subdir)):
            if "_window" in audio:
                file_id = audio.split("_window")[0]
            else:
                file_id = os.path.splitext(audio)[0]

            audio_filepath = os.path.join(input_dir,speaker_class,audio)
            duration = librosa.get_duration(filename=audio_filepath)

            if file_id not in speakers_dict[speaker_class].keys():
                if duration >= min_dur:
                    speakers_dict[speaker_class][file_id] = [audio]
            else:
                if duration >= min_dur:
                    speakers_dict[speaker_class][file_id].append(audio)
    

    out_file_dataset = os.path.join(out_dir,os.path.basename(input_dir)+"_all.json")

    if not os.path.exists(out_file_dataset):
        with open(out_file_dataset, 'w') as fp:
            json.dump(speakers_dict, fp)

    # we use all classes for testing!
    query_dict = {}
    support_dict = {}
    
    for speaker in speakers_dict.keys():
        # we shuffle the audio recordings in a class with a fixed seed. 
        # so the support and query don't have data coming from the same audio.
        # id represents the name of an audio_file from a class,
        random.seed(30)

        # at the moment we do a fixed split of 80% support and 20% query for the files in each class
        support_ids = random.sample(sorted(list(speakers_dict[speaker].keys())), int(len(list(speakers_dict[speaker].keys())) * 0.8))
        query_ids = list(set(list(speakers_dict[speaker].keys())) - set(support_ids))


        query_dict[speaker] = {}
        support_dict[speaker] = {}

        for file_id in sorted(query_ids):
            query_dict[speaker][file_id] = [i for i in speakers_dict[speaker][file_id]]

        for file_id in sorted(support_ids):
            support_dict[speaker][file_id] = [i for i in speakers_dict[speaker][file_id]]


    out_file_query = os.path.join(out_dir,os.path.basename(input_dir)+"_query.json")
    if not os.path.exists(out_file_query):
        with open(out_file_query, 'w') as fp:
            json.dump(query_dict, fp)

    out_file_support = os.path.join(out_dir,os.path.basename(input_dir)+"_support.json")
    if not os.path.exists(out_file_support):
        with open(out_file_support, 'w') as fp:
            json.dump(support_dict, fp)

    # here we will create a manifest file for support and query audios
    # it will contain: audio_filepath, label, duration, file_id
    # it can be used later for extracting features
    manifest_support = os.path.join(out_dir,"manifest_" + os.path.basename(input_dir) + "_support.json")
    for speaker in support_dict.keys():
        for file_id in support_dict[speaker].keys():
            for audio in support_dict[speaker][file_id]:

                audio_filepath = os.path.join(input_dir,speaker,audio)
                duration = librosa.get_duration(filename=audio_filepath)
                
                entry = {}
                entry["audio_filepath"] = os.path.join(input_dir,speaker,audio)
                entry["label"] = speaker
                entry["duration"] = duration
                entry["file_id"] = file_id
                
                with open(manifest_support,"a") as f:                   
                    json.dump(entry, f)
                    f.write("\n")

    manifest_query = os.path.join(out_dir,"manifest_" + os.path.basename(input_dir) + "_query.json")
    for speaker in query_dict.keys():
        for file_id in query_dict[speaker].keys():
            for audio in query_dict[speaker][file_id]:
                
                audio_filepath = os.path.join(input_dir,speaker,audio)
                duration = librosa.get_duration(filename=audio_filepath)

                entry = {}
                entry["audio_filepath"] = os.path.join(input_dir,speaker,audio)
                entry["label"] = speaker
                entry["duration"] = duration
                entry["file_id"] = file_id
                
                with open(manifest_query,"a") as f:   
                    json.dump(entry, f)
                    f.write("\n")