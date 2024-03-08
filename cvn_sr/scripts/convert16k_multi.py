import os
import librosa
import soundfile as sf
import sys
from multiprocessing import Process, Queue

def convert_audio(q):
    while True:
        item = q.get()
        if item is None:
            break
        input_file_path, output_file_path = item
        y, sr = librosa.load(input_file_path, sr=16000)
        sf.write(output_file_path, y, sr)
        print(f"Converted {input_file_path} to {output_file_path}")

def convert_to_16khz(input_dir, output_dir, num_processes):
    tasks_queue = Queue()
    processes = []

    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file in files:
            if file.endswith(".wav"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_subdir, file)
                tasks_queue.put((input_file_path, output_file_path))

    # Start processes
    for _ in range(num_processes):
        p = Process(target=convert_audio, args=(tasks_queue,))
        p.start()
        processes.append(p)

    # Add None tasks to indicate the end of tasks
    for _ in range(num_processes):
        tasks_queue.put(None)

    # Join processes
    for p in processes:
        p.join()

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_dir> <output_dir> <num_processes>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    num_processes = int(sys.argv[3])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    convert_to_16khz(input_dir, output_dir, num_processes)
