import multiprocessing as mp
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from mne.io import read_raw_fif
import msvcrt
import sys
import os

# Configuration
DEFAULT_FILE = r"data_organized\subject3_ses4_run3_20260121_210246_raw.fif"
STREAM_NAME = "test-player"
DEBUG_FILE = False

class InteractiveLSLPlayer:
    def __init__(self, raw, name, chunk_size=32):
        self.raw = raw
        self.name = name
        self.chunk_size = chunk_size
        self.sfreq = raw.info["sfreq"]
        self.n_channels = len(raw.ch_names)
        
        # Mapping for display
        self.class_names = {
            1: "Relax",
            2: "Left Hand",
            3: "Right Hand", 
            4: "Both Hands",
            5: "Both Feet"
        }
        
        # Pre-process data
        self._prepare_data_segments()
        
    def _prepare_data_segments(self):
        """Extract data segments for each class."""
        print("Preparing data segments...")
        self.segments = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        # Classes of interest
        keep_desc = ['1', '2', '3', '4', '5']
        data = self.raw.get_data()
        fs = self.sfreq
        
        # Iterate annotations
        for onset, duration, desc in zip(self.raw.annotations.onset, self.raw.annotations.duration, self.raw.annotations.description):
            if desc in keep_desc:
                class_id = int(desc)
                if duration == 0:
                    duration = 5.0
                    
                start_sample = int(onset * fs)
                end_sample = int((onset + duration) * fs)
                
                # Check bounds
                if end_sample > data.shape[1]:
                    end_sample = data.shape[1]
                
                # Extract segment (channels, samples) -> Transpose to (samples, channels) for LSL
                segment = data[:, start_sample:end_sample].T
                
                if segment.shape[0] > 0:
                    self.segments[class_id].append(segment)
                    
        # Verify we have data for all classes
        for cid in [1, 2, 3, 4, 5]:
            if not self.segments[cid]:
                raise ValueError(f"No data found for class {cid} ({self.class_names[cid]})")
            else:
                print(f"Class {cid} ({self.class_names[cid]}): {len(self.segments[cid])} segments.")

    def run(self, command_queue, start_event):
        print(f"Initializing LSL Stream: {self.name}")
        info = StreamInfo(name=self.name, type='EEG', channel_count=self.n_channels, nominal_srate=self.sfreq, channel_format='float32', source_id=self.name+'_eeg')
        
        chns = info.desc().append_child("channels")
        for label in self.raw.ch_names:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("type", "EEG")
            ch.append_child_value("unit", "microvolts") 
        
        outfile = StreamOutlet(info)
        
        annot_name = f"{self.name}-annotations"
        annot_info = StreamInfo(name=annot_name, type='Markers', channel_count=5, nominal_srate=self.sfreq, channel_format='float32', source_id=self.name+'_markers')
        achns = annot_info.desc().append_child("channels")
        for i in range(1, 6):
            achns.append_child("channel").append_child_value("label", str(i))
            
        annot_outfile = StreamOutlet(annot_info)
        
        print("Stream initialized. Ready.")
        start_event.set()
        
        current_class = 1 # Default to Relax
        current_segment_idx = 0
        current_sample_idx = 0
        
        # Pre-select segment
        current_segment = self.segments[current_class][0]
        
        dt = 1.0 / self.sfreq
        chunk_size = self.chunk_size
        
        last_t = local_clock()
        
        while True:
            # 1. Update State
            while not command_queue.empty():
                try:
                    cmd = command_queue.get_nowait()
                    if cmd in [1, 2, 3, 4, 5]:
                        if cmd != current_class:
                            print(f"[Player] Switching to {self.class_names[cmd]}")
                            current_class = cmd
                            # Randomly pick a segment or start from 0
                            # Picking random segment adds variety
                            curr_segs = self.segments[current_class]
                            import random
                            current_segment_idx = random.randint(0, len(curr_segs) - 1)
                            current_segment = curr_segs[current_segment_idx]
                            current_sample_idx = 0
                except:
                    pass
            
            # 2. Get Data Chunk
            # Need 'chunk_size' samples
            # If current segment finishes, go to next
            
            chunk_data = []
            samples_needed = chunk_size
            
            while samples_needed > 0:
                remaining = current_segment.shape[0] - current_sample_idx
                take = min(samples_needed, remaining)
                
                part = current_segment[current_sample_idx : current_sample_idx + take]
                chunk_data.append(part)
                
                current_sample_idx += take
                samples_needed -= take
                
                if current_sample_idx >= current_segment.shape[0]:
                    # Segment finished, move to next
                    curr_segs = self.segments[current_class]
                    current_segment_idx = (current_segment_idx + 1) % len(curr_segs)
                    current_segment = curr_segs[current_segment_idx]
                    current_sample_idx = 0
            
            # Concatenate parts
            final_chunk = np.vstack(chunk_data)
            
            # Re-create Annotation Chunk
            annot_vec = np.zeros(5, dtype=np.float32)
            annot_vec[current_class - 1] = 1.0
            annot_chunk = np.tile(annot_vec, (chunk_size, 1))
            
            # Ensure contiguous and float32 (matches StreamInfo)
            final_chunk = np.ascontiguousarray(final_chunk, dtype=np.float32)
            annot_chunk = np.ascontiguousarray(annot_chunk, dtype=np.float32)
            
            # if current_sample_idx < chunk_size:
            #    print(f"Debug: Data shape {final_chunk.shape}, Annot shape {annot_chunk.shape}")

            # 3. Create Annotation Chunk matches chunk_size
            if annot_chunk.shape[0] != final_chunk.shape[0]:
                # Re-create if sizes mismatch (should not happen if logic correct)
                annot_chunk = np.tile(annot_vec, (final_chunk.shape[0], 1))
                annot_chunk = np.ascontiguousarray(annot_chunk, dtype=np.float32)

            # 4. Push
            # Sleep to maintain rate
            # We push 'chunk_size' samples. Duration = chunk_size / sfreq
            
            now = local_clock()
            try:
                # Push data
                outfile.push_chunk(final_chunk)
                annot_outfile.push_chunk(annot_chunk)
            except Exception as e:
                print(f"Push Error: {e}")
                print(f"Data shape: {final_chunk.shape}, Outlet chans: {self.n_channels}")
                break
            
            # Timing control
            duration = chunk_size / self.sfreq
            
            next_t = last_t + duration
            delay = next_t - local_clock()
            if delay > 0:
                time.sleep(delay)
            
            last_t = next_t


def player_process_wrapper(raw_path, command_queue, start_event):
    print(f"Loading {raw_path}...")
    try:
        raw = read_raw_fif(raw_path, preload=True, verbose=False)
    except FileNotFoundError:
        print(f"Error: File not found: {raw_path}")
        return
        
    player = InteractiveLSLPlayer(raw, STREAM_NAME)
    player.run(command_queue, start_event)


if __name__ == "__main__":
    mp.freeze_support()
    
    file_path = DEFAULT_FILE
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    if not os.path.exists(file_path):
        if not os.path.exists(file_path):
             print(f"File {file_path} not found. Please check path.")
             if os.path.exists(os.path.join("..", file_path)):
                 file_path = os.path.join("..", file_path)
    
    start_event = mp.Event()
    command_queue = mp.Queue()
    
    process = mp.Process(target=player_process_wrapper, args=(file_path, command_queue, start_event), daemon=True)
    process.start()
    
    print("Waiting for player to initialize...")
    start_event.wait()
    print("Player Ready!")
    
    print("\n" + "="*40)
    print("INTERACTIVE EEG REPLAY CONTROLLER")
    print("="*40)
    print("Controls:")
    print("  [1] Relax")
    print("  [2] Left Hand")
    print("  [3] Right Hand")
    print("  [4] Both Hands")
    print("  [5] Both Feet")
    print("  [Q] Quit")
    print("="*40)
    print("Current State: Relax (1)")
    
    try:
        while True:
            # Blocking call to get character
            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                try:
                    char = key.decode("utf-8")
                except:
                    continue
                    
                if char == 'q':
                    print("Quitting...")
                    break
                    
                if char in ['1', '2', '3', '4', '5']:
                    cmd = int(char)
                    names = {1:"Relax", 2:"Left", 3:"Right", 4:"Both", 5:"Feet"}
                    print(f" -> Setting state to {cmd} ({names[cmd]})")
                    command_queue.put(cmd)
            
            time.sleep(0.05)
            
            if not process.is_alive():
                print("Player process died.")
                break
                
    except KeyboardInterrupt:
        print("\nAborting...")
    finally:
        process.terminate()
        process.join()
