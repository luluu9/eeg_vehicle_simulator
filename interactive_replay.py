import multiprocessing as mp
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from mne.io import read_raw_fif
import msvcrt
import sys
import os

# Configuration
DEFAULT_FILE = r"data_new\SUBJ1_runfull_20260523_215133_raw.fif"
STREAM_NAME = "test-player"
DEBUG_FILE = False

class InteractiveLSLPlayer:
    def __init__(self, raw, name, chunk_size=32):
        self.raw = raw
        self.name = name
        self.chunk_size = chunk_size
        self.sfreq = raw.info["sfreq"]
        self.n_channels = len(raw.ch_names)
        
        # Detect which motor classes are present and build display mapping.
        # Legacy files have classes 1-5 (4=Both Hands, 5=Feet).
        # Newer files have classes 1-4 with no Both Hands (4=Feet).
        self.present_classes = self._detect_classes()
        self.class_names = self._build_class_names(self.present_classes)
        self.n_marker_channels = max(self.present_classes)
        
        # Pre-process data
        self._prepare_data_segments()

    def _detect_classes(self):
        """Find which class ids (1-5) are present in the file's annotations."""
        candidates = {'1', '2', '3', '4', '5'}
        present = sorted(
            {int(d) for d in self.raw.annotations.description if d in candidates}
        )
        if not present:
            raise ValueError("No motor-imagery class annotations (1-5) found in file.")
        return present

    def _build_class_names(self, present):
        """Map class ids to labels, accounting for the missing Both Hands class."""
        if 5 in present:
            names = {1: "Relax", 2: "Left Hand", 3: "Right Hand", 4: "Both Hands", 5: "Feet"}
        else:
            names = {1: "Relax", 2: "Left Hand", 3: "Right Hand", 4: "Feet"}
        return {c: names.get(c, f"Class {c}") for c in present}
        
    def _prepare_data_segments(self):
        """Extract data segments for each class."""
        print("Preparing data segments...")
        self.segments = {cid: [] for cid in self.present_classes}
        
        # Classes of interest
        keep_desc = [str(cid) for cid in self.present_classes]
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
                    
        # Verify we have data for all detected classes
        for cid in self.present_classes:
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
        annot_info = StreamInfo(name=annot_name, type='Markers', channel_count=self.n_marker_channels, nominal_srate=self.sfreq, channel_format='float32', source_id=self.name+'_markers')
        achns = annot_info.desc().append_child("channels")
        for i in range(1, self.n_marker_channels + 1):
            achns.append_child("channel").append_child_value("label", str(i))
            
        annot_outfile = StreamOutlet(annot_info)
        
        print("Stream initialized. Ready.")
        start_event.set()
        
        current_class = 1 if 1 in self.present_classes else self.present_classes[0] # Default to Relax
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
                    if cmd in self.segments:
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
            annot_vec = np.zeros(self.n_marker_channels, dtype=np.float32)
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


def player_process_wrapper(raw_path, command_queue, start_event, info_queue):
    print(f"Loading {raw_path}...")
    try:
        raw = read_raw_fif(raw_path, preload=True, verbose=False)
    except FileNotFoundError:
        print(f"Error: File not found: {raw_path}")
        return
        
    player = InteractiveLSLPlayer(raw, STREAM_NAME)
    info_queue.put(player.class_names)
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
    info_queue = mp.Queue()
    
    process = mp.Process(target=player_process_wrapper, args=(file_path, command_queue, start_event, info_queue), daemon=True)
    process.start()
    
    print("Waiting for player to initialize...")
    start_event.wait()
    print("Player Ready!")
    
    class_names = info_queue.get()
    
    errp_info = StreamInfo(
        name="ErrP_Simulator",
        type="ErrP_Detection",
        channel_count=2,
        nominal_srate=0.0,
        channel_format='float32',
        source_id="errp_simulator_001",
    )
    errp_outlet = StreamOutlet(errp_info)
    print("ErrP outlet created.")
    
    print("\n" + "="*40)
    print("INTERACTIVE EEG REPLAY CONTROLLER")
    print("="*40)
    print("Controls:")
    for cid in sorted(class_names):
        print(f"  [{cid}] {class_names[cid]}")
    print("  [E] Trigger ErrP (error)")
    print("  [C] Trigger ErrP (correct)")
    print("  [Q] Quit")
    print("="*40)
    default_class = 1 if 1 in class_names else min(class_names)
    print(f"Current State: {class_names[default_class]} ({default_class})")
    
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
                    
                if char == 'e':
                    errp_outlet.push_sample([0.1, 0.9])
                    print(" -> ErrP triggered: ERROR (p_correct=0.1, p_error=0.9)")
                elif char == 'c':
                    errp_outlet.push_sample([0.9, 0.1])
                    print(" -> ErrP triggered: CORRECT (p_correct=0.9, p_error=0.1)")
                elif char in ['1', '2', '3', '4', '5'] and int(char) in class_names:
                    cmd = int(char)
                    print(f" -> Setting state to {cmd} ({class_names[cmd]})")
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
