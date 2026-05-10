import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

CHANNEL_NAMES = [
    "Fp1", "Fp2", "FC5", "FC1", "FC2", "FC6", "C3", "Cz",
    "C4", "CP5", "CP1", "CP2", "CP6", "P3", "Pz", "P4",
]


def main():
    srate = 2048
    n_channels = len(CHANNEL_NAMES)
    info = StreamInfo("MockEEG", "EEG", n_channels, srate, "float32", "mock_eeg_16ch")

    channels = info.desc().append_child("channels")
    for name in CHANNEL_NAMES:
        channels.append_child("channel").append_child_value("label", name)

    outlet = StreamOutlet(info)
    print(f"MockEEG stream: {n_channels}ch @ {srate}Hz. Press Ctrl+C to stop.")

    start_time = time.time()
    sent_samples = 0

    try:
        while True:
            elapsed = time.time() - start_time
            required = int(srate * elapsed) - sent_samples

            if required > 0:
                data = np.random.randn(required, n_channels).astype(np.float32) * 10
                outlet.push_chunk(data.tolist())
                sent_samples += required

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stream stopped.")


if __name__ == "__main__":
    main()
