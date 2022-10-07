# import argparse
import os
# import time
# from concurrent.futures import ProcessPoolExecutor
# from typing import List, NoReturn

import h5py
import librosa
# import musdb
import numpy as np
import pathlib

from bytesep.utils import float32_to_int16


# Source types of the MUSDB18 dataset.
SOURCE_TYPES = ["vocals", "drums", "bass", "other", "mixture"]

dataset_dir = r"D:/DL/bytedance"
#ataset_dir = r"F:\magicAudio4.0\HHDS\test"
hdf5s_dir = r"D:/DL/bytedance_hdf5"

os.makedirs(hdf5s_dir, exist_ok=True)

#
audio_name = os.listdir(dataset_dir)

for audio in audio_name:
    hdf5_path = os.path.join(hdf5s_dir, "{}.h5".format(audio))
    vocals, _ = librosa.load(pathlib.Path(dataset_dir, audio, f"{SOURCE_TYPES[0]}.wav"), mono=False, sr=44100)
    vocals = np.atleast_2d(vocals)
    other, _ = librosa.load(pathlib.Path(dataset_dir, audio, f"{SOURCE_TYPES[3]}.wav"), mono=False, sr=44100)
    other = np.atleast_2d(other)
    drums = other
    bass = other
    mixture, _ = librosa.load(pathlib.Path(dataset_dir, audio, f"{SOURCE_TYPES[4]}.wav"), mono=False, sr=44100)
    mixture = np.atleast_2d(mixture)
    accompaniment = other
    #['accompaniment', 'bass', 'drums', 'mixture', 'other', 'vocals']
    with h5py.File(hdf5_path, "w") as hf:
        hf.create_dataset(
            name="accompaniment", data=float32_to_int16(accompaniment), dtype=np.int16
        )
        hf.create_dataset(
            name="bass", data=float32_to_int16(bass), dtype=np.int16
        )
        hf.create_dataset(
            name="drums", data=float32_to_int16(drums), dtype=np.int16
        )
        hf.create_dataset(
            name="mixture", data=float32_to_int16(mixture), dtype=np.int16
        )
        hf.create_dataset(
            name="other", data=float32_to_int16(other), dtype=np.int16
        )
        hf.create_dataset(
            name="vocals", data=float32_to_int16(vocals), dtype=np.int16
        )
