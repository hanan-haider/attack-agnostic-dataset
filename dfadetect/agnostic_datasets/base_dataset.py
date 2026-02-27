import random

import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset
# from torch.utils.data.dataset import T_co

from dfadetect.datasets import AudioDataset, PadDataset


WAVE_FAKE_INTERFACE = True
WAVE_FAKE_SR = 16_000
WAVE_FAKE_TRIM = True
WAVE_FAKE_NORMALIZE = True
WAVE_FAKE_CELL_PHONE = False
WAVE_FAKE_PAD = True
WAVE_FAKE_CUT = 64_600


class SimpleAudioFakeDataset(Dataset):

    def __init__(self, fold_num, fold_subset, transform=None, return_label=True):
        self.transform = transform
        self.samples = pd.DataFrame()

        self.fold_num, self.fold_subset = fold_num, fold_subset
        self.allowed_attacks = None
        self.bona_partition = None
        self.seed = None
        self.return_label = return_label

        print(f"[INIT] fold_num={self.fold_num}, fold_subset={self.fold_subset}")

    def split_real_samples(self, samples_list):
        if isinstance(samples_list, pd.DataFrame):
            samples_list = samples_list.sort_values(by=list(samples_list.columns))
            samples_list = samples_list.sample(frac=1, random_state=self.seed)
        else:
            samples_list = sorted(samples_list)
            random.seed(self.seed)
            random.shuffle(samples_list)

        p, s = self.bona_partition
        subsets = np.split(
            samples_list,
            [int(p * len(samples_list)), int((p + s) * len(samples_list))]
        )
        chosen = dict(zip(["train", "test", "val"], subsets))[self.fold_subset]

        # show first 2 items
        if isinstance(chosen, pd.DataFrame):
            print("[split_real_samples] first 2 rows:\n", chosen.head(2))
        else:
            print("[split_real_samples] first 2 paths:", chosen[:2])

        return chosen

    def df2tuples(self):
        tuple_samples = []
        for _, elem in self.samples.iterrows():
            tuple_samples.append((str(elem["path"]), elem["label"], elem["attack_type"]))

        self.samples = tuple_samples
        print("[df2tuples] converted, first 2 samples:", self.samples[:2])
        return self.samples

    def __getitem__(self, index):

        if isinstance(self.samples, pd.DataFrame):
            sample = self.samples.iloc[index]
            if index < 2:
                print("[__getitem__] DataFrame mode, sample row:\n", sample)

            path = str(sample["path"])
            label = sample["label"]
            attack_type = sample["attack_type"]
        else:
            if index < 2:
                print("[__getitem__] tuple mode, sample:", self.samples[index])
            path, label, attack_type = self.samples[index]

        if WAVE_FAKE_INTERFACE:
            if index < 2:
                print("[__getitem__] loading with torchaudio from:", path)

            waveform, sample_rate = torchaudio.load(path, normalize=WAVE_FAKE_NORMALIZE)

            if sample_rate != WAVE_FAKE_SR:
                if index < 2:
                    print("[__getitem__] resampling from", sample_rate, "to", WAVE_FAKE_SR)
                waveform, sample_rate = AudioDataset.resample(
                    path, WAVE_FAKE_SR, WAVE_FAKE_NORMALIZE
                )

            if waveform.dim() > 1 and waveform.shape[0] > 1:
                waveform = waveform[:1, ...]

            if WAVE_FAKE_TRIM:
                waveform, sample_rate = AudioDataset.apply_trim(waveform, sample_rate)

            if WAVE_FAKE_CELL_PHONE:
                waveform, sample_rate = AudioDataset.process_phone_call(
                    waveform, sample_rate
                )

            if WAVE_FAKE_PAD:
                waveform = PadDataset.apply_pad(waveform, WAVE_FAKE_CUT)

            if self.return_label:
                label_num = 1 if label == "bonafide" else 0
                if index < 2:
                    print("[__getitem__] final label:", label, "->", label_num)
                return waveform, sample_rate, label_num
            else:
                return waveform, sample_rate

        # fallback path: soundfile
        data, sr = sf.read(path)
        if index < 2:
            print("[__getitem__] loading with soundfile, sr =", sr)

        if self.transform:
            data = self.transform(data)
            if index < 2:
                print("[__getitem__] transform applied")

        return data, label, attack_type

    def __len__(self):
        length = len(self.samples)
        print("[__len__] dataset size:", length)
        return length
