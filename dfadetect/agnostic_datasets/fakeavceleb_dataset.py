from pathlib import Path

import pandas as pd

from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset


FAKEAVCELEB_KFOLD_SPLIT = {
    0: {
        "train": ["rtvc", "faceswap-wav2lip"],
        "test": ["fsgan-wav2lip"],
        "val": ["wav2lip"],
        "bonafide_partition": [0.7, 0.15],
        "seed": 42,
    },
    1: {
        "train": ["fsgan-wav2lip", "wav2lip"],
        "test": ["rtvc"],
        "val": ["faceswap-wav2lip"],
        "bonafide_partition": [0.7, 0.15],
        "seed": 43,
    },
    2: {
        "train": ["faceswap-wav2lip", "fsgan-wav2lip"],
        "test": ["wav2lip"],
        "val": ["rtvc"],
        "bonafide_partition": [0.7, 0.15],
        "seed": 44,
    },
}


class FakeAVCelebDataset(SimpleAudioFakeDataset):

    audio_folder = "/kaggle/input/datasets/mrquadian/fakeavceleb"
    audio_extension = ".flac"
    metadata_file = Path(audio_folder) / "meta_data_selected_methods.csv"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, fold_num=0, fold_subset="train", transform=None):
        super().__init__(fold_num, fold_subset, transform)
        self.path = path

        self.fold_num, self.fold_subset = fold_num, fold_subset
        self.allowed_attacks = FAKEAVCELEB_KFOLD_SPLIT[fold_num][fold_subset]
        self.bona_partition = FAKEAVCELEB_KFOLD_SPLIT[fold_num]["bonafide_partition"]
        self.seed = FAKEAVCELEB_KFOLD_SPLIT[fold_num]["seed"]

        self.metadata = self.get_metadata()

        fake_df = self.get_fake_samples()
        real_df = self.get_real_samples()

        self.samples = pd.concat([fake_df, real_df], ignore_index=True)

    def get_metadata(self):
        md_path = Path(self.path) / self.metadata_file
        md = pd.read_csv(md_path)
        md["audio_type"] = md["type"].apply(lambda x: x.split("-")[-1])
        return md

    def get_fake_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": [],
        }

        for attack_name in self.allowed_attacks:
            fake_samples = self.metadata[
                (self.metadata["method"] == attack_name)
                & (self.metadata["audio_type"] == "FakeAudio")
            ]

            for _, sample in fake_samples.iterrows():
                samples["user_id"].append(sample["source"])
                samples["sample_name"].append(Path(sample["path"]).stem)
                samples["attack_type"].append(sample["method"])
                samples["label"].append("spoof")
                samples["path"].append(self.get_file_path(sample))

        df = pd.DataFrame(samples)
        return df

    def get_real_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": [],
        }

        samples_list = self.metadata[
            (self.metadata["method"] == "real")
            & (self.metadata["audio_type"] == "RealAudio")
        ]

        samples_list = self.split_real_samples(samples_list)

        for _, sample in samples_list.iterrows():
            samples["user_id"].append(sample["source"])
            samples["sample_name"].append(Path(sample["path"]).stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")
            samples["path"].append(self.get_file_path(sample))

        df = pd.DataFrame(samples)
        return df

    def get_file_path(self, sample):
        """
        sample['audio_path'] example:
          'FakeAVCeleb/FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.flac'
        We want:
          '<base_audio_folder>/FakeVideo-FakeAudio/African/men/id00076/...flac'
        """
        rel = sample["audio_path"]

        parts = rel.split("/")
        if parts[0] == "FakeAVCeleb":
            rel = "/".join(parts[1:])

        full_path = Path(self.audio_folder) / rel
        return full_path


if __name__ == "__main__":
    FAKEAVCELEB_DATASET_PATH = ""

    total_real = 0
    total_fake = 0

    for fold in [0, 1, 2]:
        for subset in ["train", "val", "test"]:
            ds = FakeAVCelebDataset(
                FAKEAVCELEB_DATASET_PATH,
                fold_num=fold,
                fold_subset=subset,
            )

            real_df = ds.get_real_samples()
            fake_df = ds.get_fake_samples()

            n_real = len(real_df)
            n_fake = len(fake_df)

            total_real += n_real
            total_fake += n_fake

    real = 0
    fake = 0
    for subset in ["train", "test", "val"]:
        dataset = FakeAVCelebDataset(
            FAKEAVCELEB_DATASET_PATH, fold_num=2, fold_subset=subset
        )
        dataset.get_real_samples()
        real += len(dataset)

        dataset = FakeAVCelebDataset(
            FAKEAVCELEB_DATASET_PATH, fold_num=2, fold_subset=subset
        )
        dataset.get_fake_samples()
        fake += len(dataset)
