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

        print(
            f"[FakeAVCelebDataset.__init__] path={self.path}, fold_num={fold_num}, "
            f"fold_subset={fold_subset}"
        )
        print(
            f"[FakeAVCelebDataset.__init__] allowed_attacks={self.allowed_attacks}, "
            f"bonafide_partition={self.bona_partition}, seed={self.seed}"
        )

        self.metadata = self.get_metadata()
        print(
            "[FakeAVCelebDataset.__init__] metadata loaded, shape:",
            self.metadata.shape,
        )
        print(
            "[FakeAVCelebDataset.__init__] metadata columns:",
            list(self.metadata.columns),
        )
        print(
            "[FakeAVCelebDataset.__init__] metadata head(2):\n",
            self.metadata.head(2),
        )

        fake_df = self.get_fake_samples()
        real_df = self.get_real_samples()

        print("[FakeAVCelebDataset.__init__] fake samples:", len(fake_df))
        print("[FakeAVCelebDataset.__init__] real samples:", len(real_df))
        if len(fake_df) > 0:
            print("[FakeAVCelebDataset.__init__] fake head(2):\n", fake_df.head(2))
        if len(real_df) > 0:
            print("[FakeAVCelebDataset.__init__] real head(2):\n", real_df.head(2))

        self.samples = pd.concat([fake_df, real_df], ignore_index=True)
        print(
            "[FakeAVCelebDataset.__init__] total samples:",
            len(self.samples),
            "shape:",
            self.samples.shape,
        )
        if len(self.samples) > 0:
            print(
                "[FakeAVCelebDataset.__init__] samples head(2):\n",
                self.samples.head(2),
            )
            print(
                "[FakeAVCelebDataset.__init__] label counts:\n",
                self.samples["label"].value_counts(),
            )

    def get_metadata(self):
        md_path = Path(self.path) / self.metadata_file
        print("[get_metadata] loading metadata from:", md_path)
        md = pd.read_csv(md_path)
        md["audio_type"] = md["type"].apply(lambda x: x.split("-")[-1])
        print("[get_metadata] unique audio_type:", md["audio_type"].unique())
        print("[get_metadata] unique method (first 5):", md["method"].unique()[:5])
        return md

    def get_fake_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": [],
        }

        print(
            "[get_fake_samples] allowed_attacks for this fold/subset:",
            self.allowed_attacks,
        )

        for attack_name in self.allowed_attacks:
            fake_samples = self.metadata[
                (self.metadata["method"] == attack_name)
                & (self.metadata["audio_type"] == "FakeAudio")
            ]
            print(
                f"[get_fake_samples] attack={attack_name}, rows={len(fake_samples)}"
            )
            if len(fake_samples) > 0:
                print(
                    f"[get_fake_samples] attack={attack_name} head(2):\n",
                    fake_samples.head(2),
                )

            for i, (_, sample) in enumerate(fake_samples.iterrows()):
                samples["user_id"].append(sample["source"])
                samples["sample_name"].append(Path(sample["path"]).stem)
                samples["attack_type"].append(sample["method"])
                samples["label"].append("spoof")
                samples["path"].append(self.get_file_path(sample))

                if i < 2:
                    print(
                        "[get_fake_samples] example spoof entry:",
                        samples["user_id"][-1],
                        samples["sample_name"][-1],
                        samples["attack_type"][-1],
                        samples["path"][-1],
                    )

        df = pd.DataFrame(samples)
        if len(df) > 0:
            print("[get_fake_samples] resulting DataFrame head(2):\n", df.head(2))
        else:
            print("[get_fake_samples] no fake samples for this config")
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
        print(
            "[get_real_samples] total real rows in metadata:",
            len(samples_list),
        )

        samples_list = self.split_real_samples(samples_list)
        print(
            "[get_real_samples] after split_real_samples, rows:",
            len(samples_list),
        )
        if len(samples_list) > 0:
            print("[get_real_samples] split head(2):\n", samples_list.head(2))

        for i, (_, sample) in enumerate(samples_list.iterrows()):
            samples["user_id"].append(sample["source"])
            samples["sample_name"].append(Path(sample["path"]).stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")
            samples["path"].append(self.get_file_path(sample))

            if i < 2:
                print(
                    "[get_real_samples] example bonafide entry:",
                    samples["user_id"][-1],
                    samples["sample_name"][-1],
                    samples["path"][-1],
                )

        df = pd.DataFrame(samples)
        if len(df) > 0:
            print("[get_real_samples] resulting DataFrame head(2):\n", df.head(2))
        else:
            print("[get_real_samples] no real samples for this config")
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

    print("FakeAVCeleb dataset stats")

    for fold in [0, 1, 2]:
        print("\n" + "=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)

        for subset in ["train", "val", "test"]:
            print(f"\n--- Subset: {subset} ---")

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

            print(f"Real samples : {n_real}")
            print(f"Fake samples : {n_fake}")
            print(f"Total        : {n_real + n_fake}")

    print("\nOverall totals across all folds/subsets:")
    print("Total real samples:", total_real)
    print("Total fake samples:", total_fake)

    real = 0
    fake = 0
    for subset in ["train", "test", "val"]:
        print(f"\n[SECOND PASS] subset={subset}")
        dataset = FakeAVCelebDataset(
            FAKEAVCELEB_DATASET_PATH, fold_num=2, fold_subset=subset
        )
        dataset.get_real_samples()
        real += len(dataset)
        print("[SECOND PASS] real len(dataset):", len(dataset))

        dataset = FakeAVCelebDataset(
            FAKEAVCELEB_DATASET_PATH, fold_num=2, fold_subset=subset
        )
        dataset.get_fake_samples()
        fake += len(dataset)
        print("[SECOND PASS] fake len(dataset):", len(dataset))

    print("\n[SECOND PASS] totals real, fake:", real, fake)
