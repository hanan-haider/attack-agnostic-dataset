from pathlib import Path

import pandas as pd

from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset


WAVEFAKE_KFOLD_SPLIT = {
    0: {
        "train": ['melgan_large', 'waveglow', 'full_band_melgan', 'melgan', 'hifiGAN'],
        "test": ['multi_band_melgan'],
        "val": ['parallel_wavegan'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 42,
    },
    1: {
        "train": ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'melgan', 'hifiGAN'],
        "test": ['waveglow'],
        "val": ['full_band_melgan'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 43,
    },
    2: {
        "train": ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'waveglow', 'full_band_melgan'],
        "test": ['melgan'],
        "val": ['hifiGAN'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 44,
    },
}


class WaveFakeDataset(SimpleAudioFakeDataset):

    fake_data_path = "generated_audio"
    jsut_real_data_path = "jsut_ver1.1/basic5000/wav"
    ljspeech_real_data_path = "the-LJSpeech-1.1/wavs"

    def __init__(self, path, fold_num=0, fold_subset="train", transform=None):
        super().__init__(fold_num, fold_subset, transform)
        self.path = Path(path)

        self.fold_num, self.fold_subset = fold_num, fold_subset
        self.allowed_attacks = WAVEFAKE_KFOLD_SPLIT[fold_num][fold_subset]
        self.bona_partition = WAVEFAKE_KFOLD_SPLIT[fold_num]["bonafide_partition"]
        self.seed = WAVEFAKE_KFOLD_SPLIT[fold_num]["seed"]

        print(
            f"[WaveFakeDataset.__init__] path={self.path}, fold_num={fold_num}, "
            f"fold_subset={fold_subset}"
        )
        print(
            f"[WaveFakeDataset.__init__] allowed_attacks={self.allowed_attacks}, "
            f"bonafide_partition={self.bona_partition}, seed={self.seed}"
        )

        gen_df = self.get_generated_samples()
        real_df = self.get_real_samples()

        print("[WaveFakeDataset.__init__] generated samples:", len(gen_df))
        print("[WaveFakeDataset.__init__] real samples:", len(real_df))
        if len(gen_df) > 0:
            print("[WaveFakeDataset.__init__] generated head(2):\n", gen_df.head(2))
        if len(real_df) > 0:
            print("[WaveFakeDataset.__init__] real head(2):\n", real_df.head(2))

        self.samples = pd.concat([gen_df, real_df], ignore_index=True)
        print(
            "[WaveFakeDataset.__init__] total samples:",
            len(self.samples),
            "shape:",
            self.samples.shape,
        )
        if len(self.samples) > 0:
            print("[WaveFakeDataset.__init__] samples head(2):\n", self.samples.head(2))
            print(
                "[WaveFakeDataset.__init__] label counts:\n",
                self.samples["label"].value_counts(),
            )

    def get_generated_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": [],
        }

        samples_list = list((self.path / self.fake_data_path).glob("*/*.wav"))
        print("[get_generated_samples] raw generated files:", len(samples_list))
        print("[get_generated_samples] first 2 raw paths:", samples_list[:2])

        samples_list = self.filter_samples_by_attack(samples_list)
        print(
            "[get_generated_samples] after filtering by attacks, files:",
            len(samples_list),
        )
        print("[get_generated_samples] first 2 filtered paths:", samples_list[:2])

        for i, sample in enumerate(samples_list):
            samples["user_id"].append(None)
            samples["sample_name"].append("_".join(sample.stem.split("_")[:-1]))
            samples["attack_type"].append(self.get_attack_from_path(sample))
            samples["label"].append("spoof")
            samples["path"].append(sample)

            if i < 2:
                print(
                    "[get_generated_samples] example spoof entry:",
                    samples["sample_name"][-1],
                    samples["attack_type"][-1],
                    samples["path"][-1],
                )

        return pd.DataFrame(samples)

    def filter_samples_by_attack(self, samples_list):
        filtered = [s for s in samples_list if self.get_attack_from_path(s) in self.allowed_attacks]
        return filtered

    def get_real_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": [],
        }

        jsut_list = list((self.path / self.jsut_real_data_path).glob("*.wav"))
        ljs_list = list((self.path / self.ljspeech_real_data_path).glob("*.wav"))
        samples_list = jsut_list + ljs_list

        print(
            "[get_real_samples] JSUT files:", len(jsut_list),
            "LJSpeech files:", len(ljs_list),
            "combined:", len(samples_list),
        )
        print("[get_real_samples] first 2 combined paths:", samples_list[:2])

        samples_list = self.split_real_samples(samples_list)
        print("[get_real_samples] after split_real_samples, files:", len(samples_list))
        print("[get_real_samples] first 2 after split:", samples_list[:2])

        for i, sample in enumerate(samples_list):
            samples["user_id"].append(None)
            samples["sample_name"].append(sample.stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")
            samples["path"].append(sample)

            if i < 2:
                print(
                    "[get_real_samples] example bonafide entry:",
                    samples["sample_name"][-1],
                    samples["path"][-1],
                )

        return pd.DataFrame(samples)

    @staticmethod
    def get_attack_from_path(path):
        folder_name = path.parents[0].relative_to(path.parents[1])
        attack = str(folder_name).split("_", maxsplit=1)[-1]
        return attack


if __name__ == "__main__":
    WAVEFAKE_DATASET_PATH = ""
    print("Dataset of WaveFake")

    for fold in [0, 1, 2]:
        print("\n" + "=" * 80)
        print(f"FOLD {fold}")
        print("=" * 80)

        for subset in ["train", "val", "test"]:
            print(f"\n[MAIN] Building dataset for fold={fold}, subset={subset}")
            ds = WaveFakeDataset(WAVEFAKE_DATASET_PATH, fold_num=fold, fold_subset=subset)
            df = ds.samples

            n_total = len(df)
            n_real = (df["label"] == "bonafide").sum()
            n_fake = (df["label"] == "spoof").sum()

            print(f"\nSubset: {subset}")
            print(f"  Total files : {n_total}")
            print(f"  Real  files : {n_real}")
            print(f"  Fake  files : {n_fake}")

            if n_total > 0:
                print("[MAIN] first 2 rows:\n", df.head(2))
