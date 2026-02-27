from pathlib import Path

import pandas as pd

from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset


ASVSPOOF_KFOLD_SPLIT = {
    0: {
        "train": ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10',
                  'A11', 'A12', 'A13', 'A14', 'A19'],
        "test": ['A05', 'A15', 'A16'],
        "val": ['A06', 'A17', 'A18'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 42
    },
    1: {
        "train": ['A01', 'A02', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10',
                  'A15', 'A16', 'A17', 'A18', 'A19'],
        "test": ['A03', 'A11', 'A12'],
        "val": ['A04', 'A13', 'A14'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 43
    },
    2: {
        "train": ['A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14',
                  'A15', 'A16', 'A17', 'A18', 'A19'],
        "test": ['A01', 'A07', 'A08'],
        "val": ['A02', 'A09', 'A10'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 44
    }
}


class ASVSpoofDataset(SimpleAudioFakeDataset):

    protocol_folder_name = "ASVspoof2019_LA_cm_protocols"
    subset_dir_prefix = "ASVspoof2019_LA_"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, fold_num=0, fold_subset="train", transform=None):
        super().__init__(fold_num, fold_subset, transform)
        self.path = path

        self.allowed_attacks = ASVSPOOF_KFOLD_SPLIT[fold_num][fold_subset]
        self.bona_partition = ASVSPOOF_KFOLD_SPLIT[fold_num]["bonafide_partition"]
        self.seed = ASVSPOOF_KFOLD_SPLIT[fold_num]["seed"]

        self.samples = pd.DataFrame()
        print(f"[ASVSpoofDataset.__init__] path={self.path}")
        print(f"[ASVSpoofDataset.__init__] fold_num={fold_num}, fold_subset={fold_subset}")
        print(f"[ASVSpoofDataset.__init__] allowed_attacks={self.allowed_attacks}")

        for subset in self.subsets:
            subset_dir = Path(self.path) / f"{self.subset_dir_prefix}{subset}"
            print(f"[ASVSpoofDataset.__init__] subset={subset}, subset_dir={subset_dir}")

            subset_protocol_path = self.get_protocol_path(subset)
            print(f"[ASVSpoofDataset.__init__] subset_protocol_path={subset_protocol_path}")

            subset_samples = self.read_protocol(subset_dir, subset_protocol_path)
            print(f"[ASVSpoofDataset.__init__] subset={subset}, rows={len(subset_samples)}")

            if len(subset_samples) > 0:
                print("[ASVSpoofDataset.__init__] subset first 2 rows:\n",
                      subset_samples.head(2))

            self.samples = pd.concat([self.samples, subset_samples], ignore_index=True)
            print(f"[ASVSpoofDataset.__init__] cumulative samples: {self.samples.shape}")

        if len(self.samples) > 0:
            print("[ASVSpoofDataset.__init__] final samples head(2):\n",
                  self.samples.head(2))
            print("[ASVSpoofDataset.__init__] attack_type unique:",
                  self.samples["attack_type"].unique()[:10])

        self.transform = transform

    def get_protocol_path(self, subset):
        paths = list((Path(self.path) / self.protocol_folder_name).glob("*.txt"))
        # debug first few protocol files
        print(f"[get_protocol_path] available protocol files ({len(paths)}):",
              [p.name for p in paths[:2]])
        for path in paths:
            if subset in Path(path).stem:
                print(f"[get_protocol_path] matched subset={subset} -> {path}")
                return path
        print(f"[get_protocol_path] WARNING: no protocol found for subset={subset}")
        return None

    def read_protocol(self, subset_dir, protocol_path):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        if protocol_path is None:
            print("[read_protocol] protocol_path is None, returning empty DataFrame")
            return pd.DataFrame(samples)

        real_samples = []
        fake_samples = []
        print(f"[read_protocol] reading protocol file: {protocol_path}")

        with open(protocol_path, "r") as file:
            for i, line in enumerate(file):
                parts = line.strip().split(" ")
                if len(parts) < 5:
                    continue
                attack_type = parts[3]

                if attack_type == "-":
                    real_samples.append(line)
                elif attack_type in self.allowed_attacks:
                    fake_samples.append(line)

                # only show first 2 protocol lines overall
                if i < 2:
                    print(f"[read_protocol] line {i}: attack_type={attack_type}")

        print(f"[read_protocol] real_samples={len(real_samples)}, fake_samples={len(fake_samples)}")

        for i, line in enumerate(fake_samples):
            samples = self.add_line_to_samples(samples, line, subset_dir)
            if i < 2:
                print("[read_protocol] fake sample added:", line.strip())

        real_samples = self.split_real_samples(real_samples)
        print(f"[read_protocol] real_samples after split for '{self.fold_subset}':",
              len(real_samples))

        for i, line in enumerate(real_samples):
            samples = self.add_line_to_samples(samples, line, subset_dir)
            if i < 2:
                print("[read_protocol] real sample added:", line.strip())

        df = pd.DataFrame(samples)
        if len(df) > 0:
            print("[read_protocol] resulting DataFrame head(2):\n", df.head(2))
        return df

    @staticmethod
    def add_line_to_samples(samples, line, subset_dir):
        user_id, sample_name, _, attack_type, label = line.strip().split(" ")
        samples["user_id"].append(user_id)
        samples["sample_name"].append(sample_name)
        samples["attack_type"].append(attack_type)
        samples["label"].append(label)

        path = subset_dir / "flac" / f"{sample_name}.flac"
        assert path.exists(), f"File does not exist: {path}"
        samples["path"].append(path)

        return samples


if __name__ == "__main__":
    ASVSPOOF_DATASET_PATH = ""
    dataset = ASVSpoofDataset(ASVSPOOF_DATASET_PATH, fold_num=1, fold_subset="test")
    print("[__main__] unique attack types:", dataset.samples["attack_type"].unique())
    print("[__main__] dataset.samples head(2):\n", dataset.samples.head(2))
    print("[__main__] dataset size:", len(dataset.samples))
