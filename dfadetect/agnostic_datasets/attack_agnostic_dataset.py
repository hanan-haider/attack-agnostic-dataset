import pandas as pd

from dfadetect.agnostic_datasets.asvspoof_dataset import ASVSpoofDataset
from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset
from dfadetect.agnostic_datasets.fakeavceleb_dataset import FakeAVCelebDataset
from dfadetect.agnostic_datasets.wavefake_dataset import WaveFakeDataset


class AttackAgnosticDataset(SimpleAudioFakeDataset):
    def __init__(
        self,
        asvspoof_path=None,
        wavefake_path=None,
        fakeavceleb_path=None,
        fold_num=0,
        fold_subset="val",
        transform=None,
        oversample=True,
        undersample=False,
        return_label=True,
        reduced_number=None,
    ):
        super().__init__(fold_num, fold_subset, transform, return_label)

        print(
            f"[AttackAgnosticDataset.__init__] fold_num={fold_num}, "
            f"fold_subset={fold_subset}, oversample={oversample}, undersample={undersample}"
        )

        datasets = []

        if asvspoof_path is not None:
            print("[AttackAgnosticDataset.__init__] loading ASVSpoofDataset from:", asvspoof_path)
            asvspoof_dataset = ASVSpoofDataset(
                asvspoof_path, fold_num=fold_num, fold_subset=fold_subset
            )
            print(
                "[AttackAgnosticDataset.__init__] ASVSpoofDataset size:",
                len(asvspoof_dataset.samples),
            )
            datasets.append(asvspoof_dataset)

        if wavefake_path is not None:
            print("[AttackAgnosticDataset.__init__] loading WaveFakeDataset from:", wavefake_path)
            wavefake_dataset = WaveFakeDataset(
                wavefake_path, fold_num=fold_num, fold_subset=fold_subset
            )
            print(
                "[AttackAgnosticDataset.__init__] WaveFakeDataset size:",
                len(wavefake_dataset.samples),
            )
            datasets.append(wavefake_dataset)

        if fakeavceleb_path is not None:
            print(
                "[AttackAgnosticDataset.__init__] loading FakeAVCelebDataset from:",
                fakeavceleb_path,
            )
            fakeavceleb_dataset = FakeAVCelebDataset(
                fakeavceleb_path, fold_num=fold_num, fold_subset=fold_subset
            )
            print(
                "[AttackAgnosticDataset.__init__] FakeAVCelebDataset size:",
                len(fakeavceleb_dataset.samples),
            )
            datasets.append(fakeavceleb_dataset)

        self.samples = pd.concat([ds.samples for ds in datasets], ignore_index=True)
        print(
            "[AttackAgnosticDataset.__init__] concatenated samples shape:",
            self.samples.shape,
        )
        if len(self.samples) > 0:
            print(
                "[AttackAgnosticDataset.__init__] head(2):\n",
                self.samples.head(2),
            )
            print(
                "[AttackAgnosticDataset.__init__] label counts:\n",
                self.samples["label"].value_counts(),
            )

        if oversample:
            print("[AttackAgnosticDataset.__init__] applying oversample_dataset()")
            self.oversample_dataset()
        elif undersample:
            print("[AttackAgnosticDataset.__init__] applying undersample_dataset()")
            self.undersample_dataset()

        if reduced_number is not None:
            print(
                "[AttackAgnosticDataset.__init__] reducing dataset to",
                reduced_number,
                "samples (with replacement)",
            )
            self.samples = self.samples.sample(
                reduced_number, replace=True, random_state=42
            )
            print(
                "[AttackAgnosticDataset.__init__] reduced samples shape:",
                self.samples.shape,
            )

    def oversample_dataset(self):
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        print(
            "[oversample_dataset] bona_length=",
            bona_length,
            "spoof_length=",
            spoof_length,
        )

        diff_length = spoof_length - bona_length
        print("[oversample_dataset] diff_length=", diff_length)

        if diff_length < 0:
            raise NotImplementedError("Oversampling where bonafide > spoof not implemented")

        if diff_length > 0:
            bonafide = samples.get_group("bonafide").sample(
                diff_length, replace=True, random_state=42
            )
            print(
                "[oversample_dataset] sampling",
                diff_length,
                "extra bonafide samples",
            )
            self.samples = pd.concat([self.samples, bonafide], ignore_index=True)
            print(
                "[oversample_dataset] new samples shape:",
                self.samples.shape,
            )
            print(
                "[oversample_dataset] new label counts:\n",
                self.samples["label"].value_counts(),
            )

    def undersample_dataset(self):
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        print(
            "[undersample_dataset] bona_length=",
            bona_length,
            "spoof_length=",
            spoof_length,
        )

        if spoof_length < bona_length:
            raise NotImplementedError("Undersampling where spoof < bonafide not implemented")

        if spoof_length > bona_length:
            spoofs = samples.get_group("spoof").sample(
                bona_length, replace=True, random_state=42
            )
            print(
                "[undersample_dataset] sampling",
                bona_length,
                "spoof samples to match bonafide",
            )
            self.samples = pd.concat(
                [samples.get_group("bonafide"), spoofs], ignore_index=True
            )
            print(
                "[undersample_dataset] new samples shape:",
                self.samples.shape,
            )
            print(
                "[undersample_dataset] new label counts:\n",
                self.samples["label"].value_counts(),
            )

    def get_bonafide_only(self):
        samples = self.samples.groupby(by=["label"])
        self.samples = samples.get_group("bonafide")
        print(
            "[get_bonafide_only] bonafide samples shape:",
            self.samples.shape,
        )
        return self.samples

    def get_spoof_only(self):
        samples = self.samples.groupby(by=["label"])
        self.samples = samples.get_group("spoof")
        print(
            "[get_spoof_only] spoof samples shape:",
            self.samples.shape,
        )
        return self.samples
