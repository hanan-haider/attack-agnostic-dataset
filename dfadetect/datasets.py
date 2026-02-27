"""Common preprocessing functions for audio data."""
import functools
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torchaudio
# from torchaudio.functional import apply_codec

from dfadetect.lfcc import LFCC
from dfadetect.utils import find_wav_files

LOGGER = logging.getLogger(__name__)

SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]


class AudioDataset(torch.utils.data.Dataset):
    """Torch dataset to load data from a provided directory."""

    def __init__(
        self,
        directory_or_path_list: Union[Union[str, Path], List[Union[str, Path]]],
        sample_rate: int = 16_000,
        amount: Optional[int] = None,
        normalize: bool = True,
        trim: bool = True,
        phone_call: bool = False,
    ) -> None:
        super().__init__()

        self.trim = trim
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.phone_call = phone_call

        if isinstance(directory_or_path_list, list):
            paths = directory_or_path_list
        elif isinstance(directory_or_path_list, Path) or isinstance(
            directory_or_path_list, str
        ):
            directory = Path(directory_or_path_list)
            if not directory.exists():
                raise IOError(f"Directory does ot exists: {directory}")

            paths = find_wav_files(directory)
            if paths is None:
                raise IOError(f"Directory did not contain wav files: {directory}")
        else:
            raise TypeError(
                f"Supplied unsupported type for argument directory_or_path_list {type(directory_or_path_list)}!"
            )

        if amount is not None:
            paths = paths[:amount]

        self._paths = paths
        print(f"[AudioDataset.__init__] total files: {len(self._paths)}")
        print("[AudioDataset.__init__] first 2 paths:", self._paths[:2])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self._paths[index]
        if index < 2:
            print(f"[AudioDataset.__getitem__] loading index {index}: {path}")

        waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)

        if sample_rate != self.sample_rate:
            if index < 2:
                print(
                    f"[AudioDataset.__getitem__] resampling {sample_rate} -> {self.sample_rate}"
                )
            waveform, sample_rate = self.resample(
                path, self.sample_rate, self.normalize
            )

        if self.trim:
            waveform, sample_rate = self.apply_trim(waveform, sample_rate)
            if index < 2:
                print(
                    f"[AudioDataset.__getitem__] after trim shape: {tuple(waveform.shape)}"
                )

        if self.phone_call:
            waveform, sample_rate = self.process_phone_call(waveform, sample_rate)
            if index < 2:
                print(
                    f"[AudioDataset.__getitem__] after phone_call shape: {tuple(waveform.shape)}"
                )

        return waveform, sample_rate

    @staticmethod
    def apply_trim(waveform, sample_rate):
        waveform_trimmed, sample_rate_trimmed = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, SOX_SILENCE
        )

        if waveform_trimmed.size()[1] > 0:
            waveform = waveform_trimmed
            sample_rate = sample_rate_trimmed

        return waveform, sample_rate

    @staticmethod
    def resample(path, target_sample_rate, normalize=True):
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
            path, [["rate", f"{target_sample_rate}"]], normalize=normalize
        )
        return waveform, sample_rate

    @staticmethod
    def process_phone_call(waveform, sample_rate):
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            sample_rate,
            effects=[
                ["lowpass", "4000"],
                [
                    "compand",
                    "0.02,0.05",
                    "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
                    "-8",
                    "-7",
                    "0.05",
                ],
                ["rate", "8000"],
            ],
        )
        # waveform = apply_codec(waveform, sample_rate, format="gsm")
        return waveform, sample_rate

    def __len__(self) -> int:
        length = len(self._paths)
        print(f"[AudioDataset.__len__] length: {length}")
        return length


class PadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, cut: int = 64600, label=None):
        self.dataset = dataset
        self.cut = cut
        self.label = label
        print(
            f"[PadDataset.__init__] cut={self.cut}, label={self.label}, base_len={len(self.dataset)}"
        )

    def __getitem__(self, index):
        waveform, sample_rate = self.dataset[index]
        if index < 2:
            print(f"[PadDataset.__getitem__] before pad shape: {tuple(waveform.shape)}")
        waveform = self.apply_pad(waveform, self.cut)
        if index < 2:
            print(f"[PadDataset.__getitem__] after pad shape: {tuple(waveform.shape)}")

        if self.label is None:
            return waveform, sample_rate
        else:
            return waveform, sample_rate, self.label

    @staticmethod
    def apply_pad(waveform, cut):
        waveform = waveform.squeeze(0)
        waveform_len = waveform.shape[0]

        if waveform_len >= cut:
            return waveform[:cut]

        num_repeats = int(cut / waveform_len) + 1
        padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
        return padded_waveform

    def __len__(self):
        length = len(self.dataset)
        print(f"[PadDataset.__len__] length: {length}")
        return length


class TransformDataset(torch.utils.data.Dataset):
    """A generic transformation dataset."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transformation: Callable,
        needs_sample_rate: bool = False,
        transform_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self._dataset = dataset

        self._transform_constructor = transformation
        self._needs_sample_rate = needs_sample_rate
        self._transform_kwargs = transform_kwargs

        self._transform = None
        print(
            f"[TransformDataset.__init__] needs_sample_rate={self._needs_sample_rate}, kwargs={self._transform_kwargs}"
        )

    def __len__(self):
        length = len(self._dataset)
        print(f"[TransformDataset.__len__] length: {length}")
        return length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = self._dataset[index]

        if self._transform is None:
            if self._needs_sample_rate:
                self._transform = self._transform_constructor(
                    sample_rate, **self._transform_kwargs
                )
            else:
                self._transform = self._transform_constructor(
                    **self._transform_kwargs
                )
            print(
                "[TransformDataset.__getitem__] created transform with kwargs:",
                self._transform_kwargs,
            )

        if index < 2:
            print(
                f"[TransformDataset.__getitem__] applying transform at index {index}, input shape={tuple(waveform.shape)}"
            )

        return self._transform(waveform), sample_rate


class DoubleDeltaTransform(torch.nn.Module):
    """Compute delta and double delta features."""

    def __init__(self, win_length: int = 5, mode: str = "replicate") -> None:
        super().__init__()
        self.win_length = win_length
        self.mode = mode

        self._delta = torchaudio.transforms.ComputeDeltas(
            win_length=self.win_length, mode=self.mode
        )
        print(
            f"[DoubleDeltaTransform.__init__] win_length={self.win_length}, mode={self.mode}"
        )

    def forward(self, X):
        delta = self._delta(X)
        double_delta = self._delta(delta)
        if X.shape[-1] > 1:
            print(
                "[DoubleDeltaTransform.forward] input shape:",
                tuple(X.shape),
                "output shape:",
                (X.shape[0] * 3, X.shape[1]),
            )
        return torch.hstack((X, delta, double_delta))


# =====================================================================
# Helper functions.
# =====================================================================


def _build_preprocessing(
    directory_or_audiodataset: Union[Union[str, Path], AudioDataset],
    transform: torch.nn.Module,
    audiokwargs: dict = {},
    transformkwargs: dict = {},
) -> TransformDataset:
    import dfadetect.agnostic_datasets.attack_agnostic_dataset as aa_ds

    """Generic function template for building preprocessing functions."""
    if isinstance(directory_or_audiodataset, AudioDataset) or isinstance(
        directory_or_audiodataset, PadDataset
    ) or isinstance(directory_or_audiodataset, aa_ds.AttackAgnosticDataset):
        print("[_build_preprocessing] using existing dataset instance")
        return TransformDataset(
            dataset=directory_or_audiodataset,
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
        )
    elif isinstance(directory_or_audiodataset, str) or isinstance(
        directory_or_audiodataset, Path
    ):
        print("[_build_preprocessing] creating AudioDataset from path")
        ds = AudioDataset(directory_or_path_list=directory_or_audiodataset, **audiokwargs)
        return TransformDataset(
            dataset=ds,
            transformation=transform,
            needs_sample_rate=True,
            transform_kwargs=transformkwargs,
        )
    else:
        raise TypeError("Unsupported type for directory_or_audiodataset!")


mfcc = functools.partial(_build_preprocessing, transform=torchaudio.transforms.MFCC)
lfcc = functools.partial(_build_preprocessing, transform=LFCC)


def double_delta(dataset: torch.utils.data.Dataset, delta_kwargs: dict = {}) -> TransformDataset:
    print("[double_delta] adding DoubleDeltaTransform with kwargs:", delta_kwargs)
    return TransformDataset(
        dataset=dataset, transformation=DoubleDeltaTransform, transform_kwargs=delta_kwargs
    )


def load_directory_split_train_test(
    path: Union[Path, str],
    feature_fn: Callable,
    feature_kwargs: dict,
    test_size: float,
    use_double_delta: bool = True,
    phone_call: bool = False,
    pad: bool = False,
    label: Optional[int] = None,
    amount_to_use: Optional[int] = None,
) -> Tuple[TransformDataset, TransformDataset]:
    """Load all wav files from directory, apply the feature transformation and split into test/train."""

    paths = find_wav_files(path)
    if paths is None:
        raise IOError(f"Could not load files from {path}!")

    if amount_to_use is not None:
        paths = paths[:amount_to_use]

    print(f"[load_directory_split_train_test] total files found: {len(paths)}")
    print("[load_directory_split_train_test] first 2 paths:", paths[:2])

    test_size_n = int(test_size * len(paths))
    train_paths = paths[:-test_size_n]
    test_paths = paths[-test_size_n:]

    LOGGER.info(f"Loading data from {path}...!")

    train_dataset = AudioDataset(train_paths, phone_call=phone_call)
    if pad:
        train_dataset = PadDataset(train_dataset, label=label)

    test_dataset = AudioDataset(test_paths, phone_call=phone_call)
    if pad:
        test_dataset = PadDataset(test_dataset, label=label)

    if feature_fn is None:
        return train_dataset, test_dataset

    dataset_train = feature_fn(
        directory_or_audiodataset=train_dataset, transformkwargs=feature_kwargs
    )
    dataset_test = feature_fn(
        directory_or_audiodataset=test_dataset, transformkwargs=feature_kwargs
    )

    if use_double_delta:
        dataset_train = double_delta(dataset_train)
        dataset_test = double_delta(dataset_test)

    return dataset_train, dataset_test


def apply_feature_and_double_delta(
    datasets: List[torch.utils.data.Dataset],
    feature_fn: Callable,
    feature_kwargs: dict,
    use_double_delta: bool = True,
):

    datasets_list = []
    for i, ds in enumerate(datasets):
        print(f"[apply_feature_and_double_delta] processing dataset {i}")
        ds = feature_fn(directory_or_audiodataset=ds, transformkwargs=feature_kwargs)
        if use_double_delta:
            ds = double_delta(ds)
        datasets_list.append(ds)

    return datasets_list
