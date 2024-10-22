# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultGenericWebdataset


@dataclass
class AudioAsrSample(Sample):
    """Sample type for visual question answering."""
    mel: Optional[torch.Tensor] = None
    length: Optional[int] = None
    text: Optional[str] = None


class AudioAsrWebdataset(DefaultGenericWebdataset[AudioAsrSample]):
    __sample_type__ = AudioAsrSample


########################################################################################

