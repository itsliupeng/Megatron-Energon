# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.base_webdataset import DefaultGenericWebdataset


@dataclass
class AudioTtsSample(Sample):
    """Sample type for visual question answering."""
    text: Optional[str] = None
    codec_label: Optional[torch.Tensor] = None
    speaker: Optional[str] = None


class AudioTtsWebdataset(DefaultGenericWebdataset[AudioTtsSample]):
    __sample_type__ = AudioTtsSample


########################################################################################

