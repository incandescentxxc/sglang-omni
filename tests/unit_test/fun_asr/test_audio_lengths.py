# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoProcessor
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)


@pytest.mark.parametrize(
    "frames, expected",
    [(0, 0), (1, 1), (7, 1), (8, 1), (9, 2), (16, 2), (17, 3), (499, 63)],
)
def test_historical_audio_token_length(frames, expected):
    assert fun_asr_low_frame_rate_length(frames) == expected


def test_processor_uses_historical_audio_token_lengths():
    frames = torch.tensor([0, 1, 7, 8, 9, 16, 17, 499], dtype=torch.long)
    original = frames.clone()
    processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
    lengths = processor._get_feat_extract_output_lengths(frames)
    torch.testing.assert_close(lengths, torch.tensor([0, 1, 1, 1, 2, 2, 3, 63]))
    torch.testing.assert_close(frames, original)
