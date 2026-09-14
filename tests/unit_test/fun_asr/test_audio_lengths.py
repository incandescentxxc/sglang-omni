# SPDX-License-Identifier: Apache-2.0

import torch

from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoProcessor


def test_processor_uses_compressed_audio_token_lengths():
    frames = torch.tensor([0, 1, 7, 8, 9, 16, 17, 499], dtype=torch.long)
    original = frames.clone()
    processor = FunAsrNanoProcessor.__new__(FunAsrNanoProcessor)
    lengths = processor._get_feat_extract_output_lengths(frames)
    torch.testing.assert_close(lengths, torch.tensor([0, 1, 1, 1, 2, 2, 3, 63]))
    torch.testing.assert_close(frames, original)
