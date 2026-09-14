# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


def fun_asr_low_frame_rate_length(lfr_frames):
    """Historical audio-token count, ceil(T_lfr / 8), for scalars or tensors.

    This is an embedding-prefix truncation policy, not additional frontend or
    adaptor downsampling. Preserve it for transcription quality: the full-frame
    policy regressed matched-cohort long-form WER in the PR #2121 ablation.
    """
    out = lfr_frames
    for _ in range(3):
        out = (out + 1) // 2
    return out
