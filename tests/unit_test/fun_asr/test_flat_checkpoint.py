# SPDX-License-Identifier: Apache-2.0
"""Check real loader destinations, format guards, and flat-encoder execution."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoConfig
from sglang_omni.models.fun_asr.sglang_model import (
    FunAsrNanoAdaptor,
    FunAsrNanoAudioEncoder,
    FunAsrNanoForConditionalGeneration,
)


@pytest.mark.parametrize(
    "config",
    [
        {"encoder_config": {}},
        {
            "audio_config": {
                "num_hidden_layers": 50,
                "num_timestamp_prediction_blocks": 20,
            }
        },
    ],
)
def test_legacy_config_rejected(config):
    with pytest.raises(ValueError, match="only the current flat HF checkpoint"):
        FunAsrNanoConfig(**config)


def test_nested_config_counts_and_roundtrip():
    audio = dict(
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_hidden_layers=3,
        num_timestamp_prediction_layers=1,
        num_mel_bins=2,
        num_stacked_frames=3,
    )
    config = FunAsrNanoConfig(
        audio_config=audio,
        adaptor_config=dict(
            hidden_size=8,
            intermediate_size=5,
            projector_hidden_size=12,
            num_attention_heads=2,
        ),
        text_config=dict(hidden_size=8, num_attention_heads=2),
    )
    assert config.audio_config.num_hidden_layers == 3
    assert config.audio_config.num_timestamp_prediction_layers == 1
    assert config.audio_config.input_size == 6
    assert config.adaptor_config.intermediate_size == 5
    assert config.adaptor_config.projector_hidden_size == 12
    restored = FunAsrNanoConfig.from_dict(config.to_dict())
    assert restored.audio_config.to_dict() == config.audio_config.to_dict()
    assert restored.adaptor_config.to_dict() == config.adaptor_config.to_dict()
    assert "encoder_config" not in restored.to_dict()


@pytest.mark.parametrize("total,timestamp", [(0, 0), (2, -1), (2, 2), (2, 3)])
def test_invalid_layer_counts_rejected(total, timestamp):
    with pytest.raises(ValueError, match="layer counts"):
        FunAsrNanoConfig(
            audio_config={
                "num_hidden_layers": total,
                "num_timestamp_prediction_layers": timestamp,
            }
        )


def test_adaptor_text_width_mismatch_rejected():
    with pytest.raises(ValueError, match="hidden size must match"):
        FunAsrNanoConfig(
            adaptor_config={"hidden_size": 8}, text_config={"hidden_size": 16}
        )


def test_config_save_and_reload(tmp_path):
    config = FunAsrNanoConfig()
    config.save_pretrained(tmp_path)
    restored = FunAsrNanoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert restored.audio_config.num_hidden_layers == 70
    assert restored.audio_config.num_timestamp_prediction_layers == 20
    assert restored.adaptor_config.hidden_size == restored.text_config.hidden_size


def test_model_builds_flat_layers_from_nested_config(monkeypatch):
    import sglang_omni.models.fun_asr.sglang_model as model_module

    monkeypatch.setattr(
        model_module, "Qwen3ForCausalLM", lambda *args, **kwargs: nn.Identity()
    )
    config = FunAsrNanoConfig(
        audio_config={
            "num_mel_bins": 2,
            "num_stacked_frames": 3,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 3,
            "num_timestamp_prediction_layers": 1,
            "layer_norm_eps": 1e-4,
        },
        adaptor_config={
            "hidden_size": 8,
            "intermediate_size": 5,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "projector_hidden_size": 12,
            "layer_norm_eps": 1e-3,
        },
        text_config={"hidden_size": 8, "num_attention_heads": 2},
    )
    model = FunAsrNanoForConditionalGeneration(config)
    assert len(model.audio_tower.layers) == 3
    assert model.audio_tower.layers[0].self_attn.q_proj.in_features == 6
    assert isinstance(model.audio_tower.layers[0].final_layernorm, nn.Identity)
    assert model.audio_tower.layers[1].final_layernorm.eps == 1e-4
    assert model.audio_tower.layers[2].final_layernorm.eps == 1e-4
    assert model.multi_modal_projector.linear_1.out_features == 12
    assert model.multi_modal_projector.layers[0].mlp.fc1.out_features == 5
    assert model.multi_modal_projector.layers[0].input_layernorm.eps == 1e-3


def tiny_model():
    model = FunAsrNanoForConditionalGeneration.__new__(
        FunAsrNanoForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(tie_word_embeddings=False),
    )
    model.audio_tower = FunAsrNanoAudioEncoder(
        input_size=6,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=3,
        dropout_rate=0,
        attention_dropout_rate=0,
        activation_dropout_rate=0,
    )
    model.multi_modal_projector = FunAsrNanoAdaptor(
        encoder_dim=8, llm_dim=8, ffn_dim=12, num_layers=1, attention_heads=2
    )
    return model.eval()


def test_full_audio_load_and_output():
    source, target = tiny_model(), tiny_model()
    weights = [
        ("model." + name, tensor.clone())
        for name, tensor in source.state_dict().items()
    ]
    target.load_weights(reversed(weights))
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], tensor, rtol=0, atol=0)
    x = torch.randn(2, 7, 6)
    mask = torch.tensor([[[1] * 7], [[1] * 4 + [0] * 3]], dtype=x.dtype)
    with torch.no_grad():
        torch.testing.assert_close(
            target.audio_tower(x, mask), source.audio_tower(x, mask), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "shape", "unknown_bias", "legacy_adaptor"]
)
def test_loader_rejects_incomplete_or_ambiguous_weights(failure):
    model = tiny_model()
    weights = [
        ("model." + name, tensor.clone()) for name, tensor in model.state_dict().items()
    ]
    if failure == "missing":
        weights.pop()
    elif failure == "duplicate":
        weights.append(weights[0])
    elif failure == "shape":
        weights[0] = (weights[0][0], torch.zeros(1))
    elif failure == "unknown_bias":
        weights.append(("model.audio_tower.missing.bias", torch.zeros(1)))
    else:
        weights.append(("model.audio_adaptor.blocks.0.fc1.weight", torch.zeros(1)))
    with pytest.raises(ValueError):
        model.load_weights(weights)


def test_flat_model_preserves_historical_audio_token_count():
    model = tiny_model()
    item = SimpleNamespace(
        feature=torch.randn(1, 6, 17), feature_attention_mask=torch.ones(1, 17)
    )
    assert model.get_audio_feature([item]).shape == (3, 8)


def test_feature_extractor_reads_hf_lfr_fields_without_model_config(tmp_path):
    import json

    from sglang_omni.models.fun_asr.configuration_fun_asr import (
        FunAsrNanoFeatureExtractor,
    )

    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor": {
                    "feature_extractor_type": "FunAsrNanoFeatureExtractor",
                    "feature_size": 40,
                    "num_frames_lfr": 3,
                    "stride_lfr": 2,
                }
            }
        )
    )
    extractor = FunAsrNanoFeatureExtractor.from_pretrained(
        tmp_path, local_files_only=True
    )
    assert extractor.num_frames_lfr == 3
    assert extractor.stride_lfr == 2


def test_encoder_and_projector_match_native_hf():
    # Optional reference check against the current upstream architecture.
    reference = pytest.importorskip(
        "transformers.models.fun_asr_nano.modeling_fun_asr_nano"
    )
    configs = pytest.importorskip(
        "transformers.models.fun_asr_nano.configuration_fun_asr_nano"
    )
    torch.manual_seed(9)
    model = tiny_model().to(dtype=torch.float32)
    audio_config = configs.FunAsrNanoEncoderConfig(
        num_mel_bins=2,
        num_stacked_frames=3,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=3,
        num_timestamp_prediction_layers=1,
        fsmn_kernel_size=3,
    )
    audio_config._attn_implementation = "eager"
    encoder = reference.FunAsrNanoEncoder(audio_config).eval()
    encoder.load_state_dict(
        model.audio_tower.state_dict(),
        strict=True,
    )
    adaptor_config = configs.FunAsrNanoAdaptorConfig(
        hidden_size=8,
        intermediate_size=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        projector_hidden_size=12,
    )
    adaptor_config._attn_implementation = "eager"
    projector = reference.FunAsrNanoMultiModalProjector(
        SimpleNamespace(audio_config=audio_config, adaptor_config=adaptor_config)
    ).eval()
    projector.load_state_dict(
        model.multi_modal_projector.state_dict(),
        strict=True,
    )
    x = torch.randn(2, 7, 6)
    mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3])
    with torch.no_grad():
        ours = model.audio_tower(x, mask[:, None, :])
        theirs = encoder(x, mask).last_hidden_state
        valid = mask.bool()
        tolerance = 2e-5
        torch.testing.assert_close(
            ours[valid], theirs[valid], atol=tolerance, rtol=tolerance
        )
        torch.testing.assert_close(
            model.multi_modal_projector(ours, mask[:, None, :])[valid],
            projector(theirs, mask)[valid],
            atol=tolerance,
            rtol=tolerance,
        )
