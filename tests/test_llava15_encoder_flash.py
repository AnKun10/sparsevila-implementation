"""Flash-kernel encoder salience path: equivalence vs attn-map path.

The encoder wrapper offers two paths:

* attn-map (use_flash_kernel=False or non-"cls" strategy): reads
  ``output_attentions`` from CLIPVisionModel.
* flash kernel ("cls" strategy + use_flash_kernel=True): captures the
  selected layer's pre-self_attn hidden_states via a forward-pre-hook,
  recomputes Q/K of that single layer, and calls
  ``column_salience(M=1, ...)``.

This test fixture builds a fake HF CLIPVisionModel that supports BOTH
paths (returns attentions when requested, exposes a
``vision_model.encoder.layers`` chain with the salience layer's self_attn
producing identical math). The two paths must produce the same salience.
"""
from types import SimpleNamespace
import torch
from torch import nn

from sparsevila.config import SparseVILAConfig
from sparsevila.models.llava_15 import (
    LlavaFifteenAdapter, EncoderSalienceOutput,
)


class _MiniCLIPSelfAttn(nn.Module):
    """Minimum surface of CLIPSdpaAttention used by the flash path."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _MiniCLIPLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.self_attn = _MiniCLIPSelfAttn(hidden_size, num_heads)


class _MiniCLIPEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            _MiniCLIPLayer(hidden_size, num_heads) for _ in range(n_layers)
        ])


class _MiniCLIPVisionModel(nn.Module):
    """A fake CLIPVisionModel that supports both output_attentions and
    the layer-internal Q/K extraction path."""

    def __init__(self, num_patches=4, hidden_size=8, num_heads=2, n_layers=2):
        super().__init__()
        self.num_patches = num_patches
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.weight = nn.Parameter(torch.zeros(1))    # for dtype/device
        # vision_model.encoder.layers[i].self_attn structure (matches HF).
        self.vision_model = SimpleNamespace(
            encoder=_MiniCLIPEncoder(hidden_size, num_heads, n_layers),
        )

    def forward(self, images, output_hidden_states=False, output_attentions=False):
        torch.manual_seed(0)
        b = images.shape[0]
        S1 = self.num_patches + 1
        # Pretend each "layer" outputs identical hidden states (no actual
        # attention iteration; we only care that the SAL_LAYER's input
        # equals what we'd compute Q/K from).
        h_layer0_in = torch.randn(b, S1, self.hidden_size)
        h_layer1_in = torch.randn(b, S1, self.hidden_size)
        # The hidden_states list: [embeddings, after_layer_0, after_layer_1, ...]
        # For our test the SAL_LAYER input is what we set up to compute Q/K from
        # on layer index -1 (== layers[1] in 2-layer model).
        # Simulate by storing the layer's input on the layer for the pre-hook
        # to pick up; but the wrapper relies on the pre-hook firing during
        # forward, so we need to actually invoke self_attn here.
        sal_layer = self.vision_model.encoder.layers[-1]
        # Compute attentions for the last layer using THIS hidden_state input,
        # so both paths see the same Q, K.
        h_sal_in = h_layer1_in
        # Run the same q_proj/k_proj as the wrapper would.
        H = self.num_heads
        D = self.hidden_size // H
        q = sal_layer.self_attn.q_proj(h_sal_in).view(b, S1, H, D).transpose(1, 2)
        k = sal_layer.self_attn.k_proj(h_sal_in).view(b, S1, H, D).transpose(1, 2)
        scale = 1.0 / (D ** 0.5)
        scores = q @ k.transpose(-2, -1) * scale
        attn_last = torch.softmax(scores, dim=-1)             # (b, H, S+1, S+1)

        # Earlier layers' attentions: arbitrary; only used for non-flash path
        # at salience_layer_idx if user picked something other than -1.
        attn_first = torch.zeros_like(attn_last)

        # Invoke sal_layer.self_attn so the wrapper's forward-pre-hook fires
        # and captures h_sal_in.
        sal_layer.self_attn(h_sal_in)

        # Build hidden_states tuple: index 0 = embeddings, 1..n = post-layer.
        hidden_states_tuple = (h_layer0_in, h_layer1_in, h_layer1_in)

        return SimpleNamespace(
            hidden_states=hidden_states_tuple if output_hidden_states else None,
            attentions=(attn_first, attn_last) if output_attentions else None,
        )


class _FakeCLIPTower(nn.Module):
    def __init__(self, num_patches=4, hidden_size=8):
        super().__init__()
        self.vision_tower = _MiniCLIPVisionModel(num_patches, hidden_size)
        self.select_layer = -2
        self.select_feature = "patch"

    @property
    def dtype(self):
        return self.vision_tower.weight.dtype

    @property
    def device(self):
        return self.vision_tower.weight.device


def _add_self_attn_passthrough(tower):
    """Make sal_layer.self_attn callable as a simple no-op for our fake
    (the real CLIPSdpaAttention.__call__ returns a tuple)."""
    sal_layer = tower.vision_tower.vision_model.encoder.layers[-1]
    sal_layer.self_attn.forward = lambda h: (h, None)


def test_flash_and_attn_map_paths_pick_same_kept_indices():
    """Both salience paths should rank patches the same way."""
    tower = _FakeCLIPTower(num_patches=8, hidden_size=8)
    _add_self_attn_passthrough(tower)
    adapter = LlavaFifteenAdapter()

    # Run flash path
    cfg_flash = SparseVILAConfig(
        encoder_prune_ratio=0.5, use_flash_kernel=True,
        salience_strategy="cls", salience_layer_idx=-1,
    )
    w_flash = adapter.wrap_encoder(tower, cfg_flash)
    w_flash.compat_mode = False
    out_flash = w_flash(torch.randn(1, 3, 16, 16))

    # Run attn-map path with the same fake (deterministic seed).
    cfg_attn = SparseVILAConfig(
        encoder_prune_ratio=0.5, use_flash_kernel=False,
        salience_strategy="cls", salience_layer_idx=-1,
    )
    tower2 = _FakeCLIPTower(num_patches=8, hidden_size=8)
    _add_self_attn_passthrough(tower2)
    # Sync weights so q_proj/k_proj produce the same outputs.
    tower2.vision_tower.vision_model.encoder.layers[-1].self_attn.load_state_dict(
        tower.vision_tower.vision_model.encoder.layers[-1].self_attn.state_dict()
    )
    w_attn = adapter.wrap_encoder(tower2, cfg_attn)
    w_attn.compat_mode = False
    out_attn = w_attn(torch.randn(1, 3, 16, 16))

    # Salience tensors should be numerically equal (up to FP rounding).
    torch.testing.assert_close(out_flash.salience, out_attn.salience,
                                atol=1e-5, rtol=1e-4)
    # And kept indices should be identical.
    assert torch.equal(out_flash.kept_idx, out_attn.kept_idx)
