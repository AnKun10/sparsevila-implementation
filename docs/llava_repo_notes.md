# LLaVA-1.5 Repo Reconnaissance

Pinned commit: c121f0432da27facab705978f83c4ada465e46fd (tag HEAD on `main` as of clone date; matches the requested `c121f04` prefix exactly).

**Clone method:** `git submodule add` succeeded on Windows — LLaVA lives at `third_party/LLaVA` as a proper submodule tracked by `.gitmodules`.

---

## Entry points used by SparseVILA

### Model loader

```python
from llava.model.builder import load_pretrained_model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, model_base=None, model_name="llava-v1.5-7b",
    device_map="auto"
)
```

- Returns `(tokenizer, model, image_processor, context_len)`.
- `model` is `LlavaLlamaForCausalLM` (subclasses both `LlamaForCausalLM` and `LlavaMetaForCausalLM`).
- `context_len` falls back to `2048` when `max_sequence_length` is absent from config.
- `image_processor` is `CLIPImageProcessor` extracted from the loaded vision tower (`vision_tower.image_processor`).
- The function calls `vision_tower.load_model(device_map=device_map)` after loading weights, so vision tower weights are ready immediately.

### Vision tower

- Class: `CLIPVisionTower` in `llava/model/multimodal_encoder/clip_encoder.py`.
- Built via `build_vision_tower(config)` in `llava/model/multimodal_encoder/builder.py`; dispatches to `CLIPVisionTower` for `openai/clip-*` and `laion/*` identifiers, and to `CLIPVisionTowerS2` when `config.s2=True`.
- Forward output: `image_forward_outs.hidden_states[self.select_layer]` — for LLaVA-1.5 `self.select_layer = -2` (second-to-last layer), then strips the CLS token (`[:, 1:]`) when `select_feature='patch'`.
- The forward is decorated `@torch.no_grad()` and internally calls `self.vision_tower(..., output_hidden_states=True)` so **all** CLIP hidden states are computed but only layer `-2` is returned.
- **SparseVILA patch point:** override `CLIPVisionTower.forward` to also capture the raw attention weights at `salience_layer_idx` from `image_forward_outs.attentions[salience_layer_idx]` before calling `feature_select`. Salience is computed from these weights BEFORE the mm_projector.

### Projector

- `encode_images(images)` in `LlavaMetaForCausalLM` (llava_arch.py line 140–143):
  ```python
  image_features = self.get_model().get_vision_tower()(images)   # CLIPVisionTower forward
  image_features = self.get_model().mm_projector(image_features) # 2-layer MLP
  ```
- The projector is accessed via `model.get_model().mm_projector` (a `nn.Sequential` two-layer MLP with GELU, built by `build_vision_projector`).
- Shape after projector: `(N_patches, hidden_size_llm)` where `N_patches = 576` for 336×336 images with patch_size=14.

### LLM

- `LlavaLlamaForCausalLM.forward` (llava_llama.py lines 57–102): when `inputs_embeds is None`, calls `prepare_inputs_labels_for_multimodal` to interleave visual embeddings, then delegates to `LlamaForCausalLM.forward`.
- `model.model` is a `LlavaLlamaModel` (subclasses `LlamaModel`); per-layer attention lives at `model.model.layers[l].self_attn` (a `LlamaAttention` instance).
- **SparseVILA wrap point:** replace each `LlamaAttention.forward` with a wrapper that:
  1. On the first prefill of a new query, computes salience via `column_salience` and builds `packed_kv[l]` once.
  2. On subsequent decode steps, attends only into `packed_kv[l]` (extended with the latest token KV).

---

## Visual span in input_ids

- Image token sentinel: `IMAGE_TOKEN_INDEX = -200` (defined in `llava/constants.py`).
- `prepare_inputs_labels_for_multimodal` (llava_arch.py lines 145–324):
  1. Finds positions where `cur_input_ids == IMAGE_TOKEN_INDEX` using `torch.where`.
  2. Splits the token sequence around those positions, embeds the text chunks via `self.get_model().embed_tokens(...)`, and interleaves the visual embeddings from `encode_images`.
  3. Returns `(None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels)` — note `input_ids` is set to `None` in the return; the LLM receives `inputs_embeds` only.
- `get_visual_span` implementation for SparseVILA: find the `-200` marker position in `input_ids` BEFORE calling `prepare_inputs_labels_for_multimodal`, then map that scalar index to the slice `[marker_pos : marker_pos + N_patches]` in the post-interleaved embedding sequence (accounting for any prefix text tokens).

---

## Image processing

- `image_processor` is a `CLIPImageProcessor` (from `transformers`), returned by `load_pretrained_model`.
- For LLaVA-1.5 7B/13B (`image_aspect_ratio='pad'`): images are center-cropped/padded to **336×336**, output tensor shape is `(B, 3, 336, 336)`.
- For LLaVA-1.5-HD / anyres variants (`image_aspect_ratio='anyres'`): images are tiled; `mm_patch_merge_type='spatial_unpad'` is used and `prepare_inputs_labels_for_multimodal` handles the tile merging and unpadding.
- The `image_processor` is also available directly at `model.get_vision_tower().image_processor` after the model is loaded.

---

## Key constants (llava/constants.py)

```python
IGNORE_INDEX = -100       # used in labels for masked positions
IMAGE_TOKEN_INDEX = -200  # sentinel in input_ids for image slot
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
```

---

## Notes on patching strategy

1. **Vision tower hook:** Since `CLIPVisionTower.forward` is `@torch.no_grad()`, re-enable grad only for the salience-layer output if needed, or use `register_forward_hook` on the target CLIP encoder layer to intercept attention weights.
2. **Projector location:** `model.get_model().mm_projector` — pruning visual tokens should happen BEFORE this projector to save both projector and LLM flops.
3. **Attention access:** `LlamaAttention` in HF transformers returns attention weights only when `output_attentions=True`; for LLaVA decode steps we instead use our own `column_salience` kernel on stored KV.
4. **`input_ids` is None post-interleaving:** SparseVILA must intercept at the `inputs_embeds` level, not `input_ids`, during the LLM forward pass.
