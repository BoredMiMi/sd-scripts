# Training Flowchart for sd-scripts

This is a beginner-friendly overview of how data moves through the scripts.

## 1. Tagging & Captioning
- `finetune/tag_images_by_wd14_tagger.py`
- `finetune/make_captions.py` or `finetune/make_captions_by_git.py`
- `finetune/merge_captions_to_metadata.py` or `finetune/merge_dd_tags_to_metadata.py`
- `finetune/clean_captions_and_tags.py`

## 2. Preparing Latents
- `finetune/prepare_buckets_latents.py`
- `tools/cache_latents.py`
- `tools/cache_text_encoder_outputs.py`
- `tools/resize_images_to_resolution.py`

## 3. Training
Scripts for training a model:
- `train_network.py`, `fine_tune.py`, `train_db.py`, `train_controlnet.py`
- `train_textual_inversion.py`, `train_textual_inversion_XTI.py`
- `sdxl_train.py`, `sdxl_train_network.py`, `sdxl_train_control_net_lllite.py`
- `sdxl_train_textual_inversion.py`

These call helpers in `library/` such as `train_util.py` and `sdxl_train_util.py`.

## 4. Merging Models
- `networks/merge_lora.py`, `networks/sdxl_merge_lora.py`, `networks/merge_lora_old.py`
- `networks/resize_lora.py`, `networks/svd_merge_lora.py`
- `tools/merge_models.py`

## 5. Inference (Generating Images)
- `gen_img.py`, `gen_img_diffusers.py`, `sdxl_gen_img.py`, `sdxl_minimal_inference.py`

---

## Simple Flow

```text
Images
  ↓
Tagging & Captions
  ↓
Preparing Latents
  ↓
Training
  ↓
Merging Models
  ↓
Inference (image generation)
```
