# Glossary

This glossary explains the main parts of the project in very simple language. Each entry has a tag in brackets:
- **[Main]** means you normally run this file directly.
- **[Helper]** means this file is mainly used by other scripts.

*Training* means teaching the computer to make new pictures by showing it many examples.
*Inference* means using the computer to make new pictures after it has learned.

## Top-level files

- **README.md** [Helper] - The main guide for the project.
- **README-ja.md** [Helper] - A Japanese version of the guide.
- **LICENSE.md** [Helper] - The rules for using the code.
- **requirements.txt** [Helper] - A list of other code this project needs.
- **setup.py** [Helper] - Helps install this project as a package.
- **fine_tune.py** [Main] - Trains models with your own images.
- **gen_img.py** [Main] - Makes images with a trained model.
- **gen_img_diffusers.py** [Main] - Similar to gen_img.py but uses the Diffusers library.
- **sdxl_gen_img.py** [Main] - Generates images with SDXL models, including LoRA and ControlNet.
- **sdxl_minimal_inference.py** [Main] - A tiny example of SDXL image generation.
- **sdxl_train.py** [Main] - Fine-tunes SDXL models. It can also use DreamBooth data and lets you set different learning rates for each U‑Net block.
- **sdxl_train_control_net_lllite.py** [Main] - Trains SDXL with the lightweight ControlNet‑LLLite method by passing extra images into the U‑Net.
- **sdxl_train_control_net_lllite_old.py** [Main] - Older version kept for reference.
- **sdxl_train_network.py** [Main] - Trains LoRA networks for SDXL. Works like train_network.py but understands the SDXL model structure.
- **sdxl_train_textual_inversion.py** [Main] - Learns new word embeddings for SDXL using textual inversion. It can make captions from templates.
- **train_controlnet.py** [Main] - Trains ControlNet models for earlier Stable Diffusion versions.
- **train_db.py** [Main] - DreamBooth training from a small set of images.
- **train_network.py** [Main] - Trains LoRA networks for non‑SDXL models.
- **train_textual_inversion.py** [Main] - Learns new words for non‑SDXL models.
- **train_textual_inversion_XTI.py** [Main] - Textual inversion using the XTI method.
- **XTI_hijack.py** [Helper] - Extra functions used by the XTI scripts.
- **_typos.toml** [Helper] - Settings for the typo checker.
- **.gitignore** [Helper] - Lists files Git should skip.

## Folders

### .github [Helper]
- **FUNDING.yml** - Link for project donations.
- **dependabot.yml** - Settings for automatic dependency checks.
- **workflows/typos.yml** - GitHub action for spelling checks.

### bitsandbytes_windows [Helper]
- **cextension.py** - Loader script for bitsandbytes on Windows.
- **libbitsandbytes_cpu.dll** - CPU support library.
- **libbitsandbytes_cuda116.dll** - GPU library for CUDA 11.6.
- **libbitsandbytes_cuda118.dll** - GPU library for CUDA 11.8.
- **main.py** - Helper code for building bitsandbytes binaries.

### docs [Helper]
- **config_README-en.md / config_README-ja.md** - Explain how to use config files.
- **fine_tune_README_ja.md** - Guide for fine_tune.py (Japanese).
- **gen_img_README-ja.md** - Guide for image generation (Japanese).
- **masked_loss_README.md / masked_loss_README-ja.md** - Explain masked loss training.
- **train_db_README-ja.md / train_db_README-zh.md** - DreamBooth guides.
- **train_lllite_README.md / train_lllite_README-ja.md** - Details about ControlNet‑LLLite.
- **train_network_README-ja.md / train_network_README-zh.md** - Guides for LoRA training.
- **train_README-ja.md / train_README-zh.md** - General training guides.
- **train_SDXL-en.md** - Notes about SDXL training scripts.
- **train_ti_README-ja.md** - Textual inversion guide (Japanese).
- **wd14_tagger_README-en.md / wd14_tagger_README-ja.md** - How to tag images using WD14.
- **GLOSSARY.md** - This glossary.

### finetune [Helper]
- **clean_captions_and_tags.py** [Main] - Cleans up auto‑generated captions.
- **hypernetwork_nai.py** [Main] - Special training for NovelAI hypernetworks.
- **make_captions.py** [Main] - Uses BLIP to create captions for your images.
- **make_captions_by_git.py** [Main] - Makes captions by reading git commit messages.
- **merge_captions_to_metadata.py** [Main] - Saves captions into metadata files.
- **merge_dd_tags_to_metadata.py** [Main] - Adds DeepDanbooru tags to metadata.
- **prepare_buckets_latents.py** [Main] - Precomputes SD or SDXL latents for bucket training.
- **tag_images_by_wd14_tagger.py** [Main] - Adds tags to images using the WD14 tagger.
- **blip/**
  - **blip.py / med.py / vit.py / med_config.json** - Code used by make_captions.py for generating captions.

### library [Helper]
- **adafactor_fused.py** - Optimizer that saves memory.
- **attention_processors.py** - Extra attention modules for training.
- **config_util.py** - Loads and checks config files.
- **custom_train_functions.py** - Optional training tweaks like SNR weighting.
- **deepspeed_utils.py** - Helpers for training with DeepSpeed.
- **device_utils.py** - Sets up the hardware (CPU/GPU).
- **huggingface_util.py** - Utilities for HuggingFace models.
- **hypernetwork.py** - Hypernetwork model code.
- **lpw_stable_diffusion.py** - Adds long prompt support.
- **model_util.py** - Functions to load and save models safely.
- **original_unet.py** - Re‑implementation of the original U‑Net.
- **sai_model_spec.py** - Model specifications for SAI format.
- **sdxl_lpw_stable_diffusion.py** - Long prompt support for SDXL.
- **sdxl_model_util.py** - Helpers specific to SDXL models.
- **sdxl_original_unet.py** - U‑Net code for SDXL.
- **sdxl_train_util.py** - Utility functions shared by SDXL training scripts.
- **slicing_vae.py** - VAE code that can process images in slices to save memory.
- **train_util.py** - Common training functions.
- **utils.py** - General helper functions.
- **__init__.py** - Marks this folder as a Python package.
- **ipex/**
  - **attention.py / diffusers.py / gradscaler.py / hijacks.py / __init__.py** - Intel extension for PyTorch helpers.

### networks [Helper]
- **check_lora_weights.py** [Main] - Checks a LoRA file for problems.
- **control_net_lllite.py** [Helper] - ControlNet‑LLLite network definition.
- **control_net_lllite_for_train.py** [Helper] - Used by sdxl_train_control_net_lllite.py.
- **dylora.py** [Helper] - Dynamic LoRA model.
- **extract_lora_from_dylora.py** [Main] - Makes a standard LoRA file from a DyLoRA model.
- **extract_lora_from_models.py** [Main] - Pulls LoRA weights from regular models.
- **lora.py** [Helper] - Base LoRA code.
- **lora_diffusers.py** [Helper] - Connects LoRA with the Diffusers library.
- **lora_fa.py** [Helper] - LoRA code with full attention layers.
- **lora_interrogator.py** [Main] - Finds good prompts for LoRA training.
- **merge_lora.py / merge_lora_old.py** [Main] - Combine multiple LoRA files.
- **oft.py** [Helper] - Implements Offset Noise Training (OFT).
- **resize_lora.py** [Main] - Changes the size of LoRA weights.
- **sdxl_merge_lora.py** [Main] - Merges LoRA files for SDXL models.
- **svd_merge_lora.py** [Main] - Merges LoRA files using SVD to reduce size.

### tools [Helper]
- **cache_latents.py** [Main] - Saves model latents to disk ahead of time.
- **cache_text_encoder_outputs.py** [Main] - Saves text encoder outputs to disk.
- **canny.py** [Main] - Makes edge images for ControlNet.
- **convert_diffusers20_original_sd.py** [Main] - Converts Diffusers 2.0 models to original format.
- **detect_face_rotate.py** [Main] - Finds and rotates faces in images.
- **latent_upscaler.py** [Main] - Upscales latents before decoding.
- **merge_models.py** [Main] - Combines two full models into one.
- **original_control_net.py** [Helper] - ControlNet code used by some tools.
- **resize_images_to_resolution.py** [Main] - Resizes images to a fixed size.
- **show_metadata.py** [Main] - Displays stored metadata from model files.

