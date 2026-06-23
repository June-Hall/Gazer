<div align="center">
  <h1>GAZER: Training-Free Semantic Correction for Autoregressive Visual Models</h1>
  <p>
    <a href="https://github.com/June-Hall/Gazer">
      <img src="https://img.shields.io/badge/github-repo-blue.svg" alt="github repo">
    </a>
    <a href="https://arxiv.org/abs/2606.22550">
      <img src="https://img.shields.io/badge/arXiv-2606.22550-red.svg" alt="arxiv">
    </a>
  </p>
</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Showcase](#-showcase)
- [News](#-news)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Acknowledgement](#-acknowledgement)
- [Citation](#-citation)

## 📌 Overview

**GAZER** is a **training-free** semantic correction framework for **autoregressive visual models**. It improves text-to-image and text-to-video generation by inserting multimodal large language model feedback into the autoregressive sampling loop, allowing the generator to inspect intermediate visual states and correct semantic drift before errors accumulate into the final output.

Autoregressive visual generators such as **InfinityStar**, **STAR**, and **Helios** synthesize images and videos through coarse-to-fine next-scale prediction. This design is powerful, but semantic mistakes made at early scales can become difficult to repair because later scales mainly refine an already committed trajectory. GAZER addresses this issue without updating model weights, changing the tokenizer, or retraining the generator.

GAZER is evaluated on compositional image and video benchmarks, including **T2I-CompBench** and **T2V-CompBench**, and improves semantic alignment across multiple autoregressive backbones.

## 🎬 Showcase

### Image Generation: InfinityStar

<table width="100%">
<thead>
<tr>
<th>Prompt</th>
<th>Baseline</th>
<th>GAZER</th>
</tr>
</thead>
<tbody>
<tr>
<td><em>A cat hiding behind a large cardboard box.</em></td>
<td><img src="./assets/showcase/image/infinitystar/cat_cardboard_box_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/infinitystar/cat_cardboard_box_ours.png" width="220"></td>
</tr>
<tr>
<td><em>A small purple tent placed between two large rocks</em></td>
<td><img src="./assets/showcase/image/infinitystar/purple_tent_rocks_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/infinitystar/purple_tent_rocks_ours.png" width="220"></td>
</tr>
<tr>
<td><em>A star-shaped cookie on a round ceramic plate.</em></td>
<td><img src="./assets/showcase/image/infinitystar/star_cookie_plate_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/infinitystar/star_cookie_plate_ours.png" width="220"></td>
</tr>
<tr>
<td><em>Four blue cups on a wooden table.</em></td>
<td><img src="./assets/showcase/image/infinitystar/four_blue_cups_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/infinitystar/four_blue_cups_ours.png" width="220"></td>
</tr>
<tr>
<td><em>a backpack on the top of a chicken</em></td>
<td><img src="./assets/showcase/image/infinitystar/backpack_chicken_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/infinitystar/backpack_chicken_ours.png" width="220"></td>
</tr>
</tbody>
</table>

### Image Generation: STAR

<table width="100%">
<thead>
<tr>
<th>Prompt</th>
<th>Baseline</th>
<th>GAZER</th>
</tr>
</thead>
<tbody>
<tr>
<td><em>A pink chair placed in front of a green wall</em></td>
<td><img src="./assets/showcase/image/star/pink_chair_green_wall_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/star/pink_chair_green_wall_ours.png" width="220"></td>
</tr>
<tr>
<td><em>A tiny flower growing beside a large stone</em></td>
<td><img src="./assets/showcase/image/star/tiny_flower_stone_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/star/tiny_flower_stone_ours.png" width="220"></td>
</tr>
<tr>
<td><em>Four orange pumpkins arranged in a straight line</em></td>
<td><img src="./assets/showcase/image/star/four_orange_pumpkins_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/star/four_orange_pumpkins_ours.png" width="220"></td>
</tr>
<tr>
<td><em>Two green trees on both sides of a narrow road</em></td>
<td><img src="./assets/showcase/image/star/two_green_trees_road_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/star/two_green_trees_road_ours.png" width="220"></td>
</tr>
<tr>
<td><em>a boy on the right of a pig</em></td>
<td><img src="./assets/showcase/image/star/boy_right_of_pig_baseline.png" width="220"></td>
<td><img src="./assets/showcase/image/star/boy_right_of_pig_ours.png" width="220"></td>
</tr>
</tbody>
</table>

### Video Generation: InfinityStar

<table width="100%">
<thead>
<tr>
<th>Prompt</th>
<th>Baseline</th>
<th>GAZER</th>
</tr>
</thead>
<tbody>
<tr>
<td><em>A cat pushes a ball toward a puppy</em></td>
<td><video src="./assets/showcase/video/infinitystar/cat_pushes_ball_puppy_baseline.mp4" width="220" controls muted loop playsinline></video></td>
<td><video src="./assets/showcase/video/infinitystar/cat_pushes_ball_puppy_ours.mp4" width="220" controls muted loop playsinline></video></td>
</tr>
<tr>
<td><em>A dog chases after a balloon drifting in the wind</em></td>
<td><video src="./assets/showcase/video/infinitystar/dog_chases_balloon_baseline.mp4" width="220" controls muted loop playsinline></video></td>
<td><video src="./assets/showcase/video/infinitystar/dog_chases_balloon_ours.mp4" width="220" controls muted loop playsinline></video></td>
</tr>
<tr>
<td><em>A dog runs through a field while a cat climbs a tree</em></td>
<td><video src="./assets/showcase/video/infinitystar/dog_field_cat_tree_baseline.mp4" width="220" controls muted loop playsinline></video></td>
<td><video src="./assets/showcase/video/infinitystar/dog_field_cat_tree_ours.mp4" width="220" controls muted loop playsinline></video></td>
</tr>
<tr>
<td><em>A girl is blowing bubbles in the yard, while a puppy is jumping</em></td>
<td><video src="./assets/showcase/video/infinitystar/girl_bubbles_puppy_baseline.mp4" width="220" controls muted loop playsinline></video></td>
<td><video src="./assets/showcase/video/infinitystar/girl_bubbles_puppy_ours.mp4" width="220" controls muted loop playsinline></video></td>
</tr>
<tr>
<td><em>A girl walks beside a white horse</em></td>
<td><video src="./assets/showcase/video/infinitystar/girl_white_horse_baseline.mp4" width="220" controls muted loop playsinline></video></td>
<td><video src="./assets/showcase/video/infinitystar/girl_white_horse_ours.mp4" width="220" controls muted loop playsinline></video></td>
</tr>
</tbody>
</table>

### Video Generation: Helios

<table width="100%">
<thead>
<tr>
<th>Prompt</th>
<th>Baseline</th>
<th>GAZER</th>
</tr>
</thead>
<tbody>
<tr>
<td><em>A cat sits on a windowsill and a dog plays in the yard</em></td>
<td><video src="./assets/showcase/video/helios/cat_windowsill_dog_yard_baseline.mp4" width="220" controls muted loop playsinline></video></td>
<td><video src="./assets/showcase/video/helios/cat_windowsill_dog_yard_ours.mp4" width="220" controls muted loop playsinline></video></td>
</tr>
<tr>
<td><em>A kangaroo bounds across the plain and a cow grazes</em></td>
<td><video src="./assets/showcase/video/helios/kangaroo_cow_baseline.mp4" width="220" controls muted loop playsinline></video></td>
<td><video src="./assets/showcase/video/helios/kangaroo_cow_ours.mp4" width="220" controls muted loop playsinline></video></td>
</tr>
<tr>
<td><em>Dog snatches a rolling ball</em></td>
<td><video src="./assets/showcase/video/helios/dog_snatches_ball_baseline.mp4" width="220" controls muted loop playsinline></video></td>
<td><video src="./assets/showcase/video/helios/dog_snatches_ball_ours.mp4" width="220" controls muted loop playsinline></video></td>
</tr>
</tbody>
</table>

## 📢 News

- [2026.06] Paper draft: **"Training-Free Semantic Correction for Autoregressive Visual Models"**.

## 🔧 Installation

### Environment

```bash
cd Gazer

conda create -n gazer python=3.10
conda activate gazer

# Install PyTorch according to your CUDA version first.
# Example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install transformers gradio gradio-client loguru flash-attn
```

### Model Checkpoints

The demo scripts expect the following checkpoint layout by default:

```text
../models/
├── InfinityStar/
│   ├── infinitystar_8b_720p_weights/
│   ├── infinitystar_videovae.pth
│   └── text_encoder/
│       └── flan-t5-xl-official/
└── Qwen3-VL-8B-Instruct/
```

These paths are resolved relative to the `Gazer/` directory. If your checkpoints are stored elsewhere, set `CHECKPOINTS_DIR` when running the demo and update `model_path` in `vlm/video_evaluation_server.py` for the Qwen3-VL evaluator.

## 🚀 Quick Start

### Single Demo

The simplest way to run GAZER is:

```bash
cd Gazer
./run_demo.sh
```

This script starts the Qwen3-VL evaluation server if it is not already running, waits for the server on `127.0.0.1:16660`, and then generates one 720p demo video with self-reflection enabled.

You can customize the prompt and output path from the command line:

```bash
PROMPT="A girl is blowing bubbles in the yard, while a puppy is jumping" \
OUTPUT_PATH="./results/demo.mp4" \
INFER_CUDA_VISIBLE_DEVICES=0 \
./run_demo.sh
```

### Python Entry

You can also call the single-demo entry directly:

```bash
python tools/infer_video_720p_demo.py \
    --checkpoints_dir ../models/InfinityStar/ \
    --output_path ./results/demo.mp4 \
    --prompt "A girl is blowing bubbles in the yard, while a puppy is jumping" \
    --seed 42 \
    --reflection_start_scale 0 \
    --reflection_end_scale 1.0 \
    --reflection_interval 4
```

To disable self-reflection and run the base generator through the same demo entry:

```bash
python tools/infer_video_720p_demo.py \
    --checkpoints_dir ../models/InfinityStar/ \
    --output_path ./results/baseline_demo.mp4 \
    --prompt "A girl is blowing bubbles in the yard, while a puppy is jumping" \
    --disable_self_reflection
```

### VLM Server

GAZER uses a local Gradio server for MLLM feedback:

```bash
python vlm/video_evaluation_server.py
```

The client connects to `http://localhost:16660` by default. The server loads Qwen3-VL from `../models/Qwen3-VL-8B-Instruct` unless changed in `vlm/video_evaluation_server.py`.

## 🙏 Acknowledgement

GAZER builds on recent progress in next-scale autoregressive visual generation, especially Infinity, InfinityStar, STAR, and Helios. The current demo uses Qwen3-VL as the multimodal feedback model. We thank the authors of these projects and the maintainers of the evaluation benchmarks used in the paper.

## 📚 Citation

If you find this project useful, please cite:

```bibtex
@misc{chen2026trainingfreesemanticcorrectionautoregressive,
      title={Training-Free Semantic Correction for Autoregressive Visual Models}, 
      author={Junhao Chen and Chanyu Zhu and Zheqi Lv and Keting Yin and Shengyu Zhang},
      year={2026},
      eprint={2606.22550},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.22550}, 
}
```
