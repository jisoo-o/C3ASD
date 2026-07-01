# C&sup3;ASD: Multi-Level Consistency-Driven Representation Learning for Robust Active Speaker Detection

This repository contains the official implementation of the following paper:

> **C&sup3;ASD: Multi-Level Consistency-Driven Representation Learning for Robust Active Speaker Detection**
> Jin Hong\*, Jisoo Park\*, Junseok Kwon&dagger;
> (\* Equal contribution)
> **ECCV 2026**

## Overview

C&sup3;ASD improves the robustness of lightweight Active Speaker Detection (ASD) models by introducing three levels of consistency regularization that encourage coherent representations across modalities and prediction heads, without adding inference cost.

<p align="center">
  <img src="figures/architecture.png" width="90%">
</p>

### Consistency Losses

| Loss | Description |
|------|-------------|
| **Inter-modality** | Cosine similarity between audio and visual embeddings on speaking frames, encouraging cross-modal alignment |
| **Intra-modality** | Supervised contrastive (InfoNCE) loss within each modality, pulling same-speaker same-state embeddings together |
| **Prediction-level** | Confidence-masked MSE that distills the audio-visual prediction into unimodal (audio-only, visual-only) heads |

## Requirements

- Python 3.6+
- PyTorch 1.7+
- python_speech_features
- scipy
- opencv-python
- pandas
- tqdm

```bash
pip install torch torchvision python_speech_features scipy opencv-python pandas tqdm
```

## Dataset

We use the [AVA-ActiveSpeaker](https://research.google.com/ava/download.html) dataset. To download and preprocess:

```bash
python train.py --dataPathAVA /path/to/AVA --downloadAVA
```

### Dataset Structure

```
AVADataPath/
├── clips_audios/{train,val}/{videoID}/{clipID}.wav
├── clips_videos/{train,val}/{videoID}/{clipID}/{frame}.jpg
└── csv/
    ├── {train,val}_loader.csv
    └── {train,val}_orig.csv
```

## Training

### C&sup3;ASD (default, with all consistency losses)

```bash
python train.py --dataPathAVA /path/to/AVA --savePath exps/c3asd
```

### Baseline (Light-ASD, without consistency losses)

```bash
python train.py --dataPathAVA /path/to/AVA --savePath exps/baseline \
    --lambda_inter 0 --lambda_intra_audio 0 --lambda_intra_visual 0 --lambda_pred 0
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 0.001 | Learning rate |
| `--lrDecay` | 0.95 | Learning rate decay (StepLR) |
| `--maxEpoch` | 30 | Maximum training epochs |
| `--batchSize` | 1000 | Dynamic batch size (frames) |
| `--lambda_inter` | 0.01 | Inter-modality consistency weight |
| `--lambda_intra_audio` | 0.001 | Intra-modality consistency weight (audio) |
| `--lambda_intra_visual` | 0.001 | Intra-modality consistency weight (visual) |
| `--lambda_pred` | 0.01 | Prediction-level consistency weight |
| `--intra_temperature` | 0.07 | Temperature for InfoNCE loss |

## Evaluation

### Clean Evaluation

```bash
python train.py --dataPathAVA /path/to/AVA --evaluation
```

### Corruption Robustness Evaluation

We provide a corruption testing framework following the [CAV2Vec](https://github.com/sungnyun/cav2vec) (ICLR 2025) evaluation protocol. Corruptions are applied at test time only.

#### Supported Corruption Types

| Category | Types |
|----------|-------|
| **Audio (MUSAN)** | `audio_babble`, `audio_music`, `audio_natural`, `audio_speech` |
| **Audio (DEMAND)** | `audio_demand_park`, `audio_demand_cafe`, `audio_demand_metro`, `audio_demand_river`, etc. |
| **Visual** | `visual_object`, `visual_gaussian`, `visual_blur`, `visual_hands`, `visual_pixelate` |
| **Joint** | `joint_audio_visual`, `joint_severe` |
| **Temporal** | `temporal_desync` |

#### Single Corruption Test

```bash
# Audio corruption (MUSAN babble noise at SNR=-10dB)
python test_corruption.py \
    --dataPathAVA /path/to/AVA \
    --modelPath exps/c3asd/model/model_0030.model \
    --corruption_type audio_babble \
    --audio_snr -10 \
    --musan_path /path/to/musan

# Visual corruption (object occlusion)
python test_corruption.py \
    --dataPathAVA /path/to/AVA \
    --modelPath exps/c3asd/model/model_0030.model \
    --corruption_type visual_object \
    --occlusion_path /path/to/occlusion_patch

# Temporal desynchronization (+3 frames = +120ms)
python test_corruption.py \
    --dataPathAVA /path/to/AVA \
    --modelPath exps/c3asd/model/model_0030.model \
    --corruption_type temporal_desync \
    --temporal_desync 3
```

#### Full Corruption Benchmark

Run all corruption types at once:

```bash
# Edit paths in test_all_corruption.sh first, then:
bash test_all_corruption.sh c3asd 30
```

Required external datasets for corruption testing:
- [MUSAN](https://www.openslr.org/17/) - Audio noise (babble, music, natural, speech)
- [DEMAND](https://zenodo.org/record/1227121) - Real-world audio noise environments
- Occlusion patches - COCO object crops and hand images for visual occlusion

## Results

### AVA-ActiveSpeaker

| Method | Params (M) | FLOPs (G) | mAP (%) |
|--------|-----------|-----------|---------|
| Light-ASD | 1.02 | 0.62 | 93.6 |
| **C&sup3;ASD (ours)** | **1.02** | **0.62** | **93.8** |

### WASD (In-the-Wild)

| Method | mAP (%) |
|--------|---------|
| Light-ASD | 85.3 |
| **C&sup3;ASD (ours)** | **86.1** |

## Acknowledgements

This codebase is built upon [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) (CVPR 2023). The corruption evaluation protocol follows [CAV2Vec](https://github.com/sungnyun/cav2vec) (ICLR 2025). We thank the authors for their open-source contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
