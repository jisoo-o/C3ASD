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

### In-domain Comparison on AVA-ActiveSpeaker (Table 1)

| Method | Single | Pre-train | E2E | Params (M) | FLOPs (G) | mAP (%) |
|--------|:------:|:---------:|:---:|:----------:|:---------:|:-------:|
| ASC | ✗ | ✓ | ✗ | 23.5 | 1.8 | 87.1 |
| MAAS | ✗ | ✓ | ✗ | 22.5 | 2.8 | 88.8 |
| UniCon | ✗ | ✓ | ✗ | >22.4 | >1.8 | 92.2 |
| ASDNet | ✗ | ✓ | ✗ | 51.3 | 14.9 | 93.5 |
| EASEE-50 | ✗ | ✓ | ✓ | >74.7 | >65.5 | 94.1 |
| SPELL | ✗ | ✓ | ✗ | 22.5 | 2.6 | 94.2 |
| SPELL+ | ✗ | ✓ | ✗ | 47.3 | 5.4 | 94.9 |
| LoCoNet | ✗ | ✓ | ✓ | 34.3 | 4.9 | 95.2 |
| TalkNCE | ✗ | ✓ | ✓ | 34.3 | 4.9 | 95.5 |
| | | | | | | |
| TalkNet | ✓ | ✗ | ✓ | 15.7 | 1.5 | 92.3 |
| Sync-TalkNet | ✓ | ✗ | ✓ | 15.7 | 1.5 | 89.8 |
| ASD-Trans. | ✓ | ✗ | ✓ | >13.9 | >1.5 | 93.0 |
| ADENet | ✓ | ✗ | ✓ | 33.2 | 22.7 | 93.2 |
| Light-ASD | ✓ | ✗ | ✓ | 1.02 | 0.62 | 93.6 |
| LR-ASD | ✓ | ✓ | ✓ | 0.84 | 0.51 | 94.5 |
| **C&sup3;ASD (Ours)** | **✓** | **✗** | **✓** | **1.02** | **0.62** | **93.8** |

### Cross-domain Evaluation on WASD (Table 2)

| Method | mAP (%) |
|--------|:-------:|
| TalkNet | 78.4 |
| ADENet | 85.6 |
| Light-ASD | 85.3 |
| **C&sup3;ASD (Ours)** | **86.1** |

### Audio Corruption Robustness - MUSAN (Table 3)

mAP (%) at each SNR level (dB):

| Method | -10 | -5 | 0 | 5 | 10 | AVG |
|--------|:---:|:--:|:-:|:-:|:--:|:---:|
| **Babble** | | | | | | |
| TalkNet | 85.5 | 87.6 | 89.3 | 90.5 | 91.2 | 88.8 |
| ADENet | 84.0 | 85.8 | 86.7 | 87.5 | 88.4 | 86.5 |
| Light-ASD | 88.0 | 89.7 | 91.1 | 92.1 | 92.7 | 90.7 |
| **C&sup3;ASD (Ours)** | **88.3** | **89.9** | **91.3** | **92.3** | **93.0** | **91.0** |
| **Music** | | | | | | |
| TalkNet | 84.1 | 86.5 | 88.5 | 90.0 | 91.0 | 88.0 |
| ADENet | 82.5 | 84.5 | 85.9 | 87.3 | 88.4 | 85.7 |
| Light-ASD | 86.8 | 88.7 | 90.4 | 91.6 | 92.5 | 90.0 |
| **C&sup3;ASD (Ours)** | **87.5** | **89.2** | **90.8** | **92.0** | **92.8** | **90.5** |
| **Natural** | | | | | | |
| TalkNet | 85.2 | 87.4 | 89.1 | 90.2 | 90.9 | 88.6 |
| ADENet | 83.6 | 85.3 | 86.4 | 87.4 | 88.3 | 86.2 |
| Light-ASD | 87.6 | 89.4 | 90.8 | 91.8 | 92.5 | 90.4 |
| **C&sup3;ASD (Ours)** | **88.3** | **89.9** | **91.2** | **92.2** | **92.8** | **90.9** |

### Audio Corruption Robustness - DEMAND (Table 4)

mAP (%) with random SNR in [-10, 10] dB:

| Method | Park | River | Cafe | Restau. | Cafeter. | Metro | Station | Meeting | AVG |
|--------|:----:|:-----:|:----:|:-------:|:--------:|:-----:|:-------:|:-------:|:---:|
| TalkNet | 90.3 | 89.1 | 86.8 | 86.5 | 86.8 | 91.1 | 89.6 | 85.7 | 88.2 |
| ADENet | 88.9 | 87.1 | 84.3 | 84.9 | 85.3 | 89.1 | 87.5 | 85.5 | 86.6 |
| Light-ASD | 92.1 | 91.0 | 89.0 | 88.8 | 89.0 | 92.6 | 91.2 | 87.6 | 90.2 |
| **C&sup3;ASD (Ours)** | **92.5** | **91.3** | **89.7** | **89.4** | **89.7** | **92.7** | **91.5** | **88.9** | **90.7** |

### Visual Corruption Robustness (Table 5)

mAP (%):

| Method | Object Occlusion + Noise | Pixelated |
|--------|:------------------------:|:---------:|
| TalkNet | 70.73 | 92.12 |
| ADENet | 66.36 | 89.04 |
| Light-ASD | 76.86 | 93.15 |
| **C&sup3;ASD (Ours)** | **78.90** | **93.47** |

### Ablation Study (Table 8)

mAP (%) on AVA-ActiveSpeaker:

| L_inter | L_intra | L_pred | mAP (%) |
|:-------:|:-------:|:------:|:-------:|
| ✗ | ✗ | ✗ | 93.61 |
| ✓ | ✗ | ✗ | 93.70 |
| ✗ | ✓ | ✗ | 93.62 |
| ✗ | ✗ | ✓ | 93.68 |
| ✓ | ✓ | ✓ | **93.80** |

## Acknowledgements

This codebase is built upon [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) (CVPR 2023). The corruption evaluation protocol follows [CAV2Vec](https://github.com/sungnyun/cav2vec) (ICLR 2025). We thank the authors for their open-source contributions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
