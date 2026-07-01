#!/bin/bash

# Corruption Robustness Testing Script
# Based on CAV2Vec corruption testing methodology
# Usage: bash test_all_corruption.sh <exp_name> [epoch]
#   e.g. bash test_all_corruption.sh inter
#        bash test_all_corruption.sh pred 25

# ====== ARGUMENT PARSING ======
EXP_NAME=${1:-baseline}
EPOCH=${2:-30}
EPOCH_STR=$(printf "%04d" $EPOCH)

# ====== CONFIGURATION ======
DATA_PATH="/path/to/AVA"                                                    # Change to your AVA dataset path
MODEL_PATH="exps/${EXP_NAME}/model/model_${EPOCH_STR}.model"
MUSAN_PATH="/path/to/musan"                                                 # Dir with: babble/ music/ natural/ speech/
DEMAND_PATH="/path/to/DEMAND"                                               # Dir with NPARK/ TMETRO/ etc.
OCCLUSION_PATH="/path/to/occlusion_patch"                                   # Dir with: object_image_sr/ object_mask_x4/
PATCH_SCALE=0.4                                                             # Occlusion patch size relative to frame (0.0~1.0). 1.0=full, 0.5=half
SEED=42                                                                     # Random seed for reproducibility
EVAL_DATA_TYPE="val"
SNRS=(-10 -5 0 5 10)                                                       # SNR levels for MUSAN sweep
MUSAN_TYPES=("audio_babble" "audio_music" "audio_natural")
DEMAND_TYPES=("audio_demand_park" "audio_demand_cafe" "audio_demand_metro"
              "audio_demand_river" "audio_demand_restaurant" "audio_demand_cafeteria"
              "audio_demand_public_station" "audio_demand_meeting_room")
VISUAL_JOINT=("object_occlusion" "pixelation")
TEMPORAL_DESYNCS=(1 2 3 5 10)                                              # Temporal desync in video frames (±N, 1frame=40ms)

# Results directory
RESULTS_DIR="corruption_test_results/${EXP_NAME}_${PATCH_SCALE}"
mkdir -p $RESULTS_DIR

# Verify model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    exit 1
fi

echo "==========================================="
echo "Corruption Robustness Testing"
echo "==========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH ($EVAL_DATA_TYPE)"
echo "Patch Scale: $PATCH_SCALE"
echo "Seed: $SEED"
echo "MUSAN SNR levels: ${SNRS[*]}"
echo "DEMAND SNR: random (-10, 10)"
echo "Results: $RESULTS_DIR"
echo ""

# ====== 1. CLEAN (BASELINE) ======
echo "[1] Testing: Clean (No Corruption)"
python test_corruption.py --seed $SEED \
    --dataPathAVA $DATA_PATH \
    --modelPath $MODEL_PATH \
    --evalDataType $EVAL_DATA_TYPE \
    --corruption_type clean \
    --results_save_path $RESULTS_DIR

# ====== 2. AUDIO CORRUPTIONS (MUSAN) — per SNR ======
for NOISE_TYPE in "${MUSAN_TYPES[@]}"; do
    echo ""
    echo "====== Testing: ${NOISE_TYPE} (SNR sweep) ======"
    for SNR in "${SNRS[@]}"; do
        echo "  ${NOISE_TYPE} @ SNR=${SNR}dB"
        python test_corruption.py --seed $SEED \
            --dataPathAVA $DATA_PATH \
            --modelPath $MODEL_PATH \
            --evalDataType $EVAL_DATA_TYPE \
            --corruption_type $NOISE_TYPE \
            --audio_snr $SNR \
            --musan_path $MUSAN_PATH \
            --results_save_path $RESULTS_DIR
    done
done

# ====== 3. AUDIO CORRUPTIONS (DEMAND) — random SNR (-10, 10) ======
for NOISE_TYPE in "${DEMAND_TYPES[@]}"; do
    echo ""
    echo "====== Testing: ${NOISE_TYPE} (random SNR) ======"
    python test_corruption.py --seed $SEED \
        --dataPathAVA $DATA_PATH \
        --modelPath $MODEL_PATH \
        --evalDataType $EVAL_DATA_TYPE \
        --corruption_type $NOISE_TYPE \
        --demand_path $DEMAND_PATH \
        --results_save_path $RESULTS_DIR
done

# ====== 4. VISUAL CORRUPTIONS ======
echo ""
echo "====== Testing: Visual - Object Occlusion (COCO) ======"
python test_corruption.py --seed $SEED \
    --dataPathAVA $DATA_PATH \
    --modelPath $MODEL_PATH \
    --evalDataType $EVAL_DATA_TYPE \
    --corruption_type visual_object \
    --occlusion_path $OCCLUSION_PATH \
    --patch_scale $PATCH_SCALE \
    --results_save_path $RESULTS_DIR

echo ""
echo "====== Testing: Visual - Gaussian Noise + Blur ======"
python test_corruption.py --seed $SEED \
    --dataPathAVA $DATA_PATH \
    --modelPath $MODEL_PATH \
    --evalDataType $EVAL_DATA_TYPE \
    --corruption_type visual_gaussian \
    --patch_scale $PATCH_SCALE \
    --results_save_path $RESULTS_DIR

echo ""
echo "====== Testing: Visual - Hands Occlusion ======"
python test_corruption.py --seed $SEED \
    --dataPathAVA $DATA_PATH \
    --modelPath $MODEL_PATH \
    --evalDataType $EVAL_DATA_TYPE \
    --corruption_type visual_hands \
    --occlusion_path $OCCLUSION_PATH \
    --patch_scale $PATCH_SCALE \
    --results_save_path $RESULTS_DIR

echo ""
echo "====== Testing: Visual - Pixelation ======"
python test_corruption.py --seed $SEED \
    --dataPathAVA $DATA_PATH \
    --modelPath $MODEL_PATH \
    --evalDataType $EVAL_DATA_TYPE \
    --corruption_type visual_pixelate \
    --patch_scale $PATCH_SCALE \
    --results_save_path $RESULTS_DIR

# ====== 5. JOINT CORRUPTIONS (Visual × MUSAN) — per SNR ======
for VIS in "${VISUAL_JOINT[@]}"; do
    for NOISE in "${MUSAN_TYPES[@]}"; do
        echo ""
        echo "====== Joint: ${NOISE} + ${VIS} (SNR sweep) ======"
        for SNR in "${SNRS[@]}"; do
            echo "  ${NOISE} + ${VIS} @ SNR=${SNR}dB"
            python test_corruption.py --seed $SEED \
                --dataPathAVA $DATA_PATH \
                --modelPath $MODEL_PATH \
                --evalDataType $EVAL_DATA_TYPE \
                --corruption_type $NOISE \
                --visual_corruption_type $VIS \
                --audio_snr $SNR \
                --musan_path $MUSAN_PATH \
                --occlusion_path $OCCLUSION_PATH \
                --patch_scale $PATCH_SCALE \
                --results_save_path $RESULTS_DIR
        done
    done
done

# ====== 6. JOINT CORRUPTIONS (Visual × DEMAND) — random SNR (-10, 10) ======
for VIS in "${VISUAL_JOINT[@]}"; do
    for NOISE in "${DEMAND_TYPES[@]}"; do
        echo ""
        echo "====== Joint: ${NOISE} + ${VIS} (random SNR) ======"
        python test_corruption.py --seed $SEED \
            --dataPathAVA $DATA_PATH \
            --modelPath $MODEL_PATH \
            --evalDataType $EVAL_DATA_TYPE \
            --corruption_type $NOISE \
            --visual_corruption_type $VIS \
            --demand_path $DEMAND_PATH \
            --occlusion_path $OCCLUSION_PATH \
            --patch_scale $PATCH_SCALE \
            --results_save_path $RESULTS_DIR
    done
done

# ====== 7. TEMPORAL DESYNC ======
for DESYNC in "${TEMPORAL_DESYNCS[@]}"; do
    echo ""
    echo "====== Testing: Temporal Desync +${DESYNC} frames (+$((DESYNC*40))ms) ======"
    python test_corruption.py --seed $SEED \
        --dataPathAVA $DATA_PATH \
        --modelPath $MODEL_PATH \
        --evalDataType $EVAL_DATA_TYPE \
        --corruption_type temporal_desync \
        --temporal_desync $DESYNC \
        --results_save_path $RESULTS_DIR

    echo "====== Testing: Temporal Desync -${DESYNC} frames (-$((DESYNC*40))ms) ======"
    python test_corruption.py --seed $SEED \
        --dataPathAVA $DATA_PATH \
        --modelPath $MODEL_PATH \
        --evalDataType $EVAL_DATA_TYPE \
        --corruption_type temporal_desync \
        --temporal_desync -$DESYNC \
        --results_save_path $RESULTS_DIR
done

# ====== SUMMARIZE RESULTS ======
echo ""
echo "==========================================="
echo "Testing Complete!"
echo "==========================================="
echo ""
echo "Generating summary..."
python summarize_corruption_results.py --results_dir $RESULTS_DIR

echo ""
echo "All results saved in: $RESULTS_DIR"
echo "Summary saved in: ${RESULTS_DIR}/summary.txt"
