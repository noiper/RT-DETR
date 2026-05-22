# KNDETR

KNDETR is a temporal RT-DETR variant for video detection. Key frames run the full RT-DETR path and cache encoder/query features. Non-Key frames reuse the cache, fuse it with the current backbone feature, and run a lightweight decoder.

## MOT17 Key-Model Finetuning

```bash
python rtdetrv2_pytorch/tools/train.py \
  -c rtdetrv2_pytorch/configs/kndrtr/kndetr_mot17.yml \
  -t models/rtdetrv2_r18vd_120e_coco.pth \
  --output-dir output/kndetr_mot17
```

## Temporal Training

Train the temporal model from a key-only or previous temporal checkpoint:

```bash
python rtdetrv2_pytorch/tools/train_temporal.py \
  -c rtdetrv2_pytorch/configs/kndrtr/temporal_kndetr_mot17.yml \
  -s kd \
  -t output/kndetr_mot17/best_model.pth \
  --lambda_kd 0 \
  --lambda_score 15 \
  --output_dir output/temporal_mot17
```

Useful flags:

```text
-s freeze_key|kd|joint
--same_frame              level0 / same-frame training
--lr 1e-4                 override temporal optimizer LR
--epochs 10               override config epochs
```

## Evaluation

Low incoming frame rate, alternating Key/Non-Key:

```bash
python rtdetrv2_pytorch/tools/eval_temporal_low_rate.py \
  -c rtdetrv2_pytorch/configs/kndrtr/temporal_kndetr_mot17.yml \
  -w output/temporal_mot17/best_model.pth \
  --skip 3
```

Reuse baseline for the same setting:

```bash
python rtdetrv2_pytorch/tools/eval_temporal_low_rate.py \
  -c rtdetrv2_pytorch/configs/kndrtr/temporal_kndetr_mot17.yml \
  -w output/temporal_mot17/best_model.pth \
  --skip 3 \
  --baseline
```

Streaming K/NK schedule:

```bash
python rtdetrv2_pytorch/tools/eval_temporal_stream.py \
  -c rtdetrv2_pytorch/configs/kndrtr/temporal_kndetr_mot17.yml \
  -w output/temporal_mot17/best_model.pth \
  --nk_per_key 1 \
  --frame_stride 2
```

Optional score tuning:

```bash
--tune_score
```
