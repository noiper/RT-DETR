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

## TensorRT Jetson FP32 Baseline

TensorRT export builds FP32 engines by default. Use this as the baseline for
All-Key, Key-Reuse, and KNDETR deployment.

```bash
python rtdetrv2_pytorch/tools/export_onnx.py \
  -c rtdetrv2_pytorch/configs/kndrtr/temporal_kndetr_mot17.yml \
  -r output/phase1_mot17_skip8/05_417_736.pth \
  --key_onnx onnx/mot17_skip8/key_model.onnx \
  --nonkey_onnx onnx/mot17_skip8/nonkey_model.onnx

python rtdetrv2_pytorch/tools/export_trt.py \
  -i onnx/mot17_skip8/key_model.onnx \
  -o engines/mot17_skip8/key_fp32.engine \
  -m key \
  --workspaceMB 4096

python rtdetrv2_pytorch/tools/export_trt.py \
  -i onnx/mot17_skip8/nonkey_model.onnx \
  -o engines/mot17_skip8/nonkey_fp32.engine \
  -m nonkey \
  --workspaceMB 4096
```

```bash
python rtdetrv2_pytorch/tools/infer_trt.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp32.engine \
  --mode all_key \
  -k 1 \
  --power

python rtdetrv2_pytorch/tools/infer_trt.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp32.engine \
  --mode reuse \
  -k 1 \
  -m 1 \
  --power

python rtdetrv2_pytorch/tools/infer_trt.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp32.engine \
  --nonkey_engine engines/mot17_skip8/nonkey_fp32.engine \
  --mode knk \
  -k 1 \
  -m 1 \
  --power
```

By default, `infer_trt.py` reports inference latency/FPS only. Add COCO mAP
reporting to any inference command with:

```bash
--map \
--ann_file ../dataset/mot17/val.json \
--frames_root ../dataset/mot17/val
```

## YOLO26 All-Key Baseline

For a non-temporal YOLO26 baseline that runs the detector on every evaluated
frame, see [docs/yolo26_all_key.md](docs/yolo26_all_key.md). The workflow
converts the existing COCO-format MOT17/VIRAT annotations to Ultralytics YOLO
format, fine-tunes `yolo26*.pt`, and runs All-Key validation or inference.
