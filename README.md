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

With `--map`, the script defaults to `--map_frame_source ann`, so the evaluated
frame list comes from the COCO annotation file instead of recursively scanning
every image under `--frames_dir`. This keeps mAP and latency on the same
annotated split and avoids unmapped frames when the image folder contains extra
sequences. Use `--map_frame_source frames_dir` only when you intentionally want
the older directory-scan behavior.

For MOT17 inference-only runs without `--map`, exclude the non-30-FPS validation
sequence explicitly:

```bash
--exclude_sequences MOT17-05-FRCNN
```

## TensorRT FP16-All Experiment

FP16-all uses FP16 TensorRT optimization for both the key and non-key engines.
Keep the FP32 engines/results as the accuracy baseline, then validate FP16 mAP
before reporting speedups.

```bash
python rtdetrv2_pytorch/tools/export_trt.py \
  -i onnx/mot17_skip8/key_model.onnx \
  -o engines/mot17_skip8/key_fp16.engine \
  -m key \
  --fp16 \
  --workspaceMB 4096

python rtdetrv2_pytorch/tools/export_trt.py \
  -i onnx/mot17_skip8/nonkey_model.onnx \
  -o engines/mot17_skip8/nonkey_fp16.engine \
  -m nonkey \
  --fp16 \
  --workspaceMB 4096
```

First check All-Key FP16 mAP. If this is much lower than FP32 All-Key, debug the
key engine before interpreting temporal results.

```bash
python rtdetrv2_pytorch/tools/infer_trt.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp16.engine \
  --mode all_key \
  -k 1 \
  --warmup 10 \
  --map \
  --ann_file ../dataset/mot17/val.json \
  --frames_root ../dataset/mot17/val \
  --save_json output/trt_fp16_all/all_key.json

python rtdetrv2_pytorch/tools/infer_trt.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp16.engine \
  --mode reuse \
  -k 1 \
  -m 1 \
  --warmup 10 \
  --map \
  --ann_file ../dataset/mot17/val.json \
  --frames_root ../dataset/mot17/val \
  --save_json output/trt_fp16_all/reuse_m1.json

python rtdetrv2_pytorch/tools/infer_trt.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp16.engine \
  --nonkey_engine engines/mot17_skip8/nonkey_fp16.engine \
  --mode knk \
  -k 1 \
  -m 1 \
  --warmup 10 \
  --map \
  --ann_file ../dataset/mot17/val.json \
  --frames_root ../dataset/mot17/val \
  --save_json output/trt_fp16_all/knk_m1.json
```

## TensorRT INT8 Non-Key Experiment

INT8 is implemented for the non-key engine only. Reuse the existing key FP32 or
key FP16 engine, collect real non-key calibration inputs from that key engine,
then build a calibrated non-key INT8 engine.

```bash
python rtdetrv2_pytorch/tools/collect_nonkey_calibration.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp32.engine \
  --output_dir output/calib_nonkey_int8_from_key_fp32 \
  --max_samples 512 \
  --compressed

python rtdetrv2_pytorch/tools/export_trt.py \
  -i onnx/mot17_skip8/nonkey_model.onnx \
  -o engines/mot17_skip8/nonkey_int8.engine \
  -m nonkey \
  --int8 \
  --calib_data output/calib_nonkey_int8_from_key_fp32 \
  --calib_cache engines/mot17_skip8/nonkey_int8.cache \
  --workspaceMB 4096
```

Run FP32-key + INT8-nonkey:

```bash
python rtdetrv2_pytorch/tools/infer_trt.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp32.engine \
  --nonkey_engine engines/mot17_skip8/nonkey_int8.engine \
  --mode knk \
  -k 1 \
  -m 1 \
  --warmup 10 \
  --map \
  --ann_file ../dataset/mot17/val.json \
  --frames_root ../dataset/mot17/val \
  --save_json output/trt_int8_nk/key_fp32_nonkey_int8.json
```

Run FP16-key + the same INT8-nonkey engine by changing only the key engine:

```bash
python rtdetrv2_pytorch/tools/infer_trt.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp16.engine \
  --nonkey_engine engines/mot17_skip8/nonkey_int8.engine \
  --mode knk \
  -k 1 \
  -m 1 \
  --warmup 10 \
  --map \
  --ann_file ../dataset/mot17/val.json \
  --frames_root ../dataset/mot17/val \
  --save_json output/trt_int8_nk/key_fp16_nonkey_int8.json
```

## TensorRT INT4 Non-Key Experiment

INT4 is implemented for the non-key engine only while keeping the key engine
FP32. Unlike the INT8 path, INT4 must use explicit Q/DQ weight-only quantization.
TensorRT's native `BuilderFlag.INT4` alone is diagnostic-only for this graph
because it can build an FP32-like engine with FP32-like size and latency.

Collect real non-key calibration inputs from the FP32 key engine, then build the
INT4 Q/DQ non-key engine:

```bash
python rtdetrv2_pytorch/tools/collect_nonkey_calibration.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp32.engine \
  --output_dir output/calib_nonkey_int4_from_key_fp32 \
  --max_samples 512 \
  --compressed

python rtdetrv2_pytorch/tools/export_trt.py \
  -i onnx/mot17_skip8/nonkey_model.onnx \
  -o engines/mot17_skip8/nonkey_int4.engine \
  -m nonkey \
  --int4 \
  --int4_mode qdq \
  --calib_data output/calib_nonkey_int4_from_key_fp32 \
  --int4_onnx onnx/mot17_skip8/nonkey_int4_qdq.onnx \
  --workspaceMB 4096
```

If the TensorRT build machine does not have ModelOpt installed, generate
`nonkey_int4_qdq.onnx` on another machine, copy it to the Jetson, then build with
`--int4 --int4_mode qdq --int4_prequantized`.

Run FP32-key + INT4-nonkey:

```bash
python rtdetrv2_pytorch/tools/infer_trt.py \
  --frames_dir ../dataset/mot17/val \
  --recursive \
  --key_engine engines/mot17_skip8/key_fp32.engine \
  --nonkey_engine engines/mot17_skip8/nonkey_int4.engine \
  --mode knk \
  -k 1 \
  -m 1 \
  --warmup 10 \
  --map \
  --ann_file ../dataset/mot17/val.json \
  --frames_root ../dataset/mot17/val \
  --save_json output/trt_int4_nk/key_fp32_nonkey_int4.json
```

## YOLO26 All-Key Baseline

For a non-temporal YOLO26 baseline that runs the detector on every evaluated
frame, see [docs/yolo26_all_key.md](docs/yolo26_all_key.md). The workflow
converts the existing COCO-format MOT17/VIRAT annotations to Ultralytics YOLO
format, fine-tunes `yolo26*.pt`, and runs All-Key validation or inference.
