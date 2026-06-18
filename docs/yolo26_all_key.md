# YOLO26 All-Key Baseline

This workflow runs YOLO26 as an image detector on every evaluated frame. In
KNDETR terms this is an All-Key baseline: no cached features, no Non-Key path,
and no prediction reuse.

Ultralytics documents YOLO26 detection checkpoints as `yolo26n.pt`,
`yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, and `yolo26x.pt`, with train, val,
predict, and export support:

- YOLO26 model docs: <https://docs.ultralytics.com/models/yolo26/>
- Ultralytics detection dataset format: <https://docs.ultralytics.com/datasets/detect/>
- Fine-tuning guide: <https://docs.ultralytics.com/guides/finetuning-guide/>

## 1. Install YOLO26 Support

Use an environment with a CUDA/Jetson-compatible PyTorch first, then install or
upgrade Ultralytics. YOLO26 landed in Ultralytics `8.4.0`, so use that or newer.

```bash
python -m pip install -U "ultralytics>=8.4.0"
```

On Jetson, avoid accidentally replacing a working NVIDIA PyTorch build. If
needed, install PyTorch from NVIDIA first, then install Ultralytics.

## 2. Prepare The Dataset

The repository stores MOT17 and VIRAT as COCO JSON annotations. Convert them to
the Ultralytics YOLO layout with symlinked images and normalized txt labels.
Class ids stay 0-indexed.

MOT17:

```bash
python rtdetrv2_pytorch/tools/prepare_yolo_dataset.py \
  --preset mot17 \
  --out-dir datasets/yolo_mot17
```

VIRAT 30-FPS:

```bash
python rtdetrv2_pytorch/tools/prepare_yolo_dataset.py \
  --preset virat30 \
  --out-dir datasets/yolo_virat30
```

Smoke test before the full VIRAT conversion:

```bash
python rtdetrv2_pytorch/tools/prepare_yolo_dataset.py \
  --preset virat30 \
  --out-dir datasets/yolo_virat30_smoke \
  --max-images-per-split 200
```

For a low incoming-rate All-Key validation split, keep training at full rate and
stride only validation. Example: every 3rd validation frame, matching 10 FPS for
30-FPS source video.

```bash
python rtdetrv2_pytorch/tools/prepare_yolo_dataset.py \
  --preset mot17 \
  --out-dir datasets/yolo_mot17_val_stride3 \
  --train-frame-stride 1 \
  --val-frame-stride 3
```

## 3. Fine-Tune

Start with `yolo26n.pt` or `yolo26s.pt` for edge comparisons. Scale up only if
accuracy is the priority.

```bash
yolo detect train \
  model=yolo26s.pt \
  data=datasets/yolo_mot17/data.yaml \
  imgsz=640 \
  epochs=50 \
  batch=16 \
  device=0 \
  project="$(pwd)/output/yolo26" \
  name=mot17_all_key_yolo26s
```

VIRAT:

```bash
yolo detect train \
  model=yolo26s.pt \
  data=datasets/yolo_virat30/data.yaml \
  imgsz=640 \
  epochs=50 \
  batch=16 \
  device=0 \
  project="$(pwd)/output/yolo26" \
  name=virat30_all_key_yolo26s
```

## 4. Validate All-Key mAP

Default YOLO26 uses the one-to-one end-to-end head, which is the NMS-free
deployment path.

```bash
yolo detect val \
  model=output/yolo26/mot17_all_key_yolo26s/weights/best.pt \
  data=datasets/yolo_mot17/data.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  project="$(pwd)/output/yolo26_val" \
  name=mot17_all_key_yolo26s
```

To check the traditional one-to-many head with NMS:

```bash
yolo detect val \
  model=output/yolo26/mot17_all_key_yolo26s/weights/best.pt \
  data=datasets/yolo_mot17/data.yaml \
  imgsz=640 \
  batch=16 \
  device=0 \
  end2end=False
```

## 5. Run Inference

Image-sequence inference, one YOLO call per evaluated frame:

```bash
yolo detect predict \
  model=output/yolo26/mot17_all_key_yolo26s/weights/best.pt \
  source=datasets/yolo_mot17/images/val \
  imgsz=640 \
  device=0 \
  save_txt=True \
  save_conf=True \
  project="$(pwd)/output/yolo26_pred" \
  name=mot17_all_key_yolo26s
```

Video inference at full incoming frame rate:

```bash
yolo detect predict \
  model=output/yolo26/virat30_all_key_yolo26s/weights/best.pt \
  source=../dataset/VIRAT/videos/VIRAT_S_000003.mp4 \
  imgsz=640 \
  device=0 \
  vid_stride=1
```

Video inference at 10 FPS from 30-FPS input:

```bash
yolo detect predict \
  model=output/yolo26/virat30_all_key_yolo26s/weights/best.pt \
  source=../dataset/VIRAT/videos/VIRAT_S_000003.mp4 \
  imgsz=640 \
  device=0 \
  vid_stride=3
```

## 6. Export For Jetson

Export the fine-tuned model to TensorRT on the target Jetson.

```bash
yolo export \
  model=output/yolo26/mot17_all_key_yolo26s/weights/best.pt \
  format=engine \
  imgsz=640 \
  half=True \
  device=0
```
