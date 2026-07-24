"""EXP 3 ONNX-to-TensorRT engine builder for Jetson batch-1 deployment.

Build the key and non-key engines separately on the Jetson target:

    python rtdetrv2_pytorch/tools/export_trt.py \
      -i onnx/key_model.onnx \
      -o engines/key_fp32.engine \
      -m key

    python rtdetrv2_pytorch/tools/export_trt.py \
      -i onnx/nonkey_model.onnx \
      -o engines/nonkey_fp32.engine \
      -m nonkey

FP16 is opt-in. Add --fp16 only for explicit reduced-precision experiments.
INT8 is supported for the non-key engine with calibration samples collected
from real key-engine cache tensors.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import tensorrt as trt
except ModuleNotFoundError:
    trt = None


if trt is not None:
    _INT8_CALIBRATOR_BASE = getattr(trt, "IInt8EntropyCalibrator2", None)
else:
    _INT8_CALIBRATOR_BASE = None


def _set_workspace_size(config, workspace_mb: int):
    workspace_bytes = int(workspace_mb) << 20
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def _build_shape_for_batch(name, shape, batch_size, image_h, image_w):
    resolved = []
    for axis, dim in enumerate(shape):
        d = int(dim)
        if axis == 0:
            resolved.append(batch_size if d == -1 else d)
            continue
        if d != -1:
            resolved.append(d)
            continue
        if name == 'images' and axis == 2:
            resolved.append(image_h)
        elif name == 'images' and axis == 3:
            resolved.append(image_w)
        else:
            raise ValueError(
                f"Unsupported dynamic dimension in input '{name}' at axis {axis}. "
                "Export ONNX with fixed non-batch dims, or extend this script."
            )
    return tuple(resolved)


def _validate_temporal_inputs(network, model_type):
    input_names = {network.get_input(i).name for i in range(network.num_inputs)}
    if model_type == 'nonkey':
        required = {'images', 'orig_target_sizes', 'cache_ccff_0', 'cache_ccff_1', 'cache_ccff_2', 'cache_content', 'cache_points'}
        missing = sorted(required - input_names)
        if missing:
            raise RuntimeError(
                f"Non-key ONNX is missing required inputs: {missing}. "
                "Expected inputs: images, orig_target_sizes, and all cache_* tensors."
            )
    elif model_type == 'key':
        required = {'images', 'orig_target_sizes'}
        missing = sorted(required - input_names)
        if missing:
            raise RuntimeError(f"Key ONNX is missing required inputs: {missing}")


def _list_calibration_files(calib_data):
    if calib_data is None:
        return []

    path = Path(calib_data).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Calibration file must be .npz, got: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Calibration data path does not exist: {path}")

    files = sorted(path.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz calibration samples found in: {path}")
    return files


def _network_input_dtypes(network):
    return {
        network.get_input(i).name: trt.nptype(network.get_input(i).dtype)
        for i in range(network.num_inputs)
    }


if _INT8_CALIBRATOR_BASE is not None:
    class _EntropyCalibrator(_INT8_CALIBRATOR_BASE):
        """TensorRT PTQ calibrator backed by saved non-key input blobs."""

        def __init__(self, calibration_files, input_dtypes, cache_path=None, max_batches=0):
            super().__init__()
            self.calibration_files = list(calibration_files)
            if max_batches and max_batches > 0:
                self.calibration_files = self.calibration_files[:max_batches]
            self.input_dtypes = dict(input_dtypes)
            self.cache_path = Path(cache_path).expanduser().resolve() if cache_path else None
            self.batch_index = 0
            self.device_tensors = {}
            self.batch_size = self._infer_batch_size()

            if self.calibration_files and not torch.cuda.is_available():
                raise RuntimeError("CUDA is required to run TensorRT INT8 calibration batches.")

        def _infer_batch_size(self):
            if not self.calibration_files:
                return 1
            with np.load(self.calibration_files[0]) as sample:
                for name in self.input_dtypes:
                    if name in sample:
                        value = sample[name]
                        if value.ndim > 0:
                            return int(value.shape[0])
            return 1

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.batch_index >= len(self.calibration_files):
                return None

            sample_path = self.calibration_files[self.batch_index]
            self.batch_index += 1
            self.device_tensors = {}
            bindings = []

            with np.load(sample_path) as sample:
                for name in names:
                    if name not in sample:
                        raise RuntimeError(
                            f"Calibration sample {sample_path} is missing required input '{name}'."
                        )
                    array = np.asarray(sample[name])
                    expected_dtype = self.input_dtypes.get(name)
                    if expected_dtype is not None and array.dtype != expected_dtype:
                        array = array.astype(expected_dtype, copy=False)
                    array = np.ascontiguousarray(array)
                    tensor = torch.from_numpy(array).contiguous().cuda()
                    self.device_tensors[name] = tensor
                    bindings.append(int(tensor.data_ptr()))

            return bindings

        def read_calibration_cache(self):
            if self.cache_path is not None and self.cache_path.exists():
                with self.cache_path.open("rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_path is None:
                return
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("wb") as f:
                f.write(bytes(cache))
else:
    _EntropyCalibrator = None


def main(
    onnx_path,
    engine_path,
    model_type,
    max_batchsize,
    opt_batchsize,
    min_batchsize,
    use_fp16=False,
    verbose=False,
    workspace_mb=1024,
    image_h=640,
    image_w=640,
    use_int8=False,
    calib_data=None,
    calib_cache=None,
    calib_batches=0,
):
    if trt is None:
        raise RuntimeError("TensorRT Python bindings are required to build engines. Run this on the Jetson target.")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[INFO] Loading ONNX file from {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")
    _validate_temporal_inputs(network, model_type)

    config = builder.create_builder_config()
    if hasattr(config, 'set_preview_feature') and hasattr(trt, 'PreviewFeature') and \
            hasattr(trt.PreviewFeature, 'FASTER_DYNAMIC_SHAPES_0805'):
        config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)
    _set_workspace_size(config, workspace_mb)

    if use_fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 optimization enabled.")
    elif use_fp16:
        print("[INFO] FP16 requested, but platform does not support fast FP16. Falling back to FP32.")
    else:
        print("[INFO] FP16 optimization disabled. Building FP32 TensorRT baseline.")

    profile = builder.create_optimization_profile()
    print("[INFO] Applying optimization profile:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = tuple(tensor.shape)
        min_shape = _build_shape_for_batch(tensor.name, shape, min_batchsize, image_h, image_w)
        opt_shape = _build_shape_for_batch(tensor.name, shape, opt_batchsize, image_h, image_w)
        max_shape = _build_shape_for_batch(tensor.name, shape, max_batchsize, image_h, image_w)
        profile.set_shape(tensor.name, min=min_shape, opt=opt_shape, max=max_shape)
        print(f"  - {tensor.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

    config.add_optimization_profile(profile)

    if use_int8:
        if model_type != "nonkey":
            raise RuntimeError("INT8 export is currently supported only for the non-key engine.")
        if _EntropyCalibrator is None:
            raise RuntimeError("This TensorRT build does not expose IInt8EntropyCalibrator2.")

        calibration_files = _list_calibration_files(calib_data)
        cache_path = Path(calib_cache).expanduser().resolve() if calib_cache else None
        if not calibration_files and (cache_path is None or not cache_path.exists()):
            raise RuntimeError(
                "--int8 requires --calib_data with .npz samples, or an existing --calib_cache."
            )

        if hasattr(builder, "platform_has_fast_int8") and not builder.platform_has_fast_int8:
            print("[WARN] INT8 requested, but this platform does not report fast INT8 support.")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _EntropyCalibrator(
            calibration_files,
            _network_input_dtypes(network),
            cache_path=cache_path,
            max_batches=calib_batches,
        )
        if hasattr(config, "set_calibration_profile"):
            config.set_calibration_profile(profile)
        print(
            f"[INFO] INT8 PTQ enabled for non-key engine "
            f"({len(calibration_files)} calibration sample(s), cache={cache_path})."
        )

    print("[INFO] Building TensorRT engine...")
    Path(engine_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    if hasattr(builder, 'build_serialized_network'):
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized TensorRT engine. Check unsupported nodes/plugins.")
        with open(engine_path, "wb") as f:
            f.write(bytes(serialized_engine))
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine. Check unsupported nodes/plugins.")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

    print("[INFO] Engine export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT Engine")
    parser.add_argument("--onnx", "-i", type=str, required=True, help="Path to input ONNX model file")
    parser.add_argument("--saveEngine", "-o", type=str, required=True, help="Path to output TensorRT engine file")
    parser.add_argument("--model_type", "-m", type=str, choices=['key', 'nonkey'], required=True, help="Which engine is being built?")
    parser.add_argument("--maxBatchSize", type=int, default=1)
    parser.add_argument("--optBatchSize", type=int, default=1)
    parser.add_argument("--minBatchSize", type=int, default=1)
    parser.add_argument("--workspaceMB", type=int, default=1024, help="TensorRT workspace size in MB")
    parser.add_argument("--inputH", type=int, default=640, help="Fallback input image height for dynamic ONNX dims")
    parser.add_argument("--inputW", type=int, default=640, help="Fallback input image width for dynamic ONNX dims")
    parser.add_argument("--fp16", dest="fp16", action="store_true",
                        help="Enable FP16 optimization. Default is FP32.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="Disable FP16 optimization and build the FP32 baseline (default).")
    parser.add_argument("--int8", action="store_true",
                        help="Build a calibrated INT8 non-key engine")
    parser.add_argument("--calib_data", type=str, default=None,
                        help="Directory or .npz file of non-key calibration samples")
    parser.add_argument("--calib_cache", type=str, default=None,
                        help="TensorRT INT8 calibration cache path")
    parser.add_argument("--calib_batches", type=int, default=0,
                        help="Maximum calibration samples to consume; <=0 uses all")
    parser.add_argument("--verbose", action="store_true", help="Enable TensorRT verbose logs")
    parser.set_defaults(fp16=False)

    args = parser.parse_args()
    main(
        args.onnx,
        args.saveEngine,
        args.model_type,
        args.maxBatchSize,
        args.optBatchSize,
        args.minBatchSize,
        args.fp16,
        args.verbose,
        args.workspaceMB,
        args.inputH,
        args.inputW,
        args.int8,
        args.calib_data,
        args.calib_cache,
        args.calib_batches,
    )
