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
INT8 is supported for key and non-key engines with saved calibration samples.
For non-key, collect samples from real key-engine cache tensors. INT4 is
non-key-only, and it requires explicit ONNX Q/DQ weight-only quantization
before TensorRT engine build.
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


class _NpzCalibrationDataReader:
    """Minimal ModelOpt/ORT calibration reader for saved model input blobs."""

    def __init__(self, calibration_files, max_batches=0):
        self.calibration_files = list(calibration_files)
        if max_batches and max_batches > 0:
            self.calibration_files = self.calibration_files[:max_batches]
        self.batch_index = 0

    def get_next(self):
        if self.batch_index >= len(self.calibration_files):
            return None

        sample_path = self.calibration_files[self.batch_index]
        self.batch_index += 1

        with np.load(sample_path) as sample:
            return {
                name: np.ascontiguousarray(np.asarray(sample[name]))
                for name in sample.files
            }

    def rewind(self):
        self.batch_index = 0

    def __iter__(self):
        self.rewind()
        return self

    def __next__(self):
        sample = self.get_next()
        if sample is None:
            raise StopIteration
        return sample


def _default_int4_onnx_path(onnx_path):
    path = Path(onnx_path).expanduser().resolve()
    return path.with_name(f"{path.stem}_int4{path.suffix}")


def _summarize_quantized_onnx(onnx_path):
    try:
        import onnx
    except ImportError:
        print("[WARN] onnx package unavailable; skipping INT4 Q/DQ graph summary.")
        return
    if not hasattr(onnx, "load"):
        print("[WARN] imported onnx module has no load(); skipping INT4 Q/DQ graph summary.")
        return

    model = onnx.load(str(onnx_path))
    counts = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    quantize_count = counts.get("QuantizeLinear", 0)
    dequantize_count = counts.get("DequantizeLinear", 0)
    matmul_count = counts.get("MatMul", 0)
    print(
        f"[INFO] INT4 ONNX summary: MatMul={matmul_count}, "
        f"QuantizeLinear={quantize_count}, DequantizeLinear={dequantize_count}"
    )
    if quantize_count == 0 and dequantize_count == 0:
        raise RuntimeError(
            "INT4 ONNX does not contain Q/DQ nodes. Check ModelOpt support, "
            "calibration data, and node exclusion filters."
        )


def _find_trt_int4_unsupported_weight_nodes(onnx_path):
    """Return MatMul/Gemm node names whose constant weight shape TensorRT cannot pack as INT4."""
    try:
        import onnx
    except ImportError:
        print("[WARN] onnx package unavailable; cannot auto-exclude unsupported INT4 weight nodes.")
        return []

    model = onnx.load(str(Path(onnx_path).expanduser().resolve()))
    initializer_shapes = {
        initializer.name: tuple(int(dim) for dim in initializer.dims)
        for initializer in model.graph.initializer
    }
    unsupported = []
    for node in model.graph.node:
        if node.op_type not in {"MatMul", "Gemm"} or not node.name:
            continue
        for input_name in node.input:
            shape = initializer_shapes.get(input_name)
            if shape and shape[-1] % 2 != 0:
                unsupported.append(node.name)
                break
    return sorted(set(unsupported))


def _quantize_int4_onnx(
    onnx_path,
    output_path,
    calib_data,
    calib_batches,
    calib_method,
    block_size,
    nodes_to_exclude,
    auto_exclude_unsupported=True,
):
    try:
        from modelopt.onnx.quantization import quantize as modelopt_quantize
    except ImportError as exc:
        raise RuntimeError(
            "INT4 ONNX quantization requires NVIDIA ModelOpt. Install it on a "
            "quantization/build environment with: pip install -U \"nvidia-modelopt[onnx]\". "
            "If the input ONNX is already INT4 Q/DQ quantized, use "
            "--int4_prequantized to skip ModelOpt on this machine."
        ) from exc

    calibration_files = _list_calibration_files(calib_data)
    if not calibration_files:
        raise RuntimeError("--int4 requires --calib_data with non-key .npz calibration samples.")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_reader = _NpzCalibrationDataReader(calibration_files, max_batches=calib_batches)
    nodes_to_exclude = list(nodes_to_exclude or [])
    if auto_exclude_unsupported:
        auto_excluded = _find_trt_int4_unsupported_weight_nodes(onnx_path)
        for node_name in auto_excluded:
            if node_name not in nodes_to_exclude:
                nodes_to_exclude.append(node_name)
        if auto_excluded:
            print(
                "[INFO] Auto-excluding TensorRT-unsupported INT4 weight node(s): "
                + ", ".join(auto_excluded)
            )

    print(
        f"[INFO] Quantizing non-key ONNX to INT4 Q/DQ -> {output_path} "
        f"({len(calibration_reader.calibration_files)} calibration sample(s), "
        f"method={calib_method}, block_size={block_size})."
    )
    modelopt_quantize(
        str(Path(onnx_path).expanduser().resolve()),
        quantize_mode="int4",
        calibration_method=calib_method,
        calibration_data_reader=calibration_reader,
        output_path=str(output_path),
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=nodes_to_exclude or None,
        high_precision_dtype="fp32",
        block_size=block_size,
        opset=21,
    )
    if not output_path.exists():
        raise RuntimeError(f"ModelOpt did not write expected INT4 ONNX file: {output_path}")
    _summarize_quantized_onnx(output_path)
    return output_path


if _INT8_CALIBRATOR_BASE is not None:
    class _EntropyCalibrator(_INT8_CALIBRATOR_BASE):
        """TensorRT PTQ calibrator backed by saved model input blobs."""

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
    use_int4=False,
    calib_data=None,
    calib_cache=None,
    calib_batches=0,
    int4_onnx=None,
    int4_mode="qdq",
    int4_prequantized=False,
    int4_onnx_only=False,
    allow_native_int4=False,
    int4_calib_method="awq_lite",
    int4_block_size=128,
    int4_nodes_to_exclude=None,
    int4_auto_exclude_unsupported=True,
):
    if use_int8 and use_int4:
        raise RuntimeError("--int8 and --int4 are mutually exclusive.")
    if use_int4:
        if model_type != "nonkey":
            raise RuntimeError("INT4 export is currently supported only for the non-key engine.")
        if use_fp16:
            raise RuntimeError("--int4 should not be combined with --fp16 for FP32-key/INT4-nonkey experiments.")
        if int4_mode == "native" and not allow_native_int4:
            raise RuntimeError(
                "Native TensorRT BuilderFlag.INT4 is disabled by default because it produced "
                "FP32-like engines for this non-key graph. Use --int4_mode qdq with a Q/DQ "
                "ONNX, or pass --allow_native_int4 only for diagnostics."
            )
    if use_int4 and int4_mode == "qdq":
        if int4_onnx_only and int4_prequantized:
            raise RuntimeError("--int4_onnx_only cannot be combined with --int4_prequantized.")
        if int4_prequantized:
            _summarize_quantized_onnx(Path(onnx_path).expanduser().resolve())
        else:
            int4_onnx = int4_onnx or _default_int4_onnx_path(onnx_path)
            onnx_path = str(_quantize_int4_onnx(
                onnx_path,
                int4_onnx,
                calib_data,
                calib_batches,
                int4_calib_method,
                int4_block_size,
                int4_nodes_to_exclude,
                int4_auto_exclude_unsupported,
            ))
        if int4_onnx_only:
            print(f"[INFO] Wrote INT4 Q/DQ ONNX to {onnx_path}; skipping TensorRT engine build.")
            return

    if trt is None:
        raise RuntimeError("TensorRT Python bindings are required to build engines. Run this on the Jetson target.")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')

    builder = trt.Builder(logger)
    network_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if use_int4 and int4_mode == "qdq" and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        print("[INFO] Strongly typed TensorRT network enabled for INT4 Q/DQ ONNX.")
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
    elif use_int4:
        if int4_mode == "qdq":
            print("[INFO] Building explicit INT4 Q/DQ engine; non-quantized tensors remain FP32.")
        else:
            print("[INFO] FP16 optimization disabled. Native INT4 diagnostic may keep tensors FP32.")
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
        if (
            model_type == "key"
            and hasattr(trt, "QuantizationFlag")
            and hasattr(trt.QuantizationFlag, "CALIBRATE_BEFORE_FUSION")
            and hasattr(config, "set_quantization_flag")
        ):
            config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            print("[INFO] INT8 calibration-before-fusion enabled for key engine.")
        config.int8_calibrator = _EntropyCalibrator(
            calibration_files,
            _network_input_dtypes(network),
            cache_path=cache_path,
            max_batches=calib_batches,
        )
        if hasattr(config, "set_calibration_profile"):
            config.set_calibration_profile(profile)
        print(
            f"[INFO] INT8 PTQ enabled for {model_type} engine "
            f"({len(calibration_files)} calibration sample(s), cache={cache_path})."
        )
    elif use_int4:
        if int4_mode == "native":
            if not hasattr(trt.BuilderFlag, "INT4"):
                raise RuntimeError("This TensorRT build does not expose BuilderFlag.INT4.")
            config.set_flag(trt.BuilderFlag.INT4)
            print("[INFO] Native TensorRT INT4 enabled for non-key engine.")
        else:
            print("[INFO] INT4 Q/DQ ONNX detected; building strongly typed explicit-quantized non-key engine.")

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
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument("--int8", action="store_true",
                                 help="Build a calibrated INT8 key or non-key engine")
    precision_group.add_argument("--int4", action="store_true",
                                 help="Build an INT4 non-key engine")
    parser.add_argument("--calib_data", type=str, default=None,
                        help="Directory or .npz file of calibration samples")
    parser.add_argument("--calib_cache", type=str, default=None,
                        help="TensorRT INT8 calibration cache path")
    parser.add_argument("--calib_batches", type=int, default=0,
                        help="Maximum calibration samples to consume; <=0 uses all")
    parser.add_argument("--int4_onnx", type=str, default=None,
                        help="Output path for the generated INT4 Q/DQ ONNX model")
    parser.add_argument("--int4_mode", type=str, default="qdq", choices=("qdq", "native"),
                        help="qdq builds explicit INT4 Q/DQ ONNX; native is diagnostic only")
    parser.add_argument("--int4_prequantized", action="store_true",
                        help="Treat --onnx as an already quantized INT4 Q/DQ model and skip ModelOpt")
    parser.add_argument("--int4_onnx_only", action="store_true",
                        help="Generate INT4 Q/DQ ONNX and exit without building a TensorRT engine")
    parser.add_argument("--allow_native_int4", action="store_true",
                        help="Allow diagnostic TensorRT BuilderFlag.INT4 without Q/DQ; not real INT4 for this graph")
    parser.add_argument("--int4_calib_method", type=str, default="awq_lite",
                        choices=("awq_lite", "awq_clip"),
                        help="ModelOpt INT4 calibration method")
    parser.add_argument("--int4_block_size", type=int, default=128, choices=(64, 128),
                        help="INT4 block size for ModelOpt weight-only quantization")
    parser.add_argument("--int4_nodes_to_exclude", nargs="*", default=[],
                        help="ModelOpt node-name substrings or regexes to exclude from INT4 quantization")
    parser.add_argument("--no_int4_auto_exclude_unsupported", action="store_true",
                        help="Disable automatic exclusion of TensorRT-unsupported odd-final-dim INT4 weight nodes")
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
        args.int4,
        args.calib_data,
        args.calib_cache,
        args.calib_batches,
        args.int4_onnx,
        args.int4_mode,
        args.int4_prequantized,
        args.int4_onnx_only,
        args.allow_native_int4,
        args.int4_calib_method,
        args.int4_block_size,
        args.int4_nodes_to_exclude,
        not args.no_int4_auto_exclude_unsupported,
    )
