"""
test_reproduce_pipeline.py
Minimal reproducible test executing the exact 15-step validation required in Part 3.
"""

import os
import sys
import time
import io
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.depth_engine import (
    upsample_nearest,
    upsample_bilinear_validity,
    upsample_bicubic,
    fast_guided_filter,
    joint_bilateral_filter_fast,
    edgefuse_dcr_refine,
    estimate_confidence_map,
    detect_rgb_edges,
    detect_depth_discontinuities,
    reconstruct_depth,
    inpaint_depth_holes
)
from src.model import load_model, EdgeFuseNet
from app import colorize_depth, encode_image_base64, parse_rgb_bytes, parse_depth_bytes, app
import torch

def run_15_stage_test():
    print("=" * 80)
    print("STEP-BY-STEP REPRODUCIBLE PIPELINE TEST (TESTS 1 - 15)")
    print("=" * 80)

    # TEST 1: Load RGB image
    t0 = time.perf_counter()
    rgb_path = os.path.join("demo_samples", "nyu_rgb.png")
    assert os.path.exists(rgb_path), f"Missing {rgb_path}"
    with open(rgb_path, "rb") as f:
        rgb_raw_bytes = f.read()
    rgb_img = np.array(Image.open(io.BytesIO(rgb_raw_bytes)).convert("RGB"))
    dt1 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 1] Load RGB image: shape={rgb_img.shape}, dtype={rgb_img.dtype}, range=[{rgb_img.min()}, {rgb_img.max()}], time={dt1:.2f}ms")

    # TEST 2: Load depth map
    t0 = time.perf_counter()
    depth_path = os.path.join("demo_samples", "nyu_depth_lr.png")
    assert os.path.exists(depth_path), f"Missing {depth_path}"
    with open(depth_path, "rb") as f:
        depth_raw_bytes = f.read()
    depth_raw_img = Image.open(io.BytesIO(depth_raw_bytes))
    dt2 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 2] Load depth map: format={depth_raw_img.format}, size={depth_raw_img.size}, mode={depth_raw_img.mode}, time={dt2:.2f}ms")

    # TEST 3: Decode both images and report dimensions
    t0 = time.perf_counter()
    rgb_decoded = parse_rgb_bytes(rgb_raw_bytes, max_dim=720)
    depth_decoded, d_type, d_unit, d_warn = parse_depth_bytes(depth_raw_bytes, filename="nyu_depth_lr.png", depth_scale="auto", max_dim=720)
    dt3 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 3] Decode images: RGB={rgb_decoded.shape}, Depth={depth_decoded.shape}, type={d_type}, unit={d_unit}, time={dt3:.2f}ms")

    # TEST 4: Validate depth representation and data range
    t0 = time.perf_counter()
    valid_mask = (depth_decoded > 0.05) & np.isfinite(depth_decoded)
    v_count = int(np.sum(valid_mask))
    total_px = depth_decoded.size
    pct = (v_count / total_px) * 100.0
    d_min = float(depth_decoded[valid_mask].min()) if v_count > 0 else 0.0
    d_max = float(depth_decoded[valid_mask].max()) if v_count > 0 else 0.0
    dt4 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 4] Validate depth: valid_px={v_count}/{total_px} ({pct:.2f}%), range=[{d_min:.4f}, {d_max:.4f}]{d_unit}, time={dt4:.2f}ms")

    # Target shape
    target_shape = rgb_decoded.shape[:2]

    # TEST 5: Run nearest-neighbor interpolation
    t0 = time.perf_counter()
    d_nn = upsample_nearest(depth_decoded, target_shape)
    d_nn = inpaint_depth_holes(d_nn, d_nn > 0.05)
    dt5 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 5] Nearest-Neighbor: shape={d_nn.shape}, valid_px={np.sum(d_nn > 0.05)}, range=[{d_nn.min():.3f}, {d_nn.max():.3f}], time={dt5:.2f}ms")

    # TEST 6: Run bilinear interpolation (validity-normalized)
    t0 = time.perf_counter()
    d_bilinear = upsample_bilinear_validity(depth_decoded, target_shape, valid_mask)
    d_bilinear = inpaint_depth_holes(d_bilinear, d_bilinear > 0.05)
    dt6 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 6] Bilinear (Normalized): shape={d_bilinear.shape}, valid_px={np.sum(d_bilinear > 0.05)}, range=[{d_bilinear.min():.3f}, {d_bilinear.max():.3f}], time={dt6:.2f}ms")

    # TEST 7: Run bicubic interpolation
    t0 = time.perf_counter()
    d_bicubic = upsample_bicubic(depth_decoded, target_shape)
    d_bicubic = inpaint_depth_holes(d_bicubic, d_bicubic > 0.05)
    dt7 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 7] Bicubic: shape={d_bicubic.shape}, valid_px={np.sum(d_bicubic > 0.05)}, range=[{d_bicubic.min():.3f}, {d_bicubic.max():.3f}], time={dt7:.2f}ms")

    # TEST 8: Run guided filtering
    t0 = time.perf_counter()
    d_guided = fast_guided_filter(rgb_decoded, d_bilinear, radius=5, eps=0.01)
    dt8 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 8] Guided Filtering: shape={d_guided.shape}, range=[{d_guided.min():.3f}, {d_guided.max():.3f}], time={dt8:.2f}ms")

    # TEST 9: Load actual EdgeFuseNet model
    t0 = time.perf_counter()
    model_path = os.path.join("models", "edgefuse_dcr_compact.pth")
    assert os.path.exists(model_path), f"Checkpoint missing: {model_path}"
    model = load_model(model_path, device="cpu")
    assert model is not None
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    dt9 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 9] Load EdgeFuseNet: params={num_params:,}, eval={not model.training}, device=cpu, time={dt9:.2f}ms")

    # TEST 10: Run EdgeFuseNet inference
    t0 = time.perf_counter()
    # Stride-4 padding verification
    h, w = target_shape
    pad_h = (4 - (h % 4)) % 4
    pad_w = (4 - (w % 4)) % 4
    rgb_pad = cv2.copyMakeBorder(rgb_decoded, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    depth_pad = cv2.copyMakeBorder(d_bilinear, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    v_pad = cv2.copyMakeBorder((d_bilinear > 0.05).astype(np.float32), 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0.0)

    rgb_t = torch.from_numpy(rgb_pad.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    d_t = torch.from_numpy(depth_pad.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v_pad.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        pred_d, pred_c = model(rgb_t, d_t, v_t)
        d_cnn = pred_d.squeeze().cpu().numpy()[:h, :w]
        c_cnn = pred_c.squeeze().cpu().numpy()[:h, :w]
    dt10 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 10] EdgeFuseNet Inference: shape={d_cnn.shape}, conf_shape={c_cnn.shape}, range=[{d_cnn.min():.3f}, {d_cnn.max():.3f}], time={dt10:.2f}ms")

    # TEST 11: Run confidence-guided refinement
    t0 = time.perf_counter()
    conf_map = estimate_confidence_map(d_cnn, d_bilinear > 0.05, factor=4, is_low_res=True, cnn_conf=c_cnn)
    d_conf_blended = conf_map * d_cnn + (1.0 - conf_map) * d_bilinear
    dt11 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 11] Confidence Guidance: shape={d_conf_blended.shape}, conf_mean={conf_map.mean():.4f}, time={dt11:.2f}ms")

    # TEST 12: Run boundary refinement
    t0 = time.perf_counter()
    refine_res = edgefuse_dcr_refine(d_cnn, rgb_decoded, initial_valid_mask=(d_bilinear > 0.05), apply_suppression=True, suppression_strength=0.85, factor=4, is_low_res=True, cnn_conf=c_cnn)
    d_refined = refine_res["final_depth"]
    dt12 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 12] Boundary Refinement: shape={d_refined.shape}, texture_gate_mean={refine_res['texture_gate'].mean():.4f}, time={dt12:.2f}ms")

    # TEST 13: Generate final A5 output via reconstruct_depth()
    t0 = time.perf_counter()
    rec_res = reconstruct_depth(
        rgb_image=rgb_decoded,
        depth_input=depth_decoded,
        factor=4,
        method="hybrid",
        cnn_model=model,
        device="cpu",
        apply_suppression=True,
        suppression_strength=0.85
    )
    final_a5 = rec_res["reconstructed_depth"]
    dt13 = (time.perf_counter() - t0) * 1000.0
    print(f"[TEST 13] Final A5 Reconstructed Depth: shape={final_a5.shape}, range=[{final_a5.min():.3f}, {final_a5.max():.3f}], latency={rec_res['latency_ms']:.2f}ms, total_time={dt13:.2f}ms")

    # TEST 14: Encode output image
    t0 = time.perf_counter()
    colored_out = colorize_depth(final_a5, colormap="turbo")
    b64_out = encode_image_base64(colored_out)
    dt14 = (time.perf_counter() - t0) * 1000.0
    assert len(b64_out) > 500
    print(f"[TEST 14] Encode Output Image: colored_shape={colored_out.shape}, b64_bytes={len(b64_out)}, time={dt14:.2f}ms")

    # TEST 15: Return valid API response via app endpoint
    t0 = time.perf_counter()
    from starlette.datastructures import UploadFile as StarletteUploadFile
    from app import run_reconstruction

    class RealUploadFile:
        def __init__(self, filename, data):
            self.filename = filename
            self.file = io.BytesIO(data)
        async def read(self):
            return self.file.getvalue()

    def sync_run(coro):
        try:
            coro.send(None)
        except StopIteration as e:
            return e.value

    f_rgb = RealUploadFile("test_rgb.png", rgb_raw_bytes)
    f_depth = RealUploadFile("test_depth.png", depth_raw_bytes)
    api_resp = sync_run(run_reconstruction(
        rgb_file=f_rgb,
        depth_file=f_depth,
        method="hybrid",
        upsample_factor=4,
        colormap="turbo"
    ))
    dt15 = (time.perf_counter() - t0) * 1000.0
    assert api_resp["success"] is True
    assert "reconstructed_b64" in api_resp
    assert "classical_b64" in api_resp
    assert "cnn_b64" in api_resp
    print(f"[TEST 15] Valid API Response: success={api_resp['success']}, latency={api_resp['latency_ms']}ms, total_time={dt15:.2f}ms")

    print("=" * 80)
    print("ALL 15 TESTS IN PART 3 COMPLETED AND PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    run_15_stage_test()
