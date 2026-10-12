#!/usr/bin/env python
"""Measure FPS, latency and #Params of CytoSPM.

Protocol: batch 1, 640x640 input, fp32, 50 warm-up and 300 timed iterations
of ``model.test_step`` on one pre-built batch (pre-processing, forward and
NMS; no disk IO or data pipeline). Class-name embeddings are cached.

    python tools/bench_fps.py --config configs/cytospm.py \
        --checkpoint checkpoints/cytospm_convnext_t.pth --method cytospm \
        --out work_dirs/fps.json
"""
import argparse, json, time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules

from cytospm.utils import resolve_eval_checkpoint


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--allow-unsafe-checkpoint", action="store_true",
        help="allow unrestricted deserialization for a trusted internal checkpoint")
    p.add_argument("--method", required=True, help="name written into the JSON")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    p.add_argument("--nms-included", type=int, default=1,
                   help="record whether postproc/NMS is part of the timed forward")
    return p.parse_args()


def main():
    args = parse()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    checkpoint = (
        resolve_eval_checkpoint(
            args.checkpoint,
            allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
        )
        if args.checkpoint is not None else None
    )
    # palette='random' keeps init_detector from building the dataset.
    model = init_detector(cfg, checkpoint, device=args.device, palette='random')
    model.eval()

    # One input batch from a random image; the organ is parsed from its path.
    import cv2, tempfile
    dummy_path = Path(tempfile.gettempdir()) / "cytospm_bench" / "respiratory_tract" / "dummy_640.jpg"
    dummy_path.parent.mkdir(parents=True, exist_ok=True)
    if not dummy_path.exists():
        cv2.imwrite(str(dummy_path),
                    np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8))
    test_pipeline = Compose(cfg.test_pipeline)
    data_info = dict(img_id=0, img_path=str(dummy_path),
                     ori_shape=(args.img_size, args.img_size),
                     img_shape=(args.img_size, args.img_size))
    txt_path = cfg.get("class_text_path", None)
    if txt_path and Path(txt_path).exists():
        data_info["texts"] = json.loads(Path(txt_path).read_text())
    data = test_pipeline(data_info)
    batch = dict(inputs=data["inputs"].unsqueeze(0).to(args.device),
                 data_samples=[data["data_samples"]])

    @torch.no_grad()
    def step():
        model.test_step(batch)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    fps = args.iters / dt
    latency_ms = 1000.0 * dt / args.iters

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = None
    try:
        from mmengine.analysis import get_model_complexity_info
        ana = get_model_complexity_info(
            model, inputs=batch["inputs"], show_table=False, show_arch=False)
        gflops = float(ana["flops"]) / 1e9
    except Exception as e:
        print(f"[flops] skipped ({type(e).__name__}: {e})")

    out = {
        "method": args.method, "fps": round(fps, 2), "latency_ms": round(latency_ms, 2),
        "params_M": round(params_m, 2), "gflops": round(gflops, 2) if gflops else None,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "precision": "fp32", "img_size": args.img_size,
        "warmup": args.warmup, "iters": args.iters, "nms_included": bool(args.nms_included),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
