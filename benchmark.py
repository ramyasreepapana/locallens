"""
Benchmark: measures REAL timings on whatever machine runs it.

    python benchmark.py                        # CPU, 5 runs on the sample image
    python benchmark.py --backend qnn          # only works on a Snapdragon PC with onnxruntime-qnn
    python benchmark.py --image my.png --runs 10 --output results.json

Nothing is pre-filled. Numbers from one machine are not comparable to another.
"""

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

from PIL import Image

from ocr_engine import LocalOCR, OCRError

SAMPLE = Path(__file__).parent / "sample_data" / "sample_invoice.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", choices=["cpu", "qnn"])
    parser.add_argument("--image", default=str(SAMPLE))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    engine = LocalOCR(args.backend)
    try:
        engine.load()
        image = Image.open(args.image)
        engine.run(image)                      # warm-up run (not counted)
        timings = [engine.run(image).seconds for _ in range(args.runs)]
    except OCRError as err:
        print(f"Benchmark could not run: {err}")
        return 1

    results = {
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "processor": platform.processor(), "python": platform.python_version()},
        "engine": engine.describe(),
        "image": Path(args.image).name,
        "model_load_seconds": round(engine.load_seconds, 3),
        "inference_seconds_per_run": [round(t, 3) for t in timings],
        "inference_seconds_mean": round(statistics.mean(timings), 3),
        "inference_seconds_min": round(min(timings), 3),
        "note": "Measured on this machine only (first warm-up run excluded).",
    }
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
