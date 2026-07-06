"""
WeatherAug-AV: Evaluation script for weather robustness benchmarking
"""
from ultralytics import YOLO
import argparse

def evaluate(model_path, config):
    model = YOLO(model_path)
    results = model.val(data=config, device=0)
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--config", required=True, help="Path to eval YAML config")
    args = parser.parse_args()
    evaluate(args.model, args.config)
