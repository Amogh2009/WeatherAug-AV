"""
WeatherAug-AV: Training script for weather-robust object detection on BDD100K
"""
from ultralytics import YOLO
import argparse

def train(config, model_size="s", epochs=50, batch=16, name="experiment"):
    model = YOLO(f"yolov8{model_size}.pt")
    model.train(
        data=config,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=0,
        name=name
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--model", default="s", help="YOLOv8 model size (n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="experiment")
    args = parser.parse_args()
    train(args.config, args.model, args.epochs, args.batch, args.name)
