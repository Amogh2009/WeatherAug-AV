import albumentations as A
import random
import cv2
import numpy as np

SEVERITY_LEVELS = {
    "mild": {"rain_drop_length": 10, "rain_drop_width": 1, "snow_point_lower": 0.01, "snow_point_upper": 0.05, "blur_limit": 3},
    "moderate": {"rain_drop_length": 20, "rain_drop_width": 2, "snow_point_lower": 0.05, "snow_point_upper": 0.15, "blur_limit": 5},
    "severe": {"rain_drop_length": 30, "rain_drop_width": 3, "snow_point_lower": 0.15, "snow_point_upper": 0.30, "blur_limit": 7}
}

def add_rain(image, severity="mild"):
    params = SEVERITY_LEVELS[severity]
    rain = A.RandomRain(
        drop_length=params["rain_drop_length"],
        drop_width=params["rain_drop_width"],
        blur_value=params["blur_limit"],
        brightness_coefficient=0.9,
        p=1.0
    )
    return rain(image=image)["image"]

def add_snow(image, severity="mild"):
    params = SEVERITY_LEVELS[severity]
    snow = A.RandomSnow(
        snow_point_range=(params["snow_point_lower"], params["snow_point_upper"]),
        brightness_coeff=2.0,
        p=1.0
    )
    return snow(image=image)["image"]

def get_severity_for_epoch(epoch, total_epochs=50):
    progress = epoch / total_epochs
    if progress < 0.33:
        return "mild"
    elif progress < 0.66:
        return "moderate"
    else:
        return "severe"

def weatheraug(image, epoch=None, severity=None, total_epochs=50, p_apply=0.8):
    """
    WeatherAug: Curriculum-based weather augmentation for AV object detection.
    
    Args:
        image: numpy array (RGB)
        epoch: current training epoch (for curriculum mode)
        severity: fixed severity override (mild/moderate/severe)
        total_epochs: total training epochs
        p_apply: probability of applying augmentation
    
    Returns:
        augmented image as numpy array
    """
    if random.random() > p_apply:
        return image
    
    if severity is None and epoch is not None:
        severity = get_severity_for_epoch(epoch, total_epochs)
    elif severity is None:
        severity = random.choice(["mild", "moderate", "severe"])
    
    weather_type = random.choice(["rain", "snow"])
    if weather_type == "rain":
        return add_rain(image, severity)
    else:
        return add_snow(image, severity)
