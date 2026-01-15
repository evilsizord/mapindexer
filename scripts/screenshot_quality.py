from PIL import Image
import numpy as np

# -----------------------------
# Tunable thresholds
# -----------------------------

MAX_SKY_RATIO = 0.55
MAX_WALL_RATIO = 0.65
MIN_EDGE_DENSITY = 0.015
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 215

# -----------------------------
# Image loading
# -----------------------------

def load_image(path, resize=256):
    img = Image.open(path).convert("RGB")
    img = img.resize((resize, resize))
    return np.asarray(img)

# -----------------------------
# Sky detection
# -----------------------------

def sky_ratio(img):
    """
    Skyboxes tend to be:
    - Blue / light
    - Low texture variance
    - Upper portion of image
    """
    h = img.shape[0]
    top = img[: h // 3]

    r, g, b = top[..., 0], top[..., 1], top[..., 2]

    blueish = (b > r) & (b > g)
    bright = (r + g + b) / 3 > 150

    sky_pixels = blueish & bright
    return sky_pixels.mean()

# -----------------------------
# Flat wall detection
# -----------------------------

def wall_ratio(img):
    """
    Walls tend to be:
    - Large uniform color areas
    - Low gradient magnitude
    """
    gray = img.mean(axis=2)

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))

    grad = np.pad(gx[:-1] + gy[:, :-1], ((0,1),(0,1)))

    flat = grad < 5
    return flat.mean()

# -----------------------------
# Edge density (detail check)
# -----------------------------

def edge_density(img):
    gray = img.mean(axis=2)

    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))

    edges = (gx[:-1] + gy[:, :-1]) > 20
    return edges.mean()

# -----------------------------
# Brightness check
# -----------------------------

def mean_brightness(img):
    return img.mean()

# -----------------------------
# Master evaluator
# -----------------------------

def evaluate_image(path):
    img = load_image(path)

    metrics = {
        "sky_ratio": sky_ratio(img),
        "wall_ratio": wall_ratio(img),
        "edge_density": edge_density(img),
        "brightness": mean_brightness(img)
    }

    reject = (
        metrics["sky_ratio"] > MAX_SKY_RATIO or
        metrics["wall_ratio"] > MAX_WALL_RATIO or
        metrics["edge_density"] < MIN_EDGE_DENSITY or
        metrics["brightness"] < MIN_BRIGHTNESS or
        metrics["brightness"] > MAX_BRIGHTNESS
    )

    return not reject, metrics


"""
Example Usage (with retry):

def try_capture(camera, capture_func):
    attempts = [
        camera,
        adjust(camera, pitch=-10),
        adjust(camera, yaw=30),
        adjust(camera, yaw=-30),
        adjust(camera, z=-64)
    ]

    for cam in attempts:
        img_path = capture_func(cam)
        ok, _ = evaluate_image(img_path)
        if ok:
            return img_path

    return None


"""