import argparse
import os
import time

import numpy as np
import scipy.io as sio
import torch
from scipy.ndimage import gaussian_filter

from utils.checkpoint import load_unet


def start_points(size, split_size, overlap):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1

    while True:
        point = stride * counter

        if point + split_size >= size:
            points.append(size - split_size)
            break

        points.append(point)
        counter += 1

    return points


def get_gaussian(shape, sigma=1.0 / 8):
    gaussian = np.zeros(shape)
    coords = [size // 2 for size in shape]
    sigmas = [size * sigma for size in shape]

    gaussian[tuple(coords)] = 1
    gaussian = gaussian_filter(
        gaussian,
        sigmas,
        0,
        mode="constant",
        cval=0,
    )
    gaussian /= np.max(gaussian)

    return gaussian


def get_volume_shape(mat_path, key="vol"):
    for name, shape, _ in sio.whosmat(mat_path):
        if name == key:
            return tuple(shape)

    raise KeyError(f'Key "{key}" not found in {mat_path}')


def run_batch(
    model,
    patches,
    coords,
    figure,
    normalization,
    gaussian_map,
    device,
):
    batch = np.stack(patches, axis=0).astype(np.float32)
    batch = torch.from_numpy(batch).unsqueeze(1).to(device)

    with torch.inference_mode():
        predictions = model(batch)

    predictions = predictions.squeeze(1).cpu().numpy().astype(np.float32)

    for prediction, (row, col) in zip(predictions, coords):
        prediction *= gaussian_map

        height, width = prediction.shape

        normalization[
            row:row + height,
            col:col + width,
        ] += gaussian_map

        figure[
            row:row + height,
            col:col + width,
        ] += prediction


def predict(
    model,
    image_path,
    gaussian_map,
    device,
    split_height=256,
    split_width=256,
    batch_size=32,
):
    start_time = time.time()

    image = np.load(image_path).astype(np.float32) / 255
    image[image < 0] = 0

    if image.shape[0] > 256:
        image_size = image.shape[0]

        figure = np.zeros(
            (image_size, image_size),
            dtype=np.float32,
        )
        normalization = np.zeros(
            (image_size, image_size),
            dtype=np.float32,
        )

        x_points = start_points(image_size, split_height, 0.5)
        y_points = start_points(image_size, split_width, 0.5)

        patches = []
        coords = []

        for row in y_points:
            for col in x_points:
                patches.append(
                    image[
                        row:row + split_height,
                        col:col + split_width,
                    ]
                )
                coords.append((row, col))

                if len(patches) == batch_size:
                    run_batch(
                        model,
                        patches,
                        coords,
                        figure,
                        normalization,
                        gaussian_map,
                        device,
                    )
                    patches = []
                    coords = []

        if patches:
            run_batch(
                model,
                patches,
                coords,
                figure,
                normalization,
                gaussian_map,
                device,
            )

        figure /= normalization
        prediction = figure.astype(np.float32)

    else:
        tensor = (
            torch.from_numpy(image)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        with torch.inference_mode():
            prediction = model(tensor).squeeze().cpu().numpy()

    elapsed = time.time() - start_time
    print(f"{image_path} cost time {elapsed:f} second")

    return prediction


def reconstruct_volume(
    model,
    input_path,
    target,
    volume_shape,
    batch_size=32,
):
    os.makedirs(target, exist_ok=True)

    depth = volume_shape[2]
    device = next(model.parameters()).device

    gaussian_map = get_gaussian(
        (256, 256)
    ).astype(np.float32)

    # 保持原代码的 float64 volume，避免改变最终数值。
    volume = np.zeros(volume_shape)

    for index in range(depth):
        image_path = os.path.join(input_path, f"{index}.npy")

        prediction = predict(
            model,
            image_path,
            gaussian_map,
            device,
            batch_size=batch_size,
        )

        volume[:, :, index] = prediction

    volume = volume / np.max(volume) * 255
    volume[volume < 0] = 0
    
    sio.savemat(
        os.path.join(target, "rec.mat"),
        {"vol": volume},
    )
    
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unet SV inference")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/SV6_checkpoint.pt",
        help="Unet checkpoint path",
    )
    parser.add_argument(
        "--root",
        default="./rec/",
        help="root directory containing raw/rec.mat and raw/*.npy",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = load_unet(args.checkpoint, device)

    input_path = os.path.join(args.root, "raw")
    output_path = os.path.join(args.root, "new")
    source_mat_path = os.path.join(input_path, "rec.mat")

    volume_shape = get_volume_shape(
        source_mat_path,
        key="vol",
    )

    print(f"Volume shape: {volume_shape}")

    reconstruct_volume(
        model,
        input_path,
        output_path,
        volume_shape,
        args.batch_size,
    )