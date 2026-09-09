import argparse
import os
import time

import numpy as np
import scipy.io as sio
import torch
from scipy.ndimage import gaussian_filter

from utils.checkpoint import load_unet


def testing(model, path, filename, target, volume, index):
    start_time = time.time()

    image = np.load(os.path.join(path, filename)).astype(np.float32)
    tensor = torch.from_numpy(image / 255.0).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(next(model.parameters()).device)

    with torch.inference_mode():
        prediction = model(tensor)

    prediction = prediction.squeeze().cpu().numpy()
    volume[:, :, index] = prediction

    elapsed = time.time() - start_time
    print(f"{os.path.join(path, filename)} cost time {elapsed:f} second")


def reconstruct(model, path, target, volume_shape):
    os.makedirs(target, exist_ok=True)

    depth = volume_shape[2]
    volume = np.zeros(volume_shape)

    for index in range(depth):
        testing(
            model,
            path,
            f"{index}.npy",
            target,
            volume,
            index,
        )

    volume = gaussian_filter(volume, sigma=1.0)
    volume = volume / np.max(volume) * 255
    cropped = volume[:, :, :]

    sio.savemat(
        os.path.join(target, "rec.mat"),
        {"vol": cropped},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unet DF inference")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/DF512_checkpoint.pt",
        help="Unet checkpoint path",
    )
    parser.add_argument(
        "--root",
        default="./rec/",
        help="root directory containing raw/rec.mat and raw/*.npy",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_unet(args.checkpoint, device)

    input_path = os.path.join(args.root, "raw")
    output_path = os.path.join(args.root, "new")
    source_mat_path = os.path.join(input_path, "rec.mat")

    source_volume = sio.loadmat(source_mat_path)["vol"]
    volume_shape = source_volume.shape

    print(f"Volume shape: {volume_shape}")

    reconstruct(
        model,
        input_path,
        output_path,
        volume_shape,
    )