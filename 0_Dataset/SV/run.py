import os
import numpy as np
import cv2
from scipy.spatial import ConvexHull
from skimage.transform import radon, iradon_sart


def sart(theta, img):
    height, width = img.shape

    sinogram = radon(
        img,
        theta=theta,
        circle=False
    )

    rec = iradon_sart(
        sinogram,
        theta
    )

    rec_h, rec_w = rec.shape

    y0 = (rec_h - height) // 2
    x0 = (rec_w - width) // 2

    rec = rec[
        y0:y0 + height,
        x0:x0 + width
    ]

    return rec


def apply_circular_mask(image, radius):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    masked_image = np.zeros_like(image)

    num_points = 400
    points = []

    for _ in range(num_points):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)

        x = int(center[0] + r * np.cos(angle))
        y = int(center[1] + r * np.sin(angle))

        points.append((x, y))

    points = np.array(points)

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.fillConvexPoly(
        mask,
        hull_points,
        255
    )

    masked_image[mask == 255] = image[mask == 255]

    return masked_image


def start_points(size, split_size):
    if size <= split_size:
        return [0]

    points = [0]
    stride = split_size
    counter = 1

    while True:
        pt = stride * counter

        if pt + split_size >= size:
            last = size - split_size

            if last != points[-1]:
                points.append(last)

            break

        points.append(pt)
        counter += 1

    return points


if __name__ == "__main__":

    image_path = "../examples.png"

    input_dir = "input_crop"
    mask_dir = "mask_crop"

    patch_size = 256

    theta = (
        np.arange(-66, 67, 6)
        + (np.random.rand(23) * 2 - 1) / 3
    )

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    img = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise FileNotFoundError(image_path)

    img = img.astype(np.float32)

    img = img / np.max(img) * 255

    height, width = img.shape

    radius = int(min(height, width) * 0.48)

    label = apply_circular_mask(
        img,
        radius
    )

    rec = sart(
        theta,
        label
    ).astype(np.float32)

    print(
        "label:",
        label.shape,
        label.min(),
        label.max()
    )

    print(
        "rec:",
        rec.shape,
        rec.min(),
        rec.max()
    )

    y_points = start_points(
        height,
        patch_size
    )

    x_points = start_points(
        width,
        patch_size
    )

    count = 0

    for y in y_points:
        for x in x_points:

            mask_patch = label[
                y:y + patch_size,
                x:x + patch_size
            ]

            input_patch = rec[
                y:y + patch_size,
                x:x + patch_size
            ]

            np.save(
                os.path.join(
                    mask_dir,
                    f"{count}.npy"
                ),
                mask_patch.astype(np.uint8)
            )

            np.save(
                os.path.join(
                    input_dir,
                    f"{count}.npy"
                ),
                input_patch.astype(np.float32)
            )

            cv2.imwrite(
                os.path.join(
                    mask_dir,
                    f"{count}.png"
                ),
                mask_patch.astype(np.uint8)
            )

            cv2.imwrite(
                os.path.join(
                    input_dir,
                    f"{count}.png"
                ),
                np.clip(
                    input_patch,
                    0,
                    255
                ).astype(np.uint8)
            )

            count += 1

    print("patch count:", count)