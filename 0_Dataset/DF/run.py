import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
import skimage.transform as skt
from scipy.spatial import ConvexHull
import os 


device = 'cuda:0' if torch.cuda.is_available() else "cpu"

if torch.__version__ > '1.2.0':
    affine_grid = lambda theta, size: F.affine_grid(
        theta, size, align_corners=True
    )
    grid_sample = lambda input, grid, mode='bilinear': F.grid_sample(
        input, grid, align_corners=True, mode=mode
    )
else:
    affine_grid = F.affine_grid
    grid_sample = F.grid_sample


def deg2rad(x):
    return x * PI / 180


PI = 4 * torch.ones(1).atan()
SQRT2 = (2 * torch.ones(1)).sqrt()


class Radon(nn.Module):
    def __init__(
        self,
        in_size=None,
        theta=None,
        circle=True,
        dtype=torch.float,
        voltage=200.0,
        alpha_max=25.0,
        Nxy=512,
        Nz=512,
        res=0.3105,
        cenpos=None
    ):
        super(Radon, self).__init__()

        self.circle = circle
        self.theta = theta
        self.dtype = dtype
        self.all_grids = None
        self.random_defocus = None

        self.kernel = self.CTF(
            (Nxy, 1, Nz),
            voltage=voltage,
            alpha_max=alpha_max,
            Nxy=Nxy,
            Nz=Nz,
            res=res,
            cenpos=cenpos
        )

        if in_size is not None:
            self.all_grids = self._create_grids(
                self.theta,
                in_size,
                circle
            )

    def _create_grids(self, angles, grid_size, circle):
        if not circle:
            grid_size = int((SQRT2 * grid_size).ceil())

        all_grids = []

        for theta in angles:
            theta = deg2rad(theta)

            R = torch.tensor([[
                [theta.cos(), theta.sin(), 0],
                [-theta.sin(), theta.cos(), 0],
            ]], dtype=self.dtype)

            all_grids.append(
                affine_grid(
                    R,
                    torch.Size([1, 1, grid_size, grid_size])
                )
            )

        return all_grids

    def CTF(
        self,
        v_shape,
        voltage,
        alpha_max,
        Nxy,
        Nz,
        res,
        cenpos=None
    ):
        array = torch.zeros(
            v_shape,
            dtype=torch.complex64
        ).to(device)

        if cenpos is None:
            cenpos = Nz / 2

        wavelength = 12.398 / np.sqrt(
            (2 * 511.0 + voltage) * voltage
        )

        k_max = alpha_max * 1e-3 / wavelength
        k_min = 0.0
        dk = 1.0 / (res * Nxy)

        kx = torch.linspace(
            -np.floor(Nxy / 2.0),
            np.ceil(Nxy / 2.0) - 1,
            Nxy
        )

        ky = torch.tensor([0.0])

        [kY, kX] = torch.meshgrid(kx, ky)

        kX = kX * dk
        kY = kY * dk

        kR = torch.sqrt(
            kX ** 2 + kY ** 2
        )

        df = torch.arange(
            -cenpos,
            Nz - cenpos
        ) * res

        mask = torch.ones_like(
            array,
            dtype=torch.long
        ).to(device)

        mask[kR > k_max] = 0
        mask[kR < k_min] = 0

        for i in range(0, Nz):
            defocus = df[i]

            # 原代码 c3=0 后严格剩下这一项
            chi = (
                -torch.pi
                * wavelength
                * kR ** 2
                * defocus
            )

            chi = chi.to(device)

            probe = (
                mask[:, :, i]
                * torch.exp(-1j * chi)
            )

            probe = torch.fft.fftshift(
                torch.fft.ifft2(
                    torch.fft.ifftshift(probe)
                )
            )

            probe = torch.abs(probe) ** 2

            factor = torch.sum(probe)

            probe = torch.fft.ifftshift(
                torch.fft.fft2(
                    torch.fft.fftshift(probe)
                )
            )

            probe = probe / factor

            array[:, :, i] = probe

        return array

    def conv(self, rot, kernel):
        rot = rot.squeeze().unsqueeze(1)

        for i in range(rot.shape[2]):
            slice = rot[i, :, :]

            SLICE = torch.fft.ifftshift(
                torch.fft.fft2(
                    torch.fft.fftshift(
                        slice.permute(1, 0)
                    )
                )
            )

            SLICE = SLICE * kernel[:, :, i]

            rot[i, :, :] = torch.abs(
                torch.fft.fftshift(
                    torch.fft.ifft2(
                        torch.fft.ifftshift(SLICE)
                    )
                ).permute(1, 0)
            )

        rot = rot.squeeze().unsqueeze(0).unsqueeze(0)

        return rot

    def forward(self, x):
        N, C, W, H = x.shape

        assert W == H

        w, h = (
            int(np.ceil(W * np.sqrt(1))),
            int(np.ceil(H * np.sqrt(1)))
        )

        if self.all_grids is None:
            self.all_grids = self._create_grids(
                self.theta,
                W,
                self.circle
            )

        if not self.circle:
            diagonal = SQRT2 * W
            pad = int((diagonal - W).ceil())

            new_center = (W + pad) // 2
            old_center = W // 2
            pad_before = new_center - old_center

            pad_width = (
                pad_before,
                pad - pad_before
            )

            x = F.pad(
                x,
                (
                    pad_width[0],
                    pad_width[1],
                    pad_width[0],
                    pad_width[1]
                )
            )

        N, C, W, _ = x.shape

        out = torch.zeros(
            N,
            C,
            W,
            len(self.theta),
            device=x.device,
            dtype=self.dtype
        )

        for i in range(len(self.theta)):
            rotated = grid_sample(
                x,
                self.all_grids[i]
                .repeat(N, 1, 1, 1)
                .to(x.device)
            )

            rotated = self.conv(
                rotated,
                self.kernel
            )

            out[..., i] = rotated.sum(2)

        return out.squeeze()


class AET():
    def __init__(
        self,
        img_width,
        theta,
        circle,
        voltage,
        alpha_max,
        Nxy,
        Nz,
        res,
        cenpos
    ):
        theta = torch.Tensor(theta)

        self.radon = Radon(
            in_size=img_width,
            theta=theta,
            circle=circle,
            voltage=voltage,
            alpha_max=alpha_max,
            Nxy=Nxy,
            Nz=Nz,
            res=res,
            cenpos=cenpos
        ).to(device)

    def A(self, x):
        return self.radon(x)


def apply_circular_mask(image, radius):
    image = cv2.resize(
        image,
        (512, 512),
        interpolation=cv2.INTER_CUBIC
    )

    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    masked_image = np.zeros_like(image)

    num_points = 250
    points = []

    for _ in range(num_points):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)

        x = int(
            center[0]
            + r * np.cos(angle)
        )

        y = int(
            center[1]
            + r * np.sin(angle)
        )

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


if __name__ == "__main__":

    # =============================
    # 直接修改这里
    # =============================

    image_path = "../examples.png"
    angles_path = "Angles.txt"

    voltage = 200.0
    alpha_max = 25.0

    Nz = 512

    res = 0.3185 # pixel size
    cenpos = Nz // 2

    # =============================

    kkk = [0, 1]

    theta = np.loadtxt(angles_path)

    ET = AET(
        img_width=Nz,
        theta=theta,
        circle=True,
        voltage=voltage,
        alpha_max=alpha_max,
        Nxy=Nz,
        Nz=Nz,
        res=res,
        cenpos=cenpos
    )


    path = image_path
    
    os.makedirs("input_crop/",exist_ok=True)
    os.makedirs("mask_crop/",exist_ok=True)

    for r in range(2):

        ext_path = "input_crop/%s_examples" % r
        mask_path = "mask_crop/%s_examples" % r

        img = cv2.imread(path, 0)
        img=img/np.max(img)*255

        ra = 0.9 + (
            2 * np.random.rand(1) - 1
        ) * 0.1

        img = img * ra

        img = cv2.flip(
            img,
            kkk[r]
        )

        V = apply_circular_mask(
            img,
            245
        )

        np.save(
            mask_path + ".npy",
            V.astype(np.uint8)
        )

        V = torch.tensor(
            V,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(device)

        sino = ET.A(V)

        rec = skt.iradon_sart(
            sino.cpu().numpy(),
            theta
        )

        np.save(
            ext_path + ".npy",
            rec.astype(np.float32)
        )
        
        # cv2.imwrite(mask_path + ".png",V.squeeze().numpy())
        # cv2.imwrite(ext_path + ".png",rec)