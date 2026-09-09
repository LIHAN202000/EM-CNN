import os
import cv2
import numpy as np
import torch
from torch.optim import Adam
from src import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def proj(tomo, V):
    V = V.permute(1, 2, 0).unsqueeze(0)
    return tomo.A(V).squeeze().permute(1, 0, 2)


def calc_diff(tomo, Bfactor, height, pj_obs, model, atomic_channel, fa, res, volsize, it):
    gaussian = create_vol(model, atomic_channel, height, Bfactor, volsize, fa, res)
    pj_calc = proj(tomo, gaussian) / scale

    R1 = torch.norm(pj_calc - pj_obs, p=1) / pj_obs_norm
    CALC = torch.abs(FFT2(pj_calc))
    RF = torch.norm(CALC - OBS, p=1) / OBS_norm
    print("R1 factor: %s" % R1.item(), "Rf factor: %s" % RF.item())

    pj_calc_np = pj_calc.detach().cpu().numpy()
    new = np.zeros((pj_calc.shape[0], pj_calc.shape[1] * pj_calc.shape[2]))
    for i in range(pj_calc.shape[2]):
        new[:, i * pj_calc.shape[1]:(i + 1) * pj_calc.shape[1]] = pj_calc_np[:, :, i]
    new = new / np.max(new) * 255

    # np.save(pj_calc_file, pj_calc_np)
    cv2.imwrite("%s/pj_calc_%s.png" % (output_dir, it), new)
    return R1 + 0.5*RF


def write_loss(path, loss, gradnorm, it):
    with open(path, "a") as f:
        f.write("%s %s %s\n" % (it, loss.item(), gradnorm.item()))


def write_vesta(data, H, B, atomic_channel, name):
    data = data.detach().cpu().numpy()
    H = H.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    n = np.sum(H > 0.2)
    data[:, 2] = -data[:, 2]

    with open("%s/%s.xyz" % (output_dir, name), "a") as f:
        f.write("%d \n" % n)
        f.write("refine_str \n")
        for i in range(data.shape[0]):
            if H[i] > 0.2:
                symbol = element_symbols[atomic_channel[i]]
                f.write("%s %s %s %s %s %s\n" %
                        (symbol, data[i, 0] * res, data[i, 1] * res,
                         data[i, 2] * res, H[i], B[i]))


def compute_probe(V, res, cenpos=None):
    if cenpos is None:
        cenpos = 0
    max_z = int(2 // res)
    if res * max_z > 2:
        max_z -= 1

    wavelength = 12.398 / np.sqrt((2 * 511.0 + voltage) * voltage)
    k_max = alpha_max * 1e-3 / wavelength
    Vshape = V.shape[0]
    resZ = max_z * res
    Nxy = Vshape
    dk = 1.0 / (res * Nxy)

    kx = torch.linspace(-np.floor(Nxy / 2.0), np.ceil(Nxy / 2.0) - 1, Nxy).to(device)
    ky = torch.linspace(-np.floor(Nxy / 2.0), np.ceil(Nxy / 2.0) - 1, Nxy).to(device)
    kY, kX = torch.meshgrid(kx, ky, indexing="ij")
    kX = kX * dk
    kY = kY * dk
    kR = torch.sqrt(kX ** 2 + kY ** 2)

    Nz = int(res * Vshape / resZ)
    if (max_z - V.shape[2] % max_z) % max_z > 0:
        Nz += 1

    df = torch.arange(cenpos - 0.5 * Nz * resZ, cenpos + 0.5 * Nz * resZ, resZ)
    mask = torch.ones((Nxy, Nxy, Nz), dtype=torch.complex64).to(device)
    mask[kR > k_max] = 0
    H = torch.zeros(mask.shape, dtype=torch.complex64).to(device)

    for i in range(Nz):
        defocus = df[i]
        chi = (-torch.pi * wavelength * kR ** 2 * defocus).to(device)
        probe = mask[:, :, i] * torch.exp(-1j * chi)
        probe = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(probe)))
        probe = torch.abs(probe) ** 2
        factor = torch.sum(probe)
        probe = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(probe)))
        H[:, :, i] = probe / factor
    return H


def run(data, probe, Hfactor, Bfactor, atomic_channel, num):
    xyz = torch.tensor(np.array([0, 0, 0]), dtype=torch.float32,
                       requires_grad=True, device=device)

    params = [xyz, data, Bfactor]
    if shared_element_H:
        params.append(Hfactor)

    optimizer = Adam(params, lr=lr, betas=(0.9, 0.999),
                     eps=1e-08, weight_decay=0, amsgrad=False)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=lr_gamma
    )

    start_it = iterations_per_round * num
    end_it = iterations_per_round * (num + 1)

    for it in range(start_it, end_it):
        print(it)

        channel = torch.as_tensor(
            atomic_channel, dtype=torch.long, device=device
        )

        if shared_element_H:
            H = Hfactor[channel]
        else:
            H = data[:, 0]

        if shared_element_B:
            B = Bfactor[channel]
        else:
            B = Bfactor[:data.shape[0]]

        loss = calc_diff(
            tomo, B, H, pj_obs, data[:, 1:] + xyz,
            atomic_channel, fa, res, volsize, it
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gradnorm = torch.norm(Bfactor.grad, "fro") ** 2
        write_loss(loss_file, loss, gradnorm, str(it))

        lr_scheduler.step()
        # np.save(probe_file, probe.detach().cpu().numpy())

        if it == start_it:
            if shared_element_H:
                H_update = Hfactor[channel]
                data, atomic_channel = update(
                    data, atomic_channel, res,
                    update_distance, update_intensity, surface_distance,
                    heights=H_update
                )
            else:
                data, atomic_channel = update(
                    data, atomic_channel, res,
                    update_distance, update_intensity, surface_distance
                )

        channel = torch.as_tensor(
            atomic_channel, dtype=torch.long, device=device
        )

        if shared_element_H:
            H_out = Hfactor[channel]
        else:
            H_out = data[:, 0]

        if shared_element_B:
            B_out = Bfactor[channel]
        else:
            B_out = Bfactor[:data.shape[0]]

        write_vesta(data[:, 1:], H_out, B_out, atomic_channel, it)

    return data, probe, Hfactor, Bfactor, atomic_channel


if __name__ == "__main__":
    # Files
    projections_file = "pj_obs.npy"
    angles_file = "Angles.txt"
    model_file = "example_str.xyz"

    # Microscope / model 
    res = 0.3105           # pixel size
    voltage = 200          # kV
    alpha_max = 30         # mrad
    initial_bfactor = 4.5
    shared_element_H = False    # True: same H for all atoms of each element
    shared_element_B = True    # True: same B for all atoms of each element

    # Projection crop 
    image_size = 200
    crop = 0

    # Optimization 
    num_rounds = 4
    iterations_per_round = 50
    lr = 0.05
    lr_gamma = 0.99
    output_dir = "output/"
    loss_file = "lossR.txt" # Write loss
    
    # Atom filtering 
    update_distance = 1.6
    update_intensity = 0.2
    surface_distance = 3.6
    probe_bfactor = 5.0
    
    angles = np.loadtxt(angles_file)
    print(angles)

    pj = np.load(projections_file)[crop:image_size - crop, crop:image_size - crop, :]
    pj = pj / np.max(pj)

    raw = np.zeros((pj.shape[0], pj.shape[1] * pj.shape[2]))
    for i in range(pj.shape[2]):
        raw[:, i * pj.shape[1]:(i + 1) * pj.shape[1]] = pj[:, :, i] * 255

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite("%s/raw.png" % output_dir, raw)
    pj = pj / np.max(pj)

    # Coordinates are read exactly as before; the first XYZ column is read separately for elements.
    atomic_symbols = np.atleast_1d(
        np.genfromtxt(model_file, dtype=str, skip_header=2, usecols=0)
    )
    atomic_symbols = np.array([symbol.capitalize() for symbol in atomic_symbols])

    unknown = sorted(set(atomic_symbols) - set(ELEMENT_TO_Z))
    if unknown:
        raise ValueError("Unknown element symbol(s): %s" % ", ".join(unknown))

    atomic_type = np.array([ELEMENT_TO_Z[symbol] for symbol in atomic_symbols], dtype=np.int64)
    atom_numbers = list(dict.fromkeys(atomic_type.tolist()))
    channel_map = {z: i for i, z in enumerate(atom_numbers)}
    atomic_channel = np.array([channel_map[z] for z in atomic_type], dtype=np.int64)
    element_symbols = [PERIODIC_TABLE[z] for z in atom_numbers]

    print("Elements:", element_symbols)
    print("Atomic numbers:", atom_numbers)

    model = np.genfromtxt(model_file, delimiter=" ", skip_header=2, skip_footer=0)[:, 1:4] / res
    model[:, 2] = -model[:, 2]
    Bfactor0 = np.ones(model.shape[0])
    volsize = (pj.shape[0], pj.shape[1], pj.shape[0])
    pj_obs = torch.tensor(pj).to(device)
    pj_obs_norm = torch.norm(pj_obs, p=1)
    OBS = torch.abs(FFT2(pj_obs))
    OBS_norm = torch.norm(OBS, p=1)

    fa = generate_fa(atom_numbers, volsize, res).to(device)

    Bfactor0 = torch.tensor(Bfactor0, dtype=torch.float32).to(device)
    model0 = torch.tensor(model, dtype=torch.float32)
    cx, cy, cz = torch.mean(model0[:, 0]), torch.mean(model0[:, 1]), torch.mean(model0[:, 2])
    model0 = model0 - torch.tensor([cx, cy, cz])
    model0 = model0.to(device)
    height0 = torch.tensor(np.ones_like(model[:, 0])).to(device)
    Bfactor0 = probe_bfactor * Bfactor0

    gaussian = create_vol(model0, atomic_channel, height0, Bfactor0, volsize, fa, res)
    probe = compute_probe(gaussian, res)

    tomo = AET(pj.shape[0], angles + 90, probe, voltage, alpha_max, res)
    pj_calc = proj(tomo, gaussian)
    scale = torch.max(pj_calc)

    model = model0.cpu().numpy()
    data = np.ones((model.shape[0], 4))
    data[:, 1:] = model
    data = torch.tensor(data, dtype=torch.float32, requires_grad=True, device=device)
    probe = torch.tensor(probe.cpu().numpy(), dtype=torch.complex64,
                         requires_grad=False, device=device)
    if shared_element_H:
        Hfactor = torch.ones(
            len(atom_numbers), dtype=torch.float32,
            requires_grad=True, device=device
        )
    else:
        Hfactor = None

    if shared_element_B:
        Bfactor = torch.full(
            (len(atom_numbers),), initial_bfactor, dtype=torch.float32,
            requires_grad=True, device=device
        )
    else:
        Bfactor = torch.tensor(
            initial_bfactor * np.ones(model.shape[0]), dtype=torch.float32,
            requires_grad=True, device=device
        )

        
    tomo = AET(pj.shape[0], angles + 90, probe, voltage, alpha_max, res)
    for num in range(num_rounds):
        data, probe, Hfactor, Bfactor, atomic_channel = run(
            data, probe, Hfactor, Bfactor, atomic_channel, num
        )
