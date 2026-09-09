import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
from core.unet import Unet
from utils.hparams import HParam
from utils.metrics import MetricTracker

warnings.simplefilter("ignore", (UserWarning, FutureWarning))


# These values match the parameters that were actually used by the original
# training source code. The bundled YAML files contain the same values.
ORIGINAL_LR = 0.0002
ORIGINAL_BATCH_SIZE = 24
ORIGINAL_LOGGING_STEP = 5000
ORIGINAL_NUM_WORKERS = 2
ORIGINAL_VALID_BATCH_SIZE = 1
ORIGINAL_ADAM_BETAS = (0.9, 0.999)
ORIGINAL_ADAM_EPS = 1e-8
ORIGINAL_STEP_SIZE = 10
ORIGINAL_GAMMA = 0.1


def weights_init_normal(module):
    if "Conv" in module.__class__.__name__:
        torch.nn.init.normal_(module.weight.data, 0.0, 0.0002)


def _source_equivalent_hparam(hp, name, expected):
    """Use config values while preventing accidental drift from source settings."""
    value = getattr(hp, name)
    if value != expected:
        raise ValueError(
            f"{name} must remain {expected} to match the original training source; "
            f"got {value} in the config."
        )
    return value


def save_checkpoint(model, checkpoint_dir, name, epoch, step):
    save_path = os.path.join(
        checkpoint_dir,
        f"{name}_checkpoint_{epoch}_{step}.pt",
    )
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "arch": "Unet",
            "state_dict": model.state_dict(),
        },
        save_path,
    )
    print(f"Saved checkpoint to: {save_path}")


def append_losses(checkpoint_dir, epoch, step, valid_l1, train_l1):
    loss_file = os.path.join(checkpoint_dir, "loss.txt")
    write_header = not os.path.exists(loss_file) or os.path.getsize(loss_file) == 0

    with open(loss_file, "a", encoding="utf-8") as file:
        if write_header:
            file.write("epoch,step,valid_L1,train_L1\n")
        file.write(f"{epoch},{step},{valid_l1:.6f},{train_l1:.6f}\n")


def validation(valid_loader, model, criterion, device):
    valid_l1 = MetricTracker()

    model.eval()
    with torch.inference_mode():
        for idx, data in enumerate(tqdm(valid_loader, desc="validation")):
            inputs = data["sat_img"].to(device)
            labels = data["map_img"].to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            valid_l1.update(loss.item(), outputs.size(0))

    print(f"Validation L1 Loss: {valid_l1.avg:.8f}")
    model.train()
    return valid_l1.avg


def main(hp, num_epochs, resume, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep the effective hyperparameters identical to the original source.
    learning_rate = _source_equivalent_hparam(hp, "lr", ORIGINAL_LR)
    batch_size = _source_equivalent_hparam(hp, "batch_size", ORIGINAL_BATCH_SIZE)
    logging_step = _source_equivalent_hparam(
        hp, "logging_step", ORIGINAL_LOGGING_STEP
    )

    checkpoint_dir = os.path.join(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = torch.nn.DataParallel(Unet(1)).to(device)

    if hp.MODEL == 1:
        model.apply(weights_init_normal)

    # L1 is the only loss used for optimization and reporting.
    criterion = nn.L1Loss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=ORIGINAL_ADAM_BETAS,
        eps=ORIGINAL_ADAM_EPS,
        weight_decay=0,
        amsgrad=False,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=ORIGINAL_STEP_SIZE,
        gamma=ORIGINAL_GAMMA,
    )

    start_epoch = 0
    step = 0
    if resume:
        if not os.path.isfile(resume):
            raise FileNotFoundError(f"No checkpoint found at '{resume}'")

        print(f"=> loading checkpoint '{resume}'")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        step = checkpoint.get("step", 0)
        print(f"=> loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})")

    transform = dataset.ToTensorTarget()
    train_dataset = dataset.ImageDataset(hp, transform=transform)
    valid_dataset = dataset.ImageDataset(hp, train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=ORIGINAL_NUM_WORKERS,
        shuffle=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=ORIGINAL_VALID_BATCH_SIZE,
        num_workers=ORIGINAL_NUM_WORKERS,
        shuffle=False,
    )

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Keep scheduler timing in the same position as the original code.
        lr_scheduler.step()

        train_l1 = MetricTracker()
        loader = tqdm(train_loader, desc="training")

        for data in loader:
            inputs = data["sat_img"].to(device)
            labels = data["map_img"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_l1.update(loss.item(), outputs.size(0))
            step += 1

            if step % logging_step == 0:
                save_checkpoint(model, checkpoint_dir, name, epoch, step)
                valid_l1 = validation(
                    valid_loader,
                    model,
                    criterion,
                    device,
                )
                append_losses(
                    checkpoint_dir,
                    epoch,
                    step,
                    valid_l1,
                    train_l1.avg,
                )

            loader.set_description(f"Training L1 Loss: {train_l1.avg:.8f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet with L1 loss")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="checkpoint path to resume from",
    )
    parser.add_argument(
        "--name",
        default="default",
        type=str,
        help="experiment name",
    )
    args = parser.parse_args()

    hparams = HParam(args.config)
    print(hparams)
    main(hparams, num_epochs=args.epochs, resume=args.resume, name=args.name)
