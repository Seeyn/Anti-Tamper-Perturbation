import os
from pathlib import Path
import torch
import numpy as np
import utils
import logging
from networks.HIDNet import HIDNet
import options
from dataset import *
import sys
import re
from tqdm import tqdm
import yaml

# Add project root to sys.path to import logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from logger import logger


def sorted_nicely(l):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    folder_path = Path(folder)
    last_file = sorted_nicely([f.name for f in folder_path.iterdir()])[-1]
    return folder_path / last_file


def model_from_checkpoint(hidden_net, checkpoint):
    """Restores the hidden_net object from a checkpoint object"""
    hidden_net.encoder_decoder.load_state_dict(
        checkpoint["enc-dec-model"], strict=False
    )
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint["enc-dec-optim"])
    hidden_net.discriminator.load_state_dict(checkpoint["discrim-model"], strict=False)
    hidden_net.optimizer_discrim.load_state_dict(checkpoint["discrim-optim"])


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


config = load_config("configs/authorization.yaml")

# setup_logger() is already called in logger.py import

seed = config["seed"]
utils.set_seed(seed)
message_length = config["message_length"]
args = options.get_args()
input_dir = args.input_dir
output_dir = args.output_dir
device = args.device
gamma = args.gamma
input_mask = args.input_mask
pixel_space = args.pixel_space

logger.info(f"seed: [bold cyan]{seed}[/bold cyan]")
logger.info(f"message_length: [bold cyan]{message_length}[/bold cyan]")
logger.info(f"block_size: [bold cyan]{config['block_size']}[/bold cyan]")
logger.info(f"input_folder: [bold cyan]{input_dir}[/bold cyan]")
logger.info(f"running_folder: [bold cyan]{output_dir}[/bold cyan]")

model = HIDNet(device, gamma=gamma, input_mask=input_mask, pixel_space=pixel_space)
val_dataloader = torch.utils.data.DataLoader(
    TestDataset(train=False, get_path=True, path=input_dir),
    batch_size=10,
    shuffle=False,
)
weights = config["test"]["weights"]
if weights is not None:
    model_from_checkpoint(model, torch.load(weights, map_location=device))

###########Authorization###########


val_losses = {}
transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
os.makedirs(output_dir, exist_ok=True)
(Path(output_dir) / "Authorization_messages").mkdir(parents=True, exist_ok=True)

for batch_id, (path, image) in tqdm(enumerate(val_dataloader)):
    image = image.to(device)
    message = utils.get_phis(
        phi_dimension=message_length, batch_size=image.shape[0]
    ).to(device)
    losses, (encoded_images, random_mask, decoded_messages) = model.val_one_batch(
        [image, message]
    )
    # print('bitwise-error:',losses['bitwise-error'])
    for key in losses:
        if key not in val_losses:
            val_losses[key] = 0
        val_losses[key] += losses[key]

    for id, e_img in enumerate(encoded_images):
        p = Path(path[id])
        image_name = p.stem
        id_dir = p.parts[-3]
        set_dir = p.parts[-2]
        dir_path = Path(output_dir) / id_dir / set_dir
        dir_path.mkdir(parents=True, exist_ok=True)
        transforms.ToPILImage()(((e_img.clip(-1, 1) + 1) / 2).clip(0, 1).cpu()).save(
            dir_path / f"{image_name}.png"
        )
        torch_set = {"random_mask": random_mask.cpu(), "message": message[id].cpu()}
        torch.save(
            torch_set,
            Path(output_dir) / "Authorization_messages" / f"{id_dir}_{image_name}.pt",
        )
