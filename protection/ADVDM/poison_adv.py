# Adapted from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py

import argparse
import math
import copy
import cv2
from pathlib import Path
from torch.autograd import Variable
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import os
import torch.nn as nn
from torchvision.transforms.transforms import Resize
import random
import torchvision
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available
from torch_dct import dct as tdct
from torch_dct import idct as tidct

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")
device = "cuda"


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")


    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--poison_scale",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--poison_step_num",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--input-config",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example['path'] = self.instance_images_path[index % self.num_instance_images]
        return example


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    paths = [example["path"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "path":paths 
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

DEVICE = "cuda"

def main(args):
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    # If passed along, set the training seed now.
    if args.seed is not None:
        print('seed: ',args.seed)
        set_seed(args.seed)
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    weight_dtype = torch.float32

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.

    unet, train_dataloader = accelerator.prepare(unet, train_dataloader)
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device, dtype=weight_dtype)

    img_nums = len(os.listdir(args.instance_data_dir))
    poison_noise = torch.zeros([img_nums, 3, args.resolution, args.resolution])

    data_iter = iter(train_dataloader)
    epsilon = args.poison_scale/255.
    poison_step_size = epsilon/5
    clean_dataset = []
    paths = []
    for batch in train_dataloader:
        for id_, img in enumerate(batch['pixel_values'].cpu()):
            clean_dataset.append(img.numpy())
            paths.append(str(batch['path'][id_]))
    print(paths)
    clean_dataset = np.array(clean_dataset)

    ########################################
    # Optimize the Poison Noise
    ########################################

    idx = 0
    unet.eval()
    for param in unet.parameters():
        param.requires_grad = False
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        batch_start_idx, batch_noise = idx, []
        for i, _ in enumerate(batch['pixel_values']):
            batch_noise.append(poison_noise[idx])
            idx += 1
        batch_noise = torch.stack(batch_noise).cuda()
        perturb_imgs = Variable(batch['pixel_values'] + batch_noise, requires_grad=True)
        perturb_imgs = Variable(torch.clamp(perturb_imgs, -1, 1), requires_grad=True)
        eta = batch_noise
        momentum = torch.zeros_like(perturb_imgs).detach().cuda()

        if args.input_config is not None:
            import yaml
            def load_config(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                return config
            config = load_config(args.input_config)
            random_mask = torch.load(config['mask'])
            random_mask = 1-random_mask
            random_mask = random_mask.cuda()

        for step_idx in range(args.poison_step_num):
            unet.zero_grad()
            with accelerator.accumulate(unet):
            # Convert images to latent space

                unet.zero_grad()
                perturb_imgs.requires_grad_(True)
                perturb_imgs_t = perturb_imgs
                latents = vae.encode(perturb_imgs_t.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                bsz = latents.shape[0]
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                perturb_imgs.retain_grad()
                loss.backward()
                if args.input_config is None:
                    eta = poison_step_size * perturb_imgs.grad.data.sign()
                    perturb_imgs = Variable(perturb_imgs + eta, requires_grad=True)
                    
                    eta = torch.clamp(perturb_imgs - batch["pixel_values"], -epsilon, epsilon)
                    perturb_imgs = Variable(batch["pixel_values"] + eta, requires_grad=True)
                    perturb_imgs = Variable(torch.clamp(perturb_imgs, -1, 1), requires_grad=True)
                else:
                    def zero_low_frequencies(x):
                        x_dct = dct_img(x)
                        x_dct *= random_mask
                        x_idct = idct_img(float(config['ADVDM']['step_size'])*x_dct.sign())
                        return x_idct

                    grad = zero_low_frequencies(perturb_imgs.grad.data)
                    with torch.no_grad():
                        adv_images = perturb_imgs + grad
                    
                    adv_images_dct = dct_img(adv_images)
                    original_images_dct = dct_img(batch["pixel_values"])
                    
                    eps=float(config['ADVDM']['radius'])
                    eta = torch.clamp(adv_images_dct - original_images_dct, min=-eps, max=+eps)*random_mask
                    perturb_imgs =  Variable(torch.clamp(idct_img(original_images_dct + eta), min=-1, max=+1), requires_grad=True)
                print(loss.item())

        for i, delta in enumerate(eta):
            poison_noise[batch_start_idx+i] = delta.clone().detach().cpu()
        
        ##########################
        # Save the Poisoned Images
        ##########################

        if args.input_config is None:
            poisoned_dataset = copy.deepcopy(clean_dataset)
            perturb_noise = poison_noise.cpu().numpy()
            poisoned_dataset += perturb_noise
            poisoned_dataset = np.clip(poisoned_dataset, a_min=-1, a_max=1)
            poisoned_dataset = cv2.normalize(poisoned_dataset, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        
        else:
            poisoned_dataset = torch.tensor(copy.deepcopy(clean_dataset))
            poisoned_dataset_tmp = copy.deepcopy(clean_dataset)
            for id,img in enumerate(clean_dataset):
                eta = poison_noise[id].cpu().unsqueeze(0)
                adv_img_dct = dct_img(poisoned_dataset[id].unsqueeze(0)) + eta
                adv_img = torch.clamp(idct_img(adv_img_dct), min=-1, max=+1).detach_()
                poisoned_dataset_tmp[id] = (((adv_img.squeeze()+1)/2)*255.).numpy()
            poisoned_dataset = poisoned_dataset_tmp
            
        poisoned_img_output_path = args.output_dir
        
        if os.path.exists(poisoned_img_output_path)==False:
            os.mkdir(poisoned_img_output_path)
        poisoned_dataset = np.uint8(poisoned_dataset.transpose((0, 2, 3, 1)))

        for i in range(len(poisoned_dataset)):
            name = paths[i].split('/')[-1].split('.')[0]
            poisoned_dataset_pil = Image.fromarray(np.array(poisoned_dataset[i]))
            poisoned_dataset_pil.save(poisoned_img_output_path+"/noise_{}.png".format(name))


def dct_img(x,norm='ortho'):
    # print(x.shape)
    x = block_splitting(x)
    length = len(x.shape)
    tdct_x = tdct(tdct(x, norm=norm).transpose(length-2, length-1), norm=norm).transpose(length-2, length-1)
    return block_merging(tdct_x)

def idct_img(x,norm='ortho'):
    x = block_splitting(x)
    length = len(x.shape)
    tidct_x = tidct(tidct(x.transpose(length-2, length-1), norm=norm).transpose(length-2, length-1), norm=norm)
    return block_merging(tidct_x)

def block_merging(patches, height=512, width=512):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """
        
    k = 16
    batch_size = patches.shape[0]
    image_reshaped = patches.view(batch_size, 3, height//k, width//k, k, k)
    image_transposed = image_reshaped.permute(0, 1, 2,4, 3, 5)

    return image_transposed.contiguous().view(batch_size,3, height, width)

def block_splitting(image,k=16):
    """ Splitting image into patches
    Input:
        image(tensor): batch x c x height x width
    Output: 
        patch(tensor):  batch  x h*w/64 x c x h x w
    """
    channel,height, width = image.shape[1:4]
    batch_size = image.shape[0]
    image_reshaped = image.view(batch_size,channel, height // k, k, -1, k)
    image_transposed = image_reshaped.permute(0, 1, 2,4, 3, 5)
    return image_transposed.contiguous().view(batch_size, channel,-1,k, k)

if __name__ == "__main__":
    args = parse_args()
    main(args)
