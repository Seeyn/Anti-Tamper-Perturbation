import torch
import numpy as np
import random
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from networks import HIDNet
import logging
import os

def set_seed(seed):
    
    random.seed(seed)
    
    
    np.random.seed(seed)
    
    
    torch.manual_seed(seed)
    
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    
    torch.backends.cudnn.deterministic = True
    
    
    torch.backends.cudnn.benchmark = False




class TensorBoardLogger:
    """
    Wrapper class for easy TensorboardX logging
    """
    def __init__(self, log_dir):
        self.grads = {}
        self.tensors = {}
        self.writer = SummaryWriter(log_dir)

    # def grad_hook_by_name(self, grad_name):
    #     def backprop_hook(grad):
    #         self.grads[grad_name] = grad
    #     return backprop_hook

    def save_losses(self, losses_accu: dict, step: int):
        for loss_name, loss_value in losses_accu.items():
            self.writer.add_scalar('losses/{}'.format(loss_name.strip()), loss_value, global_step=step)
    
    def save_val_losses(self, losses_accu: dict, step: int):
        for loss_name, loss_value in losses_accu.items():
            self.writer.add_scalar('val_losses/{}'.format(loss_name.strip()), loss_value, global_step=step)

    def save_images(self, images: torch.Tensor, step: int):
        # grid = vutils.make_grid(images, normalize=True)
        self.writer.add_images('images/{}'.format(step), images, global_step=step)


    # def save_grads(self, epoch: int):
    #     for grad_name, grad_values in self.grads.items():
    #         self.writer.add_histogram(grad_name, grad_values, global_step=epoch)

    def add_tensor(self, name: str, tensor):
        self.tensors[name] = tensor

    # def save_tensors(self, epoch: int):
    #     for tensor_name, tensor_value in self.tensors.items():
    #         self.writer.add_histogram('tensor/{}'.format(tensor_name), tensor_value, global_step=epoch)

def save_checkpoint(model: HIDNet, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'enc-dec-model': model.encoder_decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'mapper-model': model.mapping_network.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


def get_phis(phi_dimension, batch_size ,eps = 1e-8):
    phi_length = phi_dimension
    b = batch_size
    phi = torch.empty(b,phi_length).uniform_(0,1)
    return torch.bernoulli(phi) + eps


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name}')
    if not os.path.exists(runs_folder):
        os.makedirs(this_run_folder)
    if not os.path.exists(os.path.join(this_run_folder, 'checkpoints')):
        os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    
    return this_run_folder


import cv2
import torch
import degradations as degradations
from torchvision.transforms.functional import normalize

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


class Degradation():

    def __init__(self, noise_type,noise_args):
        super(Degradation, self).__init__()
        # degradation configurations

        assert noise_type in ['JPEG','Resize','Blur','Noise']
            
        self.noise_type = noise_type

        # if noise_type == 'Blur':
        #     self.blur_kernel_size = opt['blur_kernel_size']
        #     self.blur_sigma = opt['blur_sigma']
        # elif noise_type == 'Gaussian':
        #     self.noise_range = opt['noise_range']
        # elif noise_type == 'JPEG':
        #     self.jpeg_range = opt['jpeg_range']
        #     add_jpg_compression
        # else:
        #     self.downsample_range = opt['downsample_range']

        
        if noise_type == 'JPEG':
            self.jpeg_quality = noise_args
        elif  noise_type == 'Resize':
            self.downsample_scale = noise_args
        elif  noise_type == 'Blur':
            self.blur_scale = noise_args
        elif  noise_type == 'Noise':
            self.noise_scale = noise_args

    def degrade(self, img_gt):
        

        # ------------------------ generate lq image ------------------------ #
        if self.noise_type == 'Resize':
        # downsample
            w,h = img_gt.shape[0],img_gt.shape[1]
            img_lq = cv2.resize(img_gt, (int(w // self.downsample_scale), int(h // self.downsample_scale)), interpolation=cv2.INTER_LINEAR)
            # resize to original size
            img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        elif  self.noise_type == 'JPEG':
            img_lq = img_gt
            img_lq = degradations.add_jpg_compression(img_lq, self.jpeg_quality)
        elif  self.noise_type == 'Blur':
            kernel = degradations.bivariate_Gaussian(3, self.blur_scale, self.blur_scale, 0)
            img_lq = cv2.filter2D(img_gt, -1, kernel)
        elif  self.noise_type == 'Noise':
            img_lq = degradations.add_gaussian_noise(img_gt, self.noise_scale)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        # normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, 0.5, 0.5, inplace=True)
        
        return img_lq

