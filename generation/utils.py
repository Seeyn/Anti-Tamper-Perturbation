import torch
import numpy as np
import random

def set_seed(seed):
    # 固定 Python 随机数生成器的种子
    random.seed(seed)
    
    # 固定 NumPy 随机数生成器的种子
    np.random.seed(seed)
    
    # 固定 PyTorch 随机数生成器的种子
    torch.manual_seed(seed)
    
    # 如果使用 GPU，也要固定 CUDA 随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    
    # 确保在有确定性算法的前提下运行
    torch.backends.cudnn.deterministic = True
    
    # 禁用 cuDNN 的自动优化（这可能会导致非确定性结果）
    torch.backends.cudnn.benchmark = False






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

        assert noise_type in ['JPEG','Resize']
            
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
        else:
            self.downsample_scale = noise_args


    def degrade(self, img_gt):
        

        # ------------------------ generate lq image ------------------------ #
        if self.noise_type == 'Resize':
        # downsample
            w,h = img_gt.shape[0],img_gt.shape[1]
            img_lq = cv2.resize(img_gt, (int(w // self.downsample_scale), int(h // self.downsample_scale)), interpolation=cv2.INTER_LINEAR)
            # resize to original size
            img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            img_lq = img_gt
            img_lq = degradations.add_jpg_compression(img_lq, self.jpeg_quality)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        # normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, 0.5, 0.5, inplace=True)
        
        return img_lq

