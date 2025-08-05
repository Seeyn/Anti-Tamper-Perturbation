import torch
import torch.nn as nn
from torch_dct import dct as tdct
from torch_dct import idct as tidct

def dct_img(x,norm='ortho',block_size=16):
    # print(x.shape)
    x = block_splitting(x,block_size=block_size)
    length = len(x.shape)
    tdct_x = tdct(tdct(x, norm=norm).transpose(length-2, length-1), norm=norm).transpose(length-2, length-1)
    return block_merging(tdct_x,block_size=block_size)

def idct_img(x,norm='ortho',block_size=16):
    x = block_splitting(x,block_size=block_size)
    length = len(x.shape)
    tidct_x = tidct(tidct(x.transpose(length-2, length-1), norm=norm).transpose(length-2, length-1), norm=norm)
    return block_merging(tidct_x,block_size=block_size)

def block_merging(patches, height=512, width=512,block_size=16):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """
    
    k = block_size
    batch_size = patches.shape[0]
    image_reshaped = patches.view(batch_size, 3, height//k, width//k, k, k)
    image_transposed = image_reshaped.permute(0, 1, 2,4, 3, 5)

    return image_transposed.contiguous().view(batch_size,3, height, width)

def block_splitting(image,block_size=16):
    """ Splitting image into patches
    Input:
        image(tensor): batch x c x height x width
    Output: 
        patch(tensor):  batch  x h*w/64 x c x h x w
    """
    k = block_size
    channel,height, width = image.shape[1:4]
    batch_size = image.shape[0]
    image_reshaped = image.view(batch_size,channel, height // k, k, -1, k)
    image_transposed = image_reshaped.permute(0, 1, 2,4, 3, 5)
    return image_transposed.contiguous().view(batch_size, channel,-1,k, k)


class DCTlayer(nn.Module):

    def __init__(self,inverse = False,block_size=16) -> None:
        super().__init__()
        # hyper: if the dct layer for idct: true, otherwise, false 
        self.inverse = inverse
        self.block_size = block_size

    def forward(self, x, img_dct = None, random_mask = None):
        
        if not self.inverse:
            img_dct = dct_img(x,block_size=self.block_size)
            # generate random 0-1 mask obey bernoulli distribution (p = 0.5)
            if random_mask is None:
                random_mask  = torch.bernoulli(torch.ones(img_dct.shape) * 0.5).to(img.device)
            # get the dct element corresponding to value 1 in the mask
            img_dct_masked = (img_dct * random_mask)#.reshape(img_dct.shape[0],-1)
            return img_dct_masked, img_dct, random_mask
        
        else:
            ## reshape the encoded dct in B x dim shape
            encoded_dct = x#.reshape(img_dct.shape)
            # filter out the output in 0-value location
            encoded_dct = encoded_dct*random_mask
            # replace the value in original img_dct
            encoded_dct_mask = encoded_dct.clone()
            encoded_dct += img_dct * (-1) * (random_mask - 1)
            # idct for adv and iden loss calculation
            img = idct_img(encoded_dct,block_size=self.block_size)
            # print((encoded_dct_mask-encoded_dct).abs().sum())
            return encoded_dct_mask,img

class Pixellayer(nn.Module):

    def __init__(self,inverse = False) -> None:
        super().__init__()
        # hyper: if the dct layer for idct: true, otherwise, false 
        self.inverse = inverse

    def forward(self, x, img_dct = None, random_mask = None):
        # print('hi')
        if not self.inverse:
            img_dct = x
            # print(img_dct.max(),img_dct.min())
            # generate random 0-1 mask obey bernoulli distribution (p = 0.5)
            if random_mask is None:
                random_mask  = torch.bernoulli(torch.ones(img_dct.shape) * 0.5).to(img.device)
            # get the dct element corresponding to value 1 in the mask
            # print(random_mask.shape)
            img_dct_masked = (img_dct * random_mask)#.reshape(img_dct.shape[0],-1)
            return img_dct_masked, img_dct, random_mask
        
        else:
            ## reshape the encoded dct in B x dim shape
            encoded_dct = x#.reshape(img_dct.shape)
            # filter out the output in 0-value location
            encoded_dct = encoded_dct*random_mask
            # replace the value in original img_dct
            encoded_dct_mask = encoded_dct.clone()
            encoded_dct += img_dct * (-1) * (random_mask - 1)
            # idct for adv and iden loss calculation
            img = encoded_dct
            # print((encoded_dct_mask-encoded_dct).abs().sum())
            return encoded_dct_mask,img



