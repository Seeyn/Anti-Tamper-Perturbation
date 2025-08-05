import torch.nn as nn
from networks.encoder import Encoder
from networks.decoder import Decoder
from networks.DCTlayer import *

class EncoderDecoder(nn.Module):
   
    def __init__(self,encode_message_length = 128,decode_message_length=32,pixel_space=False,block_size=16):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(encode_message_length)
        if not pixel_space:
            self.dctlayer = DCTlayer(block_size=block_size)
            self.idctlayer = DCTlayer(inverse=True,block_size=block_size)
        else:
            self.dctlayer = Pixellayer()
            self.idctlayer = Pixellayer(inverse=True)
        self.decoder = Decoder(decode_message_length)

    def forward(self, image, message,random_mask=None):

        img_dct_masked, img_dct, random_mask = self.dctlayer(image,random_mask=random_mask)
        encoded_dct = self.encoder(img_dct_masked, message)
        encoded_dct, encoded_image = self.idctlayer(encoded_dct,img_dct,random_mask)
        decoded_message = self.decoder(encoded_dct)
        return encoded_image, decoded_message, random_mask,encoded_dct,img_dct_masked.detach()