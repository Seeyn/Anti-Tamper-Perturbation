import torch
import torch.nn as nn
from networks.utils import ConvBNRelu


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self,message_length = 30):
        super(Encoder, self).__init__()
        self.H = 512
        self.W = 512
        self.conv_channels = 64
        self.num_blocks = 4
        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(self.num_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + message_length,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, image, message):

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze(-1)

        expanded_message = expanded_message.repeat(1,1, self.H * self.W).reshape(expanded_message.shape[0],-1, self.H , self.W)
        # print(expanded_message.shape)
        encoded_image = self.conv_layers(image)
        
        # concatenate expanded message and image
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w


        