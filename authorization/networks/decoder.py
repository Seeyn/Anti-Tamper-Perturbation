import torch.nn as nn
from networks.utils import ConvBNRelu


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self,message_length = 30):

        super(Decoder, self).__init__()
        self.channels = 64
        self.num_blocks = 7
        
        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(self.num_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        
        layers.append(ConvBNRelu(self.channels, message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(message_length, message_length)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x