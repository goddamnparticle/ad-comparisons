import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator:
        Instead of Transposed Convolutions first Convolution then Upsample operations implemented for omitting checkerboard effect.
        img_size: Training dataset image size.
        latent_dim: Generator noise vector dimention.
        channels: Color channels for Generator output (RGB: 3, B&W: 1).
    """
    def __init__(self, img_size=32, latent_dim=100, channels=3):
        super().__init__()
        self.init_size = img_size // 4
        self.input_layer = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.model = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.input_layer(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.model(out)
        return img


class Discriminator(nn.Module):
    """
    Discriminator:
        channels: Color channels for Generator output (RGB: 3, B&W: 1).
        img_size: Training dataset image size.
    """
    def __init__(self, channels=3, img_size=32):
        super().__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        ds_size = img_size // 2 ** 4
        self.output_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        output = self.output_layer(out)
        return output 

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)















