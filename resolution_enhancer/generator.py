# Generator architecture must be imported to infer

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard SRGAN Residual Block
    
    Structure:
        Conv
        BatchNorm
        PReLU
        Conv
        BatchNorm
        Skip Connection
    """

    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(

            # First convolution
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3, #3x3 kernel size
                stride=1, #no reduction in image res
                padding=1 #keeps original
            ),

            # Batch Norm
            nn.BatchNorm2d(channels), #normalizes feature maps

            #PReLU
            nn.PReLU(), #keeps negative values (-ve x alpha)

            # Second convolution
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            # BatchNorm
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):

        # Save original input
        residual = x

        # Pass through convolution block
        out = self.block(x)

        # Add skip connection
        out = out + residual

        return out
    

class PixelShuffleBlock(nn.Module):
    """
    Upsampling block using PixelShuffle

    Structure:
        Conv
        PixelShuffle
        PReLU
    """

    def __init__(self, channels=64, upscale_factor=2):
        super().__init__()

        self.block = nn.Sequential(

            # Increase channel count
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels * (upscale_factor ** 2),
                kernel_size=3,
                stride=1,
                padding=1
            ),

            # Rearrange channels into pixels
            nn.PixelShuffle(upscale_factor),

            # Activation
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)
    

class Generator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_channels=64,
        num_residual_blocks=16,
        upscale_factor=2
    ):
        super().__init__()

        # -----------------------------------------
        # Initial Feature Extraction
        # Convert RGB image into feature maps
        # -----------------------------------------
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_channels,
                kernel_size=9, #9x9 kernel (capture large contexts, original SRGAN paper)
                stride=1,
                padding=4
            ),
            nn.PReLU() #activation
        )

        # -----------------------------------------
        # Deep Feature Extraction
        # 16 Residual Blocks
        # -----------------------------------------
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(num_channels)
                for _ in range(num_residual_blocks)
            ]
        )

        # -----------------------------------------
        # Feature Fusion Layer
        # Combines all learned residual features
        # -----------------------------------------
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_channels)
        )

        # -----------------------------------------
        # Upsampling block
        # Increases resolution by 2×
        # -----------------------------------------
        self.upsample = PixelShuffleBlock(
            channels=num_channels,
            upscale_factor=upscale_factor
        )

        # -----------------------------------------
        # Output Layer
        # Converts feature maps back to RGB image
        # -----------------------------------------
        self.output = nn.Conv2d(
            in_channels=num_channels,
            out_channels=in_channels,
            kernel_size=9, #9x9, original SRGAN
            stride=1,
            padding=4
        )


    def forward(self, x):
        # -----------------------------------------
        # Initial feature extraction
        # -----------------------------------------
        initial_features = self.initial(x)

        # Save features for the global skip connection
        skip_connection = initial_features

        # -----------------------------------------
        # Deep feature learning
        # -----------------------------------------
        x = self.residual_blocks(initial_features)

        # -----------------------------------------
        # Feature fusion
        # -----------------------------------------
        x = self.conv_block(x)

        # -----------------------------------------
        # Global Skip Connection
        # -----------------------------------------
        x = x + skip_connection

        # -----------------------------------------
        # Increase image resolution
        # -----------------------------------------
        x = self.upsample(x)

        # -----------------------------------------
        # Convert feature maps into RGB image
        # -----------------------------------------
        x = self.output(x)
        x = torch.tanh(x)
        return x
    
'''
Test
-----------------------
generator = Generator()
lr = torch.randn(1, 3, 96, 96)
hr = generator(lr)
print(hr.shape)
'''
