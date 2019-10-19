import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class UNetBaseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetBaseConv, self).__init__()

        self.unet_act = nn.ReLU(inplace=True)
        #self.unet_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.unet_norm = nn.BatchNorm2d(out_channels)

        self.unet_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.normal_(self.unet_conv1.weight, mean=0.0, std=np.sqrt(2/(kernel_size*kernel_size*in_channels)))

        self.unet_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.normal_(self.unet_conv2.weight, mean=0.0, std=np.sqrt(2/(kernel_size*kernel_size*in_channels)))

    def forward(self, x):

        x = self.unet_conv1(x)
        x = self.unet_norm(x)
        x = self.unet_act(x)
        
        x = self.unet_conv2(x)
        x = self.unet_norm(x)
        x = self.unet_act(x)

        return x

class UNetDownConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UNetDownConv, self).__init__()

        self.unet_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.unet_conv_block = UNetBaseConv(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.unet_pool(x)
        x = self.unet_conv_block(x)
        return x

class UNetUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, up_mode):
        super(UNetUpConv, self).__init__()
        assert up_mode in ('ConvTranspose', 'Upsample')

        self.up_mode = up_mode
        self.unet_up_conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.unet_up_conv_upsample = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True), nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.unet_conv_block = UNetBaseConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, crop):
        x = self.unet_up_conv_transpose(x) if self.up_mode == 'ConvTranspose' else self.unet_up_conv_upsample(x)
        x = torch.cat([x, crop], dim=1)
        x = self.unet_conv_block(x)
        return x


def center_crop(layer, target_size_x, target_size_y):
    lower_x = int((layer.shape[2] - target_size_x) / 2)
    upper_x = lower_x + target_size_x
    lower_y = int((layer.shape[3] - target_size_y) / 2)
    upper_y = lower_y + target_size_y

    return layer[:, :, lower_x:upper_x, lower_y:upper_y]


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, up_mode='ConvTranspose', padding=False):
        """
        Implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

        -- Args
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        up_mode (str): one of 'ConvTranspose' or 'Upsample':
                           - 'ConvTranspose' will use transposed convolutions for learned upsampling.
                           - 'Upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('ConvTranspose', 'Upsample')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_mode = up_mode
        self.pad = 1 if padding else 0
        
        self.init_conv = UNetBaseConv(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=self.pad)

        self.encode1 = UNetDownConv(64, 64*2, kernel_size=3, stride=1, padding=self.pad)
        self.encode2 = UNetDownConv(64*2, 64*2*2, kernel_size=3, stride=1, padding=self.pad)
        self.encode3 = UNetDownConv(64*2*2, 64*2*2*2, kernel_size=3, stride=1, padding=self.pad)
        self.encode4 = UNetDownConv(64*2*2*2, 64*2*2*2*2, kernel_size=3, stride=1, padding=self.pad)

        self.decode1 = UNetUpConv(64*2*2*2*2, 64*2*2*2, kernel_size=3, stride=1, padding=self.pad, up_mode=up_mode)
        self.decode2 = UNetUpConv(64*2*2*2, 64*2*2, kernel_size=3, stride=1, padding=self.pad, up_mode=up_mode)
        self.decode3 = UNetUpConv(64*2*2, 64*2, kernel_size=3, stride=1, padding=self.pad,  up_mode=up_mode)
        self.decode4 = UNetUpConv(64*2, 64, kernel_size=3, stride=1, padding=self.pad, up_mode=up_mode)

        self.exit_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        # encode (down)
        x = self.init_conv(x)
        row_1_aux = x

        x = self.encode1(x)
        row_2_aux = x

        x = self.encode2(x)
        row_3_aux = x

        x = self.encode3(x)
        row_4_aux = x

        x = self.encode4(x)

        x = nn.Dropout()(x)

        # decode (up)
        #print("x.shape: {}".format(x.shape))
        #print("row_4_aux.shape: {}".format(row_4_aux.shape))
        crop = center_crop(row_4_aux, int(x.shape[2]*2), int(x.shape[3]*2))
        #print("crop.shape: {}".format(crop.shape))
        x = self.decode1(x, crop)

        crop = center_crop(row_3_aux, int(x.shape[2]*2), int(x.shape[3]*2))
        x = self.decode2(x, crop)

        crop = center_crop(row_2_aux, int(x.shape[2]*2), int(x.shape[3]*2))
        x = self.decode3(x, crop)

        crop = center_crop(row_1_aux, int(x.shape[2]*2), int(x.shape[3]*2))
        x = self.decode4(x, crop)

        x = self.exit_conv(x)

        x = nn.Sigmoid()(x)
        
        return x;

    def name(self):
        return "UNet_IN-{}_OUT-{}_UPMODE-{}".format(self.in_channels, self.out_channels, self.up_mode)
    
    def model_input_channels(self):
        return self.in_channels
    
    def model_output_channels(self):
        return self.out_channels
    
    def model_up_mode(self):
        return self.up_mode
    
    def model_padding(self):
        return self.pad


def load_checkpoint(file_path='checkpoints/UNet_IN-1_OUT-2_UPMODE-ConvTranspose_Epoch-1000.pt'):

    # Checking for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # load the trained model
    checkpoint = torch.load(file_path) if torch.cuda.is_available() else torch.load(file_path, map_location=lambda storage, loc: storage)

    model_in_channels = checkpoint['model_in_channels']
    model_out_channels = checkpoint['model_out_channels']
    model_up_mode = checkpoint['model_up_mode']
    model_padding = keyCheck('model_padding', checkpoint, True)

    # recreate the model
    with torch.no_grad():
        model = UNet(in_channels=model_in_channels, out_channels=model_out_channels, up_mode=model_up_mode, padding=model_padding).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

    print('Model loaded on: {}'.format(device))
    return model


def keyCheck(key, arr, default):
    if key in arr.keys():
        return arr[key]
    return default