import torch
import torch.nn as nn
from torchvision.models import resnet18
#import torchsummary
from .non_local import NLBlockND
from loguru import logger
#from utils import Instance

class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels, inter_channels=None):
        '''
            args:
                input_channels : number of input channels
                output_channels : number of output channels
                inter_channels : number of intermediate channels
            description:
                initialize the decoder block
                if inter_channels is None, then inter_channels = input_channels
            
        '''
        super().__init__()
        if inter_channels is None:
            inter_channels = input_channels
        self.up_sampling = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            nn.Conv2d(inter_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x, x_skip):
        x = self.up_sampling(x)
        x_input = torch.cat([x, x_skip], 1) # skip connection
        y = self.up(x_input)
        return y


class ResNetUNet(nn.Module):
    def __init__(
        self, 
        pretrained_pth : str = None,
        model_type : str = 'testing'
    ):
        
        ## Encoder
        super(ResNetUNet, self).__init__()
        
        self.base_model = resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())
        
        self.base_layers[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.conv0 = nn.Sequential(*self.base_layers[:3])
        self.conv1 = nn.Sequential(*self.base_layers[3:5])
        self.conv2 = self.base_layers[5]
        self.conv3 = nn.Sequential(
            *self.base_layers[6],
            NLBlockND(in_channels=256, inter_channels=128, mode='dot', dimension=2),
        )
        self.conv4 = nn.Sequential(
            *self.base_layers[7],
            NLBlockND(in_channels=512, inter_channels=256, mode='dot', dimension=2),
        )

        ### change Conv2d to DilatedConv2d
        for layer in self.conv3[:-1]:
            layer.conv1.dilation = (2, 2)
            layer.conv1.padding = (2, 2)
            layer.conv2.dilation = (2, 2)
            layer.conv2.padding = (2, 2)
        
        for layer in self.conv4[:-1]:
            layer.conv1.dilation = (2, 2)
            layer.conv1.padding = (2, 2)
            layer.conv2.dilation = (2, 2)
            layer.conv2.padding = (2, 2)

        
        self.avg_pool = self.base_layers[8]
        self.upsample = nn.ConvTranspose2d(512, 512, kernel_size=8, stride=1, padding=0, bias=False)
        self.fc = self.base_layers[9]


        ## Decoder
        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec4 = DecoderBlock(64, 64, 128)

        self.final = nn.Conv2d(64, 77, kernel_size=1, stride=1, padding=0)
        
        if model_type == 'training':
            self.sigmoid = None
        else:
            self.sigmoid = nn.Sigmoid()
        
        if pretrained_pth is not None:
            self.load_state_dict(torch.load(pretrained_pth))
            logger.info('[Model] : load pretrained model from %s'%(pretrained_pth))
    
    def forward(self, x):
        x_enc0 = self.conv0(x)
        x_enc1 = self.conv1(x_enc0)
        x_enc2 = self.conv2(x_enc1)
        x_enc3 = self.conv3(x_enc2)
        x_enc4 = self.conv4(x_enc3)

        x_enc = self.avg_pool(x_enc4)
        
        x_up = self.upsample(x_enc)
        x = self.dec1(x_up, x_enc3)
        x = self.dec2(x, x_enc2)
        x = self.dec3(x, x_enc1)
        x = self.dec4(x, x_enc0)

        x = self.final(x)
        #print(x.shape)
        # x= self.base_model(x)
        return x if self.sigmoid is None else self.sigmoid(x)
        #return self.sigmoid(x)
        
    def feature_extraction(self, x):
        with torch.no_grad():
            x_enc0 = self.conv0(x)
            x_enc1 = self.conv1(x_enc0)
            x_enc2 = self.conv2(x_enc1)
            x_enc3 = self.conv3(x_enc2)
            x_enc4 = self.conv4(x_enc3)
            x_enc = self.avg_pool(x_enc4)
            x_enc = torch.flatten(x_enc, 2) # (B, 512, 1, 1) -> (B, 512, 1)
            return x_enc
