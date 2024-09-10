import torch
import torch.nn as nn
from torchvision.models import resnet18
#import torchsummary
from .non_local import NLBlockND
#from utils import Instance

class RefineEncoder(nn.Module):
    def __init__(self, n_input):
        super(RefineEncoder, self).__init__()
        
        self.base_model = resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())
        
        self.base_layers[0] = nn.Conv2d(n_input, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
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
    
    def forward(self, x):
        x_enc0 = self.conv0(x)
        x_enc1 = self.conv1(x_enc0)
        x_enc2 = self.conv2(x_enc1)
        x_enc3 = self.conv3(x_enc2)
        x_enc4 = self.conv4(x_enc3)

        x_enc = self.avg_pool(x_enc4)

        return x_enc
    
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
        self.up = nn.Sequential(
            nn.Conv2d(input_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        self.up_sampling = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=2, stride=2)

    def forward(self, x):
        y = self.up(x)
        y = self.up_sampling(y)
        return y


class RefineModel(nn.Module):
    def __init__(self, type: str, refine_model = None):
        ## Encoder
        super(RefineModel, self).__init__()
        self.image_encoder = RefineEncoder(n_input = 6)
        self.heatmap_encoder = RefineEncoder(n_input = 77)
        
        self.image_upsample = nn.ConvTranspose2d(512, 512, kernel_size=8, stride=1)
        self.heatmap_upsample = nn.ConvTranspose2d(512, 512, kernel_size=8, stride=1)


        ## Decoder
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)

        self.final = nn.Conv2d(64, 77, kernel_size=1, stride=1, padding=0)
        
        if type == "testing":
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
        
        
        if refine_model is not None:
            self.load_state_dict(torch.load(refine_model))
            print('[Model] : load pretrained model from %s'%(refine_model))
    
    def forward(self, x, pt_from):
        x = self.image_encoder(x)
        x = self.image_upsample(x)
        pt_from = self.heatmap_encoder(pt_from)
        pt_from = self.heatmap_upsample(pt_from)

        input_dec = torch.cat([x, pt_from], dim=1)

        y = self.dec1(input_dec)
        y = self.dec2(y)
        y = self.dec3(y)
        y = self.dec4(y)
        y = self.final(y)

        return y if self.sigmoid is None else self.sigmoid(y)
        #return self.sigmoid(y)
