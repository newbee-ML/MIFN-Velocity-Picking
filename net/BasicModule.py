import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.LoadData import interpolation


# Conv + BatchNorm + LeakyReLU 
class CBL(nn.Module):
    def __init__(self, InChannal=1, OutChannal=1, K=(5, 3), S=(3, 2)):
        super(CBL, self).__init__()
        layers = [
            nn.Conv2d(InChannal, OutChannal, kernel_size=K, stride=S, bias=True),
            nn.BatchNorm2d(OutChannal),
            nn.LeakyReLU(0.1, inplace=True)
            ]
        self.CBL = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.CBL(x)


# Conv + BatchNorm + ReLU 
class CBR(nn.Module):
    def __init__(self, InChannal=1, OutChannal=1, K=(5, 3), S=(3, 2)):
        super(CBR, self).__init__()
        layers = [
            nn.Conv2d(InChannal, OutChannal, kernel_size=K, stride=S, bias=True),
            nn.BatchNorm2d(OutChannal),
            nn.ReLU()
            ]
        self.CBR = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.CBR(x)


# Spatial Pyramid Pooling 
class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.CBL1 = CBL(1, 1)
        self.CBL2 = CBL(3, 2)
        self.CBL3 = CBL(2, 1)
        self.Pool1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=3 // 2)
        self.Pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
 
    def forward(self, x):
        x = self.CBL1(x)
        xP1 = self.Pool1(x)
        xP2 = self.Pool2(x)
        xCat = torch.cat([x, xP1, xP2],dim=1)
        xCat = self.CBL2(xCat)
        x = self.CBL3(xCat)
        return x


# Encoder for Stack Gather
class GatherEncoder(nn.Module):
    def __init__(self, outLen=512, mode='all'):
        super(GatherEncoder, self).__init__()
        """
        :param mode: P --> positive part
                     N --> Negative part
                     all --> all data
                     mute --> keep original signal
        """

        self.mode = mode
        self.outLen = outLen
        self.SPP = SPP()

    def forward(self, x):
        """
        :param x: gather data        | shape =  N*K*t*W  

        :return: gather encode       | shape =  N*1*outLen*1
        """
        ########### split positive and negative ######################
        # default: split = positive + (-negative)
        NegP = -torch.where(x > 0, torch.zeros_like(x), x)
        PosP = torch.where(x < 0, torch.zeros_like(x), x)
        if self.mode == 'all':
            FeaX = PosP + NegP
        else:
            FeaX = x
        
        # scale the value to [0, 1] for 'all' or [-1, 1] for 'mute'
        MaxValue = max(torch.max(NegP), torch.max(PosP))
        FeaX /= MaxValue  

        ########### encode gather information ########################
        # extract multi-scale feature by SPP
        outTensor = self.SPP(FeaX)
        # interpolate the outTensor to a setting length(self.outLen)
        reLenTensor = F.interpolate(outTensor, (self.outLen, 1), mode='bilinear')

        return reLenTensor


# Encoder for Velocity Curve
class CVEncoder(nn.Module):
    def __init__(self, tInt, opt, resize=(512, 256), ScaleSize=(256, 256)):
        super(CVEncoder, self).__init__()
        """
        :param tInt: velocity time interval             | e.g. [0, 20, 40, ..., 7000]
        :param opt: Parameters
        :param resize: the output size of CV encode     | default: (512, 256)
        :param ScaleSize: the encode resolution         | default: (256, 256)
        """
        self.tInt = tInt
        self.resize = resize
        self.opt = opt
        self.ScaleSize = ScaleSize

    def forward(self, VelPoints, VMM):
        """
        :param VMM: minimum and maximum of velocity       | shape = BS * 2
        :param VelPoints: t-v points (reference veloicty) | shape = BS * K * N * 2

        :return: Encode                                   | shape = BS * K * T * V
        """

        BS, K, H, W = VelPoints.shape[0], VelPoints.shape[1], self.ScaleSize[0], self.ScaleSize[1]

        ############### scale the t-v points from real scale to pixel scale ##################
        # map original scale to new scale on time domain
        ntInt = np.linspace(self.tInt[0], self.tInt[-1], H)
        VelPoints[..., 0] = (VelPoints[..., 0] - ntInt[0]) / (ntInt[1] - ntInt[0])

        # map original scale to new scale on velocity domain
        for batch in range(BS):
            nvInt = np.linspace(VMM[batch, 0], VMM[batch, 1], W)
            VelPoints[batch, ..., 1] = (VelPoints[batch, ..., 1] - nvInt[0]) / (nvInt[1] - nvInt[0])

        VelPoints = VelPoints.view((-1, VelPoints.shape[-2], 2)).cpu().numpy()
        VelCurve = torch.zeros((VelPoints.shape[0], H, 2), dtype=torch.int64)

        ############### transform the t-v points to velocity curve mask ###########################
        for i in range(VelCurve.shape[0]):
            # VelPoints to VelCurve
            VelPointsN = VelPoints[i, ...]
            VelPointsN = VelPointsN[VelPointsN[:, 0] > 0, :]
            VelCurve[i, ...] = torch.tensor(interpolation(VelPointsN, np.arange(H), np.arange(W)), dtype=torch.int64)

        # make the velocity curve mask (map the velocity curve to a t-v 2D map)
        VelCurveHot = VelCurve[..., 0] * W + VelCurve[..., 1]
        SoftMask = torch.ones((VelCurveHot.shape[0], H * W)) * 0.01
        MaskOne = torch.ones_like(VelCurveHot) * 0.9
        SoftMask.scatter_add_(1, VelCurveHot, MaskOne)
        SoftMask = SoftMask.view((BS, K, H, W)).cuda(self.opt.GPUNO)
        # resize the mask to the size we need 
        RescaleSoftMask = F.interpolate(SoftMask, self.resize, mode='bilinear')

        return RescaleSoftMask


# Encoder for Stack Data (consists of CVEncoder and GatherEncoder)
class STKEncoder(nn.Module):
    def __init__(self, tInt, opt, mode='all', resize=(1024, 512)):
        super(STKEncoder, self).__init__()
        """
        :param tInt: velocity time interval                     | e.g. [0, 20, 40, ..., 7000]
        :param opt: Parameters
        :param mode: the mode of GatherEncoder                  | default: 'all'
        :param resize: the output size of the STK encode        | default: (1024, 512)
        """
        self.GEncoder = GatherEncoder(outLen=resize[0], mode=mode)
        self.CVEncoder = CVEncoder(tInt, opt, resize=resize)
        self.CBL = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True), 
                                 nn.BatchNorm2d(1), 
                                 nn.LeakyReLU(0.1, inplace=True))
        self.init_weight()

    def forward(self, gather, cv, VMM):
        """
        :param gather: gather data                        | shape = BS * K * t * W
        :param cv: t-v points (reference veloicty)        | shape = BS * K * N * 2
        :param VMM: minimum and maximum of velocity       | shape = BS * 2
        
        :return: Stk Data Encode                          | shape = BS * 1 * H * W
        """
        gatherNew = gather.view((-1, 1, gather.shape[-2], gather.shape[-1]))
        GEnR = self.GEncoder(gatherNew).view((gather.shape[0], gather.shape[1], -1, 1))
        with torch.no_grad():
            CVMask = self.CVEncoder(cv, VMM)
        Multiply = torch.mul(GEnR, CVMask)
        MaxFeature = torch.sum(Multiply, dim=1).unsqueeze(1)
        MixX = self.CBL(MaxFeature)
        return MixX

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)



# Down-Sample Block for U-Net
class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=True):
        super(UNetDownBlock, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if down_sample:
            layers.append(nn.MaxPool2d(2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Up-Sample Block for U-Net
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.ReLU()
        )

    def forward(self, x, skip_input):
        x = self.up_sample(x)
        x = torch.cat((x, skip_input), 1)
        x = self.conv(x)
        return x


# U-Net Segmentation Network 
class UNet(nn.Module):
    def __init__(self, input_channel=1):
        super(UNet, self).__init__()
        self.down1 = UNetDownBlock(input_channel, 8, False)
        self.down2 = UNetDownBlock(8, 16)
        self.down3 = UNetDownBlock(16, 32)
        self.down4 = UNetDownBlock(32, 64)

        self.center = UNetDownBlock(64, 128)

        self.up4 = UNetUpBlock(128, 64)
        self.up3 = UNetUpBlock(64, 32)
        self.up2 = UNetUpBlock(32, 16)
        self.up1 = UNetUpBlock(16, 8)

        self.last = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        d1 = self.down1(x)  # 8     H       W
        d2 = self.down2(d1)  # 16    H/2     W/2
        d3 = self.down3(d2)  # 32    H/4     W/4
        d4 = self.down4(d3)  # 64    H/8     W/8

        out = self.center(d4)  # 128  H/16   W/16

        out = self.up4(out, d4)  # 64    H/8     W/8
        out = self.up3(out, d3)  # 32    H/4     W/4
        out = self.up2(out, d2)  # 16    H/2     W/2
        out = self.up1(out, d1)  # 8     H       W

        out = self.last(out)  # 1     H       W

        return out

        
if __name__ == '__main__':
    x = torch.rand((2, 1, 3500, 15))
    f = GatherEncoder()
    print(f(x).shape)
