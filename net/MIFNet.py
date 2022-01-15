from net.BasicModule import *


# Multi-Information Fusion Network (Main Work)
class MultiInfoNet(nn.Module):
    def __init__(self, tInt, opt, mode='all', in_channels=11, resize=(1024, 512)):
        """
        :param tInt: velocity time interval | e.g. [0, 20, 40, ..., 7000]
        :param opt: Parameters
        :param mode: the mode of GatherEncoder | default: 'all'
        :param in_channels: the number of the input channels | default: 11
        :param resize: the output size of the STK encode | default: (1024, 512)
        """
        super(MultiInfoNet, self).__init__()
        self.UNet = UNet(in_channels)
        self.SE_Block = SEBlock(in_channels)
        self.STKEncoder = STKEncoder(tInt, opt, resize=resize, mode=mode)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, spec, gather, cv, VMM):
        """
        :param spec: velocity spectrum matrix             | shape = BS * K'* H * W
        :param gather: gather data                        | shape = BS * K * t * W'
        :param cv: t-v points (reference veloicty)        | shape = BS * K * N * 2
        :param VMM: minimum and maximum of velocity       | shape = BS * 2
        
        :return: 
            x: Segmentation Map                           | shape = BS * 1 * H * W
            stkFeaMap: Stk Data Encode                    | shape = BS * 1 * H * W
        """ 
        stkFeaMap = self.STKEncoder(gather, cv, VMM)
        xSE = self.SE_Block(torch.cat((spec, stkFeaMap), 1))
        x = self.UNet(xSE)
        return x, stkFeaMap
