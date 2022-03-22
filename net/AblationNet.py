from net.BasicModule import *
from net.MIFNet import MultiInfoNet

class MIUNet(nn.Module):
    def __init__(self, tInt, mode='mute', in_channels=10, resize=(1024, 512), device=0):
        super(MIUNet, self).__init__()
        self.UNet = UNet(in_channels)
        self.STKEncoder = STKEncoder(tInt, resize=resize, mode=mode, device=device)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, spec, gather, cv, VMM, save=False):
        Feature = {}
        stkFeaMap = self.STKEncoder(gather, cv, VMM)
        x, feature = self.UNet(torch.cat((spec, stkFeaMap), 1), save)
        if save:
            Feature.setdefault('Stk', stkFeaMap)
            for key, tensor in feature.items():
                Feature.setdefault(key, tensor)
        return x, Feature


class MixNet(nn.Module):
    def __init__(self, tInt, resize, NetType='all', mode='mute', device=0):
        super(MixNet, self).__init__()
        if NetType == 'all':
            self.net = MultiInfoNet(tInt, mode=mode, in_channels=11, resize=resize, device=device)
        elif NetType == 'en':
            self.net = UNet(input_channel=10)
        elif NetType == 'stkS':
            self.net = MIUNet(tInt, in_channels=2, resize=resize, device=device)
        elif NetType == 'su':
            self.net = UNet(input_channel=1)
        else:
            raise 'Error NetType: %s' % NetType
        self.NetType = NetType

    def forward(self, spec, gather, cv, VMM, save=False):
        if self.NetType == 'all':
            return self.net(spec, gather, cv, VMM, save)
        elif self.NetType == 'stkS':
            return self.net(spec[:, 0:1], gather, cv, VMM, save)
        elif self.NetType == 'en':
            return self.net(spec, save)
        elif self.NetType == 'su':
            return self.net(spec[:, 0:1], save)
