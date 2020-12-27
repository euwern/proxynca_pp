import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Feature(nn.Module):
    def __init__(self, model='b0', pool='avg'):
        nn.Module.__init__(self)

        self.base = EfficientNet.from_pretrained('efficientnet-' + model)

        if pool == 'max':
            self.base._avg_pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.base._fc = nn.LayerNorm(1280, elementwise_affine=False).cuda()

    def forward(self, x):
        return self.base(x)


class Feat_efficientNet_b0_max(Feature):
    def __init__(self):
        Feature.__init__(self, model='b0', pool='max')


class Feat_efficientNet_b0_avg(Feature):
    def __init__(self):
        Feature.__init__(self, model='b0', pool='avg')


class Feat_efficientNet_b1_max(Feature):
    def __init__(self):
        Feature.__init__(self, model='b1', pool='max')


class Feat_efficientNet_b1_avg(Feature):
    def __init__(self):
        Feature.__init__(self, model='b1', pool='avg')
