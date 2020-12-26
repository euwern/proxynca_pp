import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Feature(nn.Module):
    def __init__(self, model='b0', pool='avg', use_lnorm=False):
        nn.Module.__init__(self)
        # self.model = model

        # self.base = models.__dict__[model](pretrained=True)
        # if pool == 'avg':
        #     self.pool = nn.AdaptiveAvgPool2d((1, 1))        
        # elif pool == 'max':
        #     self.pool = nn.AdaptiveMaxPool2d((1, 1))
        # else:
        #     raise Exception('pool: %s pool must be avg or max', str(pool))

        self.base = EfficientNet.from_pretrained('efficientnet-' + model)

        self.lnorm = None
        if use_lnorm:
            # self.lnorm = nn.LayerNorm(2048, elementwise_affine=False).cuda()
            self.lnorm = nn.LayerNorm(1280, elementwise_affine=False).cuda()

    def forward(self, x):
        # x = self.base.conv1(x)
        # x = self.base.bn1(x)
        # x = self.base.relu(x)
        # x = self.base.maxpool(x)

        # x = self.base.layer1(x)
        # x = self.base.layer2(x)
        # x = self.base.layer3(x)
        # x = self.base.layer4(x)

        self.base._fc = Identity()
        self.base._swish = Identity()

        x = self.base(x)

        # x1 = self.pool(x)
        # x = x1
        x = x.reshape(x.size(0), -1)

        if self.lnorm != None:
            x = self.lnorm(x)

        return x


class Feat_efficientNet_b0_max(Feature):
    def __init__(self):
        Feature.__init__(self, model='b0', pool='max')


class Feat_efficientNet_b0_avg(Feature):
    def __init__(self):
        Feature.__init__(self, model='b0', pool='avg')


class Feat_efficientNet_b0_max_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='b0', pool='max', use_lnorm=True)


class Feat_efficientNet_b0_avg_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='b0', pool='avg', use_lnorm=True)

class Feat_efficientNet_b1_max(Feature):
    def __init__(self):
        Feature.__init__(self, model='b1', pool='max')


class Feat_efficientNet_b1_avg(Feature):
    def __init__(self):
        Feature.__init__(self, model='b1', pool='avg')


class Feat_efficientNet_b1_max_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='b1', pool='max', use_lnorm=True)


class Feat_efficientNet_b1_avg_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='b1', pool='avg', use_lnorm=True)

