import torchvision.models as models
import torch.nn as nn


class Feature(nn.Module):
    def __init__(self, model='resnet50', pool='avg', use_lnorm=False):
        nn.Module.__init__(self)
        self.model = model

        self.base = models.__dict__[model](pretrained=True)
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise Exception('pool: %s pool must be avg or max', str(pool))

        self.lnorm = None
        if use_lnorm:
            self.lnorm = nn.LayerNorm(2048, elementwise_affine=False).cuda()

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x1 = self.pool(x)
        x = x1
        x = x.reshape(x.size(0), -1)

        if self.lnorm is not None:
            x = self.lnorm(x)

        return x




class Feat_resnet50_max(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet50', pool='max')


class Feat_resnet50_avg(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet50', pool='avg')


class Feat_resnet50_max_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet50', pool='max', use_lnorm=True)


class Feat_resnet50_avg_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet50', pool='avg', use_lnorm=True)


class Feat_resnet152_max_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet152', pool='max', use_lnorm=True)


class Feat_resnet152_max(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet152', pool='max')


class Feat_wide_resnet50_2_max_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='wide_resnet50_2', pool='max', use_lnorm=True)


class Feat_wide_resnet101_2_max_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='wide_resnet101_2', pool='max', use_lnorm=True)

