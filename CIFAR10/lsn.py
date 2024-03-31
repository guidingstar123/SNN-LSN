import random
import torch
import torch.nn as nn
import cupy
from spikingjelly.clock_driven import neuron, layer, functional, surrogate
import os



assert cupy is not None
# ------------------- #
#   ResNet Example    #
# ------------------- #


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(inplanes, planes, stride)
        self.bn3 = norm_layer(planes)
        self.conv4 = conv3x3(planes, planes)
        self.bn4 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = layer.SeqToANNContainer(self.conv1, self.bn1)
        self.conv2_s = layer.SeqToANNContainer(self.conv2, self.bn2)
        self.conv3_s = layer.SeqToANNContainer(self.conv3, self.bn3)
        self.conv4_s = layer.SeqToANNContainer(self.conv4, self.bn4)
        self.spikeout1 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout2 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout3 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout4 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout5 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)

        

    def forward(self, x):
        identity = x

        out = self.conv1_s(x)
        out = self.spikeout1(out)
        out = self.conv2_s(out)
        out = self.spikeout2(out)
        
        out2 = self.conv3_s(x)
        out2 = self.spikeout3(out2)
        out2 = self.conv4_s(out2)
        out2 = self.spikeout4(out2)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.spikeout5(identity)

        #output = identity*(1-out)+(1-identity)*out
        output = identity*out+(1-identity)*out*(1-out2)+identity*(1-out)*(1-out2)

        return output



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.conv4 = conv1x1(inplanes, width)
        self.bn4 = norm_layer(width)
        self.conv5 = conv3x3(width, width, stride, groups, dilation)
        self.bn5 = norm_layer(width)
        self.conv6 = conv1x1(width, planes * self.expansion)
        self.bn6 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = layer.SeqToANNContainer(self.conv1, self.bn1)
        self.conv2_s = layer.SeqToANNContainer(self.conv2, self.bn2)
        self.conv3_s = layer.SeqToANNContainer(self.conv3, self.bn3)
        self.conv4_s = layer.SeqToANNContainer(self.conv4, self.bn4)
        self.conv5_s = layer.SeqToANNContainer(self.conv5, self.bn5)
        self.conv6_s = layer.SeqToANNContainer(self.conv6, self.bn6)
        self.spikeout1 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout2 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout3 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout4 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout5 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout6 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        self.spikeout7 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)

    def forward(self, x):
        identity = x

        out = self.conv1_s(x)
        out = self.spikeout1(out)
        out = self.conv2_s(out)
        out = self.spikeout2(out)
        out = self.conv3_s(out)
        out = self.spikeout3(out)
        
        out2 = self.conv4_s(x)
        out2 = self.spikeout4(out2)
        out2 = self.conv5_s(out2)
        out2 = self.spikeout5(out2)
        out2 = self.conv6_s(out2)
        out2 = self.spikeout6(out2)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.spikeout7(identity)

        #output = identity*(1-out)+(1-identity)*out
        output = identity*out+(1-identity)*out*(1-out2)+identity*(1-out)*(1-out2)

        return output


class LSN(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(LSN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #7 2 3
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        #self.maxpool = tdLayer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 96, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 192, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 384, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
                                       
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc1 = nn.Linear(384 * block.expansion, 10)
        self.fc1_s = layer.SeqToANNContainer(self.fc1)
        #self.fc2 = nn.Linear(256, num_classes)
        #self.fc2_s = layer.SeqToANNContainer(self.fc2)
        
        self.spike1 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        #self.spike2 = neuron.MultiStepLIFNode(surrogate_function=surrogate.Tri(alpha=1), backend='cupy', decay_input=False)
        #self.spike_out = LIFSpikeOut()
        self.T = 4

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = layer.SeqToANNContainer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        

        #x = self.maxpool(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.spike1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1_s(x)
        #x = self.spike2(x)
        #x = self.fc2_s(x)
        #x = self.spike_out(x)
        return x.mean(0)

    def forward(self, x):
        return self._forward_impl(x)



def _lsn(arch, block, layers, pretrained, progress, **kwargs):
    model = LSN(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def lsn18(pretrained=False, progress=True, **kwargs):
 
    return _lsn("lsn18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
 
 
def lsn34(pretrained=False, progress=True, **kwargs):

    return _lsn("lsn34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)
 
 
def lsn50(pretrained=False, progress=True, **kwargs):

    return _lsn("lsn50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
 
 
def lsn101(pretrained=False, progress=True, **kwargs):

    return _lsn("lsn101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
 
 
def lsn152(pretrained=False, progress=True, **kwargs):

    return _lsn("lsn152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    model = lsn18(num_classes=1000)
    model.T = 3
    x = torch.rand(2,3,224,224)
    y = model(x).mean(0)
    
    print(y.shape)