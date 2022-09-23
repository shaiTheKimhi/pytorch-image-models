import torch
from torch import nn
import torch.nn.functional as F

__all__ = ['MobileNetV2', 'mobilenet_v2']



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 skip=2,
                 skip_start=True,
                 exclusive_skip=False):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        self.skip_start = skip_start
        self.skip = skip
        self.exclusive_forward = exclusive_skip
        start = 3 if skip_start else 0

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        n = [r[2] for r in inverted_residual_setting]
        s = [r[3] for r in inverted_residual_setting]
        sums = [sum(n[:i+1]) for i in range(len(n))]
        self.layer_pools = [0] + [(i+1) for i in range(sum(n)) if (s[sums.index([j for j in sums if j>i][0])]==2 and len([j for j in sums if j == i]) > 0)]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        n = sums[-1]
        input_channel = _make_divisible(input_channel * width_mult, round_nearest) + 17 * skip #TODO: + skip
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        input_channel += start
        count = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                count += 1
                stride = s if i == 0 else 1
                output_channel += (17 - count * int(self.exclusive_forward)) * skip
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer)) #TODO: make output channel + (17-previous)*skip
                input_channel = output_channel + start + count * skip #TODO: + skip*previous
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.last_channel += count + 1 + start
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x): #TODO: add skips in forward
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        #x = self.features(x)
        n = len(self.features)

        start = [x] if self.skip_start else []
        prev = []
        for i in range(n):
            prev_pooled = [start[0]] if self.skip_start else []
            for j in range(len(prev[:-1])):
                d = i - j
                if d == 0:
                    prev_pooled.append(prev[:-1][j][:,(-d-1)*self.skip:,:,:])
                else:
                    prev_pooled.append(prev[:-1][j][:,(-d-1)*self.skip:-d*self.skip,:,:])
            
            for idx in range(len(prev_pooled)):
                val = self.greater(idx, i, self.layer_pools)
               
                p = F.adaptive_avg_pool2d
                size = int(prev_pooled[idx].shape[-1] / 2**val)
                # if prev_pooled[idx].shape[1] == 0:
                #     prev_pooled = prev_pooled[:idx] + prev_pooled[idx+1:]
                #     continue
                prev_pooled[idx] = p(prev_pooled[idx], (size, size))
            
            if len(prev) == 0:
                inx = x
            else:
                temp = prev[-1] if not self.exclusive_forward else prev[-1][:,:-(n-1-i)*self.skip,:,:] if i < 8 else prev[-1]
                inx = torch.cat(prev_pooled + [temp], dim=1) 

            prev.append(self.features[i](inx))
        
        prev_pooled = start + [z[:,-n-1:-n, :, :] for z in prev[:-1]]
        for idx in range(len(prev_pooled)):
                val = self.greater(idx, n, self.layer_pools)
            
                p = F.adaptive_avg_pool2d
                
                size = int(prev_pooled[idx].shape[-1] / 2**val)
                prev_pooled[idx] = p(prev_pooled[idx], (size, size))

        
        prev_pooled = [z for z in prev_pooled if z.shape[1] > 0]
        x = torch.cat(prev_pooled + [prev[-1]], dim=1) 
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def greater(self, val1, val2, list):
        return len([x for x in list if x >= val1 and x < val2]) if self.skip_start else len([x for x in list if x > val1 and x < val2])


    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
   
    return model


if __name__ == "__main__":
    m = mobilenet_v2(skip=2)
    x = torch.randn(1,3,64,64)
    y = m(x)
    print(y.shape)
