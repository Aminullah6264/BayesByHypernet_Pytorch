import math
import numpy as np
import functools
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.distributions as dist

# Nonlinearities. Note that we bake the constant into the
# nonlinearites rather than the WS layers.
nonlinearities =    {'silu': lambda x: F.silu(x) / .5595,
                    'relu': lambda x: F.relu(x) / (0.5 * (1 - 1 / np.pi)) ** 0.5,
                    'identity': lambda x: x}


class HypernetWeight(nn.Module):
    def __init__(self, shape, units=[16, 32, 64], bias=True,
                 noise_shape=1, activation=nn.LeakyReLU(0.1)):
        super(HypernetWeight, self).__init__()
        self.shape = shape
        self.noise_shape = noise_shape

        layers = []
        in_features = noise_shape
        for out_features in units:
            layers.append(nn.Linear(in_features, out_features, bias=bias))
            layers.append(activation)
            in_features = out_features

        layers.append(nn.Linear(in_features, np.prod(shape), bias=bias))

        self.net = nn.Sequential(*layers)

    def forward(self, x=None, num_samples=1):
        if x is None:
            x = torch.randn((num_samples, self.noise_shape)).to('cuda')
        return self.net(x).reshape((x.shape[0], *self.shape))


class ScaledWSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                    stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True,
                    eps=1e-4, noise_shape= 1, units=[16, 32, 64]):
        super(ScaledWSConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups= groups
        self.gain = gain
        self.bias = bias
        self.conv1_weights_len = self.out_channels * self.kernel_size * self.kernel_size * self.in_channels
        self.conv1_w = HypernetWeight((self.conv1_weights_len, 1), noise_shape= noise_shape, units=units)
        
        if self.bias:
            self.conv1_bias_len = self.out_channels 
            self.conv1_b = HypernetWeight((self.conv1_bias_len, ), noise_shape= noise_shape, units=units)
        else: 
            self.conv1_b = None

        if self.gain:
            self.conv1_gain_len = self.out_channels 
            self.conv1_gain = HypernetWeight((self.conv1_gain_len, ), noise_shape= noise_shape, units=units)
        else:
            self.conv1_gain 
        
    def standardize_weight(self, weight, eps=1e-4):
        # Get Scaled WS weight OIHW;
        fan_in = np.prod(weight.shape[1:])
        mean = torch.mean(weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(weight, axis=[1, 2, 3], keepdims=True)
        weight = (weight - mean) / (var * fan_in + eps) ** 0.5

        if self.gain:
            _conv1_gain = self.conv1_gain()[0]
            _conv1_gain = _conv1_gain.view(self.out_channels, 1, 1, 1)
            weight = weight * _conv1_gain
        return weight

    def forward(self, x):
        
        _conv1_w = self.conv1_w()[0]
        _conv1_w = _conv1_w.view(self.out_channels, self.in_channels,
                                 self.kernel_size, self.kernel_size)
        _conv1_w = self.standardize_weight(_conv1_w)
            
        if self.bias:
            _conv1_b = self.conv1_b()[0]
        else: 
            _conv1_b = None

        res = F.conv2d(x, _conv1_w, _conv1_b,
                    self.stride, self.padding,
                    self.dilation, self.groups)

        return res

    def sample(self, num_samples=5):
        _conv1_w_samples = self.conv1_w(num_samples=num_samples).view((num_samples, -1))
        _conv1_b_samples = self.conv1_b(num_samples=num_samples).view((num_samples, -1))
        _conv1_gain_samples = self.conv1_gain(num_samples=num_samples).view((num_samples, -1))

        gen_weights = torch.cat([_conv1_w_samples, _conv1_b_samples, _conv1_gain_samples], 1)

        return gen_weights

    
class SqueezeExcite(nn.Module):
    """Simple Squeeze+Excite layers."""
    def __init__(self, in_channels, out_channels, activation, bias = True, noise_shape=1, units=[16, 32, 64]):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 1
        self.bias = bias
        self.conv1_weights_len = self.out_channels * self.kernel_size * self.kernel_size * self.in_channels
        self.conv1_w = HypernetWeight((self.conv1_weights_len, 1), noise_shape=noise_shape, units=units)
        
        if self.bias:
            self.conv1_bias_len = self.out_channels 
            self.conv1_b = HypernetWeight((self.conv1_bias_len, ), noise_shape= noise_shape, units=units)
        else: 
            self.conv1_b = None


        self.conv2_weights_len = self.out_channels * self.kernel_size * self.kernel_size * self.in_channels
        self.conv2_w = HypernetWeight((self.conv1_weights_len, 1), noise_shape= noise_shape, units=units)
        
        if self.bias:
            self.conv2_bias_len = self.in_channels
            self.conv2_b = HypernetWeight((self.conv1_bias_len, ), noise_shape= noise_shape, units=units)
        else: 
            self.conv2_b = None

        self.activation = activation

    def forward(self, x):
        
        # Mean pool for NCHW tensors
        h = torch.mean(x, axis=[2, 3], keepdims=True)
        # Apply two linear layers with activation in between

        _conv1_w = self.conv1_w()[0]
        _conv1_w = _conv1_w.view(self.out_channels, self.in_channels,
                                 self.kernel_size, self.kernel_size)
            
        if self.bias:
            _conv1_b = self.conv1_b()[0]
        else: 
            _conv1_b = None

        _conv2_w = self.conv2_w()[0]
        _conv2_w = _conv2_w.view(self.out_channels, self.in_channels,
                                 self.kernel_size, self.kernel_size)
            
        if self.bias:
            _conv2_b = self.conv2_b()[0]
        else: 
            _conv2_b = None

        res = self.activation(F.conv2d(x, _conv1_w, _conv1_b))
        h = F.conv2d(res, _conv2_w, _conv2_b)
        # Rescale the sigmoid output and return
        return (torch.sigmoid(h) * 2) * x


    def sample(self, num_samples=5):
        _conv1_w_samples = self.conv1_w(num_samples=num_samples).view((num_samples, -1))
        _conv1_b_samples = self.conv1_b(num_samples=num_samples).view((num_samples, -1))

        gen_weights = torch.cat([_conv1_w_samples, _conv1_b_samples, _conv2_w_samples, _conv2_b_samples], 1)

        return gen_weights


class NF_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=F.relu, which_conv=ScaledWSConv2d,
    beta=1.0, alpha=1.0, noise_shape= 1, units=[16, 32, 64], se_ratio=0.5):
        super(NF_BasicBlock, self).__init__()

        self.activation = activation
        self.beta, self.alpha = beta, alpha

        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = which_conv(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = which_conv(planes, planes, kernel_size=3, stride=1, padding=1)
        

        if stride != 1 or in_planes != planes:
            self.shortcut_conv = which_conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride)
        
        self.se = SqueezeExcite(self.expansion * planes, self.expansion * planes, self.activation)
        self.skipinit_gain_w = HypernetWeight((1, ), noise_shape= noise_shape, units= units)

    def forward(self, x):
        out = self.activation(x) / self.beta
        if self.stride != 1 or self.in_planes != self.planes:
            shortcut = self.shortcut_conv(out)
        else:
            shortcut = x
        out = self.conv1(out) # Initial bottleneck conv
        out = self.conv2(self.activation(out)) # Spatial conv
        out = self.se(out) # Apply squeeze + excite to middle block.
        
        skipinit_gain = self.skipinit_gain_w()[0] 

        return out * skipinit_gain * self.alpha + shortcut

    def sample(self, num_samples=5):
        _conv1_w_samples = self.conv1.sample(num_samples)
        _conv2_w_samples = self.conv2.sample(num_samples)
        if self.stride != 1 or self.in_planes != self.planes:
            _conv1_shortcut_samples = self.shortcut_conv.sample(num_samples)
        
        _skipinit_gain_w_samples = self.skipinit_gain_w(num_samples=num_samples).view((num_samples, -1))
        if self.stride != 1 or self.in_planes != self.planes:
            gen_weights = torch.cat([_conv1_w_samples, _conv2_w_samples, _conv1_shortcut_samples, _skipinit_gain_w_samples], 1)
        else:
            gen_weights = torch.cat([_conv1_w_samples, _conv2_w_samples, _skipinit_gain_w_samples], 1)

            
        return gen_weights

class NF_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, se_ratio=0.5, alpha=0.2, 
    activation='relu', drop_rate=None, stochdepth_rate=0.0, noise_shape=1, units=[16, 32, 64]):
        super(NF_ResNet, self).__init__()
        self.in_planes = 16
        self.se_ratio = se_ratio
        self.alpha = alpha
        self.activation = nonlinearities.get(activation)
        self.stochdepth_rate = stochdepth_rate
        self.which_conv = functools.partial(ScaledWSConv2d, gain=True, bias=True)

        self.conv1 = self.which_conv(3, 16, kernel_size=3, stride=1, padding=1, units=[16, 32, 64])
        expected_var = 1.0
        beta = expected_var ** 0.5
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, activation=self.activation,
                                    which_conv=self.which_conv,
                                    beta=beta, alpha=self.alpha,
                                    se_ratio=self.se_ratio, noise_shape=noise_shape, units=units)
        expected_var += self.alpha ** 2
        beta = expected_var ** 0.5
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, activation=self.activation,
                                    which_conv=self.which_conv,
                                    beta=beta, alpha=self.alpha,
                                    se_ratio=self.se_ratio, noise_shape=noise_shape, units=units)

        expected_var += self.alpha ** 2
        beta = expected_var ** 0.5
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, activation=self.activation,
                                    which_conv=self.which_conv,
                                    beta=beta, alpha=self.alpha,
                                    se_ratio=self.se_ratio, noise_shape=noise_shape, units=units)
        last_layer_weight_len = 64*num_classes
        last_layer_bias_len = num_classes


        self.linear_w = HypernetWeight((last_layer_weight_len,1), noise_shape=noise_shape,units=units)
        self.linear_b = HypernetWeight((last_layer_bias_len,), noise_shape=noise_shape, units=units)

        

    def _make_layer(self, block, planes, num_blocks, stride, activation=F.relu, which_conv=ScaledWSConv2d,
    beta=1.0, alpha=1.0, se_ratio=0.5, noise_shape = 1, units=[16, 32, 64]):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation=activation, which_conv=which_conv,
    beta=beta, alpha=alpha, se_ratio=se_ratio, noise_shape=noise_shape, units=units))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)


        _last_layer_w = self.linear_w()[0]
        _last_layer_w = _last_layer_w.view(10, 64)
        _last_layer_b = self.linear_b()[0]
        out = F.linear(out, _last_layer_w, _last_layer_b)

        return out

    
    def sample(self, num_samples=5):
        all_gen_weights = []
        _conv1_w_samples = self.conv1.sample(num_samples)
        all_gen_weights.append(_conv1_w_samples)

        for conv in self.layer1:
            all_gen_weights.append(conv.sample(num_samples))
        for conv in self.layer2:
            all_gen_weights.append(conv.sample(num_samples))
        for conv in self.layer3:
            all_gen_weights.append(conv.sample(num_samples))

        l_w_samples = self.linear_w(num_samples=num_samples).view((num_samples, -1))
        l_b_samples = self.linear_b(num_samples=num_samples).view((num_samples, -1))
        all_gen_weights.append(l_w_samples)
        all_gen_weights.append(l_b_samples)
       
        gen_weights = torch.cat(all_gen_weights, 1)

        return gen_weights

    def kl(self, num_samples=5, full_kernel=True):

        gen_weights = self.sample(num_samples=num_samples)
        gen_weights = gen_weights.transpose(1, 0)
        prior_samples = torch.randn_like(gen_weights).to('cuda')

        eye = torch.eye(num_samples, device=gen_weights.device)
        wp_distances = (prior_samples.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2
        # [weights, samples, samples]

        ww_distances = (gen_weights.unsqueeze(2) - gen_weights.unsqueeze(1)) ** 2

        if full_kernel:
            wp_distances = torch.sqrt(torch.sum(wp_distances, 0) + 1e-8)
            wp_dist = torch.min(wp_distances, 0)[0]

            ww_distances = torch.sqrt(
                torch.sum(ww_distances, 0) + 1e-8) + eye * 1e10
            ww_dist = torch.min(ww_distances, 0)[0]

            # mean over samples
            kl = torch.mean(torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8))
            kl *= gen_weights.shape[0]
            kl += np.log(float(num_samples) / (num_samples - 1))
        else:
            wp_distances = torch.sqrt(wp_distances + 1e-8)
            wp_dist = torch.min(wp_distances, 1)[0]

            ww_distances = (torch.sqrt(ww_distances + 1e-8)
                            + (eye.unsqueeze(0) * 1e10))
            ww_dist = torch.min(ww_distances, 1)[0]

            # sum over weights, mean over samples
            kl = torch.sum(torch.mean(
                torch.log(wp_dist / (ww_dist + 1e-8) + 1e-8)
                + torch.log(float(num_samples) / (num_samples - 1)), 1))

        return kl

       
def NF_ResNet18(noise_shape= 8, units=[16, 32, 64]):
    return NF_ResNet(NF_BasicBlock, [3, 3, 3], num_classes=10, noise_shape= noise_shape, units=units)


# model = NF_ResNet18(noise_shape= 8, units=[16, 32, 64]).to('cuda')
# from torchsummary import summary
# summary(model, (3, 32, 32))


