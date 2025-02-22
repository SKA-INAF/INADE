import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class ILADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, noise_nc, add_sketch):
        super().__init__()
        self.norm_nc = norm_nc
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        # wights and bias for each class
        self.weight = nn.Parameter(torch.Tensor(label_nc, norm_nc,2))
        self.bias = nn.Parameter(torch.Tensor(label_nc, norm_nc,2))
        self.reset_parameters()
        self.fc_noise = nn.Linear(noise_nc, norm_nc)
        # define the sketchE if specified
        self.add_sketch = add_sketch
        if add_sketch:
            self.sketch_conv = nn.Conv2d(1, norm_nc, kernel_size=3, padding=1) 
            self.merge_conv = nn.Conv2d(2*norm_nc, norm_nc, kernel_size=1, padding=0, bias=False)

    def reset_parameters(self):
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, segmap, input_instances=None, noise=None, sketch=None):
        # Part 1. generate parameter-free normalized activations
        # noise is [B, inst_nc, 2, noise_nc], 2 is for scale and bias
        normalized = self.param_free_norm(x)

        # Part 2. scale the segmentation mask and instance mask
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        input_instances = F.interpolate(input_instances, size=x.size()[2:], mode='nearest')

        # the segmap is concate with instance map
        inst_map = torch.unsqueeze(segmap[:,-1,:,:],1)
        segmap = segmap[:,:-1,:,:]

        # Part 3. class affine with noise
        noise_size = noise.size() # [B,inst_nc,2,noise_nc]
        noise_reshape = noise.view(-1, noise_size[-1]) # reshape to [B*inst_nc*2,noise_nc]
        noise_fc = self.fc_noise(noise_reshape) # [B*inst_nc*2, norm_nc]
        noise_fc = noise_fc.view(noise_size[0],noise_size[1],noise_size[2],-1)
        # create weigthed instance noise for scale
        class_weight = torch.einsum('ic,nihw->nchw', self.weight[...,0], segmap)
        class_bias = torch.einsum('ic,nihw->nchw', self.bias[...,0], segmap)
        # init_noise = torch.randn([x.size()[0], input_instances.size()[1], self.norm_nc], device=x.get_device())
        instance_noise = torch.einsum('nic,nihw->nchw', noise_fc[:,:,0,:], input_instances)
        scale_instance_noise = class_weight*instance_noise+class_bias
        # create weighted instance noise for bias
        class_weight = torch.einsum('ic,nihw->nchw', self.weight[..., 1], segmap)
        class_bias = torch.einsum('ic,nihw->nchw', self.bias[..., 1], segmap)
        # init_noise = torch.randn([x.size()[0], input_instances.size()[1], self.norm_nc], device=x.get_device())
        instance_noise = torch.einsum('nic,nihw->nchw', noise_fc[:,:,1,:], input_instances)
        bias_instance_noise = class_weight * instance_noise + class_bias

        out = scale_instance_noise * normalized + bias_instance_noise

        # Part 4. Other operations
        if self.add_sketch:
            assert sketch is not None, "If add sketch, sketch input should not be None !"
            sketch = F.interpolate(sketch, size=x.size()[2:], mode='nearest')
            sketch = self.sketch_conv(sketch)
            out = torch.cat([out, sketch], 1)
            out = self.merge_conv(out)

        return out