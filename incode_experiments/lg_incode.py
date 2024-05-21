import numpy as np
import torch
from torch import nn
from incode_experiments.encoding import Encoding
import torchvision.models as models
import torchvision.models.video as video
import torchaudio
import math

from incode_experiments.incode import MLP, Custom1DFeatureExtractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocallyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, groups, bias=True):
        super().__init__()
        weight_blocks = []

        for i in range(groups):
            block = torch.randn((in_features // groups, out_features // groups))
            weight_blocks.append(block)

        self.weight = nn.Parameter(torch.cat([b.unsqueeze(0) for b in weight_blocks], dim=0))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features).reshape(groups, -1))
            self.init_bias(in_features // groups)
        else:
            self.register_parameter('bias', None)

    def init_bias(self, fan_in) -> None:
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = torch.bmm(x, self.weight)
        if self.bias is not None:
            out += self.bias.unsqueeze(1)
        return out


class SineLayer(nn.Module):
    '''
    SineLayer is a custom PyTorch module that applies a modified Sinusoidal activation function to the output of a linear transformation
    with adjustable parameters.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, the linear transformation includes a bias term. Default is True.
        is_first (bool, optional): If True, initializes the weights with a narrower range. Default is False.
        omega_0 (float, optional): Frequency scaling factor for the sinusoidal activation. Default is 30.
        
    Additional Parameters:
        a_param (float): Exponential scaling factor for the sine function. Controls the amplitude. 
        b_param (float): Exponential scaling factor for the frequency.
        c_param (float): Phase shift parameter for the sine function.
        d_param (float): Bias term added to the output.

    '''
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                num_input = self.linear.weight.size(-1)
                self.linear.weight.uniform_(-1 / num_input,
                                             1 / num_input)
            else:
                num_input = self.linear.weight.size(-1)
                self.linear.weight.uniform_(-np.sqrt(6 / num_input) / self.omega_0,
                                             np.sqrt(6 / num_input) / self.omega_0)
        
    def forward(self, input, a_param=None, b_param=None, c_param=None, d_param=None):
        output = self.linear(input)
        if a_param is None and b_param is None and c_param is None and d_param is None:
            output = torch.sin(self.omega_0 * output)
        else:
            output = torch.exp(a_param) * torch.sin(torch.exp(b_param) * self.omega_0 * output + c_param) + d_param
        return output


class LocallyConnectedSineLayer(nn.Module):
    def __init__(self, in_features, out_features, groups, bias=True, is_first=False, omega_0=30):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = LocallyConnectedLayer(in_features, out_features, groups=groups, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                num_input = self.linear.weight.size(1)
                self.linear.weight.uniform_(-1 / num_input, 1 / num_input)
            else:
                num_input = self.linear.weight.size(1)
                self.linear.weight.uniform_(-np.sqrt(6 / num_input) / self.omega_0,
                                            np.sqrt(6 / num_input) / self.omega_0)

    def forward(self, input, a_param=None, b_param=None, c_param=None, d_param=None):
        output = self.linear(input)
        if a_param is None and b_param is None and c_param is None and d_param is None:
            output = torch.sin(self.omega_0 * output)
        else:
            output = torch.exp(a_param) * torch.sin(torch.exp(b_param) * self.omega_0 * output + c_param) + d_param
        return output
    

class LCINCODE(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, groups,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30, 
                 pos_encode_configs={'type':None, 'use_nyquist': None, 'scale_B': None, 'mapping_input': None},
                 MLP_configs={'model': 'resnet34', 'in_channels': 64, 'hidden_channels': [64, 32, 4], 'activation_layer': nn.SiLU}):
        super().__init__()

        # Positional Encoding
        self.pos_encode = pos_encode_configs['type']
        if self.pos_encode in Encoding().encoding_dict.keys():
            self.positional_encoding = Encoding(self.pos_encode).run(in_features=in_features, pos_encode_configs=pos_encode_configs)
            in_features = self.positional_encoding.out_dim
        elif self.pos_encode == None: 
            self.pos_encode = False
        else:
            assert "Invalid pos_encode. Choose from: [frequency, gaussian]"


        self.ground_truth = MLP_configs['GT']
        self.task = MLP_configs['task']
        self.nonlin = LocallyConnectedSineLayer
        self.hidden_layers = hidden_layers

        # Harmonizer network
        if MLP_configs['task'] == 'audio':
            self.feature_extractor = torchaudio.transforms.MFCC(
                                                sample_rate=MLP_configs['sample_rate'],
                                                n_mfcc=MLP_configs['in_channels'],
                                                melkwargs={'n_fft': 400, 'hop_length': 160,
                                                            'n_mels': 50, 'center': False})
        elif MLP_configs['task'] == 'shape':
            model_ft = getattr(video, MLP_configs['model'])()
            self.feature_extractor = nn.Sequential(*list(model_ft.children())[:MLP_configs['truncated_layer']])
        elif MLP_configs['task'] == 'inpainting':
            self.feature_extractor = Custom1DFeatureExtractor(im_chans=3, out_chans=[32, 64, 64])
        else:
            model_ft = getattr(models, MLP_configs['model'])()
            self.feature_extractor = nn.Sequential(*list(model_ft.children())[:MLP_configs['truncated_layer']])

        self.aux_mlp = MLP(MLP_configs)

        if MLP_configs['task'] == 'shape':
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            if MLP_configs['task'] != 'inpainting':
                self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Composer Network
        self.net = []
        self.net.append(self.nonlin(in_features * groups, hidden_features, groups,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, groups,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = LocallyConnectedLayer(hidden_features, out_features * groups, groups=groups)

            with torch.no_grad():
                num_input = final_linear.weight.size(1)
                const = np.sqrt(6 / num_input) / max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, groups,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        if coords.shape[0] != 1:
            raise ValueError("LocallyConnectedBlock currently supports batch_size=1")
        coords = coords.squeeze(0)

        if self.pos_encode:
            coords = self.positional_encoding(coords)
        
        extracted_features = self.feature_extractor(self.ground_truth)
        if self.task == 'shape':
            gap = self.gap(extracted_features)[:, :, 0, 0, 0]
            coef = self.aux_mlp(gap)
        elif self.task == 'inpainting':
            coef = self.aux_mlp(extracted_features)
        else:
            gap = self.gap(extracted_features.view(extracted_features.size(0), extracted_features.size(1), -1)) 
            coef = self.aux_mlp(gap[..., 0])
        a_param, b_param, c_param, d_param = coef[0]
                
        output = self.net[0](coords, a_param, b_param, c_param, d_param)
        
        for i in range(1, self.hidden_layers + 1):
            output = self.net[i](output, a_param, b_param, c_param, d_param)
        
        output = self.net[self.hidden_layers + 1](output)

        output = output.unsqueeze(0)
        return [output, coef]


class LGINCODE(LCINCODE):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, groups, global_hidden_features,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30,
                 pos_encode_configs={'type':None, 'use_nyquist': None, 'scale_B': None, 'mapping_input': None},
                 MLP_configs={'model': 'resnet34', 'in_channels': 64, 'hidden_channels': [64, 32, 4], 'activation_layer': nn.SiLU}):
        super().__init__(in_features, hidden_features, hidden_layers, out_features, groups, outermost_linear, first_omega_0, hidden_omega_0, pos_encode_configs, MLP_configs)

        self.global_net = []
        self.global_hidden_features = global_hidden_features

        block_hidden_features = (hidden_features // groups)

        first_global_layer = SineLayer(in_features, global_hidden_features, bias=True, is_first=True)
        self.global_net.append(first_global_layer)

        for i in range(hidden_layers):
            self.global_net.append(SineLayer(global_hidden_features, global_hidden_features, bias=True, is_first=False))

        self.global_net = nn.Sequential(*self.global_net)

        self.agg_func = SineLayer(global_hidden_features + block_hidden_features, block_hidden_features, bias=True,
                                  is_first=False)

    def forward(self, coords):
        if coords.shape[0] != 1:
            raise ValueError("LocalGlobalBlock currently supports batch_size=1")
        coords = coords.squeeze(0)

        if self.pos_encode:
            coords = self.positional_encoding(coords)

        extracted_features = self.feature_extractor(self.ground_truth)
        if self.task == 'shape':
            gap = self.gap(extracted_features)[:, :, 0, 0, 0]
            coef = self.aux_mlp(gap)
        elif self.task == 'inpainting':
            coef = self.aux_mlp(extracted_features)
        else:
            gap = self.gap(extracted_features.view(extracted_features.size(0), extracted_features.size(1), -1))
            coef = self.aux_mlp(gap[..., 0])

        a_param, b_param, c_param, d_param, a_global, b_global, c_global, d_global, a_agg, b_agg, c_agg, d_agg = coef[0]

        global_layer_input = coords.reshape(coords.shape[0] * coords.shape[1], coords.shape[2])
        local_layer_input = coords

        for i, (local_layer, global_layer) in enumerate(zip(self.net, self.global_net)):
            global_layer_input = global_layer(global_layer_input, a_global, b_global, c_global, d_global)
            global_features = global_layer_input.reshape(coords.shape[0], coords.shape[1], -1)

            local_features = local_layer(local_layer_input, a_param, b_param, c_param, d_param)
            local_layer_input = self.agg_func(torch.cat([local_features, global_features], dim=2).reshape(coords.shape[0] * coords.shape[1], -1), a_agg, b_agg, c_agg, d_agg).reshape(coords.shape[0], coords.shape[1], -1)

        output = self.net[-1](local_layer_input)

        output = output.unsqueeze(0)
        return [output, coef]
