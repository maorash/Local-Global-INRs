import math

import numpy as np
import torch
from torch import nn


class BatchLinear(nn.Linear):
    __doc__ = nn.Linear.__doc__

    def forward(self, input):
        output = input.matmul(self.weight.permute(*[i for i in range(len(self.weight.shape) - 2)], -1, -2))
        output += self.bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


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


class LocallyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, groups,
                 outermost_linear=False, nonlinearity='relu', weight_init=None, bias=True):
        super().__init__()

        self.groups = groups
        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None)}

        self.nl, self.nl_weight_init, self.first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = self.nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            LocallyConnectedLayer(in_features * groups, hidden_features, groups, bias), self.nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                LocallyConnectedLayer(hidden_features, hidden_features, groups, bias), self.nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(LocallyConnectedLayer(hidden_features, out_features * groups, groups, bias)))
        else:
            self.net.append(nn.Sequential(
                LocallyConnectedLayer(hidden_features, out_features * groups, groups, bias), self.nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if self.first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(self.first_layer_init)

    def forward(self, coords):
        if coords.shape[0] != 1:
            raise ValueError("LocallyConnectedBlock currently supports batch_size=1")
        coords = coords.squeeze(0)

        output = self.net(coords)

        output = output.unsqueeze(0)
        return output


class LocalGlobalBlock(LocallyConnectedBlock):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, groups,
                 outermost_linear=False, nonlinearity='relu', weight_init=None, bias=True,
                 agg_type="siren", global_hidden_features=64):
        super().__init__(in_features, out_features, num_hidden_layers, hidden_features, groups,
                         outermost_linear, nonlinearity, weight_init, bias)

        self.global_net = []
        self.agg_type = agg_type
        self.global_hidden_features = global_hidden_features

        block_hidden_features = (hidden_features // groups)

        first_global_layer = BatchLinear(in_features, global_hidden_features, bias=bias)

        if self.agg_type == "concat_and_fc":
            self.global_net.append(nn.Sequential(first_global_layer, self.nl))
        else:
            self.global_net.append(nn.Sequential(nn.Sequential(first_global_layer, self.nl),
                                                 nn.Sequential(BatchLinear(global_hidden_features,
                                                                           block_hidden_features, bias=bias), self.nl)))

        for i in range(num_hidden_layers):
            if self.agg_type == "concat_and_fc":
                self.global_net.append(
                    nn.Sequential(BatchLinear(global_hidden_features, global_hidden_features, bias=bias), self.nl))
            else:
                self.global_net.append(
                    nn.Sequential(nn.Sequential(BatchLinear(global_hidden_features, global_hidden_features, bias=bias),
                                                self.nl),
                                  nn.Sequential(BatchLinear(global_hidden_features, block_hidden_features, bias=bias),
                                                self.nl)))

        self.global_net = nn.Sequential(*self.global_net)

        if self.agg_type == "concat_and_fc":
            self.agg_func = nn.Sequential(
                BatchLinear(global_hidden_features + block_hidden_features, block_hidden_features, bias=bias), self.nl)
        elif self.agg_type == "fc_and_add":
            self.agg_func = nn.Identity()
        else:
            raise ValueError("Invalid aggregation type")

        if self.weight_init is not None:
            self.global_net.apply(self.weight_init)
            self.agg_func.apply(self.weight_init)

        if self.first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            first_global_layer.apply(self.first_layer_init)

    def forward(self, coords):
        if coords.shape[0] != 1:
            raise ValueError("LocalGlobalBlock currently supports batch_size=1")
        coords = coords.squeeze(0)

        global_layer_input = coords.reshape(coords.shape[0] * coords.shape[1], coords.shape[2])
        local_layer_input = coords

        for local_layer, global_layer in zip(self.net, self.global_net):
            if self.agg_type == "concat_and_fc":
                global_layer_input = global_layer(global_layer_input)
                global_features = global_layer_input.reshape(coords.shape[0], coords.shape[1], -1)
            else:
                global_layer_input = global_layer[0](global_layer_input)
                global_features = global_layer[1](global_layer_input).reshape(coords.shape[0], coords.shape[1], -1)

            local_features = local_layer(local_layer_input)
            if self.agg_type == "concat_and_fc":
                local_layer_input = self.agg_func(
                    torch.cat([local_features, global_features], dim=2).reshape(coords.shape[0] * coords.shape[1],
                                                                                -1)).reshape(coords.shape[0],
                                                                                             coords.shape[1], -1)
            elif self.agg_type == "fc_and_add":
                local_layer_input = local_features + global_features

        output = self.net[-1](local_layer_input)

        output = output.unsqueeze(0)
        return output


class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords):
        output = self.net(coords)
        return output


class INR(nn.Module):
    def __init__(self, out_features=1, activation_type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, groups=None, agg_type=None,
                 global_hidden_features=None, **kwargs):
        super().__init__()
        self.mode = mode

        if self.mode == 'mlp':
            self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                               hidden_features=hidden_features, outermost_linear=True, nonlinearity=activation_type)
        elif self.mode == 'lc':
            self.net = LocallyConnectedBlock(in_features=in_features, out_features=out_features,
                                             num_hidden_layers=num_hidden_layers,
                                             hidden_features=hidden_features, groups=groups, outermost_linear=True,
                                             nonlinearity=activation_type, bias=True)
        elif self.mode == 'lg':
            self.net = LocalGlobalBlock(in_features=in_features, out_features=out_features,
                                        num_hidden_layers=num_hidden_layers,
                                        hidden_features=hidden_features, groups=groups, outermost_linear=True,
                                        nonlinearity=activation_type, bias=True, agg_type=agg_type,
                                        global_hidden_features=global_hidden_features)
        else:
            raise ValueError("Unsupported mode")

        print(self)

    def forward(self, model_input):
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}

    def count_parameters(self, cropped_partition_indices=None):
        def _count_model_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        counts = {"Total Parameters": _count_model_parameters(self)}
        if self.mode in ['lg']:
            counts["Local Parameters"] = _count_model_parameters(self.net.net)
            counts["Global Parameters"] = _count_model_parameters(self.net.global_net) + \
                                          _count_model_parameters(self.net.agg_func)

        if self.mode in ['lc', 'lg'] and cropped_partition_indices is not None:
            counts["Local Parameters"] *= ((self.net.groups - len(cropped_partition_indices)) / self.net.groups)
            counts["Total Parameters"] = counts["Local Parameters"] + counts["Global Parameters"]

        return counts

    def crop(self, partition_indices):
        with torch.no_grad():
            for i in range(len(self.net.net)):
                self.net.net[i][0].weight[partition_indices] = 0




"""
Initialization methods from the original SIREN paper
"""


def init_weights_normal(m):
    if hasattr(m, 'weight'):
        if type(m) == BatchLinear or type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        elif type(m) == LocallyConnectedLayer:
            raise ValueError("Unimplemented")
        else:
            raise ValueError("Unsupported layer in init")


def init_weights_selu(m):
    if hasattr(m, 'weight'):
        if type(m) == BatchLinear or type(m) == nn.Linear:
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))
        else:
            raise ValueError("Unsupported layer in init")


def init_weights_elu(m):
    if hasattr(m, 'weight'):
        if type(m) == BatchLinear or type(m) == nn.Linear:
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))
        else:
            raise ValueError("Unsupported layer in init")


def init_weights_xavier(m):
    if hasattr(m, 'weight'):
        if type(m) == BatchLinear or type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
        else:
            raise ValueError("Unsupported layer in init")


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            if type(m) == BatchLinear or type(m) == nn.Linear:
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
            elif type(m) == LocallyConnectedLayer:
                num_input = m.weight.size(1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
            else:
                raise ValueError("Unsupported layer in init")


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            if type(m) == BatchLinear or type(m) == nn.Linear:
                num_input = m.weight.size(-1)
                # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
                m.weight.uniform_(-1 / num_input, 1 / num_input)
            elif type(m) == LocallyConnectedLayer:
                num_input = m.weight.size(1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)
            else:
                raise ValueError("Unsupported layer in init")
