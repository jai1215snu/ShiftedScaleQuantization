import torch
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, round_ste


class ChannelQuant(nn.Module):
    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, shuffle_ratio: float = 0.0, qscale: float = 0.5):
        super(ChannelQuant, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.sel_resol = None
        self.soft_targets = False
        
        self.qscale = qscale
        self.nchannel = (weight_tensor.shape[0], weight_tensor.shape[1])
        self.device = weight_tensor.device
        
        self.simpleScale = torch.zeros(self.nchannel)

        # params for sigmoid function
        self.init_resol(prob=shuffle_ratio)

    def forward(self, x):
        if len(self.delta.shape) == 4:
            # x = x * self.sel_resol.view(-1, 1, 1, 1)
            x = x * self.sel_resol.view(self.nchannel[0], self.nchannel[1], 1, 1).expand(x.shape)

            x_int = torch.round(x / self.delta)
        else:
            x_int = torch.round(x / self.delta)
        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta

        return x_float_q


    def init_resol(self, prob: float = 0.1):
        self.sel_resol = torch.multinomial(torch.tensor([1-prob, prob]), num_samples=self.nchannel[0]*self.nchannel[1], replacement=True)
        self.sel_resol = self.sel_resol.view(self.nchannel[0], self.nchannel[1])
        
        # self.sel_resol = torch.randint(0, 2, size=(x.shape[0],), device=x.device)
        # self.sel_resol = torch.pow(2, -self.sel_resol.float()).to(x.device)
        self.sel_resol = (torch.ones_like(self.sel_resol) - (self.sel_resol.float() * (1-self.qscale))).to(self.device)
        
    def resetResol(self):
        self.simpleScale = torch.zeros(self.nchannel)
        
    def changeResol(self, oc, ic):
        self.resetResol()
        self.setResol(oc, ic)
    
    def setResol(self, oc: int, ic: int, value: float = 1.0):
        self.simpleScale[oc, ic] = value
        self.sel_resol = (torch.ones_like(self.simpleScale) - (self.simpleScale.float() * (1-self.qscale))).to(self.device)
        
        