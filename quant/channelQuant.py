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
        self.device = weight_tensor.device
        
        self.shiftedScale = torch.ones_like(weight_tensor, device=self.device)
        self.soft_targets = False
        
        self.qscale = qscale
        self.nchannel = (weight_tensor.shape[0], weight_tensor.shape[1])
        
        self.simpleScale = torch.zeros(self.nchannel, device=self.device)
        
        self.shuffle_ratio = shuffle_ratio
        

        # params for sigmoid function
        # self.init_resol(prob=shuffle_ratio)

    def forward(self, x):
        if len(self.delta.shape) == 4:
            x = x * self.shiftedScale.expand(x.shape)
            x_int = torch.round(x / self.delta)
        else:
            #TODO: FC layer control
            x_int = torch.round(x / self.delta)
            
        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta / self.shiftedScale.expand(x.shape)
        
        return x_float_q
    
    def getQuantWeight(self, x):
        if len(self.delta.shape) == 4:
            x = x * self.shiftedScale.expand(x.shape)
            x_int = torch.round(x / self.delta)
        else:
            #TODO: FC layer control
            x_int = torch.round(x / self.delta)
            
        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta / self.shiftedScale.expand(x.shape)
        
        return x_float_q

    # def init_resol(self, prob: float = 0.1):
        # self.simpleScale = torch.randint(0, 2, size=(x.shape[0],), device=x.device)
        # self.simpleScale = torch.pow(2, -self.simpleScale.float()).to(x.device)
        
    def resetResol(self):
        self.simpleScale = torch.zeros(self.nchannel)
        
    def changeResol(self, oc, ic):
        self.resetResol()
        self.setResol(oc, ic)
    
    def setResol(self, oc: int, ic: int, value: float = 1.0):
        self.simpleScale[oc, ic] = value
        self.simpleScale = (torch.ones_like(self.simpleScale) - (self.simpleScale.float() * (1-self.qscale))).to(self.device)
        
    def setScale(self, selected):
        quantLevel = [1.0, 4/8, 2/8, 6/8, 3/8, 5/8, 7/8, 9/8]
        for i in range(8):
            self.shiftedScale[(selected == i)] = quantLevel[i]
                
    def run_layerRandomize(self):
        randomValue = torch.multinomial(torch.tensor([1-self.shuffle_ratio, self.shuffle_ratio]), num_samples=self.nchannel[0]*self.nchannel[1], replacement=True)
        randomValue = randomValue.reshape(self.nchannel[0], self.nchannel[1])
        self.setScale(randomValue)