import torch
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, round_ste

class ChannelQuant(nn.Module):
    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor):
        super(ChannelQuant, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.device = weight_tensor.device
        
        self.nchannel = (weight_tensor.shape[0], weight_tensor.shape[1])
        self.shiftedScale = torch.ones((weight_tensor.shape[0], weight_tensor.shape[1], 1, 1), device=self.device)
        # self.simpleScale = torch.zeros(self.nchannel, device=self.device)
        
        #optimization method
        self.opt_mode = 'none'
        self.hard_targets = False
        
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        
        self.alpha = None
        self.deltaQuant = None
        
        self.init_v(x=weight_tensor.clone().detach())

        # params for sigmoid function
        # self.init_resol(prob=shuffle_ratio)

    def forward(self, x):
        if len(self.delta.shape) != 4: #NOTE: FC layer control
            x_int = torch.round(x / self.delta)
        elif self.opt_mode == 'none':
            x_int = torch.round(x / (self.delta*self.shiftedScale))
        elif self.opt_mode in  'learned_hard_sigmoid':
            x_int = torch.round(x / self.delta)
        else:
            raise ValueError('opt_mode is not defined')
            
        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_quant = x_quant - self.zero_point
        
        if len(self.delta.shape) != 4: #NOTE: FC layer control
            x_float_q = x_quant * self.delta
        elif self.opt_mode == 'learned_hard_sigmoid':
            x_float_q = x_quant * self.delta
            soft_target = self.get_soft_targets().view(self.nchannel[0], self.nchannel[1], 1, 1)
            # print(x[0][0])
            # print(x_float_q[0][0])
            # print(self.deltaQuant[0][0])
            # print(soft_target[0][0])
            # print(self.delta[0][0])
            # print((self.deltaQuant*soft_target*self.delta)[0][0])
            if self.hard_targets:
                soft_target = torch.where(soft_target > 0.5, torch.ones_like(soft_target), torch.zeros_like(soft_target))
            x_float_q = x_float_q + self.deltaQuant*soft_target
            # print(x_float_q[0][0])
        elif self.opt_mode == 'none':
            x_float_q = x_quant * (self.delta*self.shiftedScale)
                
        return x_float_q
    
    def getQuantWeight(self, x):
        if len(self.delta.shape) == 4:
            x = x * self.shiftedScale.expand(x.shape)
            x_int = torch.round(x / self.delta)
        else:
            x_int = torch.round(x / self.delta)
            
        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * (self.delta*self.shiftedScale.expand(x.shape))
        
        return x_float_q

    def setScale(self, selected):
        quantLevel = [1.0, 4/8, 2/8, 6/8, 3/8, 5/8, 7/8, 9/8]
        for i in range(8):
            self.shiftedScale[(selected == i)] = quantLevel[i]
                
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    @torch.no_grad()
    def init_alpha(self, x: torch.Tensor, x_q: torch.Tensor, clip = [0.1, 0.9]):
        # W_Q = W_QS + DeltaQuant x H(Alpha)
        # YX = DeltaQuant x H(Alpha)
        # H(Alpha) = sum(DxYX)/sum(DxD) -> from linear regression normal equation
        yx = x - x_q
        d = self.deltaQuant
        w = torch.sum(d*yx, dim=(-1, -2))/torch.sum(d*d+1e-9, dim=(-1, -2))
        return torch.clamp(w, min=clip[0], max=clip[1])
        
    def init_v(self, x: torch.Tensor):
        #Make linear function for deltaQuant
        # W_Q = W_QS + (W_QS - W_QS/2) x H(Alpha)
        # W_Q = W_QS + DeltaQuant x H(Alpha)
        x_q = self(x)
        self.shiftedScale = self.shiftedScale / 2
        x_2q = self(x)
        self.shiftedScale = self.shiftedScale * 2
        self.deltaQuant = (x_2q - x_q)
        
        alpha = torch.zeros(self.nchannel, device=self.device)
        p = self.init_alpha(x, x_q)
        alpha = torch.log(p/(1-p))
        
        # mask = (self.deltaQuant == 0)
        # mask = torch.all(mask, dim=3)
        # mask = torch.all(mask, dim=2)
        # alpha = torch.where(mask, torch.tensor(1e3, device=self.device), alpha)
        self.alpha = nn.Parameter(alpha)