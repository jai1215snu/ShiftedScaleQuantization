import torch
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, round_ste
import torch.nn.functional as F

class ChannelQuantAct(nn.Module):
    @torch.no_grad()
    def __init__(self, uaq: UniformAffineQuantizer, shiftTarget: list=[2/2, 2/2]):
        super(ChannelQuantAct, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.device = 'cuda:0'
        
        self.shiftedScale = 1.0
        self.shiftTarget = shiftTarget
        
        #optimization method
        self.opt_mode = 'none'
        # self.round_mode = 'normal'
        self.hard_targets = False
        
        self.gamma, self.zeta = -0.1, 1.1
        
        self.alpha = None
        self.shiftedDone = False
        
        #For Shifted Scale -> Moved to layer reconstruction
        # self.init_v(x=weight_tensor.clone().detach())
        #For AdaRound
        # self.init_beta(x=weight_tensor.clone().detach())
        
    def forward(self, x):
        #If AdaRound mode (ingore shifted scale)
        if self.opt_mode == 'adaShift':
            x_floor = self.shifted_x_quant()
            if not self.hard_round:
                x_int = x_floor + self.get_soft_round()
            else:
                x_int = x_floor + (self.beta >= 0).float()
            x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - self.zero_point) * (self.delta*self.shiftedScale)
            return x_float_q 
        elif self.opt_mode == 'adaround':
            x_floor = torch.floor(x/(self.delta*self.shiftedScale))
            if not self.hard_round:
                x_int = x_floor + self.get_soft_round()
            else:
                x_int = x_floor + (self.beta >= 0).float()
            x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - self.zero_point) * (self.delta*self.shiftedScale)
            return x_float_q 
        elif self.opt_mode == 'none':
            x_int = torch.round(x / (self.delta*self.shiftedScale))
        elif self.opt_mode in  'learned_hard_sigmoid':
            return self.shifted_x_quant()
        else:
            raise ValueError('opt_mode is not defined')
            
        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_quant = x_quant - self.zero_point
        
        x_float_q = x_quant * (self.delta*self.shiftedScale)
        return x_float_q
    
    def get_sig_soft_targets(self):
        return torch.clamp(F.softmax(self.alpha, dim=-1) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def inverse_softmax(self, x):
        #Inverse of Zeta/Gamma
        x = (x - self.gamma)/(self.zeta - self.gamma)
        #Inverse softmax
        logits = torch.log(x)
        avg_logit = torch.mean(logits, dim=-1, keepdim=True)
        return logits - avg_logit
    
    def get_delta(self):
        p = self.get_sig_soft_targets()
        # self.dump_torch(p, 'p')
        if p.dim() == 2:
            p = p.unsqueeze(0)
        max_index = torch.argmax(p, dim=-1)
        # self.dump_torch(max_index, 'max_index')
        # self.dump_torch(self.delta, 'delta_ori')
        
        delta = self.delta*self.shiftTarget[0]
        for i in range(1, len(self.shiftTarget)):
            mask = max_index == i
            if not self.isFC:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            delta = torch.where(mask, self.delta*self.shiftTarget[i], delta)
        # self.dump_torch(delta, 'delta_out')
        return delta
    
    @torch.no_grad()
    def init_v(self, x: torch.Tensor):
        shiftTarget = self.shiftTarget
        for st in shiftTarget:
            self.shiftedScale = st
            self.x_q.append(torch.floor(x/(self.delta*self.shiftedScale)))
        self.shiftedScale = 1.0
        self.alpha = self.init_alpha(x, clip = (0.90-0.05*len(shiftTarget)), device=self.device)
        delta = self.get_delta()
        x_floor = torch.floor(x/delta)
        rest = (x / delta) - x_floor  # rest of rounding [0, 1)
        beta = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(beta) = rest
        self.alpha = nn.Parameter(self.alpha)
        self.beta = nn.Parameter(beta)