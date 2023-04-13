import torch
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, round_ste
import torch.nn.functional as F

class ChannelQuant(nn.Module):
    def __init__(self, delta, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, shiftTarget: list=[2/2, 2/2]):
        super(ChannelQuant, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta * delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.device = weight_tensor.device
        
        self.isFC = len(self.delta.shape) != 4
        
        self.nchannel = (weight_tensor.shape[0], weight_tensor.shape[1])
        self.shiftedScale = 1.0
        # self.shiftTarget = [1/4, 2/4, 3/4, 4/4, 5/4, 6/4, 7/4]
        # self.shiftTarget = [2/4, 3/4, 4/4, 5/4, 6/4]
        # self.shiftTarget = [1/4, 2/4, 4/4, 6/4, 7/4]
        # self.shiftTarget = [1/2, 2/2, 3/2]
        # self.shiftTarget = [1/2, 2/2]
        self.shiftTarget = shiftTarget
        self.x_q  = []
        
        #optimization method
        self.opt_mode = 'none'
        self.hard_targets = False
        
        self.gamma, self.zeta = -0.1, 1.1
        # self.beta = 2/3 #Not used?
        
        self.alpha = None
        self.deltaQuant = None
        self.shiftedDone = False
        
        self.init_v(x=weight_tensor.clone().detach())
        
    def forward(self, x):
        # if len(self.delta.shape) != 4: #NOTE: FC layer control
        #     x_int = torch.round(x / self.delta)
        # elif self.opt_mode == 'none':
        if self.opt_mode == 'none':
            x_int = torch.round(x / (self.delta*self.shiftedScale))
        elif self.opt_mode in  'learned_hard_sigmoid':
            p = self.get_sig_soft_targets()
            if self.hard_targets:
                max_index = torch.argmax(p, dim=-1)
                x_out = self.x_q[0]
                for i in range(1, len(self.shiftTarget)):
                    mask = max_index == i
                    if not self.isFC:
                        mask = mask.unsqueeze(-1).unsqueeze(-1)
                    x_out = torch.where(mask, self.x_q[i], x_out)
                return x_out
            else: #soft target
                if self.isFC:
                    p = self.get_sig_soft_targets()
                    x_out = (self.x_q[0] * p[:, :, 0])
                    for i in range(1, len(self.shiftTarget)):
                        x_out +=  (self.x_q[i] * p[:, :, i])
                else:
                    p = self.get_sig_soft_targets().unsqueeze(-1).unsqueeze(-1)
                    x_out = (self.x_q[0] * p[:, :, 0, :, :])
                    for i in range(1, len(self.shiftTarget)):
                        x_out += (self.x_q[i]* p[:, :, i, :, :])
                return x_out
        else:
            raise ValueError('opt_mode is not defined')
            
        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_quant = x_quant - self.zero_point
        
        #NOTE: FC and CV same?
        # if len(self.delta.shape) != 4: 
        #     x_float_q = x_quant * self.delta
        # elif self.opt_mode == 'learned_hard_sigmoid':
        #     x_float_q = x_quant * self.delta
        #     soft_target = self.get_soft_targets().view(self.nchannel[0], self.nchannel[1], 1, 1)
        #     if self.hard_targets:
        #         soft_target = torch.where(soft_target > 0.5, torch.ones_like(soft_target), torch.zeros_like(soft_target))
        #     x_float_q = x_float_q + self.deltaQuant*soft_target
        # elif self.opt_mode == 'none':
        #     x_float_q = x_quant * (self.delta*self.shiftedScale)
        x_float_q = x_quant * (self.delta*self.shiftedScale)
        return x_float_q
    
    # def getQuantWeight(self, x):
    #     if len(self.delta.shape) == 4:
    #         x = x * self.shiftedScale.expand(x.shape)
    #         x_int = torch.round(x / self.delta)
    #     else:
    #         x_int = torch.round(x / self.delta)
            
    #     x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
    #     x_float_q = (x_quant - self.zero_point) * (self.delta*self.shiftedScale.expand(x.shape))
        
    #     return x_float_q

    # def setScale(self, selected):
    #     quantLevel = [1.0, 4/8, 2/8, 6/8, 3/8, 5/8, 7/8, 9/8]
    #     for i in range(8):
    #         self.shiftedScale[(selected == i)] = quantLevel[i]
    
    def get_sig_soft_targets(self):
        return torch.clamp(F.softmax(self.alpha, dim=-1) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    @torch.no_grad()
    def init_alpha(self, x: torch.Tensor, clip = 0.80, device='cuda'):
        shiftNum = len(self.shiftTarget)
        mse = []
        
        for i in range(shiftNum):
            if self.isFC:
                mse.append((x - self.x_q[i])**2)
            else:
                mse.append(torch.sum((x - self.x_q[i])**2, dim=(2, 3)))
        mse = torch.stack(mse, dim=0)
        _, min_index = torch.min(mse, dim=0)
        
        remain_probability = (1.0-clip)/(shiftNum-1)
        alpha = torch.full((*min_index.shape, shiftNum), remain_probability, dtype=torch.float, device=device)
        
        mask = torch.zeros((*min_index.shape, shiftNum), dtype=torch.bool)
        # Set the values of mask based on the values of index
        for i in range(shiftNum):
            mask[:,:,i] = (min_index==i)
        alpha[mask] = clip
        return self.inverse_softmax(alpha)
        
    def inverse_softmax(self, x):
        #Inverse of Zeta/Gamma
        x = (x - self.gamma)/(self.zeta - self.gamma)
        #Inverse softmax
        logits = torch.log(x)
        avg_logit = torch.mean(logits, dim=-1, keepdim=True)
        return logits - avg_logit
    
    @torch.no_grad()
    def init_v(self, x: torch.Tensor):
        #W_Q = W_QS x P(W_QS) + W_QS/2 x P(W_QS/2)
        # self.shiftedScale.fill_(2/4)
        # self.shiftedScale.fill_(3/4)
        
        shiftTarget = self.shiftTarget
        self.opt_mode = 'none'
        for st in shiftTarget:
            self.shiftedScale = st
            self.x_q.append(self(x))
        self.shiftedScale = 1.0
        # alpha = torch.zeros((*self.nchannel, len(shiftTarget)), device=self.device)
        alpha = self.init_alpha(x, clip = (0.90-0.05*len(shiftTarget)), device=self.device)
        self.alpha = nn.Parameter(alpha)
        self.opt_mode = 'learned_hard_sigmoid'
    
        
    # def init_v2(self, x: torch.Tensor):
    #     #Make linear function for deltaQuant
    #     # W_Q = W_QS + (W_QS - W_QS/2) x H(Alpha)
    #     # W_Q = W_QS + DeltaQuant x H(Alpha)
    #     self.opt_mode = 'none'
    #     x_q = self(x)
    #     self.shiftedScale = self.shiftedScale / 2
    #     x_2q = self(x)
    #     self.shiftedScale = self.shiftedScale * 2
    #     self.deltaQuant = (x_2q - x_q)
        
    #     alpha = torch.zeros(self.nchannel, device=self.device)
    #     p = self.init_alpha(x, x_q)
    #     alpha = torch.log(p/(1-p))
        
    #     # p = torch.tensor(0.2, device=self.device)
    #     # alpha.fill_(torch.log(p/(1-p)))
        
    #     # mask = (self.deltaQuant == 0)
    #     # mask = torch.all(mask, dim=3)
    #     # mask = torch.all(mask, dim=2)
    #     # alpha = torch.where(mask, torch.tensor(1e3, device=self.device), alpha)
    #     self.alpha = nn.Parameter(alpha)
    #     self.opt_mode = 'learned_hard_sigmoid'
        