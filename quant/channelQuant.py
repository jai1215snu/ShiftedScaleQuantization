import torch
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, round_ste
import torch.nn.functional as F

class ChannelQuant(nn.Module):
    @torch.no_grad()
    def __init__(self, delta, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, shiftTarget: list=[2/2, 2/2], act=False, name='--'):
        super(ChannelQuant, self).__init__()
        self.RUN_CHANNEL_WISE = True #NOTE: small size
        # self.RUN_CHANNEL_WISE = False #NOTE: big size(use tensor wise)
        # copying all attributes from UniformAffineQuantizer
        self.act = act
        
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta * delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.device = weight_tensor.device
        
        self.isFC = len(self.delta.shape) != 4
        
        self.nchannel = (weight_tensor.shape[0], weight_tensor.shape[1])
        self.shiftedScale = 1.0
        self.shiftTarget = shiftTarget
        self.x_q  = []
        
        #optimization method
        self.opt_mode = 'none'
        # self.round_mode = 'normal'
        self.hard_targets = False
        self.hard_round = False
        
        self.gamma, self.zeta = -0.1, 1.1
        
        self.alpha = None
        self.beta = None #For adaptive rounding
        self.deltaQuant = None
        self.shiftedDone = False
        
        self.name = name
        
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
            
            if self.sym:
                x_quant = torch.clamp(x_int + self.zero_point, -self.n_levels//2, self.n_levels//2 - 1)
            else:
                x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
                
            x_float_q = (x_quant - self.zero_point) * (self.delta*self.shiftedScale)
            return x_float_q 
        elif self.opt_mode == 'adaround':
            x_floor = torch.floor(x/(self.delta*self.shiftedScale))
            if not self.hard_round:
                x_int = x_floor + self.get_soft_round()
            else:
                x_int = x_floor + (self.beta >= 0).float()
                
            if self.sym:
                x_quant = torch.clamp(x_int + self.zero_point, -self.n_levels//2, self.n_levels//2 - 1)
            else:
                x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
                
            x_float_q = (x_quant - self.zero_point) * (self.delta*self.shiftedScale)
            return x_float_q 
        elif self.opt_mode == 'none':
            x_int = torch.round(x / (self.delta*self.shiftedScale))
        elif self.opt_mode in  'learned_hard_sigmoid':
            return self.shifted_x_quant()
        else:
            raise ValueError('opt_mode is not defined')
            
        if self.sym:
            x_quant = torch.clamp(x_int + self.zero_point, -self.n_levels//2, self.n_levels//2 - 1)
        else:
            x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
            
        x_quant = x_quant - self.zero_point
        
        x_float_q = x_quant * (self.delta*self.shiftedScale)
        return x_float_q
    
    def shifted_x_quant(self):
        p = self.get_sig_soft_targets()
        if p.dim() == 2:
            p = p.unsqueeze(0)
        if self.hard_targets:#hard target
            max_index = torch.argmax(p, dim=-1)
            x_out = self.x_q[0]
            for i in range(1, len(self.shiftTarget)):
                mask = max_index == i
                if not self.isFC:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)
                x_out = torch.where(mask, self.x_q[i], x_out)
        else: #soft target
            if self.isFC:
                x_out = (self.x_q[0] * p[:, :, 0])
                for i in range(1, len(self.shiftTarget)):
                    x_out +=  (self.x_q[i] * p[:, :, i])
            else:
                p = p.unsqueeze(-1).unsqueeze(-1)
                x_out = (self.x_q[0] * p[:, :, 0, :, :])
                for i in range(1, len(self.shiftTarget)):
                    x_out += (self.x_q[i]* p[:, :, i, :, :])
        return x_out
    
    def get_sig_soft_targets(self):
        return torch.clamp(F.softmax(self.alpha, dim=-1) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def get_soft_round(self):
        return torch.clamp(torch.sigmoid(self.beta) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    # def init_alpha_ori(self, x: torch.Tensor, clip = 0.80, device='cuda'):
    #     # RUN_CHANNEL_WISE = True #NOTE: temp code Full granuarity
    #     RUN_CHANNEL_WISE = False #NOTE: use all
    #     shiftNum = len(self.shiftTarget)
    #     mse = []
        
    #     for i in range(shiftNum):
    #         if self.isFC:
    #             mse.append((x - self.x_q[i])**2)
    #         else:
    #             concat_ch = (0, 2, 3) if RUN_CHANNEL_WISE else (2, 3)
    #             mse.append(torch.sum((x - self.x_q[i])**2, dim=concat_ch))
    #     mse = torch.stack(mse, dim=0)
    #     _, min_index = torch.min(mse, dim=0)
        
    #     remain_probability = (1.0-clip)/(shiftNum-1)
    #     alpha = torch.full((*min_index.shape, shiftNum), remain_probability, dtype=torch.float, device=device)
        
    #     mask = torch.zeros((*min_index.shape, shiftNum), dtype=torch.bool)
    #     # Set the values of mask based on the values of index
    #     for i in range(shiftNum):
    #         if mask.dim() == 3:
    #             mask[:, :, i] = (min_index == i)
    #         else:
    #             mask[:, i] = (min_index == i)
    #     alpha[mask] = clip
    #     return self.inverse_softmax(alpha)
    
    
    def init_alpha(self, x: torch.Tensor, clip = 0.80, device='cuda'):
        # #TODO: Temp code min_index is always point to 1.0
        clip = 0.33
        shiftNum = len(self.shiftTarget)
        mse = []
        
        for i in range(shiftNum):
            if self.isFC:
                mse.append((x - self.x_q[i])**2)
            else:
                concat_ch = (0, 2, 3) if self.RUN_CHANNEL_WISE else (2, 3)
                mse.append(torch.sum((x - self.x_q[i])**2, dim=concat_ch))
        mse = torch.stack(mse, dim=0)
        _, min_index = torch.min(mse, dim=0)
        
        # #TODO: Temp code min_index is always point to 1.0
        # min_index = torch.zeros_like(min_index)
        
        if shiftNum == 1:
            remain_probability = 0
            clip = 1.0
        else:
            remain_probability = (1.0-clip)/(shiftNum-1)
        alpha = torch.full((*min_index.shape, shiftNum), remain_probability, dtype=torch.float, device=device)
        
        mask = torch.zeros((*min_index.shape, shiftNum), dtype=torch.bool)
        # Set the values of mask based on the values of index
        for i in range(shiftNum):
            if mask.dim() == 3:
                mask[:, :, i] = (min_index == i)
            else:
                mask[:, i] = (min_index == i)
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
        shiftTarget = self.shiftTarget
        # self.opt_mode = 'none'
        for st in shiftTarget:
            self.shiftedScale = st
            self.x_q.append(self(x))
        self.shiftedScale = 1.0
        # alpha = torch.zeros((*self.nchannel, len(shiftTarget)), device=self.device)
        alpha = self.init_alpha(x, clip = (0.90-0.05*len(shiftTarget)), device=self.device)
        self.alpha = nn.Parameter(alpha)
        self.opt_mode = 'learned_hard_sigmoid'
        
    # def dump_torch(self, x, name):
    #     x_str = str(x.detach().cpu().numpy())
    #     # Write the tensor data to a text file
    #     with open(f'{name}.txt', 'w') as f:
    #         f.write(x_str)
    
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
    def init_shift_candidates(self, x):
        num_of_candi = 3
        candidates = [i/8 for i in range(1, 16) if i!=8]
        mse_candidates = []
        for st in candidates:
            x_q = torch.round(x/(self.delta*st))
            if self.sym:
                x_q = torch.clamp(x_q + self.zero_point, -self.n_levels//2, self.n_levels//2 - 1)
            else:
                x_q = torch.clamp(x_q + self.zero_point, 0, self.n_levels - 1)
            x_float = (x_q - self.zero_point) * (self.delta*st)
            x_mse = (x_float - x).abs().pow(2.4)
            if self.RUN_CHANNEL_WISE:
                if self.isFC:
                    mse = x_mse.sum(dim=(0))
                else:
                    mse = x_mse.sum(dim=(0, -1, -2))
            else:
                if self.isFC:
                    mse = x_mse.sum(dim=(-1, -2))
                else:
                    mse = x_mse
                mse = mse.flatten()
                    
            mse_candidates.append(mse)
        
        mse_candidates = torch.stack(mse_candidates, dim=0)
        scores = dict()
        for i in range(mse_candidates.shape[0]):
            scores[i] = 0
        for i in range(mse_candidates.shape[1]):
            sorted_indices = torch.argsort(mse_candidates[:, i])
            for j in range(num_of_candi):
                scores[sorted_indices[j].item()] += num_of_candi - j
        sorted_dict = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        top_keys = list(sorted_dict.keys())[:(num_of_candi-1)]
        self.shiftTarget = [candidates[i] for i in top_keys]
        self.shiftTarget.append(1.0)
    
    @torch.no_grad()
    def init_v_beta(self, x: torch.Tensor):
        # self.init_shift_candidates(x)
        shiftTarget = self.shiftTarget
        print(f"{self.name}, Optimal shift candidates: ", shiftTarget)
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
    
    @torch.no_grad()
    def update_delta(self):
        self.delta = self.get_delta()
        
    @torch.no_grad()
    def init_beta(self, x: torch.Tensor):
        # delta = self.get_delta()
        delta = self.delta
        x_floor = torch.floor(x / delta)
        rest = (x / delta) - x_floor  # rest of rounding [0, 1)
        beta = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(beta) = rest
        self.beta = nn.Parameter(beta)