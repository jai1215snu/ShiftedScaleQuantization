import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from tqdm import tqdm
from common import *


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, tune_delta_zero: bool = False, ch: int = 64, disable_act_quant: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 1 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.raw_zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.disable_act_quant = disable_act_quant
        
        if not tune_delta_zero and not disable_act_quant:
            if leaf_param:
                self.delta = torch.nn.Parameter(torch.tensor(0.0))
                self.zero_point = torch.nn.Parameter(torch.tensor(0.0))
            elif type(ch) is int:
                self.delta = torch.nn.Parameter(torch.zeros(size=(ch,1)))
                self.zero_point = torch.nn.Parameter(torch.zeros(size=(ch,1)))
            elif len(ch) == 2:
                self.delta = torch.nn.Parameter(torch.zeros(size=(ch[0],1)))
                self.zero_point = torch.nn.Parameter(torch.zeros(size=(ch[0],1)))
            else:
                self.delta = torch.nn.Parameter(torch.zeros(size=(ch[0],1,1,1)))
                self.zero_point = torch.nn.Parameter(torch.zeros(size=(ch[0],1,1,1)))

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            # if self.leaf_param:
            #     delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            #     self.delta = torch.nn.Parameter(delta)
            #     self.zero_point = torch.nn.Parameter(zero_point)
            #     print("Making initialization")
            # else:
                # self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            delta, zero_point, self.raw_zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.delta = torch.nn.Parameter(delta)
            self.zero_point = torch.nn.Parameter(zero_point)
            self.inited = True

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels//2, self.n_levels//2 - 1)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point, raw_zero_point = None, None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            raw_zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c], raw_zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
                raw_zero_point = raw_zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
                raw_zero_point = raw_zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                if self.sym:
                    x_absmax = max(abs(x_min), x_max)
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    # warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)
                raw_zero_point = -x_min

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                if self.sym:
                    x_absmax = max(abs(x_min), x_max)
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round() if not self.sym else 0
                        raw_zero_point = -new_min if not self.sym else 0
            else:
                raise NotImplementedError

        return delta, zero_point, raw_zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        weight_quant_params['ch'] = self.weight.shape
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        act_quant_params['disable_act_quant'] = disable_act_quant
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr
        
        self.cache_features      = 'none'
        self.cached_inp_features = []
        self.cached_out_features = []
        
        if self.weight.dim() == 4:#Conv
            n_ch = self.weight.shape[0]
            self.alpha_out = torch.nn.Parameter(torch.ones(n_ch) .view(1, n_ch, 1, 1))
            self.beta_out  = torch.nn.Parameter(torch.zeros(n_ch).view(1, n_ch, 1, 1))
        else:
            n_ch = self.weight.shape[0]
            self.alpha_out = torch.nn.Parameter(torch.ones(n_ch) .view(1, n_ch))
            self.beta_out  = torch.nn.Parameter(torch.zeros(n_ch).view(1, n_ch))
        self.selection = None # for greedy selection
        
        self.selectionInited = False
        self.pathName = ''
        self.dump_cnt = 0#NOTE:debug purpose. remove after debugging
        
    def forward(self, input: torch.Tensor):
        if self.cache_features == 'if':
            self.cached_inp_features += [input.cpu().clone().detach()]
            
        if self.use_weight_quant and self.cache_features == 'none':
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
            
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        if self.use_weight_quant and self.cache_features == 'none':
            out = out * self.alpha_out + self.beta_out
            

        if self.se_module is not None:
            out = self.se_module(out)
            
        # if self.cache_features == 'if':
        #     print(out[0][1])
        out = self.activation_function(out)

        if not self.disable_act_quant:
            if self.use_act_quant:
                out = self.act_quantizer(out)
                
        if self.cache_features == 'debug':
            torch.save(out, f'fc_{self.dump_cnt}.pt')
            self.dump_cnt += 1
        
        if self.cache_features == 'of':
            self.cached_out_features += [out.cpu().clone().detach()]
            
        return out
    
    # def cal_quantLoss(self):
    #     quant_out = self.selfForward()
    #     ori_out = self.cached_out_features
        
    #     # weight_quant = 
        
    #     #Calculate Loss(difference between quantized output and original output)
    #     # loss = F.mse_loss(weight_quant, self.weight)
    #     # loss = (self.weight_quantizer(self.weight) - self.weight).abs().pow(3).sum(1).mean().detach().cpu().item()
    #     loss = self.getLoss(quant_out, ori_out, p=2.4)

    #     #TODO:Temp
    #     # self.cached_out_quant_features = quant_out
            
    #     return loss

    def set_quant_init_state(self):
        self.weight_quantizer.inited = True
        self.act_quantizer.inited = True
        
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        
    def disable_cache_features(self):
        self.cache_features = 'none'
        
    def clear_cached_features(self):
        self.cached_inp_features = []
        self.cached_out_features = []
    
    # def selfForward(self):
    #     quant_out = []
    #     for inp in self.cached_inp_features:
    #         quant_out += [self.forward(inp).detach()]
    #     return quant_out
    
    def getLoss(self, A, B, p=2.0):
        loss = 0.0
        for i in range(len(B)):
            loss += (B[i] - A[i]).abs().pow(p).sum(1).mean()
        return loss.detach().cpu().item()
        
    def run_layerGreedy(self, nc=0):
        if nc >= self.weight_quantizer.nchannel[0]:
            return 1e10
        # quant_out = []
        # for inp in self.cached_inp_features:
        #     quant_out += [self.forward(inp).detach()]
        
        # ori_out = self.cached_out_features
        n_channel = self.weight_quantizer.nchannel
        # t  = tqdm(range(n_channel[0]), desc='', position=0, dynamic_ncols=True)
        # print(f"n_channel: {n_channel}")
        if not self.selectionInited:
            self.selection = torch.zeros((n_channel[0], n_channel[1]))
            self.weight_quantizer.setScale(self.selection)
            self.minGreedyLoss = (self.weight_quantizer(self.weight) - self.weight).abs().pow(2.4).sum(1).mean().detach().cpu().item()
            self.selectionInited = True
            
        # for nc in t:
        # for ic in tqdm(range(n_channel[1]), desc='', leave=False, position=1):
        with torch.no_grad():
            for ic in range(n_channel[1]):
                minK = 0
                for k in range(0, 8):
                    self.selection[nc, ic] = k
                    self.weight_quantizer.setScale(self.selection)
                    # quant_out = self.selfForward()
                    # loss = self.getLoss(quant_out, ori_out)
                    loss = (self.weight_quantizer(self.weight) - self.weight).abs().pow(2.4).sum(1).mean().detach().cpu().item()
                    if loss < self.minGreedyLoss :
                        self.minGreedyLoss  = loss
                        minK = k
                        # t.set_description(f"best {minLoss:.6f} cur {loss:.6f} sel_cnt{sel_cnt}")
            self.selection[nc, ic] = minK
            self.weight_quantizer.setScale(self.selection)
        return self.minGreedyLoss 

    def run_layerDist(self, nc=0):
        if nc >= self.weight_quantizer.nchannel[0]:
            return 1e10 #Default Loss
        
        n_channel = self.weight_quantizer.nchannel
        if not self.selectionInited:
            self.selection = torch.zeros((n_channel[0], n_channel[1]))
            self.weight_quantizer.setScale(self.selection)
            self.selectionInited = True
            
        qParam = {0: 1.0, 1:0.5, 2:0.25, 3:0.75}
        
        with torch.no_grad():
            delta = self.weight_quantizer.delta[nc, 0, 0, 0]
            zero_point = self.weight_quantizer.zero_point[nc, 0, 0, 0]
            
            for ic in range(n_channel[1]):
                minLoss = 1e10
                minK = 0
                weight = self.weight[nc, ic, :, :]
                for k in range(0, 2):
                    quant_weight = torch.round(weight / (delta/qParam[k]))
                    quant_weight = torch.clamp(quant_weight + zero_point, 0, self.weight_quantizer.n_levels - 1)
                    quant_weight = (quant_weight - zero_point) * (delta/qParam[k])
                    loss = (quant_weight - weight).abs().pow(2.0).sum(1).mean().detach().cpu().item()
                    if loss < minLoss :
                        minLoss  = loss
                        minK = k
                self.selection[nc, ic] = minK
            self.weight_quantizer.setScale(self.selection)
        return 1e10 #Default Loss

    def selection_init(self):
        n_channel = self.weight_quantizer.nchannel
        self.selection = torch.zeros((n_channel[0], n_channel[1]))
        self.weight_quantizer.setScale(self.selection)
        self.selectionInited = True
    
    def restore_selection(self):
        sel = self.selection
        scale = self.weight_quantizer.shiftedScale
        for nc in range(sel.shape[0]):
            for ic in range(sel.shape[1]):
                sel[nc, ic] = 0 if scale[nc, ic, 0, 0] == 1.0 else 1
        self.weight_quantizer.setScale(self.selection)
        
    @torch.no_grad()
    def run_GreedyLoss(self, useWeight=False):
        cached_inp = self.cached_inp_features
        cached_out = self.cached_out_features
        
        n_channel = self.weight_quantizer.nchannel
        if not self.selectionInited:
            self.selection_init()
            
        #Run and Calculate Loss
        quant_out = []
        for inp in cached_inp:
            quant_out += [self.forward(inp).detach()]
        if useWeight:
            self.minGreedyLoss = 1e10
        else:
            self.minGreedyLoss = self.getLoss(cached_out, quant_out)
        
        for _ in range(1): # dobule greedy optimization
            initialLoss = self.minGreedyLoss
            t = tqdm(range(n_channel[0]), desc='', dynamic_ncols=True)
            for nc in t:
                for ic in range(n_channel[1]):
                    minK = self.selection[nc, ic].item()
                    if useWeight:
                        self.minGreedyLoss = 1e10
                    rng = range(0, 2)
                    for k in rng:
                        if k == minK:
                            continue
                        self.selection[nc, ic] = k
                        self.weight_quantizer.setScale(self.selection)
                        
                        if useWeight:
                            loss = (self.weight_quantizer(self.weight) - self.weight).abs().pow(2.4).sum(1).mean().detach().cpu().item()
                        else:
                            quant_out = []
                            for inp in self.cached_inp_features:
                                quant_out += [self.forward(inp).detach()]
                            loss = self.getLoss(cached_out, quant_out)
                        
                        if loss < self.minGreedyLoss :
                            self.minGreedyLoss  = loss
                            minK = k
                    self.selection[nc, ic] = minK
                    self.weight_quantizer.setScale(self.selection)
                t.set_description(f"Loss {initialLoss:.5f}->{self.minGreedyLoss:.5f}")
            count_nonzero = torch.count_nonzero(self.selection)
            total_elements = self.selection.numel()
            ratio = count_nonzero / total_elements
            print(f"{ratio*100:.3f}% of the selections are 1/2 scale")

    @torch.no_grad()
    def run_GreedyLossSorted(self, useWeight=False):
        cached_inp = self.cached_inp_features
        cached_out = self.cached_out_features
        
        
        n_channel = self.weight_quantizer.nchannel
        if not self.selectionInited:
            self.selection = torch.zeros((n_channel[0], n_channel[1]))
            self.weight_quantizer.setScale(self.selection)
            #Run and Calculate Loss
            quant_out = []
            for inp in cached_inp:
                quant_out += [self.forward(inp).detach()]
            if useWeight:
                self.minGreedyLoss = 1e10
            else:
                self.minGreedyLoss = self.getLoss(cached_out, quant_out)
            self.selectionInited = True
        
        
        initialLoss = self.minGreedyLoss
        
        #Calculate Loss Matrix
        lossMatrix = torch.zeros((n_channel[0], n_channel[1]))
        t = tqdm(range(n_channel[0]), desc='Loss Matrix', dynamic_ncols=True)
        for nc in t:
            for ic in range(n_channel[1]):
                sel = torch.zeros((n_channel[0], n_channel[1]))
                sel[nc, ic] = 1
                self.weight_quantizer.setScale(sel)
                quant_out = []
                for inp in self.cached_inp_features:
                    quant_out += [self.forward(inp).detach()]
                loss = self.getLoss(cached_out, quant_out)
                lossMatrix[nc, ic] = loss
                
        flattened_arr = lossMatrix.detach().cpu().numpy().flatten()
        argsorted = np.argsort(flattened_arr)
        sorted_row_indices, sorted_col_indices = np.unravel_index(argsorted, lossMatrix.shape)
        
        t = tqdm(range(n_channel[0]), desc='', dynamic_ncols=True)
        for nc in t:
            for ic in range(n_channel[1]):
                nc = sorted_row_indices[nc]
                ic = sorted_col_indices[ic]
                minK = 0
                if useWeight:
                    self.minGreedyLoss = 1e10
                    rng = range(0, 2)
                else:
                    rng = range(1, 2)
                for k in rng:
                    self.selection[nc, ic] = k
                    self.weight_quantizer.setScale(self.selection)
                    
                    if useWeight:
                        loss = (self.weight_quantizer(self.weight) - self.weight).abs().pow(2.4).sum(1).mean().detach().cpu().item()
                    else:
                        quant_out = []
                        for inp in self.cached_inp_features:
                            quant_out += [self.forward(inp).detach()]
                        loss = self.getLoss(cached_out, quant_out)
                    
                    if loss < self.minGreedyLoss :
                        self.minGreedyLoss  = loss
                        minK = k
                self.selection[nc, ic] = minK
                self.weight_quantizer.setScale(self.selection)
            t.set_description(f"Loss {initialLoss:.5f}->{self.minGreedyLoss:.5f}")
