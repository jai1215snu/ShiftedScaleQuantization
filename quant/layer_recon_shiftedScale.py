import torch
from quant.quant_layer import QuantModule, StraightThrough, lp_loss
import numpy as np
import pickle

def layer_recon_shiftedScale(layer: QuantModule, iters: int = 20000, lmda: float = 1.):
    # lmda = 0.0000001 #NOTE:CHECK
    warmup = 0
    p = 2.0
    b_range = (20, 2)
    device = 'cuda'
    
    opt_params = [layer.weight_quantizer.alpha]
    optimizer = torch.optim.Adam(opt_params)
    scheduler = None
    layer.weight_quantizer.opt_mode = 'learned_hard_sigmoid'
    loss_func = ScaleLossFunction(layer, round_loss='relaxation', lmda=lmda,
                            max_count=iters, b_range=b_range,
                            decay_start=0, warmup=warmup, p=p)
    
    cached_inp = layer.cached_inp_features
    cached_out = layer.cached_out_features
    

    target = []
    for i in range(iters):
        optimizer.zero_grad()
        for inp, oup in zip(cached_inp, cached_out):
            quant_out = layer(inp)
            
            err = loss_func(quant_out, oup)
            err.backward(retain_graph=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        # print(i, layer.weight_quantizer.get_soft_targets()[0])
        target.append(layer.weight_quantizer.get_soft_targets().detach().cpu().numpy())
        
    with open(f'./temp/param.pkl', 'wb') as f:
        target = np.array(target)
        pickle.dump(target, f)
        
    layer.weight_quantizer.hard_targets = True
                
    torch.cuda.empty_cache()
        
class ScaleLossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 lmda: float = 1.,
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
        self.round_loss = round_loss
        self.lmda = lmda
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecayShift(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            round_loss += self.lmda * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.6f} (rec:{:.6f}, round:{:.6f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss
    
class LinearTempDecayShift:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
