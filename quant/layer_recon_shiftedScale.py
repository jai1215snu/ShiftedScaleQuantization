import torch
from quant.quant_layer import QuantModule, UniformAffineQuantizer, lp_loss
from quant.quant_block import BaseQuantBlock
import torch.distributed as dist
import numpy as np
import pickle
import torch.nn as nn
import common
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm, trange

def block_recon_shiftedScale(block: BaseQuantBlock, iters: int = 20000, lmda: float = 1., model=None, test_loader=None, act=False, adaround=False, useShiftedScale=True):
    block.train()
    warmup = 0.2
    p = 2.0
    b_range = (20, 2)
    device = next(model.parameters()).device
    lr = 4e-4
    scheduler = None
    
    opt_params = []
    if act:
        #Init setting for all weight quantizer
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                if module.act_quantizer.disable_act_quant:
                    continue
                opt_params += [module.act_quantizer.delta]
            elif isinstance(module, UniformAffineQuantizer):
                if module.disable_act_quant:
                    continue
                opt_params += [module.delta]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)
    else:
        #Init setting for all weight quantizer
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                # if not module.weight_quantizer.shiftedDone:
                if adaround:
                    # module.weight_quantizer.delta = module.weight_quantizer.delta
                    if module.weight_quantizer.opt_mode == 'learned_hard_sigmoid':#when Shift first
                        module.weight_quantizer.update_delta()
                    weight_tensor=module.org_weight.data
                    module.weight_quantizer.init_beta(x=weight_tensor.clone().detach())
                    module.weight_quantizer.opt_mode = 'adaround'
                    opt_params += [module.weight_quantizer.beta]
                    # module.weight_quantizer.update_delta()
                else:
                    weight_tensor=module.org_weight.data
                    module.weight_quantizer.init_v(x=weight_tensor.clone().detach())
                    opt_params += [module.weight_quantizer.alpha]
                    # module.weight_quantizer.opt_mode = 'learned_hard_sigmoid'
        optimizer = torch.optim.Adam(opt_params)
        # if not adaround:
        #     scheduler = lr_scheduler.StepLR(optimizer, step_size=iters*0.3, gamma=0.3)
        
        # warmup = 0
        opt_param_num = sum([p.numel() for p in opt_params])
        print("number of elements in opt_params: {}".format(opt_param_num))
        
    loss_mode = 'none' if act else 'relaxation'
    
    # lmda = lmda * opt_param_num
    loss_func = ScaleLossBlockFunction(block, round_loss=loss_mode, lmda=lmda,
                            max_count=iters, b_range=b_range,
                            decay_start=0, warmup=warmup, p=p, adaround=adaround)
    
    cached_inp = torch.cat(block.cached_inp_features).to(device)
    cached_out = torch.cat(block.cached_out_features).to(device)
    batch_size = 32
    
    target = []
    total_idx = iters
    start_loss = 0.0
    t = tqdm(range(total_idx), desc=f'', dynamic_ncols=True)
    
    for i in t:
        permIdx = torch.randperm(cached_inp.size(0))[:batch_size]
        cur_inp = cached_inp[permIdx].to(device)
        cur_out = cached_out[permIdx].to(device)
        
        optimizer.zero_grad()
        quant_out = block(cur_inp)
        
        err = loss_func(quant_out, cur_out)
        err.backward(retain_graph=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        if i%500 == 0:
            start_loss = max(start_loss, loss_func.rec_loss)
            # print(f"{start_loss:.6f} -> {loss_func.rec_loss:.6f} {loss_func.round_loss_val:.3f}")
            t.set_description(f"{start_loss:.6f} -> {loss_func.rec_loss:.6f} {loss_func.round_loss_val:.3f} ")

    rec_loss_out = []
    cur_inp = cached_inp[:batch_size].to(device)
    cur_out = cached_out[:batch_size].to(device)
    optimizer.zero_grad()
    quant_out = block(cur_inp)
    err = loss_func(quant_out, cur_out)
    print(f"Soft Round : {start_loss:.6f} -> {loss_func.rec_loss:.6f} {loss_func.round_loss_val:.3f}")
    rec_loss_out.append(loss_func.rec_loss)
    if not act:
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                if adaround:
                    module.weight_quantizer.hard_round = True
                    weight = module.weight_quantizer(module.weight)
                    # print(module.weight_quantizer.beta[0][0])
                    # print(weight[0][0])
                else:
                    module.weight_quantizer.hard_targets = True
                    module.weight_quantizer.shiftedDone = True
    
    optimizer.zero_grad()
    quant_out = block(cur_inp)
    err = loss_func(quant_out, cur_out)
    print(f"Hard Round : {start_loss:.6f} -> {loss_func.rec_loss:.6f} {loss_func.round_loss_val:.3f}")
    rec_loss_out.append(loss_func.rec_loss)
    torch.cuda.empty_cache()
    model.eval()
    return rec_loss_out

def block_recon_shiftedScale_old(block: BaseQuantBlock, iters: int = 20000, lmda: float = 1., model=None, test_loader=None, act=False, adaround=False, useShiftedScale=True):
    block.train()
    warmup = 0.2
    p = 2.0
    b_range = (20, 2)
    device = next(model.parameters()).device
    lr = 4e-4
    
    opt_params = []
    if act:
        #Init setting for all weight quantizer
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                if module.act_quantizer.disable_act_quant:
                    continue
                opt_params += [module.act_quantizer.delta]
            elif isinstance(module, UniformAffineQuantizer):
                if module.disable_act_quant:
                    continue
                opt_params += [module.delta]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)
    else:
        #Init setting for all weight quantizer
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                # if not module.weight_quantizer.shiftedDone:
                if adaround:
                    opt_params += [module.weight_quantizer.beta]
                    module.weight_quantizer.opt_mode = 'adaround'
                    # module.weight_quantizer.update_delta()
                else:
                    weight_tensor=module.org_weight.data
                    module.weight_quantizer.init_v(x=weight_tensor.clone().detach())
                    opt_params += [module.weight_quantizer.alpha]
                    # module.weight_quantizer.opt_mode = 'learned_hard_sigmoid'
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=iters*, gamma=0.3)
        # warmup = 0
        opt_param_num = sum([p.numel() for p in opt_params])
        print("number of elements in opt_params: {}".format(opt_param_num))
        
    loss_mode = 'none' if act else 'relaxation'
    
    # lmda = lmda * opt_param_num
    loss_func = ScaleLossBlockFunction(block, round_loss=loss_mode, lmda=lmda,
                            max_count=iters, b_range=b_range,
                            decay_start=0, warmup=warmup, p=p, adaround=adaround)
    
    cached_inp = torch.cat(block.cached_inp_features)
    cached_out = torch.cat(block.cached_out_features)
    batch_size = 32
    
    target = []
    sub_iter = cached_inp.size(0)//batch_size
    total_idx = iters//sub_iter
    
    start_loss = 0.0
    t = tqdm(range(total_idx), desc=f'', dynamic_ncols=True)
    for i in t:
        permIdx = torch.randperm(cached_inp.size(0))
        for k in range(sub_iter):
            idx = permIdx[batch_size*k:batch_size*(k+1)]
            cur_inp = cached_inp[idx].to(device)
            cur_out = cached_out[idx].to(device)
            
            # cur_inp = torch.load('cur_inp.pt')
            # cur_out = torch.load('cur_out.pt')
            
            optimizer.zero_grad()
            quant_out = block(cur_inp)
            
            # print('batch', batch_size)
            # print(cur_out[0][0][0])
            # print(quant_out[0][0][0])
            # print(torch.sum(quant_out))
            
            err = loss_func(quant_out, cur_out)
            # exit(1)
            err.backward(retain_graph=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                
            run_idx = (i*sub_iter)+k
            if run_idx%500 == 0:
                start_loss = max(start_loss, loss_func.rec_loss)
                t.set_description(f"{start_loss:.6f} -> {loss_func.rec_loss:.6f} {loss_func.round_loss_val:.3f} ")
                
        # if i%50 == 0 and not act and not adaround:
            # target.append(block.conv1.weight_quantizer.get_sig_soft_targets().detach().cpu().numpy())

        # if run_idx%50 == 0:
            # model.store_quantization_state()
            # model.set_quant_state(True, act)# For Accuracy test
            # if act:
            #     hard_acc = common.validate_model(test_loader, model, simple=True)
            #     result_message = f'accuracy [{i:5d} / {run_idx:5d}] : {hard_acc:.3f} {loss_func.report()}'
            # else:
            #     soft_acc = common.validate_model(test_loader, model, simple=True)
            #     hard_acc = soft_acc
                # for name, module in block.named_modules():
                #     if isinstance(module, QuantModule):
                #         if adaround:
                #             module.weight_quantizer.hard_round = True
                #         else:
                #             module.weight_quantizer.hard_targets = True
                # hard_acc = common.validate_model(test_loader, model, simple=True)
                # for name, module in block.named_modules():
                #     if isinstance(module, QuantModule):
                #         if adaround:
                #             module.weight_quantizer.hard_round = False
                #         else:
                #             module.weight_quantizer.hard_targets = False
                # result_message = f'accuracy [{i:5d} / {run_idx:5d}] : {soft_acc:.3f}/{hard_acc:.3f} {loss_func.report()}'
                
            # print(result_message)
            # model.restore_quantization_state()
    
    if not act:
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                if adaround:
                    module.weight_quantizer.hard_round = True
                else:
                    module.weight_quantizer.hard_targets = True
                    module.weight_quantizer.shiftedDone = True
                
        # with open(f'./temp/param.pkl', 'wb') as f:
        #     target = np.array(target)
        #     pickle.dump(target, f)
    
    torch.cuda.empty_cache()
    model.eval()

def layer_recon_shiftedScale(layer: QuantModule, iters: int = 20000, lmda: float = 1., model=None, test_loader=None, act=False, adaround=False, useShiftedScale=True):
    model.train()
    warmup = 0.2
    p = 2.0
    b_range = (20, 2)
    
    device = torch.device('cuda')
    #Torch Parallel
    if adaround:
        if layer.weight_quantizer.opt_mode == 'learned_hard_sigmoid':#when Shift first
            layer.weight_quantizer.update_delta()
        weight_tensor=layer.org_weight.data
        layer.weight_quantizer.init_beta(x=weight_tensor.clone().detach())
        layer.weight_quantizer.opt_mode = 'adaround'
        opt_params = [layer.weight_quantizer.beta]
    else:
        weight_tensor=layer.org_weight.data
        layer.weight_quantizer.init_v(x=weight_tensor.clone().detach())
        opt_params = [layer.weight_quantizer.alpha]
    optimizer = torch.optim.Adam(opt_params)
    scheduler = None
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.3)
    loss_mode = 'none' if act else 'relaxation'
    loss_func = ScaleLossFunction(layer, round_loss=loss_mode, lmda=lmda,
                            max_count=iters, b_range=b_range,
                            decay_start=0, warmup=warmup, p=p, adaround=adaround)
    
    cached_inp = torch.cat(layer.cached_inp_features).to(device)
    cached_out = torch.cat(layer.cached_out_features).to(device)
    batch_size = 32
    
    target = []
    total_idx = iters
    start_loss = 0.0
    t = tqdm(range(total_idx), desc=f'', dynamic_ncols=True)
    for i in t:
        permIdx = torch.randperm(cached_inp.size(0))[:batch_size]
        cur_inp = cached_inp[permIdx].to(device)
        cur_out = cached_out[permIdx].to(device)
        
        optimizer.zero_grad()
        # for inp, oup in zip(cached_inp, cached_out):
        quant_out = layer(cur_inp)
        
        err = loss_func(quant_out, cur_out)
        err.backward(retain_graph=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        if i%500 == 0:
            start_loss = max(start_loss, loss_func.rec_loss)
            t.set_description(f"{start_loss:.6f} -> {loss_func.rec_loss:.6f} {loss_func.round_loss_val:.3f} ")

    rec_loss_out = []
    cur_inp = cached_inp[:batch_size].to(device)
    cur_out = cached_out[:batch_size].to(device)    
    optimizer.zero_grad()
    quant_out = layer(cur_inp)
    err = loss_func(quant_out, cur_out)
    print(f"Soft Round : {start_loss:.6f} -> {loss_func.rec_loss:.6f} {loss_func.round_loss_val:.3f}")
    rec_loss_out.append(loss_func.rec_loss)    
    
    if adaround:
        layer.hard_round = True
    else:
        layer.weight_quantizer.hard_targets = True
        layer.weight_quantizer.shiftedDone = True
    optimizer.zero_grad()
    quant_out = layer(cur_inp)
    err = loss_func(quant_out, cur_out)
    print(f"Hard Round : {start_loss:.6f} -> {loss_func.rec_loss:.6f} {loss_func.round_loss_val:.3f}")
    rec_loss_out.append(loss_func.rec_loss)
    
    torch.cuda.empty_cache()
    model.eval()
    return rec_loss_out
        
class ScaleLossBlockFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 lmda: float = 1.,
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.0,
                 adaround: bool = False):

        self.block = block
        self.round_loss = round_loss
        self.lmda = lmda
        self.loss_start = max_count * warmup
        self.itr = max_count
        self.p = p
        self.total_loss = self.rec_loss = self.round_loss_val = self.b = 0
        
        self.temp_decay = LinearTempDecayShift(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.adaround = adaround
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
        rec_loss = lp_loss(pred, tgt, p=self.p)
        
        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    if self.adaround:
                        round_vals = module.weight_quantizer.get_soft_round()
                        round_loss += self.lmda * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                    else:
                        # round_vals = module.weight_quantizer.get_soft_targets()
                        round_vals = module.weight_quantizer.get_sig_soft_targets()
                        # round_loss += self.lmda * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                        # round_vals[round_vals] = 1e-10 # to avoid nan
                        round_loss += self.lmda * (- torch.sum(round_vals * torch.log(round_vals+1e-10)))
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        
        #For report
        self.total_loss = total_loss.item()
        self.rec_loss = rec_loss.item()
        self.round_loss_val = round_loss
        self.b = b
        # if self.count % 500 == 0 or (self.itr-1 == self.count):
        #     print('Total loss:\t{:.6f} (rec:{:.6f}, round:{:.6f})\tb={:.2f}\tcount={}'.format(
        #           float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        self.count += 1
        return total_loss

    def report(self):
        return 'Total loss:\t{:.6f} (rec:{:.6f}, round:{:.6f})\tb={:.2f}'.format(
                float(self.total_loss), float(self.rec_loss), float(self.round_loss_val), self.b)
        
class ScaleLossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 lmda: float = 1.,
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.0,
                 adaround: bool = False
                 ):

        self.layer = layer
        self.round_loss = round_loss
        self.lmda = lmda
        self.loss_start = max_count * warmup
        self.itr = max_count
        self.p = p
        self.total_loss = self.rec_loss = self.round_loss_val = self.b = 0

        self.adaround = adaround
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
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            if self.adaround:
                round_loss = 0
                round_vals = self.layer.weight_quantizer.get_soft_round()
                round_loss += self.lmda * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                round_loss = 0
                # round_vals = self.layer.weight_quantizer.get_soft_targets()
                round_vals = self.layer.weight_quantizer.get_sig_soft_targets()
                # round_loss += self.lmda * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                # round_vals[round_vals] = 1e-10 # to avoid nan
                round_loss += self.lmda * (- torch.sum(round_vals * torch.log(round_vals+1e-10)))
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        # if self.count % 500 == 0 or (self.itr-1 == self.count):
        #     print('Total loss:\t{:.6f} (rec:{:.6f}, round:{:.6f})\tb={:.2f}\tcount={}'.format(
        #           float(total_loss), float(rec_loss), float(round_loss), b, self.count))
            
        self.total_loss = total_loss.item()
        self.rec_loss = rec_loss.item()
        self.round_loss_val = round_loss
        self.b = b
        
        self.count += 1
        return total_loss
    
    def report(self):
        return 'Total loss:\t{:.6f} (rec:{:.6f}, round:{:.6f})\tb={:.2f}'.format(
                float(self.total_loss), float(self.rec_loss), float(self.round_loss_val), self.b)
    
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
