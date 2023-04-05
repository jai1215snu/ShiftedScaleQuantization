import torch
from quant.quant_layer import QuantModule, StraightThrough, lp_loss

def layer_recon_shiftedScale(layer: QuantModule):
    
    opt_params = [layer.weight_quantizer.alpha]
    optimizer = torch.optim.Adam(opt_params)
    scheduler = None