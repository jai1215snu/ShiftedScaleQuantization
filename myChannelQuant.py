import torch.nn as nn
from common import *
from data.cifar10 import build_cifar10_data
from pretrained.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from quant import *
from myVisualize import *
import torch
import linklink as link
from quant.quant_layer import QuantModule, StraightThrough, lp_loss
from quant.quant_model import QuantModel
from quant.block_recon import LinearTempDecay
from quant.adaptive_rounding import AdaRoundQuantizer
from quant.channelQuant import ChannelQuant
from quant.data_utils import save_grad_data, save_inp_oup_data
from tqdm import tqdm
import pickle

class LossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
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
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss

def layer_channelGreedy(model: QuantModel, layer: QuantModule, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False, eval: bool = False, shuffle_ratio: float = 0.0, qscale: float = 1.0):
    # Replace weight quantizer to channelWiseQuantizer
    layer.weight_quantizer = ChannelQuant(uaq=layer.weight_quantizer, weight_tensor=layer.org_weight.data, shuffle_ratio=0.0, qscale=qscale)
    device = 'cuda'
    cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, asym, act_quant, batch_size)
    
    idx = torch.randperm(cached_inps.size(0))[:batch_size]
    cur_inp = cached_inps[idx]
    cur_out = cached_outs[idx]
    out_quant = layer(cur_inp)
    errPrv = lp_loss(out_quant, cur_out, p=2.4).detach().cpu().data
    # print("Original err: ", errPrv)
    numChannel = layer.weight.shape[0]
    errOfChannel = np.zeros(numChannel)
    for i in range(numChannel):
        layer.weight_quantizer.changeResol(i)
        out_quant = layer(cur_inp)
        errOfChannel[i] = lp_loss(out_quant, cur_out, p=2.4).detach().cpu().data
    
    # print(errOfChannel)
    sorted_idx = np.argsort(errOfChannel)
    # print(sorted_idx)
    # print(errOfChannel[sorted_idx])
    
    layer.weight_quantizer.resetResol()
    for i in sorted_idx:
        layer.weight_quantizer.setResol(i)
        out_quant = layer(cur_inp)
        scaledLoss = lp_loss(out_quant, cur_out, p=2.4).detach().cpu().data
        if scaledLoss < errPrv:
            errPrv = scaledLoss
            # print("Channel %d: err: %f" % (i, errPrv))
        else:
            layer.weight_quantizer.setResol(i, 0)
    # out_quant = layer(cur_inp)
    # errPrv = lp_loss(out_quant, cur_out, p=2.4).detach().cpu().data
    # print(">>>After err: ", errPrv)
    # print(layer.weight_quantizer.simpleScale)

def layer_channelRandomize(model: QuantModel, layer: QuantModule, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False, eval: bool = False, shuffle_ratio: float = 0.0, qscale: float = 1.0):
    # Replace weight quantizer to channelWiseQuantizer
    layer.weight_quantizer = ChannelQuant(uaq=layer.weight_quantizer, weight_tensor=layer.org_weight.data, shuffle_ratio=shuffle_ratio, qscale=qscale)

    
    # #No study, when eval mode
    # if eval:
    #     iters = 0
        
    # model.set_quant_state(False, False)
    # layer.set_quant_state(True, act_quant)
    # round_mode = 'learned_hard_sigmoid'
    
    # if not include_act_func:
    #     org_act_func = layer.activation_function
    #     layer.activation_function = StraightThrough()

    # if not act_quant:
    #     # Replace weight quantizer to AdaRoundQuantizer
    #     layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
    #                                                weight_tensor=layer.org_weight.data)
    #     layer.weight_quantizer.soft_targets = True

    #     # Set up optimizer
    #     opt_params = [layer.weight_quantizer.alpha]
    #     optimizer = torch.optim.Adam(opt_params)
    #     scheduler = None
    # else:
    #     # Use UniformAffineQuantizer to learn delta
    #     opt_params = [layer.act_quantizer.delta]
    #     optimizer = torch.optim.Adam(opt_params, lr=lr)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)
    # loss_mode = 'none' if act_quant else 'relaxation'
    # rec_loss = opt_mode

    # loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
    #                          max_count=iters, rec_loss=rec_loss, b_range=b_range,
    #                          decay_start=0, warmup=warmup, p=p)

    # # Save data before optimizing the rounding
    # if iters>0:
    #     cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, asym, act_quant, batch_size)
        
    # if opt_mode != 'mse':
    #     cached_grads = save_grad_data(model, layer, cali_data, act_quant, batch_size=batch_size)
    # else:
    #     cached_grads = None
    # device = 'cuda'
    # for i in range(iters):
    #     idx = torch.randperm(cached_inps.size(0))[:batch_size]
    #     cur_inp = cached_inps[idx]
    #     cur_out = cached_outs[idx]
    #     cur_grad = cached_grads[idx] if opt_mode != 'mse' else None

    #     optimizer.zero_grad()
    #     out_quant = layer(cur_inp)

    #     err = loss_func(out_quant, cur_out, cur_grad)
    #     err.backward(retain_graph=True)
    #     if multi_gpu:
    #         for p in opt_params:
    #             link.allreduce(p.grad)
    #     optimizer.step()
    #     if scheduler:
    #         scheduler.step()

    # torch.cuda.empty_cache()

    # # Finish optimization, use hard rounding.
    # layer.weight_quantizer.soft_targets = False

    # # Reset original activation function
    # if not include_act_func:
    #     layer.activation_function = org_act_func
    
def channelGreedyTest(qnn, test_loader, cali_data, args):
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse', eval=True, shuffle_ratio=0.0, qscale=1/2)
    
    def channelGreedy(model: nn.Module):
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    continue
                else:
                    layer_channelGreedy(qnn, module, **kwargs)
            else:
                channelGreedy(module)
                
    # Start calibration
    def printGreedy(model: nn.Module):
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    continue
                else:
                    # print(name, module.weight_quantizer.simpleScale)
                    print(module.use_weight_quant)
            else:
                printGreedy(module)
    
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    # channelGreedy(qnn)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    print(f'Quantized accuracy After  brecq: {validate_model(test_loader, qnn):.3f}')

def channelRandomizeTest(qnn, test_loader, cali_data, shuffle_ratio, args):
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse', eval=True, shuffle_ratio=shuffle_ratio, qscale=1/2)
    
    def channelRandomize(model: nn.Module):
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    continue
                else:
                    layer_channelRandomize(qnn, module, **kwargs)
                    # layer_channelGreedy(qnn, module, **kwargs)
            else:
                channelRandomize(module)
    # Start calibration
    
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    
    testResult = []
    for i in tqdm(range(3000), desc=f'random value={shuffle_ratio}'):
        channelRandomize(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=False)
        testResult.append(validate_model(test_loader, qnn, print_result=False))
    print("Best Accuracy : ", max(testResult))
    with open(f'./results/channelRandom1.2_fixed.{shuffle_ratio}.pkl', 'wb') as f:
        pickle.dump(testResult, f)
            
if __name__ == '__main__':
    args = loadArgments()
    
    seed_all(args.seed)
        # build cifar10 data loader
    train_loader, test_loader = build_cifar10_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)
    # ratios = [0.0, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]
    ratios = [0.08]
    for shuffle_ratio in ratios:
        cnn = resnet18(pretrained=True, device='cuda:0')
        # print_model_hierarchy(cnn)
        cnn.cuda()
        cnn.eval()
        print(f'accuracy of original : {validate_model(test_loader, cnn):.3f}')
        
        # build quantization parameters
        wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse', 'CWQ':True}
        aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
        qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
        # print_model_hierarchy(qnn)
        qnn.cuda()
        qnn.eval()
        if not args.disable_8bit_head_stem:
            print('Setting the first and the last layer to 8-bit')
            qnn.set_first_last_layer_to_8bit()
        cali_data = get_train_samples(train_loader, num_samples=args.num_samples)
        device = next(qnn.parameters()).device
        
        # # Initialize weight quantization parameters
        qnn.set_quant_state(True, False)
        qnn.set_quant_init_state()
        qnn.load_state_dict(torch.load(f'./checkPoint/QNN_CW_W{args.n_bits_w}_FP32.pth'))
        
        _ = qnn(cali_data[:64].to(device))
        # torch.save(qnn.state_dict(), f'./checkPoint/QNN_CW_W{args.n_bits_w}_FP32.pth')
        # st = torch.load(f'./checkPoint/QNN_CW_W{args.n_bits_w}_FP32.pth')
        print(f'Quantized accuracy before brecq: {validate_model(test_loader, qnn):.3f}')
        #Load Weight
        channelRandomizeTest(qnn, test_loader, cali_data, shuffle_ratio, args)
        # channelGreedyTest(qnn, test_loader, cali_data, args)
        # break
        


