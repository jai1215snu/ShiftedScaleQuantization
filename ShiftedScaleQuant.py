#Training Set
import torch.nn as nn
from data.cifar10 import build_cifar10_data
from data.imagenet import build_imagenet_data
from quant.quant_layer import QuantModule
from quant.quant_block import BaseQuantBlock, QuantBasicBlock
import pickle
from myScaledMethods import *
from quant.layer_recon_shiftedScale import *
from quant.layer_recon_fused_shiftedScale import *

#multi gpus
# import linklink as link
# from linklink.dist_helper import dist_init, allaverage
# import torch.distributed as dist
# import torch.multiprocessing as mp

def QuantRecursiveShiftRecon(model: nn.Module, layerEnabled, qnn, test_loader, prv_name="", ret=dict(), act=False, **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        # print(curName)
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            if curName in layerEnabled:
                ret[curName] = run_ShiftReconFused(model, curName, module, qnn, test_loader, act, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True:
                continue
            if curName in layerEnabled:
                ret[curName] = run_ShiftReconFused(model, curName, module, qnn, test_loader, act, **kwargs)
            else:
                QuantRecursiveShiftRecon(module, layerEnabled, qnn, test_loader, curName, ret, act, **kwargs)
        else:
            QuantRecursiveShiftRecon(module, layerEnabled, qnn, test_loader, curName, ret, act, **kwargs)

def restore_ShiftedChannelQuant(model, layers, prv_name='', path='./temp/greedyLoss'):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                if curName in layers:
                    with open(f'{path}/{curName}.pkl', 'rb') as f:
                        module.selection_init()
                        module.weight_quantizer.shiftedScale = pickle.load(f)
                        module.restore_selection()
                        print(f'Restored {name} shiftedScale')
        else:
            restore_ShiftedChannelQuant(module, layers, curName, path)

#Channel First
def run_ShiftReconFused(model, curName, module, qnn, test_loader, act=False, **kwargs):
    iters = kwargs['iters']
    rec_loss = []
    
    if isinstance(module, QuantModule):
        rec_loss_s = layer_recon_fused_shiftedScale(module, iters, (0.01, kwargs['lmda']), qnn, test_loader, act=act)
    elif isinstance(module, QuantBasicBlock):
        rec_loss_s = block_recon_fused_shiftedScale(module, iters, (0.01, kwargs['lmda']), qnn, test_loader, act=act)
    else:
        raise ValueError('Not supported reconstruction module type: {}'.format(type(module)))
    rec_loss += [rec_loss_s]
    return rec_loss

# #Channel First
# def run_ShiftRecon(model, curName, module, qnn, test_loader, act=False, **kwargs):
#     iters_for_shift = kwargs['iters']
#     iters_for_round = kwargs['iters']*2
#     useShiftedScale = not kwargs['bypassChannelShift']
#     if kwargs['bypassChannelShift']:
#         iters_for_shift = 0
        
#     rec_loss = []
    
#     if not act:#Weight Quantization
#         if isinstance(module, QuantModule):
#             #Channel Shifting
#             rec_loss_s = layer_recon_shiftedScale(module, iters_for_shift, kwargs['lmda'], qnn, test_loader, act=False)
#             #Rounding
#             rec_loss_r = layer_recon_shiftedScale(module, iters_for_round, 0.01, qnn, test_loader, act, adaround=True)
#         elif isinstance(module, QuantBasicBlock):
#             #Channel Shifting
#             rec_loss_s = block_recon_shiftedScale(module, iters_for_shift, kwargs['lmda'], qnn, test_loader, act=False)
#             #Rounding
#             rec_loss_r = block_recon_shiftedScale(module, iters_for_round, 0.01, qnn, test_loader, act, adaround=True)
#         else:
#             raise ValueError('Not supported reconstruction module type: {}'.format(type(module)))
#     else:
#         raise NotImplementedError('')
#     rec_loss += [rec_loss_s, rec_loss_r]
#     return rec_loss

# def run_ShiftRecon_round_first(model, curName, module, qnn, test_loader, act=False, **kwargs):
#     iters_for_shift = kwargs['iters']
#     iters_for_round = kwargs['iters']*2
#     useShiftedScale = not kwargs['bypassChannelShift']
#     if kwargs['bypassChannelShift']:
#         iters_for_shift = 0
        
#     rec_loss = []
    
#     if not act:#Weight Quantization
#         if isinstance(module, QuantModule):
#             #Rounding
#             rec_loss_r = layer_recon_shiftedScale(module, iters_for_round, 0.01, qnn, test_loader, act, adaround=True)
#             #Channel Shifting
#             rec_loss_s = layer_recon_shiftedScale(module, iters_for_shift, kwargs['lmda'], qnn, test_loader, act=False)
#         elif isinstance(module, QuantBasicBlock):
#             #Rounding
#             rec_loss_r = block_recon_shiftedScale(module, iters_for_round, 0.01, qnn, test_loader, act, adaround=True)
#             #Channel Shifting
#             rec_loss_s = block_recon_shiftedScale(module, iters_for_shift, kwargs['lmda'], qnn, test_loader, act=False)
#         else:
#             raise ValueError('Not supported reconstruction module type: {}'.format(type(module)))
#     else:
#         raise NotImplementedError('')
#     rec_loss += [rec_loss_r, rec_loss_s]
#     return rec_loss
                
def toggle_hardTarget(model, curName, layer, **kwargs):
    layer.weight_quantizer.hard_targets = not layer.weight_quantizer.hard_targets
    
def channelShift_wLoss(test_loader, train_loader, cali_data, botInfo, subArgs, args):
    kwargs = dict(
        cali_data=cali_data, 
        iters=args.iters_w, 
        weight=args.weight, 
        asym=True,
        b_range=(args.b_start, args.b_end), 
        warmup=args.warmup, 
        act_quant=False, 
        opt_mode='mse', 
        eval=True,
        returnLoss=True,
        batch_size=args.batch_size,
        bypassChannelShift=args.bypassChannelShift,
    )
    skipShiftLayer = [
        # '.model.layer4.1',
        # '.model.fc',
    ]
    k_lmda = subArgs['lmda']
    k_iters = subArgs['iters']
    #My options
    kwargs['lmda'] = k_lmda
    kwargs['iters'] = k_iters
    kwargs['shiftTarget'] = subArgs['shiftTarget']
    kwargs['skipShiftLayer'] = skipShiftLayer

    layerEnabled = [
        '.model.layer1.0',
        '.model.layer1.1',
        '.model.layer2.0',
        '.model.layer2.1',
        '.model.layer3.0',
        '.model.layer3.1',
        '.model.layer4.0',
        '.model.layer4.1',
        '.model.fc',
    ]
    
    qnn = build_qnn(args, test_loader)
    bot = telegram.Bot(token=botInfo['token']) if botInfo is not None else None
    msg = f'Starting with {k_lmda} & {k_iters} & {subArgs["shiftTarget"]} & WA{args.n_bits_w}/{args.n_bits_a}'
    if bot is not None:
        bot.sendMessage(chat_id=botInfo['id'], text=msg)
    #build_ShiftedChannelQuant(qnn, layerEnabled, '', delta=1100, **kwargs) #set All Layers
    build_ShiftedChannelQuant(qnn, layerEnabled, '', **kwargs) #set All Layers
    qnn.set_quant_state(False, False)# Default Setting
    
    accuracys = []
    loss_dic = dict()
    for layer in layerEnabled:
        print("Reconstructing Layer[Weight]: ", layer)
        kwargs['iters'] = k_iters if layer not in ['.model.layer4.1'] else k_iters*3//2
        #Before Quant
        #Cache input features(with quant state)
        set_cache_state(qnn, [layer], prv_name='', state='if')
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))

        #Cache output features(with quant state)
        qnn.store_quantization_state()
        qnn.set_quant_state(False, False)# For Output Feature store
        set_cache_state(qnn, [layer], prv_name='', state='of')
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))
        qnn.restore_quantization_state()
        set_cache_state(qnn, [layer], prv_name='', state='none')
        
        #Run Quantization optimization(Main Function)
        set_quant_state_block(qnn, [layer], '', True)
        QuantRecursiveShiftRecon(qnn, [layer], qnn, test_loader, '', loss_dic, **kwargs)#--> Main Function
        qnn.clear_cached_features()
        
        ####---- Test Area ---- Begin ####
        qnn.store_quantization_state()
        qnn.set_quant_state(True, False)# For Accuracy test, All Quantized mode
        
        # #Soft Result
        # QuantRecursiveToggle_hardTarget(qnn, [layer], '', **kwargs)
        # accuracys += [validate_model(test_loader, qnn, print_result=True).cpu().numpy()]
        # print(f'accuracy of qnn_soft{layer:28s}    : {accuracys[-1]:.3f}')
        
        # #Hard Result
        # QuantRecursiveToggle_hardTarget(qnn, [layer], '', **kwargs)
        if k_iters>0 and layer in ['.model.layer4.1', '.model.fc']:
            accuracys += [validate_model(test_loader, qnn, print_result=True).cpu().numpy()]
            print(f'accuracy of qnn_hard{layer:28s}    : {accuracys[-1]:.3f}')
        
        qnn.restore_quantization_state()
        ####---- Test Area ---- End   ####
    
    shiftTarget = subArgs['shiftTarget']
    
    skipped = '_skip' if len(skipShiftLayer) > 0 else ''
    if bot is not None:
        bot.sendMessage(chat_id=botInfo['id'], text=str(np.array(accuracys)))
    with open(f'./temp/loss_{k_lmda:.3e}_itr{k_iters}_{str(shiftTarget)}{skipped}.pkl', 'wb') as f:
        pickle.dump(loss_dic, f)
    return qnn

def channelShift_wLoss_feature(qnn, test_loader, cali_data, botInfo, subArgs, args):
    
    kwargs = dict(
        cali_data=cali_data, 
        iters=args.iters_w, 
        weight=args.weight, 
        asym=True,
        b_range=(args.b_start, args.b_end), 
        warmup=args.warmup, 
        act_quant=False, 
        opt_mode='mse', 
        eval=True,
        returnLoss=True,
        batch_size=args.batch_size,
        bypassChannelShift=args.bypassChannelShift,
    )
    skipShiftLayer = [
        # '.model.layer4.1',
        # '.model.fc',
    ]
    k_lmda = subArgs['lmda']
    k_iters = subArgs['iters']
    #My options
    kwargs['lmda'] = k_lmda
    kwargs['iters'] = k_iters
    kwargs['shiftTarget'] = subArgs['shiftTarget']
    kwargs['skipShiftLayer'] = skipShiftLayer
    # print(f'accuracy of qnn   : {validate_model(test_loader, qnn, print_result=True):.3f}')
    layerEnabled = [
        '.model.layer1.0',
        '.model.layer1.1',
        '.model.layer2.0',
        '.model.layer2.1',
        '.model.layer3.0',
        '.model.layer3.1',
        '.model.layer4.0',
        '.model.layer4.1',
        # '.model.fc',
    ]
    accuracys = []
    loss_dic = dict()
    for layer in layerEnabled:
        print("Reconstructing Layer[Feature]: ", layer)
        
        #Cache input features(with quant state)
        set_cache_state(qnn, [layer], prv_name='', state='if')
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))

        #Cache output features(with quant state)
        qnn.store_quantization_state()
        qnn.set_quant_state(False, False)# For Output Feature store
        set_cache_state(qnn, [layer], prv_name='', state='of')
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))
        qnn.restore_quantization_state()
        set_cache_state(qnn, [layer], prv_name='', state='none')
        
        #Run Quantization optimization(Main Function)
        set_quant_state_block(qnn, [layer], '', True, act=True)
        QuantRecursiveShiftRecon(qnn, [layer], qnn, test_loader, '', loss_dic, act=True, **kwargs)
        qnn.clear_cached_features()
        
        ####---- Test Area ---- Begin ####
        accuracys += [validate_model(test_loader, qnn).cpu().numpy()]
        print(f'accuracy of qnn_hard{layer:28s}    : {accuracys[-1]:.3f}')
        
        
if __name__ == '__main__':
    #Ready For Simulation
    args = loadArgments()
    device = torch.device(args.run_device if torch.cuda.is_available() else 'cpu')
    seed_all(args.seed)
    
    if args.dataset == 'cifar10':
        train_loader, test_loader = build_cifar10_data(batch_size=args.batch_size, workers=args.workers,
                                                        data_path=args.data_path)
    elif args.dataset == 'imagenet':
        train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers,
                                                        data_path=args.data_path)    
    # cnn = resnet18_imagenet()
    # state_dict = torch.load('./pretrained/Pytorch_imagenet/resnet18_imagenet.pth.tar', map_location='cpu')
    # cnn.load_state_dict(state_dict)
    cali_data  = get_train_samples(train_loader, num_samples=args.num_samples)
    if args.make_checkpoint:
        init_delta_zero(args, cali_data, test_loader)
        print("Making checkpoint data done")
        exit(1)
    #Telegram Bot setting.
    botInfo = {'token':'5820626937:AAHHsvT__T7xkCiLujwi799CyMoWtwNkbTM', 'id':'5955354823'} if args.msg_bot_enable else None

    # #Add these to common parameters
    itr = 80000
    # itr = 100
    lmda = 2
    shiftTargets = [[100]]
    # shiftTarget = [3/4, 3/2]
    # shiftTargets = [[2/2, 1/2, 3/2],[4/4, 3/4, 5/4],[8/8, 7/8, 9/8]]
    # shiftTargets = [[2/2, 1/2, 3/2]]
    # shiftTargets += [[4/4, 3/4, 5/4]]
    # shiftTarget = [8/8, 4/8, 2/8, 1/8]
    # shiftTarget = [1.5, 1.25, 1.0]
    # shiftTargets = [[2/2, 1/2], [2/2, 3/2, 1/2], [4/4, 5/4, 3/4]]
    # shiftTargets = [[2/2, 3/2, 1/2], [4/4, 5/4, 3/4], [4/4, 5/4, 3/4]]
    # # # # for itr in [1000, 4000, 8000]:
    # for lmda in range(2, 4):
    lmdas = [(i+1)/3 for i in range(6)]
    for lmda in lmdas:
        for shiftTarget in shiftTargets:
            kwargs = dict()
            kwargs['iters'] = itr
            kwargs['lmda'] = (10)**(-lmda)
            kwargs['shiftTarget'] = shiftTarget
            qnn = channelShift_wLoss(test_loader, train_loader, cali_data, botInfo, kwargs, args)
        # params_dict = {}
        # for name, param in qnn.named_parameters():    # for name, param in qnn.named_parameters():
        #     print("torch saving ", name, param.data.shape)
        #     params_dict[name] = param.data
        # torch.save(params_dict, f'./qnn_with_weight.pth')
        # param_st = torch.load('qnn_weight_quant.pth')
        # qnn.load_state_dict(param_st)
        # kwargs['iters'] = 100
        # channelShift_wLoss_feature(qnn, test_loader, cali_data, botInfo, kwargs, args)
    