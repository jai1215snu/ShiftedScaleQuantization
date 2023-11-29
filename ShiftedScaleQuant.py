#Training Set
import torch.nn as nn
import pickle
from data.cifar10 import build_cifar10_data
from data.imagenet import build_imagenet_data
from quant.quant_layer import QuantModule
from quant.quant_block import BaseQuantBlock, QuantBasicBlock
from myScaledMethods import *
from quant.layer_recon_shiftedScale import *
from quant.layer_recon_fused_shiftedScale import *

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
    
def channelShift_wMSE(test_loader, train_loader, cali_data, subArgs, args):
    kwargs = dict(
        cali_data=cali_data, 
        iters=args.iters_w, 
        weight=args.weight, 
        asym=True,
        b_range=(args.b_start, args.b_end), 
        warmup=args.warmup, 
        act_quant=False, 
        opt_mode=args.shift_quant_mode, 
        eval=True,
        returnLoss=True,
        batch_size=args.batch_size,
        bypassChannelShift=args.bypassChannelShift,
        level=args.mse_level,
        threshold=args.mse_threshold,
    )
    qnn = build_qnn(args, test_loader)
    qnn.set_quant_state(True, False)
    _ = qnn(cali_data[:64].to(device))
    if not args.skip_test:
        print(f'accuracy of qnn (with cal.)  : {validate_model(test_loader, qnn, bit=args.n_bits_w):.3f}')
        
    layerDisabled = [
        # '.model.layer1.0',
        # '.model.layer1.1',
        # '.model.layer2.0',
        # '.model.layer2.1',
        # '.model.layer3.0',
        # '.model.layer3.1',
        # '.model.layer4.0',
        # '.model.layer4.1',
        '.model.fc',
    ]
    kwargs['shiftTarget'] = subArgs['shiftTarget']
    def build_ShiftedChannelQuantMSE(model: nn.Module, layerDisabled, prv_name="", delta=1.0, **kwargs):
        for name, module in model.named_children():
            curName = prv_name+'.'+name
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    continue
                if curName not in layerDisabled:
                    # print("Build Layer : ", curName)
                    build_ShiftedChannelQuantMSELayer(model, curName, module, delta, **kwargs)
            elif isinstance(module, QuantBasicBlock):
                if module.ignore_reconstruction is True:
                    continue
                if curName in layerDisabled:
                    # print("Build Block : ", curName)
                    build_ShiftedChannelQuantMSEBlock(model, curName, module, delta, **kwargs)
                else:
                    build_ShiftedChannelQuantMSE(module, layerDisabled, curName, delta, **kwargs)
            else:
                build_ShiftedChannelQuantMSE(module, layerDisabled, curName, delta, **kwargs)
    prv_name = ''
    build_ShiftedChannelQuantMSE(qnn, layerDisabled, '', delta=1.0, **kwargs)
    
    accuracy, loss = validate_with_loss(test_loader, qnn, print_result=False, bit=args.n_bits_w)
    accuracy = accuracy.item()
    loss = loss.item()
    print(f'accuracy of qnn_hard   : {accuracy:.3f}, {loss:.3e}')
    
    
    
    return qnn, accuracy, loss
    
def channelShift_wLoss(test_loader, train_loader, cali_data, subArgs, args):
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
        # '.model.layer1.1',
        # '.model.layer2.0',
        # '.model.layer2.1',
        # '.model.layer3.0',
        # '.model.layer3.1',
        # '.model.layer4.0',
        # '.model.layer4.1',
        # '.model.fc',
    ]
    
    qnn = build_qnn(args, test_loader)
    
    #Run Calibration for weight quant.
    print(f"Calibration for weight quant. : {args.w_scale_method} scale mode")
    qnn.set_quant_state(True, False)
    _ = qnn(cali_data[:64].to(device))
    if not args.skip_test:
        print(f'accuracy of qnn (with cal.)  : {validate_model(test_loader, qnn):.3f}')
        
    #build_ShiftedChannelQuant(qnn, layerEnabled, '', delta=1100, **kwargs) #set All Layers
    build_ShiftedChannelQuant(qnn, layerEnabled, '', **kwargs) #set All Layers
    qnn.set_quant_state(False, False)# Default Setting
    
    accuracys = []
    loss_dic = dict()
    for layer in layerEnabled:
        print("Reconstructing Layer[Weight]: ", layer)
        kwargs['iters'] = k_iters
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
        # if k_iters>0 and layer in ['.model.layer4.1', '.model.fc']:
        if k_iters>0:
            accuracys += [validate_model(test_loader, qnn, print_result=True).cpu().numpy().item()]
            print(f'accuracy of qnn_hard{layer:28s}    : {accuracys[-1]:.3f}')
        
        qnn.restore_quantization_state()
        ####---- Test Area ---- End   ####
    
    # shiftTarget = subArgs['shiftTarget']
    
    # skipped = '_skip' if len(skipShiftLayer) > 0 else ''
    # with open(f'./temp/loss_{k_lmda:.3e}_itr{k_iters}_{str(shiftTarget)}{skipped}.pkl', 'wb') as f:
    #     pickle.dump(loss_dic, f)
    return qnn, accuracys

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
    
    #TODO: For Debug Code
    args.test = True
    args.n_bits = 4
    args.a_bits = 4
    args.w_scale_method = 'max'
    args.run_device ='cuda:1'
    
    device = torch.device(args.run_device if torch.cuda.is_available() else 'cpu')
    seed_all(args.seed)
    
    if args.dataset == 'cifar10':    train_loader, test_loader = build_cifar10_data (batch_size=args.batch_size, workers=args.workers, data_path=args.data_path)
    elif args.dataset == 'imagenet': train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers, data_path=args.data_path)
    # cnn = resnet18_imagenet()
    # state_dict = torch.load('./pretrained/Pytorch_imagenet/resnet18_imagenet.pth.tar', map_location='cpu')
    # cnn.load_state_dict(state_dict)
    cali_data  = get_train_samples(train_loader, num_samples=args.num_samples)
    if args.make_checkpoint:
        init_delta_zero(args, cali_data, test_loader)
        print("Making checkpoint data done")
        exit(1)
    #Normal Quant
    #Random Quant(without calibration)
    # #Add these to common parameters
    
    
    
    itr = 625
    lmdas = [1]
    shiftTargets = [[1.0-0.03125, 1.0+0.03125, 1.0]]
    for lmda in lmdas:
        for shiftTarget in shiftTargets:
            kwargs = dict()
            kwargs['iters'] = itr
            kwargs['lmda'] = (10)**(-lmda)
            kwargs['shiftTarget'] = shiftTarget
            if args.test:
                qnn, acc, loss = channelShift_wMSE(test_loader, train_loader, cali_data, kwargs, args)
            else:
                qnn, acc = channelShift_wLoss(test_loader, train_loader, cali_data, kwargs, args)
            # exit(1)
            with open(f'{args.run_device}.log', 'a') as fout:
                now = datetime.now()
                date_string = now.strftime("[%m-%d %H:%M:%S]")
                config = f'{lmda}, {shiftTarget}'
                fout.write(f'{date_string}:{config}: {acc} : {loss:3e} #{args}\n')
            exit(1)
            