#Training Set
import torch.nn as nn
from data.cifar10 import build_cifar10_data
from quant.quant_layer import QuantModule
from quant.quant_block import BaseQuantBlock, QuantBasicBlock
import pickle
from myScaledMethods import *
from quant.layer_recon_shiftedScale import *

#multi gpus
# import linklink as link
# from linklink.dist_helper import dist_init, allaverage
# import torch.distributed as dist
# import torch.multiprocessing as mp

def QuantRecursiveShiftRecon(model: nn.Module, layerEnabled, qnn, test_loader, prv_name="", act=False, **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        # print(curName)
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            if curName in layerEnabled:
                run_ShiftRecon(model, curName, module, qnn, test_loader, act, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True:
                continue
            if curName in layerEnabled:
                run_ShiftRecon(model, curName, module, qnn, test_loader, act, **kwargs)
            else:
                QuantRecursiveShiftRecon(module, layerEnabled, qnn, test_loader, curName, act, **kwargs)
        else:
            QuantRecursiveShiftRecon(module, layerEnabled, qnn, test_loader, curName, act, **kwargs)

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
        
def run_ShiftRecon(model, curName, module, qnn, test_loader, act=False, **kwargs):
    if isinstance(module, QuantModule):
        layer_recon_shiftedScale(module, kwargs['iters'], kwargs['lmda'], qnn, test_loader, act)
    elif isinstance(module, QuantBasicBlock):
        block_recon_shiftedScale(module, kwargs['iters'], kwargs['lmda'], qnn, test_loader, act)
    else:
        raise ValueError('Not supported reconstruction module type: {}'.format(type(module)))
                
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
    )
    k_lmda = subArgs['lmda']
    k_iters = subArgs['iters']
    #My options
    kwargs['lmda'] = k_lmda
    kwargs['iters'] = k_iters
    kwargs['shiftTarget'] = subArgs['shiftTarget']

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
    bot = telegram.Bot(token=botInfo['token'])
    msg = f'Starting with {k_lmda} & {k_iters} & {subArgs["shiftTarget"]}'
    bot.sendMessage(chat_id=botInfo['id'], text=msg)
    build_ShiftedChannelQuant(qnn, layerEnabled, '', delta=1100, **kwargs) #set All Layers
    qnn.set_quant_state(False, False)# Default Setting
    
    # print(cali_data.shape)
    # print(caliT_data.shape)
    # exit(1)
    
    accuracys = []
    for layer in layerEnabled:
        print("Reconstructing Layer: ", layer)
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
        QuantRecursiveShiftRecon(qnn, [layer], qnn, test_loader, '', **kwargs)
        qnn.clear_cached_features()
        
        ####---- Test Area ---- Begin ####
        qnn.store_quantization_state()
        qnn.set_quant_state(True, False)# For Accuracy test, All Quantized mode
        
        #Soft Result
        QuantRecursiveToggle_hardTarget(qnn, [layer], '', **kwargs)
        print(f'accuracy of qnn_soft{layer:28s}    : {validate_model(test_loader, qnn):.3f}')
        
        #Hard Result
        QuantRecursiveToggle_hardTarget(qnn, [layer], '', **kwargs)
        accuracys += [validate_model(test_loader, qnn).cpu().numpy()]
        print(f'accuracy of qnn_hard{layer:28s}    : {validate_model(test_loader, qnn):.3f}')
        
        qnn.restore_quantization_state()
        ####---- Test Area ---- End   ####
        
    bot.sendMessage(chat_id=botInfo['id'], text=str(np.array(accuracys)))
    with open(f'./temp/accuracy_{k_lmda:.3e}_itr{k_iters}.pkl', 'wb') as f:
        pickle.dump(np.array(accuracys), f)
    return qnn

def channelShift_wLoss_feature(qnn, test_loader, cali_data, botInfo, subArgs, args):
    print(f'accuracy of qnn   : {validate_model(test_loader, qnn):.3f}')
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
    for layer in layerEnabled:
        print("Reconstructing Layer: ", layer)

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
        QuantRecursiveShiftRecon(qnn, [layer], qnn, test_loader, '', act=True, **kwargs)
        qnn.clear_cached_features()
        
        ####---- Test Area ---- Begin ####
        qnn.store_quantization_state()
        qnn.set_quant_state(True, True)# For Accuracy test, All Quantized mode
        
        accuracys += [validate_model(test_loader, qnn).cpu().numpy()]
        print(f'accuracy of qnn_hard{layer:28s}    : {validate_model(test_loader, qnn):.3f}')
        
        qnn.restore_quantization_state()
        
        accuracys += [validate_model(test_loader, qnn).cpu().numpy()]
        print(f'accuracy of qnn_hard{layer:28s}    : {validate_model(test_loader, qnn):.3f}')
        
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Ready For Simulation
    args = loadArgments()
    seed_all(args.seed)
    args.num_samples = 512
    train_loader, test_loader = build_cifar10_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)
    cali_data  = get_train_samples(train_loader, num_samples=args.num_samples)
    
    #Telegram Bot setting.
    botInfo = {'token':'5820626937:AAHHsvT__T7xkCiLujwi799CyMoWtwNkbTM', 'id':'5955354823'}

    # channelRandomizeTest(test_loader, cali_data, args)
    # channelGreedyTest(test_loader, cali_data, args)
    # channelDistTest(test_loader, cali_data, args)
    # channelGreedyTest_wLoss(test_loader, cali_data, botInfo, args)
    # itr = 20000
    # lmda = 5
    itr = 4000
    lmda = 5
    shiftTarget = [1/2, 2/2]
    # # # for itr in [1000, 4000, 8000]:
    # for lmda in range(6, 10):
    kwargs = dict()
    kwargs['iters'] = itr
    kwargs['lmda'] = (10)**(-lmda)
    kwargs['shiftTarget'] = shiftTarget
    qnn = channelShift_wLoss(test_loader, train_loader, cali_data, botInfo, kwargs, args)
    channelShift_wLoss_feature(qnn, test_loader, cali_data, botInfo, kwargs, args)
    
    #Init Data
    # init_delta_zero(args, cali_data, test_loader)
    