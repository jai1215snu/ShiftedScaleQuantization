import torch.nn as nn
from data.cifar10 import build_cifar10_data
from quant.quant_layer import QuantModule
import pickle
from myScaledMethods import *
from quant.layer_recon_shiftedScale import *

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
        
def run_ShiftRecon(model, curName, layer, **kwargs):
    layer_recon_shiftedScale(layer, kwargs['iters'], kwargs['lmda'], model, kwargs['test_loader'])
                
def toggle_hardTarget(model, curName, layer, **kwargs):
    layer.weight_quantizer.hard_targets = not layer.weight_quantizer.hard_targets
    
def channelShift_wLoss(test_loader, cali_data, botInfo, subArgs, args):
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
        test_loader=test_loader,
    )
    k_lmda = subArgs['lmda']
    k_iters = subArgs['iters']
    #My options
    kwargs['lmda'] = k_lmda
    kwargs['iters'] = k_iters
    layerEnabled = [
        '.model.layer1.0.conv1',
        '.model.layer1.0.conv2',
        '.model.layer1.1.conv1',
        '.model.layer1.1.conv2',
        '.model.layer2.0.conv1',
        '.model.layer2.0.conv2',
        '.model.layer2.0.downsample.0',
        '.model.layer2.1.conv1',
        '.model.layer2.1.conv2',
        '.model.layer3.0.conv1',
        '.model.layer3.0.conv2',
        '.model.layer3.0.downsample.0',
        '.model.layer3.1.conv1',
        '.model.layer3.1.conv2',
        '.model.layer4.0.conv1',
        '.model.layer4.0.conv2',
        '.model.layer4.0.downsample.0',
        '.model.layer4.1.conv1',
        '.model.layer4.1.conv2',
    ]
    qnn = build_qnn(args, test_loader)
    bot = telegram.Bot(token=botInfo['token'])
    # bot.sendMessage(chat_id=botInfo['id'], text=f'Starting with {k_lmda} & {k_iters}')
    build_ShiftedChannelQuant(qnn, layerEnabled, '', **kwargs) #only one layer
    qnn.set_quant_state(False, False)# Default Setting
    
    
    accuracys = []
    for layer in layerEnabled:
        #Cache input features(with quant state)
        set_cache_state(qnn, [layer], prv_name='', state='if')
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))
            
        #Cache output features(with quant state)
        qnn.store_quantization_state()
        qnn.set_quant_state(False, False)# For Output Feature store
        print("Layer: ", layer)
        set_cache_state(qnn, [layer], prv_name='', state='of')
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))
        qnn.restore_quantization_state()
        set_cache_state(qnn, [layer], prv_name='', state='none')
        
        #Run Quantization optimization
        set_weight_quant(qnn, [layer], '', True)

        QuantRecursiveRun(qnn, run_ShiftRecon, [layer], '', **kwargs) #Main Function
        qnn.clear_cached_features()
        
        ####---- Test Area ---- Begin ####
        qnn.store_quantization_state()
        
        qnn.set_quant_state(True, False)# For Accuracy test
        QuantRecursiveRun(qnn, toggle_hardTarget, [layer], '', **kwargs)
        result_message = f'accuracy of qnn_soft{layer:28s}    : {validate_model(test_loader, qnn):.3f}'
        print(result_message)
        QuantRecursiveRun(qnn, toggle_hardTarget, [layer], '', **kwargs)
        
        accuracys += [validate_model(test_loader, qnn).cpu().numpy()]
        result_message = f'accuracy of qnn_hard{layer:28s}    : {validate_model(test_loader, qnn):.3f}'
        print(result_message)
        qnn.restore_quantization_state()
        ####---- Test Area ---- End   ####

    bot.sendMessage(chat_id=botInfo['id'], text=str(np.array(accuracys)))
    with open(f'./temp/accuracy_{k_lmda}_itr{k_iters}.pkl', 'wb') as f:
        pickle.dump(np.array(accuracys), f)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Ready For Simulation
    args = loadArgments()
    seed_all(args.seed)
    train_loader, test_loader = build_cifar10_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)
    cali_data = get_train_samples(train_loader, num_samples=args.num_samples)
    
    #Telegram Bot setting.
    botInfo = {'token':'5820626937:AAHHsvT__T7xkCiLujwi799CyMoWtwNkbTM', 'id':'5955354823'}

    # channelRandomizeTest(test_loader, cali_data, args)
    # channelGreedyTest(test_loader, cali_data, args)
    # channelDistTest(test_loader, cali_data, args)
    # channelGreedyTest_wLoss(test_loader, cali_data, botInfo, args)
    itr = 1000
    lmda = 0.0001
    for itr in [1000, 4000, 8000]:
        for i in range(0, 10):
            kwargs = dict()
            kwargs['iters'] = itr
            kwargs['lmda'] = 100*((10)**(-i))
            channelShift_wLoss(test_loader, cali_data, botInfo, kwargs, args)
    
    #Init Data
    # init_delta_zero(args, cali_data, test_loader)
    