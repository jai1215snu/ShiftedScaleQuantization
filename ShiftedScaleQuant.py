import torch.nn as nn
from data.cifar10 import build_cifar10_data
from quant.quant_layer import QuantModule
import telegram
import pickle
from myScaledMethods import *


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

def run_GreedyLoss(model, curName, layer, **kwargs):
    layer.run_GreedyLoss()
    # layer.run_GreedyLossSorted()
    #Dump Current shifted Scale.
    with open(f'./temp/greedyLoss/{curName}.pkl', 'wb') as f:
        pickle.dump(layer.weight_quantizer.shiftedScale, f)

def channelGreedyTest_wLoss(test_loader, cali_data, botInfo, args):
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
        shuffle_ratio=0, 
        qscale=1/2, #NOTE: Not Used. should be set on channelQuant
        returnLoss=True,
        batch_size=args.batch_size,
        test_loader=test_loader,
    )
    layerEnabled = [
        '.model.layer1.0.conv1',
        '.model.layer1.0.conv2',
        '.model.layer1.1.conv1',
        '.model.layer1.1.conv2',
        '.model.layer2.0.conv1',
        '.model.layer2.0.conv2',
        '.model.layer2.0.downsample.0',
        '.model.layer2.1.conv1',
    ]
    
    layerWeightGreedy = [
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
    print(f'accuracy of qnn    : {validate_model(test_loader, qnn):.3f}')
    bot = telegram.Bot(token=botInfo['token'])
    # bot.sendMessage(chat_id=botInfo['id'], text='Starting GreedyTest with Loss')
    build_ShiftedChannelQuant(qnn, layerEnabled+layerWeightGreedy, '', **kwargs) #only one layer
    qnn.set_quant_state(False, False)# Default Setting
    
    #NOTE: Temporal code for test
    for layer in layerEnabled+layerWeightGreedy:
        set_weight_quant(qnn, [layer], '', True)
        restore_ShiftedChannelQuant(qnn, [layer], '', './temp/greedyLoss/greedy_loss_L4')
        quant_state = store_quant_state(qnn)
        qnn.set_quant_state(True, False)# For Accuracy test
        result_message = f'accuracy of qnn{layer:28s}    : {validate_model(test_loader, qnn):.3f}'
        print(result_message)
        restore_quant_state(qnn, quant_state)
        
    exit(1)
    for layer in layerEnabled+layerWeightGreedy:
        #Cache input features(with quant state)
        set_cache_state(qnn, [layer], prv_name='', state='if')
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))
            
        #Cache output features(with quant state)
        quant_state = store_quant_state(qnn)
        qnn.set_quant_state(False, False)# For Accuracy test
        set_cache_state(qnn, [layer], prv_name='', state='of')
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))
        restore_quant_state(qnn, quant_state)
        set_cache_state(qnn, [layer], prv_name='', state='none')
        
        #Run Quantization optimization
        set_weight_quant(qnn, [layer], '', True)
        QuantRecursiveRun(qnn, run_GreedyLoss, [layer], '', **kwargs)
        qnn.clear_cached_features()
        quant_state = store_quant_state(qnn)
        qnn.set_quant_state(True, False)# For Accuracy test
        result_message = f'accuracy of qnn{layer:28s}    : {validate_model(test_loader, qnn):.3f}'
        print(result_message)
        bot.sendMessage(chat_id=botInfo['id'], text=result_message)
        restore_quant_state(qnn, quant_state)
        

def set_weight_quant(model, layers, prv_name='', state=False):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                if curName in layers:
                    module.use_weight_quant = state
        else:
            set_weight_quant(module, layers, curName, state)
        

def set_cache_state(model, layers, prv_name='', state='none'):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                if curName in layers:
                    module.cache_features = state
        else:
            set_cache_state(module, layers, curName, state)

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
    channelGreedyTest_wLoss(test_loader, cali_data, botInfo, args)
    