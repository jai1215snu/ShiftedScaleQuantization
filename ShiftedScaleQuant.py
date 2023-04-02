import torch.nn as nn
from common import *
from data.cifar10 import build_cifar10_data
from pretrained.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from quant.quant_model import QuantModel
from quant.quant_layer import QuantModule
from tqdm import tqdm
from quant.channelQuant import ChannelQuant
from myData_Utils import save_inp_oup_data
import pandas as pd
import pickle
import telegram


def build_qnn(args):
    cnn = resnet18(pretrained=True, device='cuda:0')
    cnn.cuda()
    cnn.eval()
    if not args.skip_test:
        print(f'accuracy of original : {validate_model(test_loader, cnn):.3f}')
    
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse', 'CWQ':True}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
    #Only Weight Quantization
    qnn.load_state_dict(torch.load(f'./checkPoint/QNN_CW_W{args.n_bits_w}_FP32.pth'))
    # qnn.set_quant_state(True, False)# For weight scale/zp initialization
    # _ = qnn(cali_data[:64].to(device))
    qnn.set_quant_init_state() #weight_quantizer.inited = True
    
    if not args.skip_test:
        print(f'accuracy of qnn    : {validate_model(test_loader, qnn):.3f}')
    return qnn

def build_ShiftedChannelQuant(model: nn.Module, layerEnabled, prv_name="", **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                if curName in layerEnabled:
                    build_ShiftedChannelQuantLayer(model, curName, module, **kwargs)
        else:
            build_ShiftedChannelQuant(module, layerEnabled, curName, **kwargs)


def QuantRecursiveRun(model: nn.Module, func, layerEnabled, prv_name="", result=dict(), **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                if curName in layerEnabled:
                    ret = func(model, curName, module, **kwargs)
                    result[curName] = ret
        else:
            QuantRecursiveRun(module, func, layerEnabled, curName, result, **kwargs)

def build_ShiftedChannelQuantLayer(model, curName, layer, **kwargs):
    shuffle_ratio = kwargs['shuffle_ratio']
    qscale = kwargs['qscale']
    layer.weight_quantizer = ChannelQuant(uaq=layer.weight_quantizer, weight_tensor=layer.org_weight.data, shuffle_ratio=shuffle_ratio, qscale=qscale)
    layer.use_weight_quant = True
    layer.cache_features   = 'none'
    
def run_layerRandomize(model, curName, layer, **kwargs):
    layer.weight_quantizer.run_layerRandomize()
    return layer.cal_quantLoss()

def run_layerGreedy(model, curName, layer, **kwargs):
    return layer.run_layerGreedy(kwargs['nc'])

def run_layerDist(model, curName, layer, **kwargs):
    return layer.run_layerDist(kwargs['nc'])

def dump_quant_feature(model, curName, layer, **kwargs):
    return layer.cached_out_quant_features

def dump_ori_feature(model, curName, layer, **kwargs):
    return layer.cached_out_features

def channelDistTest(test_loader, cali_data, args):
    # args.skip_test = False
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
        qscale=1/2,
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
    qnn = build_qnn(args)
    build_ShiftedChannelQuant(qnn, layerEnabled, '', **kwargs)
    #Run Calibration set for cached_inps, cached_outs
    # for i in range(len(cali_data)//args.batch_size):
    #     _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))
    qnn.disable_cache_features()
    print(f'Quantized accuracy before shifted scale: {validate_model(test_loader, qnn):.3f}')
    loss = dict()
    totalResult = []
    bestResult = 0.0
    t = tqdm(range(512), desc=f'desc', dynamic_ncols=True)
    for i in t:
        kwargs['nc'] = i
        QuantRecursiveRun(qnn, run_layerDist, layerEnabled, '', loss, **kwargs)
        testResult = validate_model(test_loader, qnn).data.cpu().item()
        totalResult += [testResult]
        if testResult > bestResult:
            bestResult = testResult
        t.set_description(f"best {bestResult:.3f} cur {testResult:.3f}")
    
    with open(f'./temp/DistResult.pkl', 'wb') as f:
        pickle.dump(totalResult, f)
    

def channelGreedyTest(test_loader, cali_data, args):
    # args.skip_test = False
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
    qnn = build_qnn(args)
    build_ShiftedChannelQuant(qnn, layerEnabled, '', **kwargs)
    #Run Calibration set for cached_inps, cached_outs
    # for i in range(len(cali_data)//args.batch_size):
    #     _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))
    qnn.disable_cache_features()
    print(f'Quantized accuracy before shifted scale: {validate_model(test_loader, qnn):.3f}')
    testLayer = layerEnabled[-1]
    loss = dict()
    totalResult = []
    bestResult = 0.0
    t = tqdm(range(512), desc=f'desc', dynamic_ncols=True)
    for i in t:
        kwargs['nc'] = i
        QuantRecursiveRun(qnn, run_layerGreedy, layerEnabled, '', loss, **kwargs)
        testResult = validate_model(test_loader, qnn).data.cpu().item()
        totalResult += [testResult]
        if testResult > bestResult:
            bestResult = testResult
        t.set_description(f"best {bestResult:.3f} cur {testResult:.3f}")
    
    with open(f'./temp/greedyResult.pkl', 'wb') as f:
        pickle.dump(totalResult, f)
    
        

def channelRandomizeTest(test_loader, cali_data, args):
    # ratio = [0.01,0.02,0.04,0.08,0.1,0.2,0.3,0.4,0.5,0.75,1.0]
    ratio = [0.01]
    
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
        qscale=1/2,
        returnLoss=True,
        batch_size=args.batch_size
    )
    
    # if not args.skip_test:
    
    for shuffle_ratio in ratio:
        bestResult = 0.0
        bestLoss   = 1e10
        testResult = []
        lossResult = []
        t = tqdm(range(30), desc=f'random value={shuffle_ratio}', dynamic_ncols=True)
        qnn = build_qnn(args)
        kwargs['shuffle_ratio'] = shuffle_ratio
        kwargs['base_model'] = qnn
        layerEnabled = [
            # '.model.layer1.0.conv1', 
            '.model.layer4.1.conv2'
        ]
        build_ShiftedChannelQuant(qnn, layerEnabled, '', **kwargs)
        #Run Calibration set for cached_inps, cached_outs
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))

        qnn.disable_cache_features()
        print(f'Quantized accuracy before shifted scale: {validate_model(test_loader, qnn):.3f}')
        
        testLayer = '.model.layer1.0.conv1' #'.model.layer4.1.conv2'
        
        #TODO: Temp Codes
        dumpData = dict()
        QuantRecursiveRun(qnn, dump_ori_feature, layerEnabled, '', dumpData, **kwargs)
        with open(f'./temp/quant_oriout.pkl', 'wb') as f:
            pickle.dump(dumpData[testLayer][0].cpu().numpy(), f)
        
        for i in t:
            loss = dict()
            QuantRecursiveRun(qnn, run_layerRandomize, layerEnabled, '', loss, **kwargs)
            testResult.append(validate_model(test_loader, qnn, print_result=False).data.cpu().item())
            lossResult.append(loss[testLayer])
            if testResult[-1] > bestResult:
                bestResult = testResult[-1]
                #TODO: Temp Codes
                dumpData = dict()
                QuantRecursiveRun(qnn, dump_quant_feature, layerEnabled, '', dumpData, **kwargs)
                with open(f'./temp/quant_maxAcc.pkl', 'wb') as f:
                    pickle.dump(dumpData[testLayer][0].cpu().numpy(), f)

            if lossResult[-1] < bestLoss:
                bestLoss = lossResult[-1]
                #TODO: Temp Codes
                dumpData = dict()
                QuantRecursiveRun(qnn, dump_quant_feature, layerEnabled, '', dumpData, **kwargs)
                with open(f'./temp/quant_minLoss.pkl', 'wb') as f:
                    pickle.dump(dumpData[testLayer][0].cpu().numpy(), f)
            t.set_description(f"best {bestResult:.3f} cur {testResult[-1]:.3f}")
        
        import matplotlib.pyplot as plt
        df = pd.DataFrame({'Loss': lossResult, 'Accuracy': testResult})
        plt.scatter(df['Loss'], df['Accuracy'])
        plt.xlabel('Loss')
        plt.ylabel('Accuracy')
        plt.title('Loss vs Accuracy')
        plt.savefig('./temp/scatter_plot.jpg', dpi=300, figsize=(8, 6))
                
        print("Best Accuracy : ", max(testResult))
        loss_min_index = lossResult.index(min(lossResult))
        print("Best Accuracy with loss: ", testResult[loss_min_index])
        
        # with open(f'./results/channelRandomL0_only_1.2_fixed.{shuffle_ratio}.pkl', 'wb') as f:
        #     pickle.dump(testResult, f)

def run_GreedyLoss(model, curName, layer, **kwargs):
    layer.run_GreedyLoss()
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
    qnn = build_qnn(args)
    print(f'accuracy of qnn    : {validate_model(test_loader, qnn):.3f}')
    bot = telegram.Bot(token=botInfo['token'])
    # bot.sendMessage(chat_id=botInfo['id'], text='Starting GreedyTest with Loss')
    build_ShiftedChannelQuant(qnn, layerEnabled, '', **kwargs) #only one layer
    qnn.set_quant_state(False, False)# Default Setting
    
    for layer in layerEnabled:
        # build_ShiftedChannelQuant(qnn, [layer], '', **kwargs) #only one layer
        
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


def store_quant_state(model):
    qState = []
    for m in model.modules():
        if isinstance(m, (QuantModule)):
            qState += [m.use_weight_quant]
    return qState

def restore_quant_state(model, qState):
    idx = 0
    for m in model.modules():
        if isinstance(m, (QuantModule)):
            m.use_weight_quant = qState[idx]
            idx += 1

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
    