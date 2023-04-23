import pickle
import torch.nn as nn
from quant.quant_model import QuantModel
from quant.quant_layer import QuantModule, UniformAffineQuantizer
from quant.quant_block import BaseQuantBlock, QuantBasicBlock
import pandas as pd
from pretrained.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from models.resnet import resnet18 as resnet18_imagenet

from quant.channelQuant import ChannelQuant
from tqdm import tqdm
import telegram
from common import *

def build_ShiftedChannelQuantLayer(model, curName, layer, delta=1.0, **kwargs):
    shiftTarget = kwargs['shiftTarget']
    layer.weight_quantizer = ChannelQuant(delta, uaq=layer.weight_quantizer, weight_tensor=layer.org_weight.data, shiftTarget=shiftTarget)
    layer.use_weight_quant = True
    layer.cache_features   = 'none'
    
def build_ShiftedChannelQuantBlock(model, prv_name, block, delta=1.0, **kwargs):
    shiftTarget = kwargs['shiftTarget']
    for name, layer in model.named_children():
        curName = prv_name+'.'+name
        # print("block const name: ", curName, type(layer))
        if isinstance(layer, QuantModule):
            if isinstance(layer.weight_quantizer, UniformAffineQuantizer):
                layer.weight_quantizer = ChannelQuant(delta, uaq=layer.weight_quantizer, weight_tensor=layer.org_weight.data, shiftTarget=shiftTarget)
                layer.use_weight_quant = True
                layer.cache_features   = 'none'
        else:
            build_ShiftedChannelQuantBlock(layer, curName, block, **kwargs)
    
def build_ShiftedChannelQuant(model: nn.Module, layerEnabled, prv_name="", delta=1.0, **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        # print("Building shifted channel quant for: ", curName, type(module))
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            if curName in layerEnabled:
                build_ShiftedChannelQuantLayer(model, curName, module, delta, **kwargs)
        elif isinstance(module, QuantBasicBlock):
            if module.ignore_reconstruction is True:
                continue
            if curName in layerEnabled:
                build_ShiftedChannelQuantBlock(model, curName, module, delta, **kwargs)
            else:
                build_ShiftedChannelQuant(module, layerEnabled, curName, **kwargs)
        else:
            build_ShiftedChannelQuant(module, layerEnabled, curName, **kwargs)

# def store_quant_state(model):
#     qState = []
#     for m in model.modules():
#         if isinstance(m, (QuantModule)):
#             qState += [m.use_weight_quant]
#     return qState

# def restore_quant_state(model, qState):
#     idx = 0
#     for m in model.modules():
#         if isinstance(m, (QuantModule)):
#             m.use_weight_quant = qState[idx]
#             idx += 1
            
def set_quant_state_block(model, layers, prv_name='', state=False, act=False):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            if curName in layers:
                if act:
                    module.use_act_quant = state
                else:
                    module.use_weight_quant = state
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True:
                continue
            if curName in layers:
                module.set_quant_state_block(state, act)
        else:
            set_quant_state_block(module, layers, curName, state, act)
        

def set_cache_state(model, layers, prv_name='', state='none'):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if curName in layers:
            if module.ignore_reconstruction is True:
                continue
            module.cache_features = state
        elif isinstance(module, QuantModule):
            continue
        else:
            set_cache_state(module, layers, curName, state)     
            

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
    qnn = build_qnn(args, test_loader)
    bot = telegram.Bot(token=botInfo['token'])
    # bot.sendMessage(chat_id=botInfo['id'], text='Starting GreedyTest with Loss')
    build_ShiftedChannelQuant(qnn, layerEnabled, '', **kwargs) #only one layer
    qnn.set_quant_state(False, False)# Default Setting
    
    # #NOTE: Temporal code for test
    # for layer in layerEnabled:
    #     set_weight_quant(qnn, [layer], '', True)
    #     restore_ShiftedChannelQuant(qnn, [layer], '', './temp/greedyLoss/greedy_loss_L4')
    #     quant_state = store_quant_state(qnn)
    #     qnn.set_quant_state(True, False)# For Accuracy test
    #     result_message = f'accuracy of qnn{layer:28s}    : {validate_model(test_loader, qnn):.3f}'
    #     print(result_message)
    #     restore_quant_state(qnn, quant_state)
        
    for layer in layerEnabled:
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
        QuantRecursiveRun(qnn, run_GreedyLoss, [layer], '', **kwargs) #Main Function
        qnn.clear_cached_features()
        quant_state = store_quant_state(qnn)
        qnn.set_quant_state(True, False)# For Accuracy test
        result_message = f'accuracy of qnn{layer:28s}    : {validate_model(test_loader, qnn):.3f}'
        print(result_message)
        bot.sendMessage(chat_id=botInfo['id'], text=result_message)
        restore_quant_state(qnn, quant_state)

def run_GreedyLoss(model, curName, layer, **kwargs):
    layer.run_GreedyLoss()
    # layer.run_GreedyLossSorted()
    #Dump Current shifted Scale.
    with open(f'./temp/greedyLoss/{curName}.pkl', 'wb') as f:
        pickle.dump(layer.weight_quantizer.shiftedScale, f)
        
def init_delta_zero(args, cali_data, test_loader):
    if args.dataset == 'cifar10':
        cnn = resnet18(pretrained=True, device=args.run_device)
    elif args.dataset == 'imagenet':
        cnn = resnet18_imagenet()
        state_dict = torch.load(
            './pretrained/Pytorch_imagenet/resnet18_imagenet.pth.tar', map_location='cpu'
        )
        cnn.load_state_dict(state_dict)
        print("model Loaded",  './pretrained/Pytorch_imagenet/resnet18_imagenet.pth.tar', 'cpu')
        
    # cnn.cuda()
    cnn.to(args.run_device)
    cnn.eval()
    if not args.skip_test:
        print(f'accuracy of original : {validate_model(test_loader, cnn):.3f}')
    
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse', 'tune_delta_zero':True, 'leaf_param':True} #NOTE: tune_delta_zero is True
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'tune_delta_zero':True, 'leaf_param': args.act_quant}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    # qnn.cuda()
    qnn.to(args.run_device)
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
    # qnn.set_quant_state(True, False)# For weight scale/zp initialization
    # _ = qnn(cali_data[:64].to(args.run_device))
    
    # params_dict = {}
    # for name, param in qnn.named_parameters():
    #     print("torch saving ", name)
    #     params_dict[name] = param.data

    # # Save the parameters dictionary
    # torch.save(params_dict, f'./QNN_CW_W{args.n_bits_w}_FP32.pth')
    
    qnn.set_quant_state(True, True)# For weight scale/zp initialization
    _ = qnn(cali_data[:64].to(args.run_device))
    params_dict = {}
    for name, param in qnn.named_parameters():    # for name, param in qnn.named_parameters():
        print("torch saving ", name, param.data.shape)
        params_dict[name] = param.data
    
    if args.dataset == 'cifar10':
        prefix = 'CIFAR10'
    elif args.dataset == 'imagenet':
        prefix = 'IMAGENET'
    torch.save(params_dict, f'./{prefix}_QNN_CW_W{args.n_bits_w}_A{args.n_bits_a}.pth')
        
def build_qnn(args, test_loader):
    if args.dataset == 'cifar10':
        cnn = resnet18(pretrained=True, device=args.run_device)
    elif args.dataset == 'imagenet':
        cnn = resnet18_imagenet()
        state_dict = torch.load('./pretrained/Pytorch_imagenet/resnet18_imagenet.pth.tar', map_location='cpu')
        cnn.load_state_dict(state_dict)
    # cnn.cuda()
    cnn.to(args.run_device)
    cnn.eval()
    if not args.skip_test:
        print(f'accuracy of original : {validate_model(test_loader, cnn):.3f}')
    
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse', 'tune_delta_zero':False}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'tune_delta_zero':False, 'leaf_param': True}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    # qnn.cuda()
    qnn.to(args.run_device)
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
    #Only Weight Quantization
    # qnn.load_state_dict(torch.load(f'./checkPoint/QNN_CW_W{args.n_bits_w}_FP32.pth'))
    if args.dataset == 'cifar10':
        prefix = 'CIFAR10'
    elif args.dataset == 'imagenet':
        prefix = 'IMAGENET'
    qnn.load_state_dict(torch.load(f'./checkPoint/{prefix}_QNN_CW_W{args.n_bits_w}_A{args.n_bits_a}.pth'))
    qnn.set_quant_init_state() #set weight_quantizer.inited = True
    
    if not args.skip_test:
        print(f'accuracy of qnn    : {validate_model(test_loader, qnn):.3f}')
    return qnn

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


def QuantRecursiveToggle_hardTarget(model: nn.Module, layerEnabled, prv_name="", result=dict(), **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        # print(curName)
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            if curName in layerEnabled:
                module.weight_quantizer.hard_targets = not module.weight_quantizer.hard_targets
        elif isinstance(module, QuantBasicBlock):
            if module.ignore_reconstruction is True:
                continue
            if curName in layerEnabled:
                module.toggleHardTarget()
        else:
            QuantRecursiveToggle_hardTarget(module, layerEnabled, curName, result, **kwargs)

def QuantRecursiveRun(model: nn.Module, func, layerEnabled, prv_name="", result=dict(), **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        # print(curName)
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                if curName in layerEnabled:
                    ret = func(model, curName, module, **kwargs)
                    result[curName] = ret
        else:
            QuantRecursiveRun(module, func, layerEnabled, curName, result, **kwargs)

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

def channelRandomizeTest(test_loader, cali_data, args):
    # ratio = [0.01,0.02,0.04,0.08,0.1,0.2,0.3,0.4,0.5,0.75,1.0]
    ratio = [0.201]
    
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
        qscale=1/2,
        returnLoss=True,
        batch_size=args.batch_size
    )
    
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