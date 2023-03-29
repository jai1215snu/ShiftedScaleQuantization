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

def build_ShiftedChannelQuant(model: nn.Module, prv_name="", **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                if curName == '.model.layer4.1.conv2':
                # if curName == '.model.layer1.0.conv1':
                    build_ShiftedChannelQuantLayer(model, curName, module, **kwargs)
        else:
            build_ShiftedChannelQuant(module, curName, **kwargs)


def QuantRecursiveRun(model: nn.Module, func, prv_name="", result=dict(), **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                if curName == '.model.layer4.1.conv2':
                # if curName == '.model.layer1.0.conv1':
                    ret = func(model, curName, module, **kwargs)
                    result[curName] = ret
        else:
            QuantRecursiveRun(module, func, curName, result, **kwargs)

def build_ShiftedChannelQuantLayer(model, curName, layer, **kwargs):
    shuffle_ratio = kwargs['shuffle_ratio']
    qscale = kwargs['qscale']
    layer.weight_quantizer = ChannelQuant(uaq=layer.weight_quantizer, weight_tensor=layer.org_weight.data, shuffle_ratio=shuffle_ratio, qscale=qscale)
    layer.use_weight_quant = True
    layer.cache_features   = True
    
def run_layerRandomize(model, curName, layer, **kwargs):
    layer.weight_quantizer.run_layerRandomize()
    return layer.cal_quantLoss()

def run_layerGreedy(model, curName, layer, **kwargs):
    layer.run_layerGreedy()
    return layer.cal_quantLoss()

def dump_quant_feature(model, curName, layer, **kwargs):
    return layer.cached_out_quant_features

def dump_ori_feature(model, curName, layer, **kwargs):
    return layer.cached_out_features


def channelGreedyTest(test_loader, cali_data, args):
    args.skip_test = False
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
    qnn = build_qnn(args)
    build_ShiftedChannelQuant(qnn, '', **kwargs)
    #Run Calibration set for cached_inps, cached_outs
    for i in range(len(cali_data)//args.batch_size):
        _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))
    qnn.disable_cache_features()
    print(f'Quantized accuracy before shifted scale: {validate_model(test_loader, qnn):.3f}')
    testLayer = '.model.layer4.1.conv2' # '.model.layer1.0.conv1'
    loss = dict()
    QuantRecursiveRun(qnn, run_layerGreedy, '', loss, **kwargs)

def build_ShiftedChannelQuant(model: nn.Module, prv_name="", **kwargs):
    for name, module in model.named_children():
        curName = prv_name+'.'+name
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                continue
            else:
                if curName == '.model.layer4.1.conv2':
                # if curName == '.model.layer1.0.conv1':
                    build_ShiftedChannelQuantLayer(model, curName, module, **kwargs)
        else:
            build_ShiftedChannelQuant(module, curName, **kwargs)

def channelRandomizeTest(test_loader, cali_data, args):
    # ratio = [0.01,0.02,0.04,0.08,0.1,0.2,0.3,0.4,0.5,0.75,1.0]
    ratio = [0.1]
    
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
        t = tqdm(range(3000), desc=f'random value={shuffle_ratio}')
        qnn = build_qnn(args)
        kwargs['shuffle_ratio'] = shuffle_ratio
        kwargs['base_model'] = qnn
        build_ShiftedChannelQuant(qnn, '', **kwargs)
        #Run Calibration set for cached_inps, cached_outs
        for i in range(len(cali_data)//args.batch_size):
            _ = qnn(cali_data[i*args.batch_size:(i+1)*args.batch_size].to(device))

        qnn.disable_cache_features()
        print(f'Quantized accuracy before shifted scale: {validate_model(test_loader, qnn):.3f}')
        
        testLayer = '.model.layer4.1.conv2' # '.model.layer1.0.conv1'
        
        #TODO: Temp Codes
        dumpData = dict()
        QuantRecursiveRun(qnn, dump_ori_feature, '', dumpData, **kwargs)
        with open(f'./temp/quant_oriout.pkl', 'wb') as f:
            pickle.dump(dumpData[testLayer][0].cpu().numpy(), f)
        
        for i in t:
            loss = dict()
            QuantRecursiveRun(qnn, run_layerRandomize, '', loss, **kwargs)
            testResult.append(validate_model(test_loader, qnn, print_result=False).data.cpu().item())
            lossResult.append(loss[testLayer])
            if testResult[-1] > bestResult:
                bestResult = testResult[-1]
                #TODO: Temp Codes
                dumpData = dict()
                QuantRecursiveRun(qnn, dump_quant_feature, '', dumpData, **kwargs)
                with open(f'./temp/quant_maxAcc.pkl', 'wb') as f:
                    pickle.dump(dumpData[testLayer][0].cpu().numpy(), f)

            if lossResult[-1] < bestLoss:
                bestLoss = lossResult[-1]
                #TODO: Temp Codes
                dumpData = dict()
                QuantRecursiveRun(qnn, dump_quant_feature, '', dumpData, **kwargs)
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
        
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Ready For Simulation
    args = loadArgments()
    seed_all(args.seed)
    train_loader, test_loader = build_cifar10_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)
    cali_data = get_train_samples(train_loader, num_samples=args.num_samples)

    # channelRandomizeTest(test_loader, cali_data, args)
    channelGreedyTest(test_loader, cali_data, args)
        
    #Load Weight
    # channelRandomizeTest(qnn, test_loader, cali_data, shuffle_ratio, args)
    # channelGreedyTest(qnn, test_loader, cali_data, args)
    # break