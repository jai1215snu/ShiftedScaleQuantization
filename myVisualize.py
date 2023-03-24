import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from myQuant import *

def fold_bn_into_conv(dct, name, stage):
    convName = name + f'.conv{stage}.weight'
    bnName = name + f'.bn{stage}.'
    #Loading Data
    w = dct[convName].detach().cpu().numpy()
    y_mean = dct[bnName + 'running_mean'].detach().cpu().numpy()
    y_var  = dct[bnName + 'running_var'].detach().cpu().numpy()
    safe_std = np.sqrt(y_var + 1e-5)
    bn_weight = dct[bnName + 'weight'].detach().cpu().numpy()
    bn_bias = dct[bnName + 'bias'].detach().cpu().numpy()
    w_view = (w.shape[0], 1, 1, 1)
    
    weight = w * (bn_weight / safe_std).reshape(w_view)
    beta = bn_bias - bn_weight * y_mean / safe_std
    
    return weight
    # return w
    
def alpha_quantize(data_alpha):
    data_alpha = QTW['model.layer1.1.conv1.weight_quantizer.alpha']
    zeta, gamma = -0.1, 1.1
    data_alpha = torch.clamp(torch.sigmoid(data_alpha) * (zeta - gamma) + gamma, 0, 1)
    data_alpha = np.array(data_alpha.detach().cpu().numpy()).flatten()
    
def gen_bins(data, num=80):
    minData = min([np.min(d) for d in data])
    maxData = max([np.max(d) for d in data])
    data_range = [minData, maxData]
    bins = np.histogram_bin_edges(data_range, bins=num)
    return bins

def adaQuant(data, layer, stage):
    n_bits = 4
    n_levels = 2 ** n_bits
    layer = f'model.{layer}.conv{stage}'
    weight = np.array(data[f'{layer}.weight'].detach().cpu().numpy())
    alpha = data[f'{layer}.weight_quantizer.alpha'].detach().cpu().numpy()
    delta, zero_point = init_delta(weight)
    x_floor = np.floor(weight / delta)
    x_int = x_floor + np.float32(alpha >= 0)
    x_quant = np.clip(x_int + zero_point, 0, n_levels - 1)
    x_float_q = (x_quant - zero_point) * delta
    return x_float_q

def quantRound(data, layer, stage):
    n_bits = 4
    n_levels = 2 ** n_bits
    layer = f'model.{layer}.conv{stage}'
    weight = np.array(data[f'{layer}.weight'].detach().cpu().numpy())
    alpha = data[f'{layer}.weight_quantizer.alpha'].detach().cpu().numpy()
    delta, zero_point = init_delta(weight)
    x_int = np.round(weight / delta)
    x_quant = np.clip(x_int + zero_point, 0, n_levels - 1)
    x_float_q = (x_quant - zero_point) * delta
    return x_float_q


if __name__ == '__main__':
    layer = 'layer1.1'
    stage = 1
    
    ORI = torch.load("pretrained/PyTorch_CIFAR10/cifar10_models/state_dicts/resnet18.pt")
    QTW = torch.load("./checkPoint/QNN_W4_A4.pth")
    
    #Draw QTW histogram with seaborn
    data1 = fold_bn_into_conv(ORI, layer, stage)
    data2 = np.array(QTW[f'model.{layer}.conv{stage}.weight'].detach().cpu().numpy())
    # data2_q = adaQuant(QTW, layer, stage)[0]
    # data2_r = quantRound(QTW, layer, stage)[0]
    
    # data = [data2, data2_q, data2_r]
    plt.figure(figsize=(10,7))
    # bins = gen_bins(data)
    # for d in data:
    # sns.histplot(data=[d.flatten() for d in data], kde=True, bins=bins, alpha=0.5)
    # sns.histplot(data_alpha, hist=True, kde=True, 
    #             bins=80,
    #             hist_kws={'color': 'blue', 'alpha': 0.5})
    # for i in range(5):
    #     sns.scatterplot(x=data1[i].flatten(), y=data2[i].flatten())

    # sns.kdeplot(data=data1.flatten(), color='red', linewidth=2)
    # sns.kdeplot(data=data2.flatten(), color='red', linewidth=2)
    
    ch = 16
    labels = [f'{i} ch' for i in range(ch)]
    sns.violinplot(data=[d.flatten() for d in data2[0][:ch]], labels=labels, scale='width', palette='Set3')
    # sns.stripplot(data=[d.flatten() for d in data2[0][:ch]])

    delta, zero_point = init_delta(data2)
    qdata = np.zeros(shape=(data2.shape[0], 2**4))
    for i in range(2**4):
        qdata[:, i] = i
    qdata = (qdata - zero_point[:, 0, 0, 0].reshape(-1, 1)) * delta[:, 0, 0, 0].reshape(-1, 1)
    
    for i in range(16):
        plt.axhline(y=qdata[0][i], color='gray', linestyle='--', linewidth=0.3)
    
    plt.xlabel('channel number', fontname='Arial', fontsize=14, fontweight='bold')
    plt.ylabel('weight distribution', fontname='Arial', fontsize=14, fontweight='bold')

    plt.title('Weight distribution of each channel', fontname='Arial', fontsize=16, fontweight='bold')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
    sns.despine(top=True, right=True)
    
    plt.show()
    # plt.savefig('result.png', dpi=300)
    exit(1)

    # #plt histogram
    # #seaborn histogram with numpy
    # sns.distplot(df['column_name'], hist=True, kde=False, 
    #              bins=int(180/5), color = 'darkblue', 
    #              hist_kws={'edgecolor':'black'})

    # # Add labels
    # plt.title('Histogram')
    # plt.xlabel('column_name')
    # plt.ylabel('Count')


    # def plot_hist(arr, bins, title, xlabel, ylabel, save_path):
    #     plt.hist(arr, bins=bins)
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.savefig(save_path)
    #     plt.clf()