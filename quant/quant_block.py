import torch
import torch.nn as nn
import torch.nn.functional as F

from quant.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough
from models.resnet import BasicBlock, Bottleneck
from models.regnet import ResBottleneckBlock
from models.mobilenetv2 import InvertedResidual
import sys
sys.path.append('..')
from pretrained.PyTorch_CIFAR10.cifar10_models.resnet import BasicBlockCIFAR

class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        act_quant_params['disable_act_quant'] = False
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False
        self.cache_features      = 'none'
        self.cached_inp_features = []
        self.cached_out_features = []
        
        self.selectionInited = False
        self.pathName = ''

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)
                
    def set_quant_init_state(self):
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_init_state()
            
    def disable_cache_features(self):
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.disable_cache_features()
                
    def clear_cached_features(self):
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.clear_cached_features()
                
    def set_quant_state_block(self, state, act=False):
        for m in self.modules():
            if isinstance(m, QuantModule):
                if act:
                    m.use_act_quant = state
                else:
                    m.use_weight_quant = state
    
    def disable_cache_features(self):
        self.cache_features = 'none'
        
    def clear_cached_features(self):
        self.cached_inp_features = []
        self.cached_out_features = []


class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = basic_block.relu2

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        # copying all attributes in original block
        self.stride = basic_block.stride
        
        self.dump_cnt = 0

    def forward(self, x):
        if self.cache_features == 'if':
            self.cached_inp_features += [x.to('cpu').clone().detach()]
            
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        
        # if self.cache_features == 'debug':
        #     torch.save(out, f'layer4.1_{self.dump_cnt}.pt')
        #     self.dump_cnt += 1
        
        if self.cache_features == 'of':
            self.cached_out_features += [out.to('cpu').clone().detach()]
        return out

    def setPathName(self, curName):
        self.pathName = curName
        self.conv1.pathName = curName+'.conv1'
        self.conv2.pathName = curName+'.conv2'
        if self.downsample is not None:
            self.downsample.pathName = curName+'.downsample'
            
    def toggleHardTarget(self):
        self.conv1.weight_quantizer.hard_targets = not self.conv1.weight_quantizer.hard_targets
        self.conv2.weight_quantizer.hard_targets = not self.conv2.weight_quantizer.hard_targets
        if self.downsample is not None:
            self.downsample.weight_quantizer.hard_targets = not self.downsample.weight_quantizer.hard_targets
        

class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu3

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        # copying all attributes in original block
        self.stride = bottleneck.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].activation_function = nn.ReLU6()

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


specials = {
    BasicBlock: QuantBasicBlock,
    BasicBlockCIFAR: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
}
