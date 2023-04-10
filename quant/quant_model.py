import torch.nn as nn
from quant.quant_block import specials, BaseQuantBlock
from quant.quant_layer import QuantModule, StraightThrough
from quant.fold_bn import search_fold_and_remove_bn


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        search_fold_and_remove_bn(model)
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.qState = []

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, depth=0, moduleName=''):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            curName = moduleName+'.'+name
            # print("quant", "-"*depth, curName)
            # if name in ['relu', 'relu2'] and curName not in ['.layer1.0.relu']:#'.layer1.0.relu']:
            # if name in ['relu2'] and curName not in ['.layer1.0.relu']:#'.layer1.0.relu']:
            #     continue
            if name in ['relu2']:
                continue
            # print(type(child_module), specials)
            if type(child_module) in specials:
                # print("Making Special block")
                setattr(module, name, specials[type(child_module)](child_module, weight_quant_params, act_quant_params))
                qmodule = getattr(module, name)
                qmodule.setPathName(curName)

            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
                prev_quantmodule.pathName = curName

            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                    # print(">>>>>>Found ReLU, adding to :", prev_quantmodule.pathName)
                else:
                    # print(">>>>>>Found ReLU - no quant module before")
                    continue
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, depth+1, moduleName=curName)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)
    
    def set_quant_init_state(self):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_init_state()
    
    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-2].act_quantizer.bitwidth_refactor(8)
        # ignore reconstruction of the first layer
        module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True
        
    # def synchorize_activation_statistics(self):
    #     import linklink.dist_helper as dist
    #     for m in self.modules():
    #         if isinstance(m, QuantModule):
    #             if m.act_quantizer.delta is not None:
    #                 dist.allaverage(m.act_quantizer.delta)
    
    def disable_cache_features(self):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.disable_cache_features()
                
    def clear_cached_features(self):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.clear_cached_features()

    def store_quantization_state(self):
        self.qState = []
        for m in self.modules():
            if isinstance(m, QuantModule):
                self.qState += [m.use_weight_quant]
            
    def restore_quantization_state(self):
        idx = 0
        for m in self.modules():
            if isinstance(m, (QuantModule)):
                m.use_weight_quant = self.qState[idx]
                idx += 1