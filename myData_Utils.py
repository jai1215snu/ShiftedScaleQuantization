import torch
from quant.quant_layer import QuantModule, Union
from quant.quant_model import QuantModel
from quant.quant_block import BaseQuantBlock

class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass

class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException

class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, model_input):
        self.model.eval()
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass

            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    _ = self.model(model_input.to(self.device))
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        # self.model.train()

        return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()

def save_inp_oup_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                      asym: bool = False, act_quant: bool = False, batch_size: int = 32, keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, asym=asym, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_inp, cur_out = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
    return cached_inps, cached_outs