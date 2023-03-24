import torch.nn as nn
from common import *
from data.cifar10 import build_cifar10_data
from pretrained.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from quant import *
from myVisualize import *

if __name__ == '__main__':
    args = loadArgments()
    
    seed_all(args.seed)
        # build cifar10 data loader
    train_loader, test_loader = build_cifar10_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)
    
    cnn = resnet18(pretrained=True, device='cuda:0')
    # print_model_hierarchy(cnn)
    cnn.cuda()
    cnn.eval()
    print(f'accuracy of original : {validate_model(test_loader, cnn):.3f}')
    
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    # print_model_hierarchy(qnn)
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()
    cali_data = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)
    _ = qnn(cali_data[:64].to(device))
    print("Validate")
    print(f'Quantized accuracy before brecq: {validate_model(test_loader, qnn):.3f}')
    
    
    #Load Weight
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse', eval=True)

    #TODO: Load Torch qnn

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print(f'Ignore reconstruction of layer {format(name)}')
                    continue
                else:
                    print(f'Reconstruction for layer {format(name)}')
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print(f'Ignore reconstruction of block {format(name)}')
                    continue
                else:
                    print(f'Reconstruction for block {format(name)}')
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)
    # Start calibration
    recon_model(qnn)
    
    print(f"loading : ./checkPoint/QNN_W{args.n_bits_w}_FP32.pth")
    st = torch.load(f'./checkPoint/QNN_W{args.n_bits_w}_FP32.pth')
    qnn.load_state_dict(st)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    print(f'Quantized accuracy after weight: {validate_model(test_loader, qnn):.3f}')
    
    if args.act_quant:
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            _ = qnn(cali_data[:64].to(device))
        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr, p=args.p, eval=True)
        recon_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        st = torch.load(f"./checkPoint/QNN_W{args.n_bits_w}_A{args.n_bits_a}.pth")
        qnn.load_state_dict(st)
        print(f'Full quantization (W{args.n_bits_w}A{args.n_bits_a}) accuracy: {validate_model(test_loader, qnn)}')
