import torch
import torch.nn as nn

from quant import *
from data.cifar10 import build_cifar10_data
from pretrained.PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from common import *
# from pretrained.resnet import resnet18

if __name__ == '__main__':
    args = loadArgments()
    seed_all(args.seed)
    # build cifar10 data loader
    train_loader, test_loader = build_cifar10_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)
    # load model
    # cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch))
    cnn = resnet18(pretrained=True, device='cuda:0')
    # cnn = resnet18(pretrained=False, device='cuda:0')
    cnn.cuda()
    cnn.eval()
    print(f'accuracy of original : {validate_model(test_loader, cnn):.3f}')
    
    skip_weight_recon = False
    skip_act_recon = False
    
    # # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse', 'tune_delta_zero':True}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    cali_data = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device

    # # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)
    _ = qnn(cali_data[:64].to(device))
    
    args.test_before_calibration = True

    if args.test_before_calibration:
        # print(f'          accuracy of original : {validate_model(test_loader, cnn):.3f}')
        print(f'Quantized accuracy before brecq: {validate_model(test_loader, qnn):.3f}')
    
    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse', eval=skip_weight_recon)

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    # print(f'Ignore reconstruction of layer {format(name)}')
                    continue
                else:
                    print(f'Reconstruction for layer {format(name)}')
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    # print(f'Ignore reconstruction of block {format(name)}')
                    continue
                else:
                    print(f'Reconstruction for block {format(name)}')
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)
    # Start calibration
    recon_model(qnn)
    # if skip_weight_recon:
    #     print(f"loading : ./checkPoint/QNN_W{args.n_bits_w}_FP32.pth")
    #     st = torch.load(f'./checkPoint/QNN_W{args.n_bits_w}_FP32.pth')
    #     qnn.load_state_dict(st)
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    print('Weight quantization accuracy: {}'.format(validate_model(test_loader, qnn)))
    exit(1)
    
    #Save torch model
    torch.save(qnn.state_dict(), f'./checkPoint/QNN_W{args.n_bits_w}_FP32.pth')

    if args.act_quant:
        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            _ = qnn(cali_data[:64].to(device))
        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr, p=args.p, eval=skip_act_recon)
        recon_model(qnn)
        if skip_act_recon:
            print(f"./checkPoint/QNN_W{args.n_bits_w}_A{args.n_bits_a}.pth")
            st = torch.load(f"./checkPoint/QNN_W{args.n_bits_w}_A{args.n_bits_a}.pth")
            qnn.load_state_dict(st)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a,
                                                               validate_model(test_loader, qnn)))
    torch.save(qnn.state_dict(), f'./checkPoint/QNN_W{args.n_bits_w}_A{args.n_bits_a}.pth')
