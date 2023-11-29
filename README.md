
# How to Use
1. Obtain the pre-trained Floating Point (FP) model.
- The FP models utilized in our work are derived from the [BRECQ](https://github.com/yhhhli/BRECQ) project. 
- These models are available for download via this [link](https://github.com/yhhhli/BRECQ/releases/tag/v1.0). 
- After obtaining the models, make sure to adjust the path of the pre-trained model in your hubconf.py file to match their location on your system.


## verifed environment
The code has been tested and verified to work on the following software versions:

- numpy: 1.21.5
- PyTorch: 1.11.0
- torchvision: 0.12.0
- Python: 3.7.13

## Usage

```bash
python main_imagenet.py --device_gpu='cuda:0' --arch='resnet18' --n_bits_w=2 --n_bits_a=4  --weight=1.0 --bias_cal=True --bias_ch_quant=True
```

You can get the following output:

```bash
Weight quantization accuracy: 66.32799530029297
Full quantization (W2A4) accuracy: 65.21199798583984
```

## Key Options for IOSO
- arch : Defines the architecture ('resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet').
- weight : Sets the lambda value of rounding and group policy.
- bias_cal : Enables learning of output channel scale gamma^z and output channel offset varphi^z.
- bias_ch_quant : Activates learning of input channel group R.

## Acknowledgements
We wish to express our sincere thanks to @yhhhli, whose [BRECQ](https://github.com/yhhhli/BRECQ) repository has greatly influenced our code.
