import numpy as np

def lp_loss(pred, tgt, p=2.0):
    return  np.mean(np.power(np.abs(pred-tgt), p))

def quantize_ch(x, n_bits):
    x_max = np.max(x)
    x_min = np.min(x)
    best_score = 1e+10
    for i in range(80):
        new_max = x_max * (1.0 - (i * 0.01))
        new_min = x_min * (1.0 - (i * 0.01))
        x_q = quantize(x, new_max, new_min, n_bits)
        score = lp_loss(x, x_q, p=2.4)
        if score < best_score:
            best_score = score
            delta = (new_max - new_min) / (2 ** n_bits - 1)
            zero_point = (- new_min / delta).round()
    return delta, zero_point

def quantize(x, max, min, n_bits):
    n_levels = 2 ** n_bits
    delta = (max - min) / (2 ** n_bits - 1)
    zero_point = (- min / delta).round()
    # we assume weight quantization is always signed
    x_int = np.round(x / delta)
    x_quant = np.clip(x_int + zero_point, 0, n_levels - 1)
    x_float_q = (x_quant - zero_point) * delta
    return x_float_q


def init_delta(weight):
    n_ch = weight.shape[0]
    x_max = np.amax(weight, axis=(1, 2, 3))

    print(x_max)
    delta = np.array(x_max)
    zero_point = np.array(x_max)

    for c in range(n_ch):
        delta[c], zero_point[c] = quantize_ch(weight[c], 4)
    delta = np.reshape(delta, (-1, 1, 1, 1))
    zero_point = np.reshape(zero_point, (-1, 1, 1, 1))
    return delta, zero_point