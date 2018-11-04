import numpy as np
import keras as kr
import tensorflow as tf

from keras.models import load_model
from skimage.util import img_as_ubyte
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

def generate_baseline(img):
    smol = rescale(img, 0.5, anti_aliasing=True, multichannel=False, mode='reflect')
    baseline = imresize(smol, (94, 104), interp="bicubic") #baseline do artigo é interpolação bicubica
    return baseline

def generate_prediction(baseline, model, black=False):
    if (black):
        baseline = invert(baseline)

    output = np.zeros(baseline.shape)
    overlap = np.zeros(baseline.shape)

    #Previmos cada patch de 20x20 na imagem, somamos e tiramos a média
    for y, x in itertools.product(range(0, smol.shape[0] - 19), range(0, smol.shape[1] - 19)):
        predpatch = smol[y:y + 20, x: x + 20]
        predpatch = imresize(predpatch, (40, 40), interp='bicubic')
        predpatch = black_net.predict(predpatch.reshape((1,) + predpatch.shape + (1,)))[0]
        predpatch = predpatch.reshape(predpatch.shape[0:2])
        output[y * 2: y * 2 + 40, x * 2: x * 2 + 40] += predpatch
        overlap[y * 2: y * 2 + 40, x * 2: x * 2 + 40] += 1

    output = output / overlap

    #Converte para 8bit como as imagens originais
    output_int = (output - np.min(output))
    output_int = output_int / np.max(output_int)
    output_int = img_as_ubyte(output_int)

    if (black):
        output = invert(output)

    return output

def mse(y, x):
    return np.sum((y - x) ** 2) / np.prod(y.shape)
    
#Carrega Dados
dataset = np.load("test_dataset.npy")
model = load_model("CNNSISR_black")

mse_base, psnr_base, ssim_base = (0, 0, 0)
mse, psnr, ssim = (0, 0, 0)

for test in dataset:
    baseline = generate_baseline(test)
    preds = generate_prediction(baseline, model, black=True)

    mse_base += mse(baseline, test)
    mse += mse(preds, test)
    
    psnr_base += psnr(baseline, test)
    psnr += psnr(preds, test)
    
    ssim_base += ssim(baseline, test)
    ssim += ssim(preds, test)

mse_base /= len(dataset)
mse /= len(dataset)

psnr_base /= len(dataset)
psnr /= len(dataset)

ssim_base /= len(dataset)
ssim /= len(dataset)

print("Resultados")
print("")
print("Baseline")
print("MSE: %f " % mse_base)
print("PSNR: %f " % psnr_base)
print("SSIM: %f " % ssim_base)
print("")
print("Modelo")
print("MSE: %f " % mse)
print("PSNR: %f " % psnr)
print("SSIM: %f " % ssim)

