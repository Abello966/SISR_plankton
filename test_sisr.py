import numpy as np
import keras as kr
import tensorflow as tf

from keras.models import load_model
from skimage.util import img_as_ubyte
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

from skimage.transform import rescale as rescale
from scipy.misc import imresize as imresize
from skimage.util import invert

import itertools
import time

def generate_baseline(img):
    smol = rescale(img, 0.5, anti_aliasing=True, multichannel=False, mode='reflect')
    baseline = imresize(smol, img.shape, interp="bicubic") #baseline do artigo é interpolação bicubica
    return baseline

def generate_prediction(img, model, black=False, tolerance=1):
    smol = rescale(img, 0.5, anti_aliasing=True, multichannel=False, mode='reflect')
    output = np.zeros(img.shape)
    overlap = np.zeros(img.shape)

    #define cor de background
    if black:
        bg = 0
        smol = invert(smol)
    else:
        bg = 255

    numiters = 0
    numpreds = 0

    #Previmos cada patch de 20x20 na imagem, somamos e tiramos a média
    for y, x in itertools.product(range(0, smol.shape[0] - 19), range(0, smol.shape[1] - 19)):
        predpatch = smol[y:y + 20, x: x + 20]
        predpatch = imresize(predpatch, (40, 40), interp='bicubic')

        #só executamos a previsão em lugares que a rede está treinada
        if (np.sum(predpatch == bg) / np.prod(predpatch.shape)) < tolerance: 
            predpatch = model.predict(predpatch.reshape((1,) + predpatch.shape + (1,)))[0]
            predpatch = predpatch.reshape(predpatch.shape[0:2])
            numpreds += 1

        output[y * 2: y * 2 + 40, x * 2: x * 2 + 40] += predpatch
        overlap[y * 2: y * 2 + 40, x * 2: x * 2 + 40] += 1
        numiters += 1

    output = output / overlap

    #Converte para 8bit como as imagens originais
    output_int = (output - np.min(output))
    output_int = output_int / np.max(output_int)
    output_int = img_as_ubyte(output_int)

    if (black):
        output_int = invert(output_int)

    print("predicted %d of %d possible patches" % (numpreds, numiters))

    return output_int, numpreds, numiters

def mse(y, x):
    return np.sum((y - x) ** 2) / np.prod(y.shape)

def test_and_metrics(test, model, black=True, tolerance=1):  
    if (test.shape[0] % 2 == 1):
        test = test[0:-1, :]
    if (test.shape[1] % 2 == 1):
        test = test[:, 0:-1]

    if (test.shape[0] < 40 or test.shape[1] < 40):
        return None
    

    timebef = time.time()

    baseline = generate_baseline(test)
    preds, numpreds, numiters = generate_prediction(test, model, black=black, tolerance=tolerance)

    mse_base = mse(baseline, test)
    mse_pred = mse(preds, test)
    
    psnr_base = psnr(baseline, test)
    psnr_pred = psnr(preds, test)
    
    ssim_base = ssim(baseline, test)
    ssim_pred = ssim(preds, test)
        
    timeaft = time.time()
    print("w shape %d x %d, iteration took %f seconds" % (test.shape[0], test.shape[1], (timeaft - timebef)))
    return mse_base, psnr_base, ssim_base, mse_pred, psnr_pred, ssim_pred, numpreds, numiters
    
dataset = np.load("test_dataset.npy")
dataset = dataset[0:100]
model = load_model("models/CNNSISR_black_1_tol")

#Carrega Dados
if __name__ == "__main__":

    import time
    import sys

    if (sys.argv[1] == "black"):
        black = True
    else:
        black = False

    tolerance = float(sys.argv[2])

    mse_base, psnr_base, ssim_base = (0, 0, 0)
    mse_pred, psnr_pred, ssim_pred = (0, 0, 0)
    numpreds, numiters = (0, 0)

    adjust = 0

    for test in dataset:
        metrics = test_and_metrics(test, model, black, tolerance)
        if metrics is None:
            adjust += 1
            continue
        mse_base += metrics[0]
        psnr_base += metrics[1]
        ssim_base += metrics[2]
        mse_pred += metrics[3]
        psnr_pred += metrics[4]
        ssim_pred += metrics[5]
        numpreds += metrics[6]
        numiters += metrics[7]
    

    mse_base /= (len(dataset) - adjust)
    mse_pred /= (len(dataset) - adjust)

    psnr_base /= (len(dataset) - adjust)
    psnr_pred /= (len(dataset) - adjust)

    ssim_base /= (len(dataset) - adjust)
    ssim_pred /= (len(dataset) - adjust)


    print("Resultados - %d de %d previsões - %f p100" % (numpreds, numiters, float(numpreds) / numiters * 100))
    print("")
    print("Baseline")
    print("MSE: %f " % mse_base)
    print("PSNR: %f " % psnr_base)
    print("SSIM: %f " % ssim_base)
    print("")
    print("Modelo")
    print("MSE: %f " % mse_pred)
    print("PSNR: %f " % psnr_pred)
    print("SSIM: %f " % ssim_pred)

