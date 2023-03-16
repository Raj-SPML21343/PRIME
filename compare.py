# Imports
import numpy as np
import os, cv2, random, time
import matplotlib.pyplot as plt

from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utility import WaveletTransform, p_omega, p_omega_t, l1_prox, tv_norm

def ISTA(fx, gx, gradf, proxg, params):
    """
        Implementation of ISTA or Iterative Soft Thresholding Algorithm
    """
    print("ISTA : Starting")
    time_start = time.time()

    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['L']

    info = {'iter': maxit, 'fx': np.zeros(maxit)}

    x_k = x0
    for k in range(maxit):
        y = x_k - alpha * gradf(x_k)
        x_k_next = proxg(y, alpha)
        x_k = x_k_next
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)

    print("ISTA : Time Taken : {}".format(time.time() - time_start))
    return x_k, info

def APGD(fx, gx, gradf, proxg, params):
    """
        Implementation of APGD or Accelerated Proximal Gradient Descent
    """
    print("APGD : Starting")
    time_start = time.time()

    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['L']

    info = {'iter': maxit, 'fx': np.zeros(maxit)}

    x_k = x0
    y_k = x0
    for k in range(maxit):
        y_k_next = proxg(x_k - alpha * gradf(x_k), alpha)
        x_k_next = y_k_next + ((k-1)/(k+2)) * (y_k_next - y_k)
        y_k = y_k_next
        x_k = x_k_next
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)

    print("APGD : Time Taken : {}".format(time.time() - time_start))
    return x_k, info

def FISTA(fx, gx, gradf, proxg, params):
    """
        Implementation of FISTA or Fast Iterative Soft Thresholding Algorithm
    """
    print("FISTA : Starting")
    time_start = time.time()

    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['L']
    restart_fista = params['restart']

    info = {'iter': maxit, 'fx': np.zeros(maxit)}

    t_k = 1
    x_k = x0
    y_k = x0
    for k in range(maxit):
        x_k_next = proxg(y_k - alpha * gradf(y_k), alpha)
        t_k_next = (1 + np.sqrt(4 * (t_k ** 2) + 1)) / 2
        y_k_next = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)
        if restart_fista and restart_condition(x_k.reshape(x_k.shape[0],), x_k_next.reshape(x_k_next.shape[0],), y_k.reshape(y_k.shape[0],)):
            y_k = x_k
        else:
            y_k = y_k_next
            t_k = t_k_next
            x_k = x_k_next
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)

    print("FISTA : Time Taken : {}".format(time.time() - time_start))
    return x_k, info

def restart_condition(x_k, x_k_next, y_k):
    """
        Gradient Based Restart Condition for FISTA based on https://arxiv.org/pdf/1906.09126.pdf
    """
    return (y_k - x_k_next) @ (x_k_next - x_k) > 0

def optimize_L1(image, indices, optimizer, params):
    """
        Optimzation using the L1 loss (Formulation 1)
    """
    w = WaveletTransform(m=params["m"])

    forward_operator = lambda x: p_omega(w.WT(x), indices)                  # P_Omega.W^T
    adjoint_operator = lambda x: w.W(p_omega_t(x, indices, params['m']))    # W.P_Omega^T

    b = p_omega(image, indices)

    fx = lambda x: 0.5 * np.linalg.norm(b - forward_operator(x)) ** 2
    gx = lambda x:  np.linalg.norm(x, 1)
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: adjoint_operator(forward_operator(x) - b)

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return w.WT(x).reshape((params['m'], params['m'])), info

def optimize_TV(image, indices, optimizer, params):
    """
        Optimzation using the TV loss (Formulation 2)
    """
    forward_operator = lambda x: p_omega(x, indices)                        # P_Omega
    adjoint_operator = lambda x: p_omega_t(x, indices, params['m'])         # P_Omega^T

    b = forward_operator(image)

    fx = lambda x: 0.5 * np.linalg.norm(b - forward_operator(x)) ** 2
    gx = lambda x: tv_norm(x, optimizer)
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m'])),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              max_num_iter=50).reshape((params['N'], 1))
    gradf = lambda x: adjoint_operator(forward_operator(x) - b).reshape(x.shape[0],1)

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return x.reshape((params['m'], params['m'])), info

def mouse_callback(event, x, y, flags, param):
    """
        Mouse Callback function to draw a rectangle to generate noisy patches
    """
    global image, drawing, mask, rect_start
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = image.copy()
        cv2.rectangle(img_copy, rect_start, (x,y), (0, 255, 0), 2)
        cv2.imshow('Image', img_copy * mask)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end = (x, y)
        mask[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]] = 0
        noise = np.random.random(mask[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]].shape)
        noise[noise > 0.25] = 0.0
        noise[noise > 0.] = 1
        mask[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]] = noise

if __name__ == "__main__":
    # Select a random image from the ImageNet-Mini Dataset
    folder = random.choice(os.listdir('data/ImageNet-Mini/images'))
    file = random.choice(os.listdir('data/ImageNet-Mini/images/' + folder))
    image = cv2.imread('data/ImageNet-Mini/images/' + folder + '/' + file, 0)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_CUBIC)
    image_copy = image

    cv2.namedWindow('Image')
    cv2.imshow('Image', image)
    drawing = False
    mask = np.ones_like(image)
    cv2.setMouseCallback('Image', mouse_callback)
    cv2.waitKey(0)

    cv2.imshow('Mask', np.uint8(255 * mask))
    cv2.imshow('Masked Image', image*mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Initialize parameters for the Optimization problem
    shape = (256, 256)
    params = {
        'maxit': 200,
        'L': 1,
        'lambda': 0.01,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart': False, #True
        'm': shape[0],
        'N': shape[0] * shape[1]
    }

    image_masked = image * mask
    image = image_copy

    image_masked = image_masked/255.0
    image = image/255.0

    indices = np.nonzero(mask.flatten())[0]
    params['indices'] = indices

    aa, bb = optimize_TV(image, indices, ISTA, params)
    cc, dd = optimize_TV(image, indices, FISTA, params)
    ee, ff = optimize_TV(image, indices, APGD, params)

    plt.subplot(1,2,1)
    plt.plot(np.abs(bb['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'ISTA')
    plt.plot(np.abs(dd['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'FISTA')
    plt.plot(np.abs(ff['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'NAG')
    plt.title('Normal Scale')
    plt.ylim([0,20])
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.subplot(1,2,2)
    plt.plot(np.abs(bb['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'ISTA')
    plt.plot(np.abs(dd['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'FISTA')
    plt.plot(np.abs(ff['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'APGD')
    plt.title('Log Scale')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.yscale("log")
    plt.legend()
    plt.suptitle("Comparison for TV Loss")
    plt.show()

    aa, bb = optimize_L1(image, indices, ISTA, params)
    cc, dd = optimize_L1(image, indices, FISTA, params)
    ee, ff = optimize_L1(image, indices, APGD, params)

    plt.subplot(1,2,1)
    plt.plot(np.abs(bb['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'ISTA')
    plt.plot(np.abs(dd['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'FISTA')
    plt.plot(np.abs(ff['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'APGD')
    plt.title('Normal Scale')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.subplot(1,2,2)
    plt.plot(np.abs(bb['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'ISTA')
    plt.plot(np.abs(dd['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'FISTA')
    plt.plot(np.abs(ff['fx'] - dd['fx'][-1])/(dd['fx'][-1]), label = 'APGD')
    plt.title('Log Scale')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.yscale("log")
    plt.legend()
    plt.suptitle("Comparison for L1 Loss")
    plt.show()
