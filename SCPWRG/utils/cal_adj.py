import numpy as np
import numba
from sklearn.preprocessing import normalize


def dot_dist(t1, t2):
    sum = np.sum(np.dot(t1, t2))
    return np.sqrt(sum)

def psnr_dist(t1, t2):
    mse = np.mean((t1-t2)**2)
    if mse == 0:
        return np.sqrt(100)
    else:
        return np.sqrt(10 * np.log10(255/np.sqrt(mse)))

def ssim_dist(t1, t2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_adj(patches, method):
    # patches: B × h × w
    # return: adj
    assert method in ['dot', 'psnr', 'ssim']
    patches_flatten = patches.reshape([-1, 32*32])
    n = patches_flatten.shape[0]
    adj = np.empty((n,n), dtype=np.float32)

    if method == 'dot':
        for i in numba.prange(n):
            for j in numba.prange(n):
                adj[i][j] = dot_dist(patches_flatten[i], patches_flatten[j])
    elif method == 'psnr':
        for i in numba.prange(n):
            for j in numba.prange(n):
                adj[i][j] = psnr_dist(patches_flatten[i], patches_flatten[j])
    elif method == 'ssim':
        for i in numba.prange(n):
            for j in numba.prange(n):
                adj[i][j] = ssim_dist(patches_flatten[i], patches_flatten[j])

    return normalize(adj)