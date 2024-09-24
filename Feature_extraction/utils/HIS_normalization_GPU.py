import torch
import cv2
import numpy as np


def read_target(path):
    target = cv2.imread(path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB) # opencv default color space is BGR, change it to RGB
    p = np.percentile(target, 90)
    target = np.clip(target * 255.0 / p, 0, 255).astype(np.uint8)
    return target

def complement_stain_matrix1(w):
    stain0 = w[:, 0]
    stain1 = w[:, 1]
    stain2 = torch.cross(stain0, stain1)
    # Normalize new vector to have unit norm
    stain2 = stain2 / torch.norm(stain2)
    return torch.stack([stain0, stain1, stain2], dim=1)

def convert_image_to_matrix1(im):
    return im.permute(2, 0, 1).reshape(3, -1)

def convert_matrix_to_image1(m,shape):
    # convert_matrix_to_image1(torch.tensor(test11), shape=(512, 512, 3)).cpu().permute(0, 2, 3, 1).numpy()[0, :, :, :]
    m = m.T.reshape((-1,) + shape[-3:])
    return m.permute(0, 3, 1, 2).contiguous()

def rgb_to_sda1(im_rgb, I_0, allow_negatives=False):
    is_matrix = im_rgb.ndim == 2
    if is_matrix:
        im_rgb = im_rgb.T
    if I_0 is None:  # rgb_to_od compatibility
        im_rgb = im_rgb.float() + 1
        I_0 = torch.tensor(256)
    if not allow_negatives:
        im_rgb = torch.minimum(im_rgb,I_0)
    im_sda = -torch.log(im_rgb/(1.*I_0)) * 255/torch.log(I_0)
    return im_sda.T if is_matrix else im_sda


def sda_to_rgb1(im_sda, I_0):
    is_matrix = im_sda.ndim == 2
    if is_matrix:
        im_sda = im_sda.T
    od = I_0 is None
    if od:
        I_0 =  256
        im_rgb = I_0 ** (1 - im_sda / 255)
        return (im_rgb.T if is_matrix else im_rgb) - 1
    else:
        im_rgb = I_0 ** (1 - im_sda / 255)
        return (im_rgb.T if is_matrix else im_rgb)


def get_principal_components1(m):
    m_t = m
    u, s, v = torch.svd(m_t)
    return u[:, :3]


def magnitude1(m):
    return torch.norm(m, dim=0)

def normalize1(m):
    return m / magnitude1(m)


def color_convolution1(im_stains, w, I_0=None):
    m = convert_image_to_matrix1(im_stains)
    sda_fwd = rgb_to_sda1(m, 255 if I_0 is not None else None, allow_negatives=True)
    sda_conv = torch.matmul(w.double(), sda_fwd.double())
    sda_inv = sda_to_rgb1(sda_conv, I_0)
    im_rgb  = torch.clamp(convert_matrix_to_image1(sda_inv, im_stains.shape), 0, 255).to(torch.uint8)
    return im_rgb

def exclude_nonfinite1(m):
    is_finite = torch.isfinite(m).all(dim=0)
    return m[:, is_finite]

def _get_angles1(m):
    m = normalize1(m)
    return (1 - m[1]) * torch.sign(m[0])


def argpercentile1(arr: torch.Tensor,p):
    i = int(p * arr.shape[0] + 0.5)
    return np.argpartition(arr.cpu().numpy(), i)[i]

def separate_stains_macenko_pca1(im_sda, minimum_magnitude=16, min_angle_percentile=0.01,max_angle_percentile=0.99, mask_out=None):
    m = convert_image_to_matrix1(im_sda)
    # get rid of NANs and infinities
    m = exclude_nonfinite1(m)
    # Principal components matrix
    pcs = get_principal_components1(m)
    # Input pixels projected into the PCA plane

    proj = torch.mm(pcs.T[:-1], m)

    # Pixels above the magnitude threshold
    filt = proj[:, magnitude1(proj) > minimum_magnitude]
    # The "angles"
    angles = _get_angles1(filt)


    def get_percentile_vector1(p):
        return pcs[:, :-1].matmul(filt[:, argpercentile1(angles, p)])



    min_v = get_percentile_vector1(min_angle_percentile)

    max_v = get_percentile_vector1(max_angle_percentile)

    # The stain matrix
    w = complement_stain_matrix1(normalize1(torch.stack([min_v, max_v]).T))

    return w


def find_stain_index1(reference, w):
    dot_products = torch.matmul(reference.double(), w.double())
    return torch.argmax(torch.abs(dot_products))


def _reorder_stains1(W, stain_color_map_hematoxylin):
    def _get_channel_order(W):
        first = find_stain_index1(stain_color_map_hematoxylin, W)
        second = 1 - first
        third = 2
        return first, second, third

    def _ordered_stack(mat, order):
        return torch.stack([mat[..., j] for j in order], -1)
    return _ordered_stack(W, _get_channel_order(W))

def rgb_separate_stains_macenko_pca1(im_rgb, I_0 =None, *args, **kwargs):
    im_sda = rgb_to_sda1(im_rgb, I_0)

    return separate_stains_macenko_pca1(im_sda, *args, **kwargs)


def stain_unmixing_routine1(im_rgb,stain_color_map_hematoxylin):

    stain_unmixing_params = {}
    stain_unmixing_params['I_0'] = None
    W_source1 =  rgb_separate_stains_macenko_pca1(im_rgb, **stain_unmixing_params)


    W_source1 = _reorder_stains1(W_source1,stain_color_map_hematoxylin)


    return W_source1

def color_deconvolution1(im_rgb, w, I_0=None):
    # complement stain matrix if needed
    if torch.norm(w[:, 2]) <= 1e-16:
        wc = complement_stain_matrix1(w)
    else:
        wc = w
    # normalize stains to unit-norm
    wc = normalize1(wc)
    # invert stain matrix
    Q = torch.inverse(wc)
    # transform 3D input image to 2D RGB matrix format
    m = convert_image_to_matrix1(im_rgb)[:3]
    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    sda_fwd = rgb_to_sda1(m, I_0)
    sda_deconv = torch.matmul(Q.float(), sda_fwd.float())
    sda_inv = sda_to_rgb1(sda_deconv,255 if I_0 is not None else None)
    # reshape output
    StainsFloat = convert_matrix_to_image1(sda_inv, im_rgb.shape)
    # transform type
    Stains = torch.clamp(StainsFloat, 0, 255).to(torch.uint8)
    # return
    return Stains, StainsFloat, wc


def color_deconvolution_routine1( im_rgb,stain_color_map_hematoxylin):
    W_source1 = stain_unmixing_routine1(im_rgb,stain_color_map_hematoxylin)
    Stains, StainsFloat, wc = color_deconvolution1(im_rgb, w=W_source1, I_0=None)
    return Stains, StainsFloat, wc



def deconvolution_based_normalization1( im_src,  W_target , stain_color_map_hematoxylin):
    _, StainsFloat, _ = color_deconvolution_routine1(im_src,stain_color_map_hematoxylin)
    StainsFloat = StainsFloat.permute(0, 2, 3, 1)[0,:,:,:]
    im_src_normalized = color_convolution1(StainsFloat, W_target)
    return im_src_normalized

class his_normalization_pytorch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stain_color_map_hematoxylin =  torch.tensor([0.65, 0.70, 0.29]).cuda()
    def forward(self,img,W_target):
        img_list11122 = []
        for irr in range(img.shape[0]):
            pred_img = img[irr,:,:,:]
            try:
                image = deconvolution_based_normalization1(pred_img, W_target,self.stain_color_map_hematoxylin)[0, :, :, :]
                img_list11122.append(image)
            except:
                img_list11122.append(pred_img.permute(2, 0 , 1))
        return torch.stack(img_list11122)





