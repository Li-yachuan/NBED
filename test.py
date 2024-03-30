from os.path import join
import os
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import scipy.io as sio


def test(model, test_loader, save_dir):
    print("single scale test")
    png_save_dir = os.path.join(save_dir, "png")
    mat_save_dir = os.path.join(save_dir, "mat")
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    if not os.path.exists(mat_save_dir):
        os.makedirs(mat_save_dir)

    model.eval()
    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        with torch.no_grad():
            result = model(image).squeeze().cpu().numpy()
        result_png = Image.fromarray((result * 255).astype(np.uint8))
        result_png.save(join(png_save_dir, "%s.png" % filename))

        sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': result}, do_compression=True)


import cv2


def multiscale_test(model, test_loader, save_dir, scale_num=7):
    png_save_dir = os.path.join(save_dir, "png")
    mat_save_dir = os.path.join(save_dir, "mat")
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    if not os.path.exists(mat_save_dir):
        os.makedirs(mat_save_dir)

    model.eval()
    if scale_num == 7:
        print("7 scale test")
        scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    else:
        print("3 scale test")
        scale = [0.5, 1.0, 1.5]

    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))
        _, H, W = image.shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = torch.from_numpy(im_.transpose((2, 0, 1))).unsqueeze(0)
            with torch.no_grad():
                result = model(im_.cuda()).squeeze().cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)

        result_png = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_png.save(join(png_save_dir, "%s.png" % filename))

        sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': multi_fuse}, do_compression=True)


from functools import partial


def __identity(x):
    return x


def enhence_test(model, test_loader, save_dir):
    print("rotate enhence test")
    png_save_dir = os.path.join(save_dir, "png")
    mat_save_dir = os.path.join(save_dir, "mat")
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    if not os.path.exists(mat_save_dir):
        os.makedirs(mat_save_dir)

    model.eval()
    funcs = [partial(__identity),
             partial(cv2.rotate, rotateCode=cv2.ROTATE_90_CLOCKWISE),
             partial(cv2.rotate, rotateCode=cv2.ROTATE_180),
             partial(cv2.rotate, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)]

    funcs_t = [partial(__identity),
               partial(cv2.rotate, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE),
               partial(cv2.rotate, rotateCode=cv2.ROTATE_180),
               partial(cv2.rotate, rotateCode=cv2.ROTATE_90_CLOCKWISE)]

    for idx, (image, filename) in enumerate(tqdm(test_loader)):

        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))

        H, W, _ = image_in.shape

        multi_fuse = np.zeros((H, W), np.float32)

        for func, funct in zip(funcs, funcs_t):
            img = func(image_in)
            edge = __enhence_test_single(img, model)
            edge = funct(edge)
            multi_fuse += edge

        image_inf = cv2.flip(image_in, 1)  # shuiping fanzhuan
        multi_fuse_f = np.zeros((H, W), np.float32)

        for func, funct in zip(funcs, funcs_t):
            img = func(image_inf)
            edge = __enhence_test_single(img, model)
            edge = funct(edge)
            multi_fuse_f += edge

        multi_fuse = multi_fuse + cv2.flip(multi_fuse_f, 1)

        multi_fuse = multi_fuse / 8

        result_png = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_png.save(join(png_save_dir, "%s.png" % filename))

        sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': multi_fuse}, do_compression=True)


def bright_enhence_test(model, test_loader, save_dir):
    print("bright enhence test")
    png_save_dir = os.path.join(save_dir, "png")
    mat_save_dir = os.path.join(save_dir, "mat")
    if not os.path.exists(png_save_dir):
        os.makedirs(png_save_dir)
    if not os.path.exists(mat_save_dir):
        os.makedirs(mat_save_dir)

    model.eval()
    for idx, (image, filename) in enumerate(tqdm(test_loader)):
        image = image[0]
        image_in = image.numpy().transpose((1, 2, 0))

        H, W, _ = image_in.shape

        multi_fuse = np.zeros((H, W), np.float32)
        bright_intals = [(0, 0.5), (0.25, 0.75), (0.5, 1)]
        for internl in bright_intals:
            img = __bright_func(image_in, internl)
            edge = __enhence_test_single(img, model)
            multi_fuse += edge

        multi_fuse = multi_fuse / len(bright_intals)

        result_png = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_png.save(join(png_save_dir, "%s.png" % filename))

        sio.savemat(join(mat_save_dir, "%s.mat" % filename), {'result': multi_fuse}, do_compression=True)


def __bright_func(image_in, internl):
    threshold_min = image_in.min() + (image_in.max() - image_in.min())*internl[0]
    threshold_max = image_in.min() + (image_in.max() - image_in.min())*internl[1]

    enh_image = np.clip(image_in,threshold_min,threshold_max)

    scale_factor = (image_in.max() - image_in.min()) / (threshold_max - threshold_min)
    offset = image_in.min() - scale_factor * threshold_min
    enh_image = scale_factor * enh_image + offset

    image = (image_in + enh_image) / 2
    return image


def __enhence_test_single(image_in, model):
    scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    H, W, _ = image_in.shape
    multi_fuse = np.zeros((H, W), np.float32)

    for k in range(0, len(scale)):
        im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
        im_ = torch.from_numpy(im_.transpose((2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            result = model(im_.cuda()).squeeze().cpu().numpy()
        fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
        multi_fuse += fuse
    multi_fuse = multi_fuse / len(scale)
    return multi_fuse
