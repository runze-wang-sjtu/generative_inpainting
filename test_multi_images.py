# -*- coding: utf-8 -*-
# @Author  : runze.wang
# @Time    : 2020/8/25 2:39 PM
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import SimpleITK as sitk

from inpaint_model import InpaintCAModel

REALITY=True

if REALITY:
    FILENAME = '4634640'

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='/data/rz/codes/generative_inpainting/test/reality/image/{}.nii.gz'.format(FILENAME), type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask', default='/data/rz/codes/generative_inpainting/test/reality/mask/{}.nii.gz'.format(FILENAME), type=str,
                        help='Where to read implant.')
    parser.add_argument('--image_real', default='/data/rz/codes/generative_inpainting/test/reality/output/{}_real.nii.gz'.format(FILENAME), type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--image_fake', default='/data/rz/codes/generative_inpainting/test/reality/output/{}_fake.nii.gz'.format(FILENAME), type=str,
                        help='Where to write output.')
    parser.add_argument('--image_implant', default='/data/rz/codes/generative_inpainting/test/reality/output/{}_implant.nii.gz'.format(FILENAME), type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='/data/rz/codes/generative_inpainting/model_logs/spine_place_pretrain', type=str,
                        help='The directory of tensorflow checkpoint.')
    # parser.add_argument('--implant_threshold', default=2000, type=int, help='The threshold for segment implant from spine image')
    parser.add_argument('--show', default=False, type=bool, help='If save slice image in 2D')

else:
    FILENAME = 'verse065'

    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        default='/data/rz/codes/generative_inpainting/test/simulation/image/{}_512.nii.gz'.format(
                            FILENAME), type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask',
                        default='/data/rz/codes/generative_inpainting/test/simulation/mask/{}-Mask.nii'.format(
                            FILENAME.split('verse')[-1]), type=str,
                        help='Where to read implant.')
    parser.add_argument('--image_real',
                        default='/data/rz/codes/generative_inpainting/test/simulation/output_placetrain/{}_real.nii.gz'.format(
                            FILENAME), type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--image_fake',
                        default='/data/rz/codes/generative_inpainting/test/simulation/output_placetrain/{}_fake.nii.gz'.format(
                            FILENAME), type=str,
                        help='Where to write output.')
    parser.add_argument('--image_implant',
                        default='/data/rz/codes/generative_inpainting/test/simulation/output_placetrain/{}_implant.nii.gz'.format(
                            FILENAME), type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='/data/rz/codes/generative_inpainting/model_logs/spine_place_pretrain',
                        type=str,
                        help='The directory of tensorflow checkpoint.')
    # parser.add_argument('--implant_threshold', default=2000, type=int, help='The threshold for segment implant from spine image')
    parser.add_argument('--show', default=False, type=bool, help='If save slice image in 2D')

def nii_to_array(img_sitk):

    img_array = sitk.GetArrayFromImage(img_sitk)
    img_array = img_array - np.min(img_array)
    img = img_array / np.max(img_array) * 255
    ###shape: z*x*y  range:[0,255]
    return img.astype('uint8')

def implant_seg(img_sitk, threshold):

    img_array = sitk.GetArrayFromImage(img_sitk)
    array_copy = img_array.copy()
    array_copy[array_copy<threshold] = 0
    array_copy[array_copy>=threshold] = 255

    return array_copy

def dilated(x, iterations=2):

    # 'x: np.array'
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.dilate(x, kernel, iterations=iterations)

    return erosion

def reference(img_sitk, reference_sitk):

    img_sitk.SetOrigin(reference_sitk.GetOrigin())
    img_sitk.SetSpacing(reference_sitk.GetSpacing())
    img_sitk.SetDirection(reference_sitk.GetDirection())

    return img_sitk

if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    model = InpaintCAModel()

    img_sitk = sitk.ReadImage(args.image)
    image_3d = nii_to_array(img_sitk)
    mask_sitk = sitk.ReadImage(args.mask)
    mask_3d = implant_seg(mask_sitk, threshold=0.5)

    mask_zero_one = mask_3d / 255
    image_implant_ = image_3d * (1 - mask_zero_one) + mask_zero_one * 255

    image_real = sitk.GetImageFromArray(image_3d)
    image_real = reference(image_real, reference_sitk=img_sitk)

    sitk.WriteImage(image_real, args.image_real)

    # mask_3d = implant_seg(img_sitk, threshold=2000)

    assert image_implant_.shape == mask_3d.shape


    z, h, w = image_implant_.shape
    grid = 8
    image_implant_ = image_implant_[:, :h//grid*grid, :w//grid*grid]
    mask_3d = mask_3d[:, :h//grid*grid, :w//grid*grid]
    print('Shape of image: {}'.format(image_implant_.shape))

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(tf.float32, shape=(1, h, 2*w, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    results = []
    masks = []
    for i in range(image_implant_.shape[0]):

        image = image_implant_[i, :, :]
        mask = mask_3d[i, :, :]
        if REALITY:
            mask = dilated(mask)
            pass
        masks.append(mask)
        image = np.stack([image, image, image], axis=0)
        image = image.transpose((1,2,0)).astype('uint8')
        mask = np.stack([mask, mask, mask], axis=0)
        mask = mask.transpose((1,2,0)).astype('uint8')
        assert image.shape == mask.shape

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        result = sess.run(output, feed_dict={input_image_ph: input_image})
        if args.show:
            cv2.imwrite(os.path.join(args.output, '{}'.format(i)), result)

        results.append(result[0,:,:,0])
        print('Processed: level_{}/{}'.format(i, image_implant_.shape[0]))

    results = np.array(results)
    print('Final results shape: {}'.format(results.shape))

    results = sitk.GetImageFromArray(results)
    results = reference(results, reference_sitk=img_sitk)
    sitk.WriteImage(results, args.image_fake)

    masks = np.array(masks)
    mask_zero_one = masks / 255
    image_implant_ = image_3d * (1 - mask_zero_one) + mask_zero_one * 255
    image_implant = sitk.GetImageFromArray(image_implant_)
    image_implant = reference(image_implant, reference_sitk=img_sitk)
    sitk.WriteImage(image_implant, args.image_implant)

    print('{} has Done'.format(FILENAME))

