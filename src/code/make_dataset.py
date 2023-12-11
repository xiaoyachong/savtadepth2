#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#
# See https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset


"""Helper script to convert the NYU Depth v2 dataset Matlab file into a set of PNG and JPEG images.
Receives 3 Files from argparse:
<h5_file> - Contains the original images, depths maps, and scene types
<train_test_split> - contains two numpy arrays with the index of the
                    images based on the split to train and test sets.
<out_folder> - Name of the folder to save the original and depth images.

Every image in the DB will have it's twine B&W image that indicates the depth
in the image. the images will be read, converted by the convert_image function
and finally saved to path based on train test split and Scene types.
"""

from __future__ import print_function

import h5py
import numpy as np
import os
import scipy.io
import sys
import cv2
from tqdm import tqdm


def convert_image(index, depth_map, img, output_folder):
    """Processes data images and depth maps
    :param index: int, image index
    :param depth_map: numpy array, image depth - 2D array.
    :param img: numpy array, the original RGB image - 3D array.
    :param output_folder: path to save the image in.

    Receives an image with it's relevant depth map.
    Normalizes the depth map, and adds a 7 px boundary to the original image.
    Saves both image and depth map to the appropriate processed data folder.
    """

    # Normalize the depth image
    # normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    img_depth = depth_map * 25.0
    cv2.imwrite("%s/%05d_depth.png" % (output_folder, index), img_depth)

    # Adding black frame to original image
    img = img[:, :, ::-1]  # Flipping the image from RGB to BGR for opencv
    image_black_boundary = np.zeros(img.shape, dtype=np.uint8)
    image_black_boundary[7:image_black_boundary.shape[0] - 6, 7:image_black_boundary.shape[1] - 6, :] = \
        img[7:img.shape[0] - 6, 7:img.shape[1] - 6, :]
    cv2.imwrite("%s/%05d.jpg" % (output_folder, index), image_black_boundary)


if __name__ == "__main__":

    # Check if got all needed input for argparse
    if len(sys.argv) != 4:
        print("usage: %s <h5_file> <train_test_split> <out_folder>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    # load arguments to variables
    h5_file = h5py.File(sys.argv[1], "r")
    train_test = scipy.io.loadmat(sys.argv[2])  # h5py is not able to open that file. but scipy is
    out_folder = sys.argv[3]

    # Extract images *indexes* for train and test data sets
    test_images = set([int(x) for x in train_test["testNdxs"]])
    train_images = set([int(x) for x in train_test["trainNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    # Grayscale
    depth = h5_file['depths']
    print("Reading", sys.argv[1])
    images = h5_file['images']  # (num_channels, height, width)

    # Extract all sceneTypes per image - "office", "classroom", etc.
    scenes = [u''.join(chr(c[0]) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]

    for i, image in tqdm(enumerate(images), desc="Processing images", total=len(images)):
        idx = int(i) + 1
        if idx in train_images:
            train_test = "train"
        else:
            assert idx in test_images, "index %d neither found in training set nor in test set" % idx
            train_test = "test"

        # Create path to save image in
        folder = "%s/%s/%s" % (out_folder, train_test, scenes[i])
        if not os.path.exists(folder):
            os.makedirs(folder)

        convert_image(i, depth[i, :, :].T, image.T, folder)

    print("Finished")
