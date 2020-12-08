// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    Instance segmentation using the PASCAL VOC2012 dataset.

    Instance segmentation sort-of combines object detection with semantic
    segmentation. While each dog, for example, is detected separately,
    the output is not only a bounding-box but a more accurate, per-pixel
    mask.

    For introductions to object detection and semantic segmentation, you
    can have a look at dnn_mmod_ex.cpp and dnn_semantic_segmentation.h,
    respectively.

    Instructions how to run the example:
    1. Download the PASCAL VOC2012 data, and untar it somewhere.
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    2. Build the dnn_instance_segmentation_train_ex example program.
    3. Run:
       ./dnn_instance_segmentation_train_ex /path/to/VOC2012
    4. Wait while the network is being trained.
    5. Build the dnn_instance_segmentation_ex example program.
    6. Run:
       ./dnn_instance_segmentation_ex /path/to/VOC2012-or-other-images

    An alternative to steps 2-4 above is to download a pre-trained network
    from here: http://dlib.net/files/instance_segmentation_voc2012net_v2.dnn

    It would be a good idea to become familiar with dlib's DNN tooling before reading this
    example.  So you should read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp
    before reading this example program.
*/

#ifndef DNN_H_
#define DNN_H_

#include "structures.h"

#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QObject>
#include <QtCore>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/opencv.h>
#include "dlib/cuda/gpu_data.h"

#include <iostream>
#include <iterator>
#include <thread>
#include <stdlib.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct image_info
{
    std::string image_filename;
    std::string gt_filename;
};

struct truth_instance
{
    dlib::matrix<unsigned char> input_image;
    dlib::matrix<float> label_image;
};

struct seg_training_sample
{
    dlib::matrix<unsigned char> input_image;
    dlib::matrix<float> label_image; // The ground-truth label of each pixel. (+1 or -1)
};

using net_type2 = dlib::loss_binary_log <
    dlib::fc < 1,
    dlib::relu<dlib::fc<10,
    dlib::relu<dlib::fc<20,
    dlib::relu<dlib::con<50, 5, 5, 1, 1,
    dlib::relu<dlib::con<50, 5, 5, 1, 1,
    dlib::input<dlib::matrix<unsigned char>>
    >>>>>>>>>>;

using net_type = dlib::loss_binary_log_per_pixel <
    dlib::cont < 1, 1, 1, 1, 1,
    dlib::relu<dlib::con<50, 5, 5, 1, 1,
    dlib::relu<dlib::con<50, 5, 5, 1, 1,
    dlib::relu<dlib::con<50, 5, 5, 1, 1,
    dlib::relu<dlib::con<50, 5, 5, 1, 1,
    dlib::input<dlib::matrix<unsigned char>>
    >>>>>>>>>>;

#endif // DNN_H_
