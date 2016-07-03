# SOP for Building Residual Network-50 for Food Classification
---

0. Do not plug GPU
0. Install `Ubuntu 16.04 LTS`
(If upgraded from 14.04, see [here](https://www.digitalocean.com/community/tutorials/how-to-upgrade-to-ubuntu-16-04-lts))
0. Install `nvidia-367` graphics driver, and then plug GPU back

    ```
    sudo apt-add-repository ppa:graphics-drivers/ppa
    sudo apt-get update
    sudo apt-get install nvidia-367
    ```
0. Install `CUDA 8.0` without installing graphics driver (download [here](https://developer.nvidia.com/cuda-toolkit), use `.run` to optionally disable installing the graphics driver)

    ```
    sudo sh cuda_8.0.27_linux.run
    ```
0. Install `cuDNN 5` (download [here](https://developer.nvidia.com/cudnn))

    ```
    sudo tar -zxvf cudnn-8.0-linux-x64-v5.0-ga.tgz --directory /usr/local/
    ```
0. `apt-get` dependencies

    ```
    sudo apt-get install python-pip python-dev python-wheel python-numpy git zlib1g-dev swig 
    ``` 
    
0. `pip` dependencies

    ```
    pip install scipy
    ```  
0. Install `JDK 8`

    ```
    sudo add-apt-repository ppa:webupd8team/java
    sudo apt-get update
    sudo apt-get install oracle-java8-installer
    ```
0. Install Bazel (follow [here](http://www.bazel.io/docs/install.html))
    
    ```
    wget https://github.com/bazelbuild/bazel/releases/download/0.3.0/bazel-0.3.0-installer-linux-x86_64.sh
    chmod +x bazel-0.3.0-installer-linux-x86_64.sh
    ./bazel-0.3.0-installer-linux-x86_64.sh --user
    ```
0. Install TensorFlow from source (follow [here](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#installing-from-sources))

    ``` 
    git clone https://github.com/tensorflow/tensorflow
    cd tensorflow
    ./configure [CUDA: 8.0, cuDNN: 5, compute capability: 6.1]
    bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    pip install /tmp/tensorflow_pkg/tensorflow-* [the name depends]
    ```

0. Clone project repository

    ```
    git clone http://dev.2bite.com:14711/stmharry/ResidualNetwork.git
    ```

0. Set `data.py/RESNET_MAT_PATH` pointing to the attached `.mat` pretrained weight. (`resource/ResNet-50-params.mat` in this case)
0. Put training images in a custom directory `TRAINING_DIR`
0. Run `./resnet_main.py --command train --training_dir TRAINING_DIR --working_dir WORKING_DIR` to train with a pre-scheduled scheme, storing reports and models in `WORKING_DIR`. `TRAINING_DIR` points to a directory containing the images. The script traverses through all images (but limited to `.jpg` due to TensorFlow infrastructure), grouped in classes by the immediate subdirectory names. For example, we may have several images shown below, making a total of three classes. Training should be finished for 10000 iterations within 6 hours with a decent machine.
```
TRAINING_DIR/class-0/foo/aaaaa.jpg
TRAINING_DIR/class-0/foo/bar/bbbbb.jpg
TRAINING_DIR/class-1/ccccc.jpg
TRAINING_DIR/class-2/bababa/dadada/rarara/ddddd.jpg
```

Detailed usage of `resnet_main.py` is illustrated below, but except for fiddling with the parameters for fun, you should not touch it.

```
usage: resnet_main.py [-h] [--batch_size BATCH_SIZE]
                      [--num_test_crops NUM_TEST_CROPS] [--num_gpus NUM_GPUS]
                      [--train_iteration TRAIN_ITERATION]
                      [--lr_half_per LR_HALF_PER] [--lr LR]
                      [--lr_slow LR_SLOW] [--weight_decay WEIGHT_DECAY]
                      [--command COMMAND] [--log_file LOG_FILE]
                      [--working_dir WORKING_DIR] [--train_dir TRAIN_DIR]
                      [--test_dir TEST_DIR] [--test_attrs TEST_ATTRS]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        mini-batch size (over all GPUs). [default: 64]
  --num_test_crops NUM_TEST_CROPS
                        random cropping at testing time. [default: 1]
  --num_gpus NUM_GPUS   number of GPUs used. [default: 1]
  --train_iteration TRAIN_ITERATION
                        number of iterations if used with --command=train.
                        [default: 10000]
  --lr_half_per LR_HALF_PER
                        learning rate half life. [default: 1500]
  --lr LR               initial learning rate. [default: 1e-1]
  --lr_slow LR_SLOW     relative learning rate for shallow layers. (so its
                        learning rate would be (lr * lr_slow). [default: 0.0]
  --weight_decay WEIGHT_DECAY
                        weight decay for convolutional kernels. [default: 0.0]
  --command COMMAND     "train" to run the pre-schuduled training scheme,
                        "test" to run tests and generate log file, and "none"
                        to do nothing. [default: none]
  --log_file LOG_FILE   log file name for test, used with --command=test.
                        [default: test_log.csv]
  --working_dir WORKING_DIR
                        working directory for saving logs and models.
                        [default: /tmp]
  --train_dir TRAIN_DIR
                        directory for files to be trained.
  --test_dir TEST_DIR   directory for files to be tested, leave blank for
                        validation split. [default: none]
  --test_attrs TEST_ATTRS
                        names for test logging.
```
