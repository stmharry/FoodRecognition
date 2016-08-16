# SOP for Building Residual Network-50 for Food Classification

0. Do not plug GPU
0. Install `Ubuntu 16.04 LTS`
(If upgraded from 14.04, see [here](https://www.digitalocean.com/community/tutorials/how-to-upgrade-to-ubuntu-16-04-lts))
0. Install `nvidia-367` graphics driver, and then plug GPU back

    ```bash
    sudo apt-add-repository ppa:graphics-drivers/ppa
    sudo apt-get update
    sudo apt-get install nvidia-367
    ```
0. Install `CUDA 8.0` without installing graphics driver (download [here](https://developer.nvidia.com/cuda-toolkit), use `.run` to optionally disable installing the graphics driver), and install patch for `gcc 5.4`

    ```bash
    sudo sh cuda_8.0.27_linux.run --silent --toolkit --override
    sudo sh cuda_8.0.27.1_linux.run --silent --accept-eula
    ```
0. Install `cuDNN 5.1.5` (download [here](https://developer.nvidia.com/cudnn))

    ```bash
    sudo tar -zxvf cudnn-8.0-linux-x64-v5.1.tgz --directory /usr/local/
    ```
0. `apt-get` dependencies

    ```bash
    sudo apt-get install python-pip python-dev python-wheel python-numpy git zlib1g-dev swig imagemagick
    ``` 
    
0. `pip` dependencies

    ```bash
    pip install scipy
    ```  
0. Install `JDK 8`

    ```bash
    sudo add-apt-repository ppa:webupd8team/java
    sudo apt-get update
    sudo apt-get install oracle-java8-installer
    ```
0. Install Bazel (follow [here](http://www.bazel.io/docs/install.html))
    
    ```bash
    wget https://github.com/bazelbuild/bazel/releases/download/0.3.0/bazel-0.3.0-installer-linux-x86_64.sh
    chmod +x bazel-0.3.0-installer-linux-x86_64.sh
    ./bazel-0.3.0-installer-linux-x86_64.sh --user
    ```
0. Install TensorFlow from source (follow [here](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#installing-from-sources))

    ```bash 
    git clone https://github.com/tensorflow/tensorflow 
    cd tensorflow
    git checkout ea9e00a630f91a459dd5858cb22e8cd1a666ba4e
    git pull
    ./configure [CUDA: 8, cuDNN: 5.1.5, compute capability: 6.1]
    bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    pip install /tmp/tensorflow_pkg/tensorflow-* [the name depends]
    ```

0. Clone project repository

    ```bash
    git clone http://dev.2bite.com:14711/stmharry/ResidualNetwork.git
    ```

# Network Training and Testing

### Important Environment Variables
0. `TRAIN_DIR`: points to a directory containing the images. The script traverses through all images (but limited to `.jpg` due to TensorFlow infrastructure), grouped in classes by the immediate subdirectory names. For example, we may have several images shown below, making a total of three classes.

    ```
    TRAIN_DIR/noodle/noodle/12784128_207636649590235_1679896454_n.jpg
    TRAIN_DIR/noodle/noodle/12783295_1046368188759598_2085538866_n.jpg
    TRAIN_DIR/noodle/noodle/12750294_476618115869897_698948571_n.jpg
    TRAIN_DIR/hamburger/漢堡F/10598203_188236178197767_960837616_n.jpg
    TRAIN_DIR/hamburger/漢堡F/12353398_1235016886514606_1440304590_n.jpg
    TRAIN_DIR/hamburger/漢堡F/12501907_1690013094589276_1866062794_n.jpg
    TRAIN_DIR/seafood/seafood/10838352_1580574975598436_1902676267_n.jpg
    TRAIN_DIR/seafood/seafood/10723687_1684791655143462_1717926865_n.jpg
    TRAIN_DIR/seafood/seafood/seafood_shrimp/12328493_531414733706067_178747078_n.jpg
    TRAIN_DIR/seafood/seafood/seafood_shrimp/1516706_570560493110098_276822549_n.jpg
    TRAIN_DIR/seafood/seafood/salmon/12748472_1105513629515405_985511985_n.jpg
    TRAIN_DIR/seafood/seafood/salmon/12748353_1549746635354356_870418168_n.jpg
    ```
    
0. `TEST_DIR`: contains all images for training, and the script as well traverses through the folder recursively.

0. `WORKING_DIR`: stores reports and models.

0. `LOG_FILE`: file name for testing output.

### Important In-script Variables
0. `data.py/RESNET_MAT_PATH`: points to the attached `.mat` pretrained weight. (`resource/ResNet-50-params.mat` in this case)

### Training
Run 

```bash
./resnet_main.py 
    --command train 
    --train_dir TRAIN_DIR 
    --working_dir WORKING_DIR
```
to train with a pre-scheduled scheme.  Training should be finished for 10000 iterations within 6 hours with a decent machine.

### Monitoring
In `tensorflow`, we can monitor the training progress with `tensorboard` through

```bash
tensorboard
    --logdir WORKING_DIR
```

Then, use browser to view `http://localhost:6006` to inspect training progress.

### Testing
Run

```bash
./resnet_main.py
    --command test
    --train_dir TRAIN_DIR
    --test_dir TEST_DIR
    --working_dir WORKING_DIR
    --log_file LOG_FILE
    --num_test_crops 4
```
to output a `.csv` file in the following format (the encoding below is messed up, but not problematic)

```
key,pred_name
TEST_DIR/Kkperson Hsiao_銋暸__葛撘_璅_dish/12a2eaee3a8c420086a2214e4b4e28a4.jpg,street_food
TEST_DIR/Kkperson Hsiao_銋暸__葛撘_璅_dish/90fc78c3e3884489acf7397721edf484.jpg,street_food
TEST_DIR/Kkperson Hsiao_銋暸__葛撘_璅_dish/956623d753354e909a11953f0e299103.jpg,street_food
...
```

### Debugging
Detailed usage of `resnet_main.py` is illustrated below, but except for fiddling with the parameters for fun, you should not touch it.

```
usage: resnet_main.py [-h] [--batch_size BATCH_SIZE]
                      [--num_train_pipelines NUM_TRAIN_PIPELINES]
                      [--num_test_pipelines NUM_TEST_PIPELINES]
                      [--num_test_crops NUM_TEST_CROPS] [--num_gpus NUM_GPUS]
                      [--train_iteration TRAIN_ITERATION]
                      [--lr_half_per LR_HALF_PER]
                      [--subsample_ratio SUBSAMPLE_RATIO] [--lr LR]
                      [--lr_slow LR_SLOW] [--weight_decay WEIGHT_DECAY]
                      [--command COMMAND] [--log_file LOG_FILE]
                      [--working_dir WORKING_DIR] [--train_dir TRAIN_DIR]
                      [--test_dir TEST_DIR] [--test_attrs TEST_ATTRS]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        mini-batch size (over all GPUs). [default: 64]
  --num_train_pipelines NUM_TRAIN_PIPELINES
                        number of parallel training pipelines. [default: 8]
  --num_test_pipelines NUM_TEST_PIPELINES
                        number of parallel testing pipelines. [default: 1]
  --num_test_crops NUM_TEST_CROPS
                        random cropping at testing time. [default: 1]
  --num_gpus NUM_GPUS   number of GPUs used. [default: 1]
  --train_iteration TRAIN_ITERATION
                        number of iterations if used with --command=train.
                        [default: 10000]
  --lr_half_per LR_HALF_PER
                        learning rate half life. [default: 1500]
  --subsample_ratio SUBSAMPLE_RATIO
                        subsample ratio for training images. [default: 1.0
  --lr LR               initial learning rate. [default: 1e-1]
  --lr_slow LR_SLOW     relative learning rate for shallow layers. (so its
                        learning rate would be (lr * lr_slow). [default: 0.0]
  --weight_decay WEIGHT_DECAY
                        weight decay for convolutional kernels. [default: 0.0]
  --command COMMAND     "train" to run the pre-schuduled training scheme,
                        "test" to run tests and generate log file, and "none"
                        to do nothing. [default: none]
  --log_file LOG_FILE   log file name for test, used with --command=test.
                        [default: /tmp/test_log.csv]
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
