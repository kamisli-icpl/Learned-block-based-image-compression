# End-to-End Learned Block-Based Image Compression with Block-Level Masked Convolutions and Asymptotic Closed-Loop Training

## Abstract
Learned image compression research has achieved state-of-the-art compression performance with auto-encoder based neural network architectures, where the image is mapped via convolutional neural networks (CNN) into a latent representation that is quantized and processed again with CNN to obtain the reconstructed image. CNN operate on entire input images. On the other hand, traditional state-of-the-art image and video compression methods process images with a block-by-block processing approach for various reasons. Very recently, work on learned image compression with block based approaches have also appeared, which use the auto-encoder architecture on large blocks of the input image and introduce additional neural networks that perform intra/spatial prediction and deblocking/post-processing functions. This paper explores and proposes an alternative learned block-based image compression approach in which neither an explicit intra prediction neural network nor an explicit deblocking neural network is used. A single auto-encoder neural network with block-level masked convolutions is used and the block size is much smaller (8x8). By using block-level masked convolutions, each block is processed using reconstructed neighboring left and upper blocks both at the encoder and decoder. Hence, the mutual information of adjacent blocks is exploited during compression and each block is reconstructed using neighboring blocks, resolving the need for explicit intra prediction and deblocking neural networks. Since the explored system is a closed-loop system, a special optimization procedure, the asymptotic closed-loop design, is used with standard stochastic gradient descent based training. The experimental results indicate competitive image compression performance.

## Citation
    @article{kamisli2024end,
      title={End-to-end learned block-based image compression with block-level masked convolutions and asymptotic closed-loop training},
      author={Kamisli, Fatih},
      journal={Multimedia Tools and Applications},
      pages={1--23},
      year={2024},
      publisher={Springer}
    }



## Installation
0) Requirements are the same as for CompressAI (https://github.com/InterDigitalInc/CompressAI)
1) Create a project directory on your machine:  
    ```bash
    mkdir lbbic
    ```
2) Install CompressAI in the project directory using 'From source' directions (https://github.com/InterDigitalInc/CompressAI)  
    ```bash
    cd lbbic
    python3 -m venv .venv
    source .venv/bin/activate
    git clone git@github.com:InterDigitalInc/CompressAI.git compressai
    cd compressai
    pip install -U pip && pip install -e .
    ```
3) Ensure that CompressAI works by evaluating a pre-trained model:  
    ```bash
    python3 -m compressai.utils.eval_model pretrained /path/to/images/folder/ -a mbt2018 -q 1,8 --cuda
    ```
4) Clone this repository in the project directory such that compressai and this project (LearnedCompressionV3) are in side-by-side directories:  
    ```bash
    cd ..
    git clone git@github.com:kamisli-icpl/Learned-block-based-image-compression.git LearnedCompressionV3
    ```
5) Install the requirements for this project inside the virtual environment:  
    ```bash
    cd LearnedCompressionV3
    pip install -r requirements.txt
    ```


## Compression/decompression (inference) with the models in the paper
1) Download the pre-trained model weights for the results in the paper from Releases v0.1  
(https://github.com/kamisli-icpl/Learned-block-based-image-compression/releases/tag/v0.1)
and copy the downloaded `pth.tar` files into the correspondig `checkpoints` folders under the experiments folder in your project directory, such as  
`lbbic/LearnedCompressionV3/experiments/blkbsdimgcomp_B8_KS3111_N768M96_v9/exp_117.045/checkpoints/` 
2) To run compression/decompression with the 8x8 block low-rate model for the lowest $\lambda$, run:
    ```bash
    python3 main.py configs/blkbsdimgcomp_B8_lowrate.json
    ```
    Note, that you should first update the dataset paths at the bottom of the json file. During compression/decompression, the png files in the valid_data path will be compressed/decompressed, however, the other paths still need to point to valid paths. 
3) To run compression/decompression with the other models and/or $\lambda$ values, pick the corresponding json file under the `configs` directory, update the `lambda_` parameter (if desired) and the dataset paths in the json file and run the above command with the json file. If you want to update the lambda_ parameter, you must choose a value that corresponds to the value in a `exp_` folder name under the `experiments/"multi_exp_name"` directory. 


## Training
1) To train a model, pick one of the provided json files and make a copy. In the new json file, set `mode="train_all_acl"` and update many other parameters, such as  `multi_exp_name, mode, block_size, N, M, lambda_, acl_itr0_rdloss_threshold` and dataset paths to appropriate values. 
2) Run:  
    ```bash
    python3 main.py configs/new-json-file.json
    ```


