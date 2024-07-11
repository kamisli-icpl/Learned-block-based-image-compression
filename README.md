# End-to-End Learned Block-Based Image Compression with Block-Level Masked Convolutions and Asymptotic Closed-Loop Training

## Abstract
Learned image compression research has achieved state-of-the-art compression performance with auto-encoder based neural network architectures, where the image is mapped via convolutional neural networks (CNN) into a latent representation that is quantized and processed again with CNN to obtain the reconstructed image. CNN operate on entire input images. On the other hand, traditional state-of-the-art image and video compression methods process images with a block-by-block processing approach for various reasons. Very recently, work on learned image compression with block based approaches have also appeared, which use the auto-encoder architecture on large blocks of the input image and introduce additional neural networks that perform intra/spatial prediction and deblocking/post-processing functions. This paper explores and proposes an alternative learned block-based image compression approach in which neither an explicit intra prediction neural network nor an explicit deblocking neural network is used. A single auto-encoder neural network with block-level masked convolutions is used and the block size is much smaller (8x8). By using block-level masked convolutions, each block is processed using reconstructed neighboring left and upper blocks both at the encoder and decoder. Hence, the mutual information of adjacent blocks is exploited during compression and each block is reconstructed using neighboring blocks, resolving the need for explicit intra prediction and deblocking neural networks. Since the explored system is a closed-loop system, a special optimization procedure, the asymptotic closed-loop design, is used with standard stochastic gradient descent based training. The experimental results indicate competitive image compression performance.

## Citation
    @article{kamisli2022end,
        title={End-to-End Learned Block-Based Image Compression with Block-Level Masked Convolutions and Asymptotic Closed Loop Training},
        author={Kamisli, Fatih},
        journal={arXiv preprint arXiv:2203.11686},
        year={2022}
    }


## Installation
0) Requirements are the same as for CompressAI (https://github.com/InterDigitalInc/CompressAI)
1) Create a project directory in your machine:  
    ```bash
    mkdir lbbic
    ```
2) Install CompressAI in the project directory using 'From source' directions (https://github.com/InterDigitalInc/CompressAI)  
    ```bash
    cd lbbic
    python3 -m venv .venv
    source .venv/bin/activate
    git clone https://github.com/InterDigitalInc/CompressAI compressai
    cd compressai
    pip install -U pip && pip install -e .
    ```
3) Ensure that CompressAI works by evaluating a pre-trained model:  
    ```bash
    python3 -m compressai.utils.eval_model pretrained /path/to/images/folder/ -a mbt2018 -q 1,8
    ```
4) Clone this repository in the project directory such that compressai and this project (LearnedCompressionV3) are in side-by-side directories:  
    ```bash
    cd ..
    git clone https://github.com/kamisli-icpl/Learned-block-based-image-compression.git LearnedCompressionV3
    ```
5) Install the requirements for this project inside the virtual environment:  
    ```bash
    pip install -r requirements.txt
    ```


## Compression/decompression (inference) with the models in the paper
1) .

## Training
1)

## Compression/Decompression
1) 
    

