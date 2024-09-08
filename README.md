# Privacy-Preserving Neural Style Transfer 

This repo contains the implementation of the [Neural Style Transfer](https://arxiv.org/pdf/1508.06576) algorithm in a privacy-preserving way. We use FHE(Fully Homomorphic Encryption) scheme to perform image style-transfer on encrypted images. This repos consists of - 
* **checkpoints/Mosaic-last_checkpoint.pth :-** A pre-trained model for a style image(Mosaic). The model is trained on CIFAR-100 dataset. It takes input an image and returns a stylized Mosaic image.
* **cifar-dataset/cifar100 :-** Contains 8 CIFAR-100 images that have been used to generate results inside notebook.
* **style/Mosaic.jpg :-** The style image used to train the model.
* **Neural_Style_Transfer_Metrics.ipynb :-** A notebook that contains the analysis of different metrics for stylized images.
* **Private Style Transfer.ipynb :-** The notebook containing details about privacy-preserving image style transfer.

We have created a FHE-compatible style transfer model from a pre-trained model using Concrete-ML. Concrete-ML uses TFHE encryption scheme internally.