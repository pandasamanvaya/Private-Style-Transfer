# Privacy-Preserving Neural Style Transfer 

This repo contains the implementation of the [Neural Style Transfer](https://arxiv.org/pdf/1508.06576) algorithm in a privacy-preserving way. We use FHE(Fully Homomorphic Encryption) scheme to perform image style-transfer on encrypted images. This repository consists of - 
* **checkpoints/Mosaic-last_checkpoint.pth :-** A pre-trained model for a style image(Mosaic). The model is trained on CIFAR-100 dataset. It takes input an image and returns a stylized Mosaic image.
* **cifar-dataset/cifar100 :-** Contains 8 CIFAR-100 images that have been used to generate results inside notebook.
* **style/Mosaic.jpg :-** The style image used to train the model.
* **Neural_Style_Transfer_Metrics.ipynb :-** A notebook that contains the analysis of different metrics for stylized images.
* **Private Style Transfer.ipynb :-** The notebook containing details about privacy-preserving image style transfer.

We have created a FHE-compatible style transfer model from a pre-trained model using Concrete-ML. Concrete-ML uses TFHE encryption scheme internally. To learn the details about the implementaion of privacy-preserving style transfer, please refer to the [Private Style Transfer](Private Style Transfer.ipynb) notebook.

Pre-processing and post-processing steps - 
* We normalize the content(input) image as a pre-processing step. Similarly, we have to denormalise the output(result) image to obtain the final stylized image. 
* Since our model is tranined on CIFAR-100, the input image needs to be resized to 32X32.


Few of the decisions taken to solve image style transfer problem within FHE/Concrete ML constraints- 
* **Network Architecture Changes:-**  The size of the layers has been made smaller. The size of ResidualBlock has been changed from 128 to 32. The size of ConvBlocks at the start and end have also been reduced i.e. (3,32) -> (3, 8), (32, 64) -> (8, 16), (64, 128) -> (16, 32). The kernel_size in the first and last layers has also been reduced from 9 to 3. These changes helped to reduce the no.of parameters from ~1.6 million in the original network to ~0.1 million(~105 thousand).
* **Padding Changes:-** In the original paper, it is recommended to use reflection padding for better results. However, support for reflection padding is not present in Concrete ML. So, we used constant padding instead. Constant padding introduces an edge/border effect while working with convolutional layers. But, since we trained our network on tiny images, this effect was not evident.
* **Normalization Changes:-** In the original paper, it is recommended to use Instance normalization for better results. However, support, for instance, normalization is not present in Concrete ML. So, we used Batch normalization instead.
* **Upsampling changes:-** In the penultimate layer and the layer before that, we need to upsample the input before applying ConvBlock. However, native support for upsampling(interpolation or Resize operator in ONNX graph) is not present in Concrete ML. So, we had to add custom logic for upsampling(nearest neighbour interpolation) which can be compiled using Concrete ML.

### Hack used to fix to some bug in Concrete ML/ONNX that can't infer size of `x` dynamically 
While compiling a torch model using Concrete-ML, if we try to infer the shape of the input `x` and use it as a value, we encounter a weird issue. Somehow, after ONNX export of the model, the value of `x.shape` is not properly propagated. To fix this issue, we had to manually hard-code variables that use dimension of the input. We have done this for two layers of our model that require upsampling.Here is the code snippet that does the fix - 
```
if x.shape[1] == 32:
    batch_size, channels, height, width = (1, 32, 8, 8)
else:
    batch_size, channels, height, width = (1, 16, 16, 16)
```