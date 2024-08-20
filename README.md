# 2D Differentiable Rendering Example

[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=matlab-deep-learning/2D-Differentiable-Rendering-Example)

*Keywords: differentiable rendering, autodifferentiation, differentiable rasterization, inverse rendering*

This repository provides an example of performing differentiable vector graphics rendering using MATLAB.

The example demonstrates how to use the differentiable rendering pipeline to approximate a target image with multiple vector shapes. By differentiating an image-based loss function with respect to the shapes' parameters—such as size, location, and color—you can iteratively adjust the shapes to closely match the target image.

![animation](example_animation.gif)

## What is Differentiable Rendering?
Differentiable rendering is a process that allows the computation of gradients through the rasterization pipeline. In traditional rasterization, vector graphics are converted into a raster image (a grid of pixels), but this process is typically non-differentiable due to the discontinuities in the rasterization process. Differentiable rasterization, on the other hand, introduces a smooth approximation to this process, enabling the computation of gradients with respect to the input parameters of the vector shapes.

Differentiable rasterization opens up new possibilities in various fields, such as computer graphics, machine learning, and computer vision. Some of the key benefits include:

- Optimization: By enabling gradient-based optimization techniques, differentiable rasterization allows for fine-tuning vector graphics to achieve specific visual goals.
- Inverse Graphics: It facilitates the reconstruction of vector graphics from raster images, which is useful in applications like image vectorization and graphic design.
- Learning-Based Methods: Differentiable rasterization can be integrated into neural networks, enabling end-to-end training for tasks that involve rendering, such as image synthesis and style transfer.

## Setup and Getting Started
To Run:
1. Open `differentiable_rendering_example.mlx` live script in MATLAB&reg;
2. Run the file.

### MathWorks Products (https://www.mathworks.com)

1. Requires MATLAB&reg; release R2024a or newer
2. [Deep Learning Toolbox&trade;](https://www.mathworks.com/products/deep-learning.html) 

## License

The license is available in the License.txt file in this GitHub repository.

## Community Support
[MATLAB Central](https://www.mathworks.com/matlabcentral)

Copyright 2024 The MathWorks, Inc.
