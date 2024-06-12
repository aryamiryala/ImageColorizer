# Image Colorization

## CSE 144: Applied Machine Learning: Deep Learning

### Authors: Aneesh Thippa, Arya Miryala, Matthew Lo, Shaan Mistry
### Date: 06.11.2024
### Professor: Yi Zhang

---

## Abstract

This project employs deep learning techniques to develop an image colorizer, transforming grayscale images into colorized versions. Utilizing Hugging Face for dataset deployment, the Stability AI Stable Diffusion model, and fine-tuning with ControlNet, we enhance the model's performance, ensuring that the colorized images align closely with desired color schemes. This project demonstrates the effectiveness of combining state-of-the-art diffusion models with fine-tuning techniques in image colorization tasks.

## Introduction

Image colorization enhances the visual appeal of images and has practical applications such as restoring old photographs and improving medical imaging. The challenge lies in generating realistic and contextually accurate colors. Our approach leverages the Stability AI Stable Diffusion model and ControlNet to address this challenge. Using Hugging Face for dataset management, we aim to produce visually appealing and contextually accurate colorized images.

## Methodology

### 1. Data Preparation & Preprocessing

- **Datasets**: COCO and Kaggle
- **Deployment**: Hugging Face Hub
- **Preprocessing**: Conversion to grayscale, resizing to 512x512 pixels, and creation of text prompts

### 2. Model Selection

- **Initial Approach**: Generative Adversarial Networks (GANs)
- **Final Approach**: Stable Diffusion model for its stability and consistency

### 3. Training

- **Platform**: Google Cloud Platform with Nvidia L4 GPU
- **Models Trained**: Three models with varying parameters and datasets
- **Final Model**: Trained on 50,000 images for 3 epochs, resulting in satisfactory colorization

### 4. Cielab Color Space

Combined the L* channel of the input image with the (a* b*) channels of the output image to preserve light levels and improve clarity.

## Results

### Evaluation Metrics

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Higher values indicate better quality.
   - Results improved with detailed prompts.

2. **SSIM (Structural Similarity Index)**
   - Measures structural information, luminance, and contrast.
   - Results improved with detailed prompts.

3. **CIEDE2000**
   - Measures perceived color differences.
   - Results improved with detailed prompts.


## Limitations

### 1. Storage Limitations

Limited dataset size and need for checkpoint training.

### 2. Computational Power

Limited access to powerful GPUs affected training efficiency.

### 3. Time Constraints

Long training times delayed progress and iterations.

## Conclusion

Our project successfully developed an image colorizer using Stability AI and ControlNet. This approach highlights the potential of deep learning techniques for practical applications. Future work includes further training and incorporating object recognition models for more accurate colorizations.

## Team Member Contributions

- **Aneesh Thippa & Shaan Mistry**: Model training, literature review, parameter tuning, inference.
- **Arya Miryala**: Dataset creation, Hugging Face deployment, preprocessing, model training.
- **Matthew Lo**: Evaluation algorithms, metric research, initial preprocessing.

## References

- [ControlNet Documentation](https://huggingface.co/docs/diffusers/en/training/controlnet)
- [Isola, P., et al., 2018](https://arxiv.org/abs/1611.07004)
- [Shu-Yu Chen, 2022](https://www.sciencedirect.com/science/article/pii/S2468502X22000389)
- [Sortino, R., 2023](https://medium.com/@rensortino/colorizenet-stable-diffusion-for-image-colorization-bdc9c35121fa)
- [Zhang, R., et al., 2016](https://arxiv.org/abs/1603.08511)

---

