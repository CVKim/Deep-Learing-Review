
[In-Domain GAN Inversion for Real Image Editing](https://arxiv.org/abs/2004.00049)  
[Project](https://genforce.github.io/idinvert/)  
[Github](https://github.com/genforce/idinvert)

# Introduction
__Inverted code가 원래 GAN의 latent space에 적합한지  
Inverted code가 target 이미지를 의미론적으로 잘 표현하는지  
Inverted code가 GAN에서 학습한 지식을 재사용하여 이미지 편집을 지원하는지  
잘 학습된 GAN을 사용하여 임의의 이미지의 inverted code를 찾을 수 있는지__

## 🔖 전체 학습 과정은 아래와 같음
1. **Encoder에 의해 생서된 모든 latent code가 in-domain하기 위해, image spcae to latent space에 mapping 되도록 domain-guided encoder를 우선적으로 학습**
2. **이후, inverted code에 의해 semantic 속성이 달라지지 않고, pixel을 더 잘 보존? 재구성하기 위해 Encoder를 Regularizer로 하여 instance-level domain regularized optimization을 수행**
3. ㅇㅇ


## Reference
[idinvert](https://github.com/genforce/idinvert?tab=readme-ov-file)  
[In-Domain GAN Inversion for Real Image Editing](https://genforce.github.io/idinvert/)  
[Sample](https://github.com/abdulium/gan-inversion-stylegan2)  
[High-Fidelity GAN inversion for Image Attribute Editing 리뷰](https://www.youtube.com/watch?v=AL_vjJHGdUU)  
[Awesome-Inpainting-Tech](https://github.com/zengyh1900/Awesome-Image-Inpainting)
[Diverse Inpainting and Editing with GAN Inversion](https://openaccess.thecvf.com/content/ICCV2023/papers/Yildirim_Diverse_Inpainting_and_Editing_with_GAN_Inversion_ICCV_2023_paper.pdf)



