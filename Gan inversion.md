
# Introduction
__Inverted code가 원래 GAN의 latent space에 적합한지  
Inverted code가 target 이미지를 의미론적으로 잘 표현하는지  
Inverted code가 GAN에서 학습한 지식을 재사용하여 이미지 편집을 지원하는지  
잘 학습된 GAN을 사용하여 임의의 이미지의 inverted code를 찾을 수 있는지__

## 🔖 전체 학습 과정은 아래와 같음
1. **Encoder에 의해 생서된 모든 latent code가 in-domain하기 위해, image spcae to latent space에 mapping 되도록 domain-guided encoder를 우선적으로 학습**
2. **이후, inverted code에 의해 semantic 속성이 달라지지 않고, pixel을 더 잘 보존? 재구성하기 위해 Encoder를 Regularizer로 하여 instance-level domain regularized optimization을 수행**

# Experiments

GAN inversion 방식을 사용하여 이미지에 디펙트를 생성하는 실험을 설계하고 진행하는 과정은 다음과 같습니다:

## 1. Generator Freeze
- Generator의 모든 파라미터를 고정(freeze)합니다.

## 2. Latent z Instantiation
- Latent z를 파라미터로 instantiate한 후, 이를 옵티마이저에 넣어 학습 가능하도록 설정합니다.
  - 옵션으로, z 대신 w나 w+를 업데이트하여 로스를 최소화할 수 있습니다.

## 3. Loss Definition
로스를 최소화하기 위해 다음과 같은 항목을 정의합니다:
  - **Euclidean Distance for Binary Mask**:
    - 해당 z로부터 생성되는 binary mask와 타겟 binary mask 간의 유클리디안 거리를 계산합니다.
  - **Euclidean Distance for Images**:
    - 생성되는 헤이즐넛 이미지와 타겟 노멀 헤이즐넛 이미지 간의 유클리디안 거리를 계산합니다.
    - 중요: 생성할 디펙트 부분의 픽셀들은 이 로스 계산에서 제외합니다.
  - **Additional Perceptual Loss for Defects**:
    - 헤이즐넛 이미지 디펙트 위치의 픽셀들에 대해 perceptual loss를 적용할 수 있으며, 이는 distillation이나 실제 defect를 해당 위치에 stitching하여 줄일 수 있습니다.
    - 이 부분은 (b)와 (c)의 로스 계산 이후 추가 논의가 필요합니다.




## Reference
[idinvert](https://github.com/genforce/idinvert?tab=readme-ov-file)  
[In-Domain GAN Inversion for Real Image Editing](https://genforce.github.io/idinvert/)  
[In-Domain GAN Inversion for Real Image Editing](https://arxiv.org/abs/2004.00049)  
[Sample](https://github.com/abdulium/gan-inversion-stylegan2)  
[High-Fidelity GAN inversion for Image Attribute Editing 리뷰](https://www.youtube.com/watch?v=AL_vjJHGdUU)  
[Awesome-Inpainting-Tech](https://github.com/zengyh1900/Awesome-Image-Inpainting)
[Diverse Inpainting and Editing with GAN Inversion](https://openaccess.thecvf.com/content/ICCV2023/papers/Yildirim_Diverse_Inpainting_and_Editing_with_GAN_Inversion_ICCV_2023_paper.pdf)



