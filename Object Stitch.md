[Paper - ObjectStitch : Generative Object Compositing](https://arxiv.org/abs/2212.00932 "arxiv")

# ObjectStitch: Generative Object Compositing (CVPR 2023)

**Contributions**
- 다양한 compositing 작업을 다룰 수 있게 최초로 diffusion base framework이다. viewpoint, geometry, lighting and shadow까지 고려함
- 이 논문에서 선형적인 multi-modal embedding을 활용하는 content adaptor module을 제안. 이를 통해 diffusion model대한 image guidance 가능해짐.
- task 특정 annotations 없이 self-supervised 방식으로 train되며, 생산의 정확도를 향상시키기 위해 augmentation 방식을 이용함
- high-resolution real world data

**Image Compositing**
- Geometric correction
  - ST-GAN
- Image harmonization
- Shadow generation
- GCE-GAN
  - Geometry
  - Light

**Guided Image synthetics**
- Stable Diffusion
- GLIDE
- SDEdit
- SDG

**Architecture**
![Alt text](/Img/image-20240128-062535.jpg "Optional title")

**Conclusion**
- object를 합성하는 데 있어, diffusion base model로 첫 시도
- text to image generation network의 content adaptor module
- fully self-supervised framework

**Limitations & Future**
- 이 논문은 초창기 단계, BG가 제한적임
- 마스터 전달력 용량의 한계 때문에 분명한 상실이 아쉽지만, 전체를 바꾸기에 한계가 분명
- Artifacts가 존재 하지만 한계를 넘어서 나올 것이라고 믿음
