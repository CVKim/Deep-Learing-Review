
# [Paper Review] Blended Diffusion for Text-driven Editing of Natural Images

[Paper] https://omriavrahami.com/blended-diffusion-page/

[Github] https://github.com/omriav/blended-diffusion

Introduction
---

이미지 생성 분야에서 인상적인 결과를 남긴 것은 GAN Model이다.
하지만 실제 이미지를 조작하기 위해 먼저 GAN의 Latent Space단으로 변경 해야 하는 부분이 존재 했으며,
많은 연구가 이루어지고 있지만, Reconstruction에 대한 정확성과 Editing 능력 사이의 Trade-off가 존재 한다.
그리고 이미지 조작을 특성 이미지 영역으로 제한 하는 것도 문제가 야기 되었다.

본 논문에선 Natural Text Guidance를 사용하여 일반적인 실제 이미지의 영역에 대한 편집을 위한 새로운 접근 방법을 제시했다.

![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/e8bf4f87-43a5-4673-912e-24ad84425716)

논문의 소개 부분을 다시 한 번 정리해보면,

2개의 Pre-train model을 사용하며, 하나는 DDPM이고 다른 하나는 CLIP이다. DDPM은 최근 SOTA GAN보다 나은 이미지 생성 품질을 보여주는
확률적 생성 모데이고, 본 논문의 저자들은 DDPM을 backbon으로 사용하여 자연스러운 결과를 보장 할 수 있다고 말한다.
CLIP은 인터넷에서 수집된 약 4억 개의의 Image + Text Pair로 Training되어 Image와 Text에 대한 풍부한 Share Embedding Space를 학습하게 된다.

저자들은 CLIP를 사용하여 유저가 제공한 Text Prompt를 일치되도록 이미지를 조작하는 Guide를 제공한다.
DDPM과 CLIP의 naive한 조합이 이미지 배경을 보존에 대한 성능 보징이 안 되지만, 각 Diffusion step마다 CLIP-Guided diffusion latent를 noise가 적절하게 추가된
입력 이미지와 혼합하는 diffusion process를 활용하여, 입력의 변경되지 않은 부분과 일관성 있는 퀄리티 있는 생성 이미지를 생성 할 수 있다고 말한다.

또한 diffusion process의 각 Step에서 extending augmentation를 사용하면 적대적 결과가 감소하며, 추가 학습 없이 사전 학습된 DDPM과 CLIP 모델을 활용한다.

---
