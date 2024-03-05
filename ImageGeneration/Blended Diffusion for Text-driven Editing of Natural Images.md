
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


Application
---


Text-driven object editingPermalink

![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/e263ff00-e6cf-4ef6-af67-f6aef6c31511)

Background replacementPermalink

![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/e02eb4e7-7de6-46f1-9a92-5055d7587c60)

Scribble-guided editingPermalink

![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/5724a499-ef7e-43e6-8678-6d7b8fbd9087)

Text-guided image extrapolationPermalink

![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/7ec2a35e-860d-4830-b0a0-ac69223d1075)

텍스트 설명으로 왼쪽은 “hell”이 주어졌고 오른쪽은 “heaven”이 주어졌다.




Limitations
---

다른 DDPM 모델들과 같이 가장 큰 한계점은 Inference time이 오래 걸린다는 문제가 존재하며, 논문 발표 당시 GPU로 이미지 한 장을 연산 하는 데,
약 30초가 걸린다고 한다. 본 논문에서는 여러 샘플을 생성하고 순위를 매겨 가장 높은 순위를 기록한 샘플을 선택하기 때문에 실시간 Application과 모바일 기기와 같은 약한 End-user device에 적용하는 데에는 한계가 보인다.

또한 이미지의 전체 컨텍스트가 아닌 편집된 영역 즉, Mask 영역에 대해서만 순위를 매기기 때문에 랭킹 시스템이 완벽하지 않다.

마지막으로 본 논문은 CLIP 기반으로 하기 때문에 CLIP의 약점과 편향을 모두 가지고 있으며, CLIP의 타이포그래피 공격에 취약하여 손글씨 사진만으로도 모델을 속일 수 있다고 한다. 이러한 현상이 아래 사진 2번에서 볼 수 있으며, 아래 예시에서는 "rubber toy"를 생성하라고 했는 데, "rubber"라는 단어 자체를 생성하는 문제가 발생 되었다.

![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/ff479abd-3857-4ebb-bc2d-9d04e554be38)
