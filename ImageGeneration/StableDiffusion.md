
# Latent diffusion model 정리

## github
https://github.com/CompVis/latent-diffusion
## paper
https://arxiv.org/abs/2112.10752


# Introduction

Diffusion model은 이미지 합성과 super-resolution 분야에서 굉장히 좋은 성능을 보였다. 하지만, 이러한 모델들은 데이터의 감지할 수 없는 세부 정보를 모델링하는 데 과도한 자원을 소비하는 경향이 있다. DDPM(논문리뷰)의 재가중된 목적 함수는 초기 노이즈 제거 단계에서 적게 샘플링하여 이 문제를 해결하고자 하지만, 이러한 모델을 학습시키고 평가하는 과정은 여전히 계산적으로 까다롭다. 이러한 모델을 학습시키려면 방대한 컴퓨팅 리소스가 필요하며, 이미 학습된 모델을 평가하기 위해 동일한 모델 아키텍처를 여러 단계에 대해 순차적으로 실행하기 때문에 시간과 메모리가 많이 든다. 강력한 diffusion model에 접근성을 높이고 동시에 상당한 리소스 사용을 줄이려면 학습 및 샘플링 모두에 대한 계산 복잡성을 줄여야 한다. 따라서 diffusion model의 접근성을 높이기 위해서는 성능을 저하시키지 않으면서 계산 요구량을 줄이는 것이 핵심이다.

## 저자들의 접근 방식

저자들의 접근 방식은 픽셀 공간에서 이미 학습된 diffusion model의 분석으로 시작한다. 다른 likelihood 기반 모델과 마찬가지로 학습은 대략 두 단계로 나눌 수 있다.

1. **Perceptual compression**: high-frequency detail들을 제거하지만 의미(semantic)는 거의 학습하지 않는 단계
2. **Semantic compression**: 실제 생성 모델이 데이터의 의미론적(semantic) 구성과 개념적(conceptual) 구성을 학습하는 단계

따라서 저자들은 먼저 perceptual하게 동등하지만 계산적으로 더 적합한 space를 찾는 것을 목표로 하며 고해상도 이미지 합성을 위해 diffusion model을 학습시킨다.

먼저, 데이터 space와 perceptual하게 동일한 저차원 representational space로 보내는 autoencoder를 학습한다. 중요한 것은 학습된 잠재 공간에서 diffusion model을 학습시키므로 이전 모델들과 달리 과도한 space 압축에 의존할 필요가 없다는 것이다. 또한 감소된 복잡성으로 인해 네트워크를 한 번만 통과하여 latent space에서 효율적인 이미지 생성이 가능하다. 이 모델을 LDM(Latent Diffusion Models)이라고 한다.

이 접근 방식의 주목할만한 장점은 범용 autoencoding 단계를 한 번만 학습하면 되므로 여러 diffusion model 학습에 재사용하거나 완전히 다른 task를 탐색할 수 있다는 것이다. 이를 통해 다양한 image-to-image 및 text-to-image task를 위한 여러 diffusion model을 효율적으로 탐색할 수 있다. 후자의 경우 트랜스포머를 diffusion model의 UNet backbone에 연결하고 임의의 유형의 토큰으로 조건을 주는 아키텍처를 설계한다.

### 아래 그림은 학습된 모델의 rate-distortion trade-off를 보여준다
![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/ba5fb9c6-823e-455e-b31a-687bc8161c87)
****


### Architecture
![Architecture](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/d5000e9a-d721-4b09-aa37-b63255dca346)

# Latent Diffusion 모델 개요
![https://pitas.tistory.com/9 참고 blog](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/12724388-63e8-4703-a977-0b3bc4d07b2c)

Latent Diffusion 모델은 텍스트 프롬프트를 입력받아 해당 텍스트에 해당하는 이미지를 생성하는 과정을 거치는 딥러닝 아키텍처입니다.
이 모델은 크게 세 가지 주요 컴포넌트로 구성됩니다: 텍스트 인코더(Text Encoder), U-net, 그리고 변분 오토인코더(VAE). 각 컴포넌트는 이미지 생성 과정에서 특정 역할을 수행하며, 이들의 상호 작용을 통해 고해상도 이미지를 효율적으로 생성합니다.

## 1. 텍스트 인코더 (Text Encoder)
![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/14edda17-133f-4547-821b-96e8cb763b53)
### 역할
입력된 텍스트 프롬프트를 숫자로 변환하는 토큰화 과정을 수행하고, 이를 텍스트 임베딩(text embedding) 형태의 latent vector로 만듭니다. 이 임베딩은 이미지 생성 과정에서 조건으로 사용됩니다.

### 구성
CLIP과 같은 모델을 사용하여, 언어를 이해하고 관련된 이미지 특성을 추출할 수 있는 임베딩을 생성합니다.

## 2. U-net (+ Scheduler)
![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/df214cbb-d99d-4eda-9333-e5766a45c274)
### 역할
텍스트 인코더에서 생성된 텍스트 임베딩을 기반으로 조건화된 이미지 생성 과정을 관리합니다. 이 과정에서는 random latent vector를 여러 번 반복하여 denoise(노이즈 제거)합니다. 이는 diffusion 모델의 기본 원리에 따른 것입니다.

### Scheduler의 역할
반복 과정에서 사용할 노이즈의 세기, 종류 및 확률적 접근 방식을 결정합니다. 다양한 스케줄러(예: DDPM, DDIM, PNDM)는 이 과정의 세부적인 조정을 담당합니다.

## 3. VAE (Variational Auto Encoder/Decoder)
![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/5ae4ae76-ed7f-4961-a5f6-be75e4730541)
![image](https://github.com/CVKim/Deep-Learing-Review/assets/90014998/779b7ab4-1132-4203-ad66-52071160ca17)
### 역할
U-net에서 생성된 저해상도 latent vector를 고해상도 이미지로 복원합니다. Encoder는 입력 데이터의 특징을 추출하여 latent vector로 변환하고, Decoder는 이 latent vector를 사용하여 원본 데이터를 복원하는 과정을 담당합니다.

### Latent Diffusion 모델과의 연계
Latent Diffusion 모델에서는 U-net에서 생성된 latent vector가 VAE를 통해 고해상도 이미지로 복원됩니다. 이 과정은 전통적인 Diffusion 모델과 비교하여 효율적으로 고해상도 이미지를 생성하는 데 도움을 줍니다.

# Latent Diffusion 모델의 특징

- **효율성**: Latent 공간에서의 작업을 통해, 모델은 더 낮은 차원에서 이미지의 복잡한 패턴을 학습하고 처리할 수 있어, 고해상도 이미지 생성을 위한 계산 비용을 줄일 수 있습니다.
- **유연성**: 다양한 텍스트 프롬프트에 대응하여 관련 이미지를 생성할 수 있으며, 스케줄러를 통해 생성 과정의 세부적인 조정이 가능합니다.
- **고해상도 이미지 생성**: VAE를 통한 복원 과정을 통해, 저해상도의 latent vector에서 고해상도 이미지를 효과적으로 생성할 수 있습니다.
- ** Latent Diffusion 모델은 이러한 고유한 구성 요소들을 통합하여, 텍스트 기반 프롬프트에서 시작하여 고품질의 이미지를 생성하는 강력한 프레임워크를 제공합니다. 이 모델은 컴퓨터 비전과 자연어 처리의 최신 기술을 결합하여, 다양한 창작물과 응용 분야에서 활용될 수 있는 높은 유연성과 효율성을 보여줍니다.

## LDM의 한계점

LDM(Latent Diffusion Models)은 픽셀 기반의 방식에 비해 계산 요구량을 크게 줄이지만, 샘플링 속도는 여전히 GAN(Generative Adversarial Networks)보다 느립니다. 또한, 높은 정밀도가 필요한 경우 LDM의 사용은 의문의 여지가 있습니다.

- **품질 손실**: `f=4` autoencoding model에서는 이미지 품질 손실이 매우 적지만, 픽셀 space에서 세밀한 정확도가 필요한 작업의 경우 재구성 기능이 병목될 수 있습니다.
- **Super-resolution 모델의 제한**: 저자들은 super-resolution 모델이 이미 이 점에서 어느 정도 제한되어 있다고 생각합니다. 특히, 고해상도 이미지를 생성할 때 필요한 높은 정밀도를 달성하기 어려울 수 있습니다.

이러한 한계에도 불구하고, LDM은 여전히 다양한 이미지 생성 작업에 유용한 도구로 간주됩니다. 그러나 사용 사례에 따라 이러한 제한 사항을 고려할 필요가 있습니다.



## reference

https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ldm/
