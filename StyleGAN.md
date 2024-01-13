GAN Summary

Generator와 Discriminator 2 network로 구성된 Generatived Model

Objective function을 통해 Generator는 Image Distribution을 학습

Discriminator - maximum(Real 1  ~ Fake : 0), Generator - minimum

Generator는 목적은 Random한 Noise로부터 주어진 잠재 변수 z를 입력으로 받아 실제 데이터와 유사한 분포를 가지는 이미지를 생성하는 것입니다. 이를 위해 Generator는 학습 과정에서 Discriminator를 속 일 수 있도록 생성된 이미지가 진짜 이미지와 유사하도록 학습됩니다. Generator의 목적 함수에서는 생성 된 이미지의 진짜와 가짜를 구분하는 Discriminator의 출력값을 1로 만들기 위해 D(G(z))의 값을 최대화 하도록 G를 학습합니다. 이 과정에서 G는 실제 데이터 분포 P_data(x)를 학습하게 됩니다.

Discriminator의 목적은 실제 데이터와 Generator가 생성한 가짜 데이터를 구분하는 것.Discriminator는 학습을 통해 실제 데이터의 분포 P_data(x) 와 Generator가 생성한 가짜 데이터의 분포 P_G(X) 를 구분할 수 있도록 학습됩니다. Discriminator의 목적 함수에서는 Discriminator의 출력값 D(x)를 1로 만들기 위해 실제 데이터 P_data를 입력으로 할 때 D(x) 값을 최대화하고, Generator가 생성한 가짜 데이터 G(z)를 입력으로 할 때는 D(G(z)) 값을 최소화하도록 D를 학습합니다. 이 과정에서 D는 실제 데이터와 가짜 데이터를 정확하게 분류하는 것이 목적

Case #1 : D(G(Z)) = 1일 경우 : 완벽히 속일 수 있는 경우 i. Generator는 Discriminator를 속이기 위해 가짜 데이터를 생성해야 합니다. 따라서 D(G(z))는 1에 가까 워져야 합니다. 목적함수에서 D(G(z)) 값을 최대화하기 위해서는 두 번째 항인 log(1-D(G(z)))의 값이 최소화되어야 합니다. 그러면 다음과 같이 최적화 문제가 정의

Case #2 : D(X) = 1일 경우 : 완벽히 분류해 낼 수 있는 경우 i. Discriminator가 실제 데이터 X가 진짜 임을 정확히 판단해야 함. 따라서 D(x)는 1에 가까워져야 합니다. 목적함수에서 D(x) 값을 최대화하기 위해서는 첫 번째 항인 log(D(x))의 값이 최대화되어야 하고 아래의 최적화 문제가 정의



Style GAN Paper summary



Style GAN 1

A Style-Based Generator Architecture for Generative Adversarial Networks

StyleGAN — Official TensorFlow Implementation

https://www.youtube.com/watch?v=kSLJriaOumA&feature=youtu.be 


▶ Summary / Contribution 

PGGAN baseline architecture를 기반으로 하여 성능과 더불어 Disentanglement feature을 향상 시켰다.
추가로, 고해상도 face dataset(FFHG)를 처음으로 발표함!
Disentanglement? 다양한 특징들이 잘 분리 되어 있다는 것을 의미 하나의 생성자를 학습 하고 난 후, 다양한 컨트롤이 가능하게 하는 개념? 생성자가 disentanglement를 만족해야 여러 semantic feature? image를 control 할 수 있음

→ 본 논문에서 기존 GAN의 Discriminator와 loss fucntion쪽은 최대한 그대로 유지하고, Generator 에서 idea를 제안하여 성능을 개선 한 것도 contribution

 Paper Review

Introduction

 PGGAN 방식은 Train 과정에서 전체 layers를 한 번에 학습 하지 않고, 점진적으로 layer를 늘려가면서 학습 하는 방식이며, 해당 방식을 통해 고해상도 이미지를 학습 하는 데 기여하였다. 그러나 image 단위로 semantic feature control이 어렵다는 한계점이 존재했지만, Style GAN에서 극복함

Style-based generator

StyleGAN Architecture + baseline model

PGGAN(ProGAN)은 학습을 진행하는 과정에서 점진적으로(Progressively) network layer를 추가하여 학습하는 방식이며, Loss fucntion은 WGAN-GP를 사용(안정적인 학습 과정이 가능, 고해상도 이미지 생성이 가능)

전반적인 구조는 latent vector sampling을 통해 Z를 생성하고, W Space(총 18개)로 mapping하여 affine trasformation를 거친 후, 각 block마다 style이 전달되고, 이 과정을 통해 고해상도 이미지를 점차 생성하게 되는 구조이며, stochastic variation을 반영 할 수 있도록 noise에 대한 정보를 사용함(linear 하며, entangled에 둔감하다)





StyleGAN Method

Add Mapping and styles

실제 train dataset이 존재하는 distribution-. Mapping을 하기 위한 후보 data

 Gaussian distribution / Z Vector-. 가우시안 분포를 따르는 latent vector를 생성하여, Generator에 input으로 사용 (← 기존 방식)-. train distibution data에서 가우시안 분포로 샘플링 되어 interpolation 했을 때, feature들이 바뀔 수 있는 문제가 있으며, 이를 entangle(꼬임) 되어 있다고 한다.

Mapping W Vector-. interpolation을 수행 할 때, linear한 space에서 feature이 각각 잘 분리 되어 mapping 할 수 있음-. maapping W를 사용하게 되면 특정 분포를 따라야 하는 dependency가 제거되며, 결국 W는 Z처럼 고정된 분포를 따르지 않게 된다-. 다시 한 번 정리하면 Factors of variation은 더욱 liear하고, disentangle하여 기존 Z에서 mapping 할 때 발생된 curved 되서 entangle한 문제가 해결 됨!





Removing Traditional Input

init input을 contant로 대체하며, 성능 향상을 기대

Stochastic Variation

별도의 노이즈 인풋을 넣어서 각각의 레이어마다 노이즈를 집어 넣어주는 방식

다양한 확률적인 측면 컨트롤이 가능

AdaIN Layer 직전에 noise data를 넣어줌

Style(synthesis network) : high-level global attributes

Noise : Stochastic variation



a : noise from all layerb : noise not appliedc : fine layer appliedd : coarse layer applied



X = All activation val, Ys = Statics를 바꾸기 위해 scaling + bias(Yb) 연산 → style transfer 방식





a : PGGANb : Style GAN

AdaIn (Adaptive Instance Normalization) - 해당 operation을 통해 style을 control

다수의 Style이 적용되어 Layer를 거치면 이미지의 다양성이 보장 

각기 다른 datas로 부터 style feature를 이용하여 생성 할 수 있으며, train 시 사용되는 parameter가 필요 하지 않음 (batch norm에서 사용 하는 γ,β를  사용하지 않음)

style transfer는 특정 이미지에서 style을 뽑고, 다른 이미지에서 contents를 뽑아 합

feed-forward 방식의 style transfer network에서 사용되어 좋은 성능을 보임(본 논문에서 등장한 새로운 idea는 아님)

하나의 이미지를 생성 할 때, 다수의 style 정보가 layer를 지나 갈 때 마, 변환 시킬 수 있도록? 해주는 방식? 

본 논문에선 매 layer마다 AdaIn이 수행되며, 해당 방식을 통해 scale-specific control이 가능함!

하나의 feature의 Statistics를 바꿀 수 있도록, scale과 bias를 적용함으로써 conv를 통해 얻은 fearture의 Statistics를 바꾸는 역할을 수행



Style Mixing

인접 layer간의 style의 correlation을 감소

Mixing Regularization 동작 방식

두 개의 서로 다른 input vector를 기반으로 Crossover 포인트를 설정

Crossover 이전은 w1, 이후는 w2를 사용(w1,2는 두 개의 vector)

결국 localized 특징을 가지고 있어 vector 안에 있는 data들 간의 상관 관계를 줄여 다양한 style 변화에 기여함

Disentanglement studies

Perceptual Path Length : 두 vector를 interpolation 할 때, 얼마나 급격하게 이미지 특징이 변화 하는 지?

latent space 상에서 interpolation을 했을 때, 얼마나 큰 변화가 있는 지 측정

interpolation을 했을 때, 일어나는 변화는 disentanglement와 관련이 있기 때문이며, 보간 시 non-linear한 변화가 이미지에 발생한다면 latent space가 entangled 할 수 있다는 뜻이기에…https://github.com/y33-j3T/Coursera-Deep-Learning/blob/master/Build%20Better%20Generative%20Adversarial%20Networks%20(GANs)/Week%201%20-%20Evaluation%20of%20GANs/PPL.ipynb 

linear Separability : latent space 상에서 attributes이 얼마나 linear하게 분류가 가능 한 지 평가?

CelebA-HQ - Gender 등의 40여개의 binaryattributes가 명시 되어 있는 measure dataset

Conclusion

Acknowledgements

Problem Definition

해당 논문 등장 전, GAN Model에서는 이미지 생성 과정의 Computation이 높고, Black box라는 문제가 존재하고 있었음, 또 다른 문제로 Latent Space 보간 기법에서 서로 다른 생성자들 간에 비교 할 수 있는 정량적인 방법론이 제공되지 않음

Motivation

Style Transfer에서 Generative Adversarial Networks을 위한 alternative generator architecture을 제안

기존 D (discriminator) 는 학습된 상수 입력값으로부터 시작하여 중간 vector(w)를 조정하여 각 계층에서 이미지의 스타일을 조절하는 방식

각 scale에서 이미지의 특징을 조절하는 데 직접적으로 관여하기 때문에 Low ~ High을 Style 방식으롷 저절이 가능



Style GAN 2

Analyzing and Improving the Image Quality of StyleGAN

StyleGAN2 — Official TensorFlow Implementation

https://youtu.be/jJDEQNMMwN0?si=kPTUOu8qjY4atw3y  

Contribution

Artifact Reduction

Normalization Techniques

Refined Network, Enhanced Style and Noise Control

▶ Contribution

StyleGAN1에서 제기 되었던 blob-like artifact, phase artifact

Blob-like (Droplet) Artifacts

문제 : StyleGAN1에서 약 64x64 해상도 이상에서 생성된 이미지에서 물방울 모양의 artifact이 나타나는 현상이 있었습니다. 이러한 artifacts는 대부분의 이미지에서는 문제가 되지 않았지만, 일부 이미지에서는 Noise처럼 보이기 때문에 제대로된 생성형 이미지 판단이 불가함

원인 : 이 문제는 AdaIN (Adaptive Instance Normalization)의 사용으로 인해 각 feature map의 평균과 표준편차가 개별적으로 정규화되면서, 서로 연관된 feature들의 정보가 손실 되는 것 때문에 발생함

변경점 : StyleGAN2에서는 AdaIN의 대체 방법으로 standard deviation만을 변경하는 접근을 채택 style block 외부에서 feature map의 값들을 변경하고, convolution 연산 결과로 나온 feature map에 대해 직접 modulation을 적용하지 않고 weight 값에 대해서 modulation을 진행함으로써, 서로 연관된 feature들의 정보 손상을 방지. 결과적으로, normalization을 demodulation로 대체함으로써 blob-like artifact가 제거 됨 - 논문 상에서는 줄었다? 라고 나오긴 하는데, 근데 DFMGAN 돌렸을 때 유사한 부자연스러운 artifact 발생함… (normal data + bin mask train)

Phase Artifact

문제 : StyleGAN1에서는 progressive growing 방법을 사용했습니다. 이 방식에서는 특정 요소(치아? 머리카락?)들이 latent manipulation 과정에서 고정되는 'phase artifact'가 발생 할 수 있음..

원인 : Progressive growing은 저해상도에서 시작하여 점차 해상도를 높여가는 방식으로, 각 해상도에서 고주파 디테일을 생성하는 데 중점을 두었습니다. 이 과정에서 중간 레이어에서 과도한 high frequency detail이 유발되었고, 일부 형태가 초기 단계에서 고정되어 변경하기 어려움

변경점 : StyleGAN2에서는 progressive growing을 제거하고, 단순한 feedforward 네트워크를 사용하는 방식을 채택했습니다. 처음부터 완전한 형태의 아키텍처를 구성하고, end-to-end 방식으로 학습을 진행하여, high-resolution 레이어에 의해 크게 영향을 받지 않는 low resolution 이미지를 생성합니다. 이러한 접근 방식은 phase artifact를 제거하는 데 기여를 함





 Reference

https://medium.com/analytics-vidhya/from-gan-basic-to-stylegan2-680add7abe82 

https://www.samsungsds.com/kr/insights/blogger/1232486_4637.html 



 TO DO Paper List : 

PGGAN

https://arxiv.org/abs/1710.10196 

https://github.com/tkarras/progressive_growing_of_gans 



DCGAN - Depp Conv Layers를 이용한 방법론이며, Image domain에서 high acc를 보임
-. D(판별자)는 Strided Conv 연산(W/H를 감소), G(생성자)는 Transposed Conv 연산(W/H를 증가)

https://arxiv.org/abs/1511.06434  

https://tutorials.pytorch.kr/beginner/dcgan_faces_tutorial.html 

https://github.com/Natsu6767/DCGAN-PyTorch 



WGAN-GP - Gradient penalty를 이용하여 기존 WGAN의 성능을 개선

-. Style GAN 논문도 WGAN-GP를 사용
-. WGAN은 Function이 1-Lipshichtz 조건을 만족하도록 하여 안정적인 학습을 유도하는 방법론을 제시

https://arxiv.org/abs/1704.00028 



용어

Latent vector / GAN 관점

-. 통상 z를 latent vector라고 부르며, 차원이 줄어든 채로 데이터를 잘 설명할 수 있는 잠재 공간에서의 vector를 의미하며 결국 차원이 줄어든 채로 데이터의 distribution을 잘 설명할 수 있는 잠재 공간의 vector를 의미



instance normalization

-. conv layer의 output을 정규화 하는 방법으로 각 이미지 instance마다 별도로 적용되어, 주로 style transfer에 사용된다.
-. StyleGAN에서 AdaIN이라는 Core idea가 등장 하는데, 해당 idea에서는 style transfer에 대한 idea를 가져와서 사용한 방법론으로 각 layer의 feature map을 style input에 따라 dynamic하게 조절하는 방식입니다. 이를 통해 생성된 이미지의 style을 세밀하게 조절이 가능하다. 
-. 하나에 이미지에 대해서 정규화를 수행, 각 채널 단위로



Group normalization (ECCV 2018) Paper Ref

Style Type 

Coarse style같은 경우, 극소적인 semantic feature들을 다루기 보다 이미지 전반적인 style 변화에 기여를 함
Middle style은 Coarse보단 골고루 이미지 전반적인 ? local feature들도 다루긴 하지만 세밀도가 떨어짐
Fine style은 머리카락? 피부와 같이 미세하고 정교한 style 변화에 기여를 함 



interpolation 시 관점 - entangle, disentangle

-. entangle
서로 얽혀 있는 상태여서 특징 구분이 어려운 상태. 즉, 각 특징들이 서로 얽혀있어서 구분이 안됨

-. disentangle
각 style들이 잘 구분 되어있는 상태여서 어느 방향으로 가면 A라는 특징이 변하고 B라는 특징이 변하게 되어서 특징들이 잘 분리가 되어있다는 의미.
선형적으로 변수를 변경했을 때 어떤 결과물의 feature인지 예측할 수 있는 상태.
