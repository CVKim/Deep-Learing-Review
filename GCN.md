
[Paper]
https://arxiv.org/abs/1609.02907

[github]
https://github.com/dmlc/dgl/tree/master/examples/mxnet/gcn

# Semi-Supervised Classification with Graph Convolutional Networks (GCN)

## Contribution

GCN (Graph Convolutional Networks)은 기존의 CNN (Convolutional Neural Networks)이 제공하는 정규 그리드 데이터 처리 방식을 그래프 구조 데이터에 적용함으로써, 노드 간 복잡한 관계를 효율적으로 학습합니다. 이 리뷰는 GCN의 개념적 차이점, 수학적 기반, 방법론적 접근, 실험적 검증, 그리고 종합적인 논의를 제공합니다.

![image](https://github.com/CVKim/PaperReview/assets/90014998/24921736-bf0e-43f5-a77e-c56e5f55bf1b)

## Introduction

- **CNN 대비 GCN의 접근 방식**: CNN은 이미지와 같은 공간적 데이터 처리에 초점을 맞춘 반면, GCN은 복잡한 그래프 구조 데이터 처리에 적합하게 설계되었습니다.
- **노드 특성의 집계**: GCN은 이웃 노드의 특성을 집계하여 강력한 노드 레벨 특성을 추출하는 데 초점을 둡니다. 이는 그래프 기반 데이터에 대한 효과적인 분석 방법을 제공합니다.


### 방법론: Fast Approximate Convolution on Graphs

GCN (Graph Convolutional Networks)의 핵심 방법론은 복잡한 그래프 구조에 대한 효과적인 특성 학습을 가능하게 하는 두 가지 주요 개념, 즉 **Spectral Graph Convolution**과 **Layer-wise Linear Model**을 기반으로 합니다. 이를 통해 **Semi-supervised node Classification** 같은 작업을 수행할 수 있습니다.

#### 1. Spectral Graph Convolution

**정의**: Spectral Graph Convolution은 그래프 데이터의 Fourier 변환을 통해 정의됩니다. 이는 그래프의 Laplacian 행렬 \( L \)을 중심으로 이루어집니다. \( L \)은 \( L = D - A \)로 정의되며, 여기서 \( D \)는 degree matrix이고 \( A \)는 adjacency matrix입니다.

![image](https://github.com/CVKim/PaperReview/assets/90014998/7e762072-9db0-49c9-bd59-05e49da8ab7a)

- 먼저 그래프의 normalized Laplacian \( \widetilde{L} \)은 \( \widetilde{L} = I - D^{-1/2}AD^{-1/2} \)로 정의됩니다. 여기서 \( I \)는 단위행렬입니다.
- \( \widetilde{L} \)은 고유값 분해를 통해 \( \widetilde{L} = U\Lambda U^T \)로 표현될 수 있습니다. 여기서 \( U \)는 고유벡터의 행렬이고, \( \Lambda \)는 고유값의 대각행렬입니다.
- Spectral Graph Convolution은 이러한 고유값 분해를 기반으로, 입력 신호 \( x \)에 대한 그래프 Fourier 변환 \( U^T x \)와 필터 \( g_\theta \)의 곱으로 표현됩니다: \( U g_\theta(\Lambda) U^T x \).

#### 2. Layer-wise Linear Model

**정의**: 이 모델은 GCN의 각 레이어가 이전 레이어의 출력을 입력으로 받아 선형 변환을 수행한다는 개념에 기반합니다.

- 각 레이어는 다음과 같이 표현됩니다: \( H^{(l+1)} = \sigma(\widetilde{A}H^{(l)}W^{(l)}) \).
- 여기서 \( H^{(l)} \)은 \( l \)번째 레이어의 출력, \( W^{(l)} \)은 가중치 행렬, \( \sigma \)는 활성화 함수, \( \widetilde{A} \)는 정규화된 인접행렬 (renormalized adjacency matrix)입니다.
- \( \widetilde{A} \)는 \( \widetilde{A} = \widetilde{D}^{-1/2}\widetilde{A}\widetilde{D}^{-1/2} \)로 구성되며, \( \widetilde{D} \)는 \( \widetilde{A} \)의 degree matrix입니다.

#### Semi-supervised Node Classification

![image](https://github.com/CVKim/PaperReview/assets/90014998/7abf012c-9416-4537-b146-480f587f4be3)

이 방법론은 레이블이 할당된 소수의 노드를 사용하여 전체 그래프의 나머지 노드에 대한 레이블을 예측하는 semi-supervised learning 문제에 적용됩니다.

- GCN은 레이블이 있는 노드의 특성과 그래프 구조를 이용하여 레이블이 없는 노드의 특성을 학습합니다.
- 이 과정은 그래프 전체에 대한 깊은 통찰력을 제공하며, 복잡한 네트워크에서의 노드 분류 및 표현 학습에 매우 효과적입니다.

GCN의 이러한 수학적 접근은 그래프 데이터의 복잡한 구조와 관계를 이해하고, 그 안에서 유의미한 패턴과 특성을 추출하는 데 큰 도움이 됩니다.

## Experiments

- **데이터셋 성능 비교**: 다양한 그래프 데이터셋에서 GCN의 성능을 분석합니다.
- **예측 정확도**: GCN의 예측 정확도에 대한 검증을 포함합니다.
- **하이퍼파라미터 설정에 따른 성능**: 다양한 하이퍼파라미터 설정에 따른 GCN의 성능 변화를 분석합니다.

![image](https://github.com/CVKim/PaperReview/assets/90014998/29076002-0a47-464a-8c8f-89cd897a9317)
![image](https://github.com/CVKim/PaperReview/assets/90014998/e4784757-50c2-4578-982a-adee8832bdf3)
![image](https://github.com/CVKim/PaperReview/assets/90014998/ee73d43a-ae36-4533-9057-dc865fa7a93a)
![image](https://github.com/CVKim/PaperReview/assets/90014998/355ffcf3-17c4-44a8-bd0d-a9f89ed0a924)


## Future Work: GCN의 발전 방향과 개선점

- **Memory Requirement**
  - **문제점**: 현재 실험에서는 모든 Neighbor를 담아야 하기 때문에 Full Batch Gradient Descent 방식을 사용했습니다. 이 방식은 많은 메모리를 요구합니다.
  - **개선 방향**: 향후 연구에서는 메모리 요구량을 줄이기 위해 Minibatch 처리 방식으로의 전환을 실험할 필요가 있습니다.

- **Directed edges and edge features**
  - **현 상태**: GCN은 현재 Undirected Graph에 한정적으로 적용 가능합니다(Weighted or Unweighted).
  - **연구 필요성**: Directed edges와 edge features를 포함하는 그래프에 대한 적용 가능성을 탐구하는 것이 중요합니다.

- **Limiting assumptions**
  - **현재 한계**: GCN은 Self-connection과 Neighborhood Edge 간에 같은 중요도를 설정해두고 계산을 진행하고 있습니다.
  - **개선 제안**: 이 둘 간의 중요도를 조절할 수 있는 파라미터를 도입하면 보다 우수한 성능을 달성할 수 있을 것입니다.

