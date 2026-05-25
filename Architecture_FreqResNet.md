# FreqRDN 아키텍처 및 논리적 배경 (LR-Only SR Model)

본 문서는 성능 극대화를 위해 업그레이드된 **FreqRDN (Frequency Residual Dense Network)** 모델의 시각적 구조와, 새롭게 도입된 모듈(RDB, Frequency-Aware Branches)을 채택한 논리적 근거(Justification) 및 논문 형식의 구조 설명을 정리한 문서입니다.

---

## 1. FreqRDN 시각적 아키텍처 (Visual Architecture)

```text
========================================================================
                 [ Input LR Wavelets (LL, LH, HL, HH) ]
========================================================================
                                   |
                                   v
                      +-------------------------+
                      |     Head: Conv 3x3      |=================+
                      +-------------------------+                 ||
                                   |                              ||
                                   v                              || (Global
                    +=============================+               ||  Feature
                    | Residual Dense Block (x 8)  |               ||  Skip)
                    |                             |               ||
                    |   +---> [ Dense Layer 1 ]   |               ||
                    |   |                         |               ||
                    |   +---> [ Dense Layer 2 ]   |               ||
                    |   |                         |               ||
                    |   +---> [ Dense Layer 3 ]   |               ||
                    |   |                         |               ||
                    |   +---> [ Dense Layer 4 ]   |               ||
                    |   |                         |               ||
                    |   |     [ LFF Conv 1x1]     |               ||
                    |   |                         |               ||
                    |   |     [     CBAM    ]     |               ||
                    |   |                         |               ||
                    |   +-----(+)<-+              |               ||
                    |    (Local Skip)             |               ||
                    +=============================+               ||
                                   |                              ||
                                   v                              ||
                      +-------------------------+                 ||
                      |     GFF: Conv 3x3       |                 ||
                      +-------------------------+                 ||
                                   |                              ||
                                   v                              ||
                                  (+) <===========================++
                                   |
             +---------------------+---------------------+
             |                     |                     |
             v                     v                     v
    +-----------------+   +-----------------+   +-----------------+
    |    LH Branch    |   |    HL Branch    |   |    HH Branch    |
    | Conv-ReLU-Conv  |   | Conv-ReLU-Conv  |   | Conv-ReLU-Conv  |
    +-----------------+   +-----------------+   +-----------------+
             |                     |                     |
             +---------------------+---------------------+
                                   |
                                   v
                             [ Concatenate ]
                                   |
                                   v
   [ Extract LR HF ] -----------> (+) <-- (Global HF Skip)
                                   |
                                   v
========================================================================
                  [ Output HF Residuals (LH, HL, HH) ]
========================================================================
```

---

## 2. Proposed Architecture (논문식 구조 설명)

초해상도(Super-Resolution, SR) 작업에 있어 공간 해상도의 보존은 디테일 복원에 필수적입니다. 제안하는 **FreqRDN** 아키텍처는 웨이블릿 변환(Wavelet Transform)을 통해 얻어진 저해상도 이미지의 4가지 주파수 대역(LL, LH, HL, HH)을 입력으로 받아, 고해상도 복원에 필요한 3가지 고주파 대역(LH, HL, HH)의 잔차(Residual)를 예측하도록 설계되었습니다. 전체 네트워크는 크게 세 부분(Head, Body, Tail)으로 구성되며, 업샘플링 연산 없이 공간 해상도를 네트워크 끝까지 유지합니다. 기존 FreqResNet에서 한 단계 진화하여 밀집 연결(Dense Connection)과 주파수 분기(Frequency-Aware Branches) 구조를 새롭게 결합하였습니다.

1. **Feature Extraction (Head)**
   네트워크의 시작점인 Head 모듈은 4채널의 입력 웨이블릿 계수를 받아 $3 \times 3$ Convolution 연산을 통해 64채널의 얕은 특징 맵(Shallow Feature Map)을 추출합니다.

2. **Deep Feature Mapping (Body with RDB & CBAM)**
   추출된 특징 맵은 8개의 밀집 잔차 블록(Residual Dense Blocks, RDB)으로 구성된 Body 모듈을 통과합니다. 단일 RDB 내부에는 4개의 밀집 계층(Dense Layer)이 존재하며, 이전 계층의 모든 특징 맵을 현재 계층의 입력으로 결합(Concatenation)하는 방식을 취합니다. 이를 통해 네트워크는 풍부한 국소 특징(Abundant Local Features)을 추출하며 미세한 주파수 정보의 유실을 완벽하게 방지합니다. 
   각 블록의 끝에는 1x1 Convolution 기반의 국소 특징 융합(Local Feature Fusion, LFF)을 수행하여 채널 수를 통제하고, 이어서 **CBAM (Convolutional Block Attention Module)**을 배치하였습니다. CBAM은 공간 및 채널 주의력을 통해 평탄 영역의 노이즈 생성을 억제하고 날카로운 엣지 성분만을 선택적으로 강화합니다. 

3. **Reconstruction & Global Skip Connections (Tail)**
   RDB를 통과한 특징들은 전역 특징 융합(Global Feature Fusion, GFF) 계층을 거친 후, Head 모듈의 초기 출력과 더해지는 **Global Feature Skip Connection**을 형성합니다.
   가장 핵심적인 변화는 네트워크의 출력단(Tail)에 도입된 **Frequency-Aware Branches**입니다. 기존 구조가 하나의 컨볼루션 모듈에서 모든 대역을 동시에 처리했던 것과 달리, 본 구조에서는 추출된 깊은 특징(Deep Feature)을 3개의 독립적인 Branch로 전달합니다. 가로(LH), 세로(HL), 대각선(HH) 대역은 각기 다른 구조적 특성을 지니므로, 이들을 개별적인 모듈에서 독립적으로 예측하게 함으로써 엣지 복원의 전문성(Specialization)을 극대화하였습니다.
   마지막으로 3개의 Branch 출력을 결합(Concatenation)한 뒤, Bicubic 보간된 입력 신호의 고주파 성분(LR HF)을 베이스라인으로 더해주는 **Global High-Frequency Skip Connection**을 적용하여 희소 표현(Sparse representation) 학습의 이점을 이어갑니다.

---

## 3. 업그레이드 요소 채택 논리 (Logical Justification)

*   **🚫 U-Net(다운샘플링) 배제**: 공간 정보를 압축하면 고주파 엣지의 위치 정보가 훼손되므로 해상도를 100% 유지합니다.
*   **🕸️ Residual Dense Blocks (RDB)**: 기존 ResBlock이 직렬 통과하며 잃어버릴 수 있는 미세 특징들을, 내부 Dense Connection을 통해 하나도 빠짐없이 다음 레이어로 넘겨주어 극도로 풍부한 특징(Rich Features)을 추출합니다.
*   **🌿 Frequency-Aware Branches (주파수 전담 분기)**: LH(가로), HL(세로), HH(대각선) 엣지는 생김새와 분포가 완전히 다릅니다. 이들을 억지로 하나의 필터가 학습하게 하지 않고, 각각의 방향만을 전담하는 독립된 네트워크 Branch를 구성하여 주파수 도메인에서만 가능한 가장 논리적이고 정교한 예측을 수행합니다.
*   **🎯 CBAM 및 Global HF Skip Connection**: 노이즈 억제(Attention)와 밑그림 기반 잔차 학습(Residual Learning)의 이점은 그대로 유지하였습니다.
