---
title:  "ServalShell"
excerpt: "NL to Bash AI project "

categories:
  - Natural Language Processing
tags:
  - Transformer
  - Project
toc: true
toc_sticky: true
last_modified_at: 2024-09-29T08:06:00-05:00
---

# Paper: Attention Is All You Need

<p align="center"><img src="https://github.com/user-attachments/assets/c63c4c3e-1342-412e-a31c-6aa574e133d6"></p>

본 논문은 Attention기법을 활용한 Transformer모델을 제안한다. Attention기법이 나오기 이전 seq2seq모델 같은 경우 encoding에서 decoding으로 넘어갈 때 고정된 크기의 context vector를 사용하기 때문에 소스 문장을 고정된 크기로 압축하는 과정에서 병목현상이 발생할 가능성이 높고, 하나의 context vector가 소스 문장의 모든 정보를 담고 있어야 해서 성능 저하의 요인이 되었다.   

이를 해결하기 위해 하나의 context vector대신 매번 소스 문장에서의 출력을 전부 모아 하나의 행렬로 만들어 decoder에 넣어주는 Attention 메커니즘이 제안되었다. 이렇게 하면 디코더는 출력 단어를 처리할 때마다 매번 인코더의 모든 출력 중 어떤 정보가 중요한지를 계산하게 되는데 이를 Energy 값이라 한다. 

$$e_{i,j} = a(s_{i-1}, h_j)$$

$$\alpha_{i,j} = \frac{\exp({e_{ij}})}{\sum^{T_ x}_ {k=1} \exp({e_ {ik}})}$$

여기서 $s$는 디코더가 이전에 출력한 단어를 만들기 위해 사용했던 hidden state이고, $h$는 인코더에서 가져온 각각의 hidden state이다. 이렇게 구해진 energy 값에 softmax를 취해 각각의 $s$가 $h$에 대해 확률적으로 얼마나 연관이 있는지를 구하고, 이 가중치 값을 $h$와 곱하여 가중치가 반영된 인코더의 출력 결과를 활용하게 된다. 

논문에서 제안하는 Transformer 모델은 RNN, CNN을 사용하지 않고 Attention 기법만을 활용한 자연어 처리 모델이다. 

## Embedding, Positional Encoding

어떠한 단어 정보를 네트워크에 넣기 위해 실수 값으로 이루어진 벡터로 임베딩 과정을 거친다. 임베딩 과정으로 통해 input embedding matrix가 만들어 지는데, 이 행렬의 행은 단어의 갯수 만큼, 열은 임베딩 차원값(embed_dim)과 동일한 값으로 만들어져 각각의 단어에 대한 임베딩 값을 저장하게 된다. 

Transformer는 RNN을 사용하지 않기 때문에 입력으로 넣는 단어들에 대한 순서정보를 추가로 넣어줘야 하는데, 이를 Positional Encoding이라 한다. Positional Encoding은 input embedding matrix와 동일한 임베딩 차원을 가지면서 위치에 대한 encoding 정보를 갖고 있어 input embedding matrix와 element wise 합으로 위치 정보를 심어준다. 

Positional Encoding은 다음과 같이 주기 함수를 활용하여 구해진다. 

$$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{model}})$$   

$$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d_{model}})$$   

$pos$는 각각의 단어 번호, $i$는 각 단어의 embedding 값 위치를 의미한다. 

Positional Encoding은 네트워크가 각 단어의 상대적 위치를 파악할 수 있게만 하면 되므로 위 처럼 $\sin, \cos$ 으로 정해진 함수를 이용할 수도 있지만, 위치에 대한 embedding 값을 따로 학습하도록 하여 네트워크에 넣을 수도 있고 성능상의 차이는 거의 없다. 

## Multi-Head Attention

<p align="center"><img src="https://github.com/user-attachments/assets/b589dd53-7d2f-4e15-8b68-32c7d3d65e82"></p>

Attention 기법은 어떤 단어들이 또 다른 단어들과 어떠한 연관성을 갖는지를 알아내기 위한 기법이다. 이 Attnetion 값을 계산하기 위해서 Query, Key, Value라는 3가지 정보가 필요한데 Query는 각 단어들에 대한 연관성에 대한 값을 알아내기 위해 넣는 값이고, 연관성을 계산하는데 쓰인느 단어들이 Key이다. 이렇게 Query와 Key로 Attention 가중치를 구하면 여기에 실제 Value값을 곱해서 최종 값을 산출한다. 이 과정이 Scaled Dot-Product Attention이고, 실제 모델에 학습시킬 때는 h개의 head로 구분하여 Multi-Head Attention을 수행한다. 이렇게 h개의 서로다른 Query, Key, Value를 만듦으로써 다양한 Attention값을 얻어낼 수 있다. 

Multi-Head Attention은 다음과 같이 계산된다. 

$$Attention(Q, K, V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V$$

$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

$$MultiHead(Q, K, V) = Concat(head_1, \cdots, head_h)W^O$$

<p align="center"><img src="https://github.com/user-attachments/assets/3873d29a-cd2d-4c3b-9a28-1b1d9a60427f"></p>


위는 단어 갯수(seq)가 6개, model dimension이 512라 가정했을 때 softmax를 구하는 과정이다. Query와 Key에 대해 각각의 단어에 대한 embedding 값들을 곱해주어 Attention Energy를 얻게 된다. 이때 나눠주는 $\sqrt{d_k}$ softmax를 구할 때, 0주변에서 떨어진 값으로 인해 gradient vanishing 문제가 발생하는 것을 방지한다. 이렇게 얻어진 Attention 
Energy는 Query의 각 단어가 Key의 각 단어와 얼마나 연관성을 갖는지를 나타낸 값으로 이 가중치에 value값을 곱하여 실제 Attention 값을 구할 수 있다. 

$$Mask = 
\begin{pmatrix}
? & 0 & 0 & 0 & \cdots & 0 \\
? & ? & 0 & 0 & \cdots & 0 \\ 
? & ? & ? & 0 & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\   
? & ? & ? & ? & \cdots & ?
\end{pmatrix}$$

Attention Energy를 구할 때 위와 같이 Mask matrix를 이용해 특정 단어를 무시하게 할 수 있다. Mask matrix는 Attention Energy에 element wise로 더해 구해지며, mask 값으로 $-\infty$ 값을 넣어주어 $softmax$의 출력이 0%에 가깝도록 할 수 있다.   
Encoder과정에서 self-attention에 쓰이는 mask는 보통 의미없는 padding token을 무시하는 용도로 사용되고, Decoder과정에서 self-attention과 cross-attention에 쓰이는 mask는 보통 모델이 학습할 때나 번역할 때, 현재 만들고 있는 token 이후의 token을 참조하지 못하게 하는 용도로 사용된다. 

<img src="https://github.com/user-attachments/assets/f64f4a37-c567-45ff-b865-f08d58dabf55">

전체 과정은 위와 같다. 최종 결과를 보면 MultiHead Attention을 수행한 뒤에도 입력 차원과 동일하게 차원이 유지되는 것을 확인할 수 있다. 앞서 transformer의 모델 diagram을 다시 살펴보면 encoder부분에서 사용되는 self-attention은 각각의 단어가 서로에게 어떤 연관성을 갖는지를 계산하여 전체 문장에 대한 representation을 학습할 수 있도록 한다. decoder부분에서 사용되는 self-attention은 뒤에 해당하는 단어들은 masked되어 앞에 등장했던 단어들만 참고하여 attention을 계산하게 한다. 이는 모델이 단어를 만들어낼 때 뒤에 등장할 단어를 이미 참조해버리면 학습이 의미 없어지게 되기 때문이다. 마지막으로 decoder부분에서 사용되는 cross-attention은 Query는 decoder에서 입력받고 Key, Value는 encoder에서 받아온다. 번역의 경우 번역할 단어들에 대한 정보를 Key, Value로 가져와 각 단어와의 연관성을 계산하고 번역을 수행한다. 


## How a Transformer works at inference vs training time
Transformer는 train할 때와 inference를 할 때 동작 방식에 약간 차이가 있다. 

#### inference time

inference time에서는 우선 Decoder input으로 <SOS> 토큰이 들어간다. 이후 <SOS> 토큰 다음으로 나올 것이라 예측되는 단어들 중 확률이 높은 단어가 Decoder output으로 나오고, <SOS> 토큰과 Decoder output으로 나온 단어를 이어 붙인 값을 다시 Decoder input으로 넣는다. 이렇게 하면 결과로 2번째 단어를 예측하게 되고, 이를 <EOS>가 나올 때 가지 반복하여 결과들을 이어 붙이면 최종 번역 결과가 나오게 된다. 

#### train time

반면 train time에서는 Decoder input으로 번역 결과인 target label들 앞에 <SOS> 토큰을 붙인 값을 넣는다. 이렇게 값을 넣으면 모델이 각 위치에 대해 softmax 확률값을 결과로 도출하고 이 값들은 각각 해당 위치의 ground truth에 해당하는 label 값과 cross entropy를 계산하게 된다. 이렇게 각 단어의 위치에 따라 구해진 cross entropy loss값들의 평균이 최종 loss가 되어 모델의 학습에 반영된다.  이를 위해 train time에서는 encoder, decoder input에 길이가 일정하도록 padding token을 붙이고, label id에도 해당 길이가 되도록 -100을 넣어주는데, -100은 해당값을 가지고 있는 index에 대해선 cross entropy loss를 구하지 않겠다는 것을 의미한다. 따라서 paddding 부분을 제외한 모델이 예측한 값들에 대한 loss가 산출되어 모델이 학습하게 된다. 

## Greedy Search & Beam Search

앞서 Transformer의 inference time에 Decoder가 예측한 확률 값 중 가장 높은 단어가 output으로 나온다고 했다. 이때 확률이 높은 값을 선정하는 방법에 따라 Greedy Search를 적용할 수도 있고, Beam Search 적용할 수도 있다. 

Greedy Search는 매우 간단하다. output으로 단어를 선택할 때 현 시점에서 가장 확률이 높은 단어를 가져온다. 이는 시간복잡도 면에서는 좋지만, 다양한 경우의 수를 고려하지 못해 최종적인 정확도 면에서 좋지 않을 수 있다. 반면 Beam Search는 이를 보완하기 위해 매 순간 고려할 빔의 수를 정하고 해당 경우의 수 만큼 누적확률을 계산해 나간다. 

<p align="center"><img src="https://github.com/user-attachments/assets/beb19a9a-82a6-4675-bc82-194a8dbce44a" height="55%" width="55%"></p>

위는 빔의 수를 2로 설정했을 때이다. 빔의 수를 k라 가정하면, 우선 현 시점에서 확률이 가장 높은 k를 뽑는다. 그 다음 이전에 선택한 k개의 경우에 대해 각각 다시 확률이 가장 높은 경우 k개를 고른다. 이런 식으로 누적 확률을 계산하다가 특정 빔이 <EOS> 토큰을 만나면 해당 빔은 후보에 오른다. 그리고 후보에 오른 빔의 자리를 대신해서 이전에 k+1번째로 확률이 높았던 빔이 활성화 되어 k개의 빔을 유지한다. 이렇게 이어 가다 후보 빔의 수가 k개가 되면 서치를 종료하고 최종 k개의 후보 중 누적 확률이 가장 높은 빔을 선택하게 된다. 

하지만 이렇게 하면 매 빔마다 누적확률을 계산하기 때문에 빔의 길이가 긴 후보가 확률이 낮을 수 밖에 없다. 이러한 문제를 해소하기 위해 Length Penalty를 구해 확률값에 나누어 주며 아래와 같이 구해진다. 

$$lp(Y) = \frac{(5 + |Y|)^{\alpha}}{(5 + 1)^{\alpha}}$$

알파는 보통 1.2를 사용하고, minimum length인 5도 변경 가능한 하이퍼 파라미터이다. 

# Project: NL to Bash translation with Transformer

```
/content# ./servalshell.sh
Bashlint grammar set up (124 utilities)

                            
       \`*-.                    
        )  _`-.                 
       .  : `. .                
       : _   '  \               
       ; *` _.   `*-._          
       `-.-'          `-.       
         ;       `       `.     
         :.       .        \    
         . \  .   :   .-'   .   
         '  `+.;  ;  '      :   
         :  '  |    ;       ;-. 
         ; '   : :`-:     _.`* ;
      .*' /  .*' ; .*`- +'  `*' 
      `*-*   `*-*  `*-*'


ServalShell:~$ -h
If you enter a command in natural language, the program automatically translates it into a bash command and executes it.
If execution fails because the bash command translated by the model is incorrect, It will recommend several command structures.
Additionally, the following options are available.

-d [cmd],  --direct [cmd]              Execute bash command directly
-r [nl],  --recommend [nl]             Even if the command execution is successful, Recommended Command Structure is displayed
-h,  --help                            Describes usage and options
-q,  --quit                            Quit Servalshell

ServalShell:~$ ▯
```

## Pre-Process & Post-Process

프로젝트를 진행하던 중 문제가 발생했다. Transformer 모델을 완성한 후 테스트 겸 영어를 이탈리아어로 번역하도록 학습시켰더니 괜찮은 결과가 나왔다. 하지만 본래 목표였던 영어를 bash 명령어로 바꾸도록 학습시켰을 때는 학습 데이터에 있던 파일명이나 경로가 아닌 새로운 파일명, 경로 등을 넣었을 때 모델이 아에 엉뚱한 답을 내놓았다. 단순 특정 언어를 다른 언어로 번역하는 작업이 아니라 bash 명령어로 번역해야 했기에 이러한 값들에 대해서도 잘 동작할 수 있도록 해당 값들을 일반화하여 학습시킬 필요가 있엇다.
조사를 해보니 **NL2CMD: An Updated Workflow for Natural Language to Bash Commands Translation** 논문에서는 학습 데이터에 있는 파일명, 경로, 변수 등을 **Generic Forms** 으로 바꾸어 전처리 하였다고 했다. 예를 들면 아래와 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/bb8997ba-9786-4b18-a12b-a420a0aec45a"></p>

**Tellina Tool**을 이용해서 Dataset의 Sorce 문장과 Target 문장 모두 위와 같은 전처리를 해 주었다. 이렇게 학습시켜본 결과 bash 명령어의 형태를 잘 유지하면서 파일명, 경로, 변수 등이 있을 자리에는 각각에 해당하는 **Generic Forms**으로 출력할 수 있게 되었다. 
마지막으로 해당 **Generic Forms**을 원래 기존 파일명, 경로, 변수 등으로 바꾸는 후처리 작업만 해주면 되었고, 이는 Tellina의 Slot filling 함수를 사용하여 구현하였다. 실제 모델이 출력한 결과를 subprocess를 이용해 실행해본 결과 실행이 되는 코드들이 어느정도 나왔다. 


## Attention Score 

**sentence:** moves "file.txt" to "./bin"   
**translation:** mv "file.txt" "./bin"

<img src="https://github.com/user-attachments/assets/edd50e16-985c-4a5d-b355-3802c2e073c1" height="85%" width="85%">

## BLEU Score

```ini
# config
batch_size: 8
num_epochs: 20
learning_rate: 1e-4
seq_len: 100
d_model: 512

# BLEU Score
Total BLEU Score = 40.36
Individual BLEU1 score = 61.51
Individual BLEU2 score = 47.06
Individual BLEU3 score = 33.74
Individual BLEU4 score = 27.18
Cumulative BLEU1 score = 61.51
Cumulative BLEU2 score = 53.80
Cumulative BLEU3 score = 46.05
Cumulative BLEU4 score = 40.36
```

## Github Repository
[ServalShell](https://github.com/em-1001/ServalShell)

# Reference 

## Web Link
Transformer : https://www.youtube.com/watch?v=bCz4OMemCcA  
　　　　　　https://www.youtube.com/watch?v=AA621UofTUA&t=2s   

## Paper
Attention Is All You Need : https://arxiv.org/pdf/1706.03762  
NL2CMD : https://arxiv.org/pdf/2302.07845  

