---
title:  "ServalShell"
excerpt: "NL to Bash AI project "

categories:
  - Natural Language Processing
tags:
  - Transformer
  - Project
last_modified_at: 2024-09-29T08:06:00-05:00
---

# Paper: Attention Is All You Need

<p align="center"><img src="https://github.com/user-attachments/assets/addaed68-d49a-4bf7-9c83-00e025250365" height="40%" width="40%"></p>

본 논문은 Attention기법을 활용한 Transformer모델을 제안한다. Attention기법이 나오기 이전 seq2seq모델 같은 경우 encoding에서 decoding으로 넘어갈 때 고정된 크기의 context vector를 사용하기 때문에 소스 문장을 고정된 크기로 압축하는 과정에서 병목현상이 발생할 가능성이 높고, 하나의 context vector가 소스 문장의 모든 정보를 담고 있어야 해서 성능 저하의 요인이 되었다.   

이를 해결하기 위해 하나의 context vector대신 매번 소스 문장에서의 출력을 전부 모아 하나의 행렬로 만들어 decoder에 넣어주는 Attention 메커니즘이 제안되었다. 이렇게 하면 디코더는 출력 단어를 처리할 때마다 매번 인코더의 모든 출력 중 어떤 정보가 중요한지를 계산하게 되는데 이를 Energy 값이라 한다. 

$$e_{i,j} = a(s_{i-1}, h_j)$$

$$\alpha_{i,j} = \frac{\exp({e_{ij}})}{\sum^{T_ x}_ {k=1} \exp({e_ {ik}})}$$

여기서 $s$는 디코더가 이전에 출력한 단어를 만들기 위해 사용했던 hidden state이고, $h$는 인코더에서 가져온 각각의 hidden state이다. 이렇게 구해진 energy 값에 softmax를 취해 각각의 $s$가 $h$에 대해 확률적으로 얼마나 연관이 있는지를 구하고, 이 가중치 값을 $h$와 곱하여 가중치가 반영된 인코더의 출력 결과를 활용하게 된다. 

논문에서 제안하는 Transformer 모델은 RNN, CNN을 사용하지 않고 Attention 기법만을 활용한 자연어 처리 모델이다. 

## Embedding, Positional Encoding

어떠한 단어 정보를 네트워크에 넣기 위해 실수 값으로 이루어진 벡터로 임베딩 과정을 거친다. 임베딩 과정으로 통해 input embedding matrix가 만들어 지는데, 이 행렬의 행은 단어의 갯수 만큼, 열은 임베딩 차원값(embed_dim)과 동일한 값으로 만들어져 각각의 단어에 대한 임베딩 값을 저장하게 된다. 

Transformer는 RNN을 사용하지 않기 때문에 입력으로 넣는 단어들에 대한 순서정보를 추가로 넣어줘야 하는데, 이를 Positional Encoding이라 한다. Positional Encoding은 input embedding matrix와 동일한 임베딩 차원을 가지면서 위치에 대한 encoding 정보를 갖고 있어 input embedding matrix와 element wise 합으로 위치 정보를 심어준다. 

## Multi-Head Attention

<p align="center"><img src="https://github.com/user-attachments/assets/b589dd53-7d2f-4e15-8b68-32c7d3d65e82"></p>

Attention 기법은 어떤 단어들이 또 다른 단어들과 어떠한 연관성을 갖는지를 알아내기 위한 기밥이다. 이 Attnetion 값을 계산하기 위해서 Query, Key, Value라는 3가지 정보가 필요한데 Query는 각 단어들에 대한 연관성에 대한 값을 알아내기 위해 넣는 값이고, 연관성을 계산하는데 쓰인느 단어들이 Key이다. 이렇게 Query와 Key로 Attention 가중치를 구하면 여기에 실제 Value값을 곱해서 최종 값을 산출한다. 이 과정이 Scaled Dot-Product Attention이고, 실제 모델에 학습시킬 때는 h개의 head로 구분하여 Multi-Head Attention을 수행한다. 이렇게 h개의 서로다른 Query, Key, Value를 만듦으로써 다양한 Attention값을 얻어낼 수 있다. 

Multi-Head Attention은 다음과 같이 계산된다. 

$$Attention(Q, K, V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V$$

$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

$$MultiHead(Q, K, V) = Concat(head_1, \cdots, head_h)W^O$$



https://www.youtube.com/watch?v=bCz4OMemCcA

https://www.youtube.com/watch?v=AA621UofTUA&t=2s

# Project: NL to Bash translation with Transformer

```sh
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
