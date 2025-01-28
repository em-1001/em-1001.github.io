---
title:  "Latent Dynamics"
excerpt: "Paper: Learning Latent Dynamics for Planning from Pixels"

categories:
  - Statistics
tags:
  - Paper
  - Statistics
last_modified_at: 2025-01-19T08:06:00-05:00
---

# Background

## State transition probability
어떠한 상태에서 다음 단계의 상태로 전이(transition)하는 것에 대한 확률을 상태전이확률(State transition probability)라 한다. 
상태전이확률은 아래와 같은 수식으로 나타낸다. 

$$a_{ij} = P(o_t = v_j \vert o_{t-1} = v_i)$$

$$a_{ij} > 0 \ and \ \sum_{j=1}^m a_{ij} = 1$$

위 식은 상태 $v_i$에서 $v_j$로 이동할 확률에 대해 나타낸다. 

상태 A, B, C가 있고 각각의 상태 전이 확률이 정의되어 있다고 가정할 때, 상태 전이도(State transition diagram)은 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/754d2f27-bc3a-48e0-a4ef-a12086ab5e52"></p>

위 상태 전이도에 대한 전이확률은 다음과 같다. 

$$P = 
\begin{pmatrix}
P_{aa} & P_{ab} & P_{ac} \\  
P_{ba} & P_{bb} & P_{bc} \\  
P_{ca} & P_{cb} & P_{cc} \\  
\end{pmatrix} = 
\begin{pmatrix}
0 & 0.8 & 0.2 \\  
0.5 & 0.1 & 0.4 \\  
0.5 & 0 & 0.5 \\  
\end{pmatrix}$$


## Markov decision process(MDP)

결정적 시스템(Deterministic Process) 이란 어떤 상태($s$)에서 행동($a$)를 선택할 때 결과가 한 가지로 정해져있는 시스템이다. 
현실의 대부분은 결정적 시스템이 아닌 확률적 시스템(Stochastic System)이다. 

어떤 변수가 1시점 이전의 변수로부터만 영향을 받고, 확률적으로 변화하는 성질을 가질 때, Markov Property를 갖는다고 가정한다. 
Markov Property는 다음과 같은 식이 성립한다. 

$$P(s_{t+1} \vert s_t, s_{t-1}, \cdots, s_1) = P(s_{t+1} \vert s_{1:t}) = P(s_{t+1} \vert s_t)$$



# Reference 
https://benban.tistory.com/62     
https://benban.tistory.com/63   
https://velog.io/@qkrdbwls191/Markov-chain      
https://planetrl.github.io/  
