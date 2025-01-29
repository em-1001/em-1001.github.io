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

# POMDP

## Markov Process 

어떤 변수가 1시점 이전의 변수로부터만 영향을 받고, 확률적으로 변화하는 성질을 가질 때, Markov Property를 갖는다고 가정한다. 
Markov Property는 다음과 같은 식이 성립한다. 

$$P(s_{t+1} \vert s_t, s_{t-1}, \cdots, s_1) = P(s_{t+1} \vert s_{1:t}) = P(s_{t+1} \vert s_t)$$

Markov Process(Markov chain)은 마코브 성질을 가지는 랜덤 상태 $S_1, S_2, \cdots$ 들의 시퀀스이다. Finite Markov Process인 경우 상태들의 집합은 유한개로 구성된다. 

**A Markov Process (or Markov Chain) is a tuple** $<\mathcal{S}, \mathcal{P}>$  
- $\mathcal{S}$ is a (finite) set of states
- $\mathcal{P}$ is a state transition probability matrix,
  $\mathcal{P}_{ss^{\prime}} = P\left[s _{t+1} = s^{\prime} \vert S_t = s \right]$

상태들간의 변환 확률 행렬(state transition matrix)은 현재 상태에서 다른 상태로 갈 확률을 모든 상태에 대해 행렬 형태로 나타낸 것이다. 상태 변환 확률 행렬 $P$는 아래와 같으며, 각 행의 합은 1이 된다. 

$$P = 
\begin{pmatrix}
P_{11} & \cdots & P_{1n} \\ 
\vdots & \ddots & \vdots \\
P_{n1} & \cdots & P_{nn} \\ 
\end{pmatrix}$$

## Markov Reward Process 

Markov Reward Process(MRP)는 Markov chain에 reward가 더해진 것이다. 임의의 state들의 시퀀스를 상태 변환 확률에 따라 지나가면서 각 상태에 도착할 때마다 보상을 얼마나 받는지도 시퀀스로서 파악하는 것이다. 

**A Markov Reward Process is a tuple** $<\mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma>$  
- $mathcal{R}$ is a reward function, $\mathcal{R}_s = \mathbb{E}[R _{t+1} \vert S_t = s]$
- $\gamma$ is a discount factor, $\gamma \in [0, 1]$

$R_s$는 보상함수로, 상태 $S_s$일 때, 받을 수 있는 즉각적인 보상에 대한 기댓값이다. 중요한 점은 앞으로 받을 보상들을 고려한 누적 보상값이 아닌 즉각적으로 받는 보상(immediate reward)이다. 

현재 상태 $t$에서 다음 step에 받을 보상과 상태가 $r$과 $s^{\prime}$이 될 확률은 다음과 같이 표현한다. 

$$p(s^{\prime}, r \vert s) = P[S_{t+1} = s^{\prime}, R_{t+1} = r \vert S_t = s]$$




## Markov decision process(MDP)

결정적 시스템(Deterministic Process) 이란 어떤 상태($s$)에서 행동($a$)를 선택할 때 결과가 한 가지로 정해져있는 시스템이다. 
현실의 대부분은 결정적 시스템이 아닌 확률적 시스템(Stochastic System)이다. 

어떤 변수가 1시점 이전의 변수로부터만 영향을 받고, 확률적으로 변화하는 성질을 가질 때, Markov Property를 갖는다고 가정한다. 
Markov Property는 다음과 같은 식이 성립한다. 

$$P(s_{t+1} \vert s_t, s_{t-1}, \cdots, s_1) = P(s_{t+1} \vert s_{1:t}) = P(s_{t+1} \vert s_t)$$

Markov Property를 갖는 상태 전이 모형을 Markove Model이라 하고, Markove Model으로 생성되는 상태 전이 연속열을 마르코프 체인(Markov Chain)이라 한다. 마르코프 모형은 결국 확률 변수 사이에서 영향을 주고 받는 관계라 생각할 수 있다. 

$$S_1 \to S_2 \to \cdots S_{t-1} \to S_{t} \to S_{t+1}$$

Graphical Model을 사용하면 시간방향 전이 외에도 다양한 변수 사이의 상호관계를 나타낼 수 있다. 아래 그림처럼 state외에도 action space를 고려하는 경우가 대표적이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/a4861c03-6e52-4707-82a6-5cfeb04ef7f0"></p>

이처럼 행동 $a_t$에도 의존하여 상태 $s_{t+1}$가 결정되는 확률적 시스템 $P(s_{t+1} \vert s_t, a_t)$ 을 마르코프 결정 프로세스(Markov Decision Process, MDP)라 한다. 

## partially observable Markov decision process(POMDP)




# Reference 
POMDP : https://www.davidsilver.uk/teaching/        
https://ralasun.github.io/reinforcement%20learning/2020/07/12/mdp/  

https://planetrl.github.io/  
