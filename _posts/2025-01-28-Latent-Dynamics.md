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

## Markov process(Markov Chain)

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

Markov process에서 sampling된 시퀀스를 에피소드(episode)라 한다. 

## Markov reward process(MRP) 

Markov Reward Process(MRP)는 Markov chain에 reward가 더해진 것이다. 임의의 state들의 시퀀스를 상태 변환 확률에 따라 지나가면서 각 상태에 도착할 때마다 보상을 얼마나 받는지도 시퀀스로서 파악하는 것이다. 

**A Markov Reward Process is a tuple** $<\mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma>$  
- $\mathcal{S}$ is a (finite) set of states
- $\mathcal{P}$ is a state transition probability matrix,
  $\mathcal{P}_{ss^{\prime}} = P\left[s _{t+1} = s^{\prime} \vert S_t = s \right]$  
- $\mathcal{R}$ is a reward function, $\mathcal{R}_s = \mathbb{E}[R _{t+1} \vert S_t = s]$
- $\gamma$ is a discount factor, $\gamma \in [0, 1]$

$R_s$는 보상함수로, 상태 $S_s$일 때, 받을 수 있는 즉각적인 보상에 대한 기댓값이다. 중요한 점은 앞으로 받을 보상들을 고려한 누적 보상값이 아닌 즉각적으로 받는 보상(immediate reward)이다. 

현재 상태 $t$에서 다음 step에 받을 보상과 상태가 $r$과 $s^{\prime}$이 될 확률은 다음과 같이 표현한다. 

$$p(s^{\prime}, r \vert s) = P[S_{t+1} = s^{\prime}, R_{t+1} = r \vert S_t = s]$$


## Markov decision process(MDP)

Markov Decision Process(MDP)는 MRP에 행동(actions)이 더해진 것이다. 

**A Markov Decision Process is a tuple** $<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma>$  
- $\mathcal{S}$ is a (finite) set of states
- $\mathcal{A}$ is a finite set of actions  
- $\mathcal{P}$ is a state transition probability matrix,
  $\mathcal{P}_{ss^{\prime}}^a = P\left[s _{t+1} = s^{\prime} \vert S_t = s, A_t = a \right]$  
- $\mathcal{R}$ is a reward function, $\mathcal{R}_s^a = \mathbb{E}[R _{t+1} \vert S_t = s, A_t = a]$
- $\gamma$ is a discount factor, $\gamma \in [0, 1]$

행동(action)까지 고려한 MDP에서의 환경 모델은 다음과 같다. 

$$p(s^{\prime}, r \vert s, a) = P[S_{t+1} = s^{\prime}, R_{t+1} = r \vert S_t = s, A_t = a]$$

### Policies

MDP에서 좋은 의사결정을 하기 위해, 에이전트 내부에 행동 전략(policy)을 가지고 있어야 한다. 

$$\pi(a \vert s) = P[A_t = a | S_t = s]$$

policy의 정의는 위 처럼 현재 상태 $S_t = s$에서, 모든 행동들에 대한 확률 분포이다. 상태는 마르코프 성질을 가지므로, 현재 상태로만으로도 의사결정 시 충분한 근거가 될 수 있다. 따라서, 현재 상태만 조건으로 가진 조건부 확률분포가 된다. 또한 MDP의 policy는 시간에 따라 변하지 않는다(stationary). 이 말은 시간이 지남에 따라 에이전트가 동일한 상태를 여러번 지나도 그 상태에 있을 때의 policy는 변하지 않는다는 뜻이다. 

MDP와 명시적인 policy가 있다면, 이는 MRP문제와 동일하며 MDP의 보상함수는 policy와의 가중평균으로 MRP의 보상함수가 된다. 

$$R^{\pi}(s) = \sum_{a \in A} \pi(a \vert s)R(s, a)$$

마찬가지로, MDP의 상태변이함수도 policy와의 가중평균으로 구해지며 MRP의 상태변이함수가 된다. 

$$P^{\pi}(s^{\prime} \vert s) = \sum_{a \in A} \pi(a \vert s)P(s^{\prime} \vert s, a)$$

## partially observable Markov decision process(POMDP)

<p align="center"><img src="https://github.com/user-attachments/assets/e53d2d91-8fe6-4e8d-9d36-71d33eab0fec" height="300px" width="300px">    <img src="https://github.com/user-attachments/assets/c5464dc8-bb99-4113-b488-173426fceae7" height="300px" width="300px"></p>

위 환경은 9x9 gridworld로 agent가 파란 박스를 움직여 빨간 박스를 피해 초록 박스에 도달하는 policy를 찾는 예시이다. 왼쪽 그림의 경우 환경의 모든 조건을 관찰 가능하기 때문에 agent가 optimal policy를 찾는 데 알아야할 것들을 그대로 이용할 수 있고 이러한 환경이 앞서 언급한 Markov decision process(MDP)이다. 하지만 real world environment에서는 system의 full state를 agent에 제공하는 경우가 거의 없다. 오른쪽 그림은 해당 world에서 3x3 region만 관측할 수 있고, 이처럼 state의 observability가 보장되지 못하는 환경을 POMDP(Partial Observability MDP)라고 한다. 

partially observable Markov decision process(POMDP)에서는 현 상태에 대한 온전한 정보가 없기 때문에, 상태를 유추할 관찰 정보 observation이 추가된다. 

**A POMDP is a tuple** $<\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{P}, \mathcal{R}, \mathcal{Z}, \gamma>$   
- $\mathcal{S}$ is a (finite) set of states
- $\mathcal{A}$ is a finite set of actions
- $\mathcal{O}$ is a finite set of observations   
- $\mathcal{P}$ is a state transition probability matrix,
  $\mathcal{P}_{ss^{\prime}}^a = P\left[s _{t+1} = s^{\prime} \vert S_t = s, A_t = a \right]$  
- $\mathcal{R}$ is a reward function, $\mathcal{R}_s^a = \mathbb{E}[R _{t+1} \vert S_t = s, A_t = a]$
- $\mathcal{Z}$ is an observation function, $\mathcal{Z}_{s^{\prime} o}^a = P[O _{t+1} = o \vert S _{t+1} = s^{\prime}, A_t = a]$
- $\gamma$ is a discount factor, $\gamma \in [0, 1]$

MDP와 POMDP의 차이를 정리하면 MDP는 $s_t = o_t$이지만 POMDP는 $s_t \neq o_t$라 할 수 있다. 

POMDP의 history $H_t$는 actions, observations and rewards의 시퀀스로 다음과 같이 표현된다. 

$$H_t = A_0, O_1, R_1, \cdots, A_{t-1}, O_t, R_t$$

belief state $b(h)$는 history $H$에 따른 state의 확률분포로 다음과 같이 표현된다. 

$$b(h) = (P[S_t = s^1 \vert H_t = h], \cdots, P[S_t = s^n \vert H_t = h])$$


# Latent Dynamics

본 논문은 dynamics model로 Deep Planning Network (PlaNet)를 제안한다. 모델의 Key contributions은 다음과 같다. 

1. Planning in latent spaces
고차원의 입력 데이터로를 latent space로 매핑하는 encoder를 학습하고, 현재 latent state와 action을 입력으로 받아 다음 latent state를 예측하는 dynamics model을 학습하여 DeepMind의 다양한 task(Cartpole, Reacher, Cheetah, Finger, Cup, Walker...)를 수행한다. 

2. Recurrent state space model(RSSM)  
Deterministic System은 어떤 상태 $s$에서 행동 $a$를 선택할 때 결과가 한 가지로 정해진 시스템이다. 반면 Stochastic System은 같은 $s$와 $a$를 취해도 확률적(노이즈, 관측의 불완전성..)으로 다른 결과가 나올 수 있는 시스템이다. 본 논문은 latent dynamics model에 deterministic과 stochastic components를 모두 사용한다.

3. Latent overshooting
Latent overshooting은 한 단계 앞의 상태를 예측하는 것이 아닌, multi-step을 학습 목표로 포함시켜 단기 예측뿐만 아니라 장기 예측에도 안정적인 성능을 발휘할 수 있게 한다.

## Latent Space Planning

본 논문에서는 일반적으로 관측된 데이터가 환경의 full state를 보여주지 못하기 때문에 POMDP를 사용하고, 다음과 같은 stochastic dynamics를 정의한다. 

- Transition function: $s_t \sim P(s_t \vert s_{t-1}, a_{t-1})$  
- Observation function: $o_t \sim P(o_t \vert s_t)$  
- Reward function: $r_t \sim P(r_t \vert s_t)$  
- Policy: $a_t \sim P(a_t \vert o_{\leq t}, a_{< t})$  

PlaNet이 동작하는 알고리즘은 다음과 같다. 

**Algorithm 1:** Deep Planning Network (PlaNet)  
**Input:**  
$R$ &nbsp;Action repeat&nbsp;&nbsp;&nbsp;&nbsp; $p(s_t \vert s_{t-1}, a_{t-1})$ &nbsp;Transition model  
$S$ &nbsp;Seed episodes&nbsp;&nbsp;&nbsp; $p(o_t \vert s_t)$ &emsp;&emsp;&emsp;&emsp;Observation model  
$C$ &nbsp;Collect interval&nbsp;&nbsp;  $p(r_t \vert s_t)$ &emsp;&emsp;&emsp;&emsp;Reward model  
$B$ &nbsp;Batch size&emsp;&emsp;&nbsp; $q(s_t \vert o_{\leq t}, a_{< t})$ &emsp;Encoder   
$L$ &nbsp;Chunk length&nbsp;&nbsp;&nbsp;&nbsp; $p(\epsilon)$ &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Exploration noise  
$\alpha$ &nbsp;Learning rate




# Reference 
POMDP : https://www.davidsilver.uk/teaching/        
https://ralasun.github.io/reinforcement%20learning/2020/07/12/mdp/  
https://benban.tistory.com/63  


https://planetrl.github.io/  
