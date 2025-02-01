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

Markov process에서 sampling된 시퀀스를 **에피소드(episode)** 라 한다. 

## Markov reward process(MRP) 

Markov Reward Process(MRP)는 Markov chain에 **reward** 가 더해진 것이다. 임의의 state들의 시퀀스를 상태 변환 확률에 따라 지나가면서 각 상태에 도착할 때마다 보상을 얼마나 받는지도 시퀀스로서 파악하는 것이다. 

**A Markov Reward Process is a tuple** $<\mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma>$  
- $\mathcal{S}$ is a (finite) set of states
- $\mathcal{P}$ is a state transition probability matrix,
  $\mathcal{P}_{ss^{\prime}} = P\left[s _{t+1} = s^{\prime} \vert S_t = s \right]$  
- $\mathcal{R}$ is a reward function, $\mathcal{R}_s = \mathbb{E}[R _{t+1} \vert S_t = s]$
- $\gamma$ is a discount factor, $\gamma \in [0, 1]$

$R_s$는 보상함수로, 상태 $S_s$일 때, 받을 수 있는 즉각적인 보상에 대한 기댓값이다. 중요한 점은 앞으로 받을 보상들을 고려한 누적 보상값이 아닌 **즉각적으로 받는 보상(immediate reward)** 이다. 

현재 상태 $t$에서 다음 step에 받을 보상과 상태가 $r$과 $s^{\prime}$이 될 확률은 다음과 같이 표현한다. 

$$p(s^{\prime}, r \vert s) = P[S_{t+1} = s^{\prime}, R_{t+1} = r \vert S_t = s]$$


## Markov decision process(MDP)

Markov Decision Process(MDP)는 MRP에 **행동(actions)** 이 더해진 것이다. 

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

MDP에서 좋은 의사결정을 하기 위해, 에이전트 내부에 행동 **전략(policy)** 을 가지고 있어야 한다. 

$$\pi(a \vert s) = P[A_t = a | S_t = s]$$

policy의 정의는 위 처럼 현재 상태 $S_t = s$에서, 모든 행동들에 대한 확률 분포이다. 상태는 마르코프 성질을 가지므로, 현재 상태로만으로도 의사결정 시 충분한 근거가 될 수 있다. 따라서, 현재 상태만 조건으로 가진 조건부 확률분포가 된다. 또한 MDP의 policy는 시간에 따라 변하지 않는다(stationary). 이 말은 시간이 지남에 따라 에이전트가 동일한 상태를 여러번 지나도 그 상태에 있을 때의 policy는 변하지 않는다는 뜻이다. 

MDP와 명시적인 policy가 있다면, 이는 MRP문제와 동일하며 MDP의 보상함수는 policy와의 가중평균으로 MRP의 보상함수가 된다. 

$$R^{\pi}(s) = \sum_{a \in A} \pi(a \vert s)R(s, a)$$

마찬가지로, MDP의 상태변이함수도 policy와의 가중평균으로 구해지며 MRP의 상태변이함수가 된다. 

$$P^{\pi}(s^{\prime} \vert s) = \sum_{a \in A} \pi(a \vert s)P(s^{\prime} \vert s, a)$$

## partially observable Markov decision process(POMDP)

<p align="center"><img src="https://github.com/user-attachments/assets/e53d2d91-8fe6-4e8d-9d36-71d33eab0fec" height="300px" width="300px">    <img src="https://github.com/user-attachments/assets/c5464dc8-bb99-4113-b488-173426fceae7" height="300px" width="300px"></p>

위 환경은 9x9 gridworld로 agent가 파란 박스를 움직여 빨간 박스를 피해 초록 박스에 도달하는 policy를 찾는 예시이다. 왼쪽 그림의 경우 환경의 모든 조건을 관찰 가능하기 때문에 agent가 optimal policy를 찾는 데 알아야할 것들을 그대로 이용할 수 있고 이러한 환경이 앞서 언급한 Markov decision process(MDP)이다. 하지만 real world environment에서는 system의 full state를 agent에 제공하는 경우가 거의 없다. 오른쪽 그림은 해당 world에서 3x3 region만 관측할 수 있고, 이처럼 state의 **observability** 가 보장되지 못하는 환경을 **POMDP(Partial Observability MDP)** 라고 한다. 

partially observable Markov decision process(POMDP)에서는 현 상태에 대한 온전한 정보가 없기 때문에, 상태를 유추할 관찰 정보 **observation** 이 추가된다. 

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

**PlaNet**은 Latent State 만으로 Planning을 진행하고, 별도의 Policy Network 없이 Planning의 결과만을 바탕으로 Action을 결정하는 **Model-Based Algorithm**이다. 
이미지와 같은 고차원의 state를 갖는 경우 planning에 많은 비용이 들어가고, 정확도가 떨어지는 문제가 있다. 이러한 문제를 해결하기 위해 데이터를 저차원의 의미 있는 표현(latent representation)으로 encoding하여 Latent Space 상에서 Prediction과 Loss 계산을 수행한다. 

## Latent Space Planning

본 논문에서는 일반적으로 관측된 데이터가 환경의 full state를 보여주지 못하기 때문에 **POMDP**를 사용하고, 다음과 같은 stochastic dynamics를 정의한다. 

- Transition model(Gaussian): $s_t \sim P(s_t \vert s_{t-1}, a_{t-1})$    
- Observation model(Gaussian): $o_t \sim P(o_t \vert s_t)$    
- Reward model(Gaussian): $r_t \sim P(r_t \vert s_t)$   
- Encoder: $s_t \sim q(s_t \vert o_{\leq t}, a_{< t})$    
- Policy: $a_t \sim P(a_t \vert o_{\leq t}, a_{< t})$  

$o_t$는 Environment로부터 전달받은 이미지를, $s_t$는 $o_t$를 encoding하여 얻은 Latent State를 의미한다.

**Algorithm 1:** Deep Planning Network (PlaNet)  
**Input:**  
$R$ &nbsp;Action repeat&nbsp;&nbsp;&nbsp;&nbsp;   
$S$ &nbsp;Seed episodes&nbsp;&nbsp;&nbsp;   
$C$ &nbsp;Collect interval&nbsp;&nbsp;     
$B$ &nbsp;Batch size&emsp;&emsp;&nbsp;      
$L$ &nbsp;Chunk length&nbsp;&nbsp;&nbsp;&nbsp;     
$\alpha$ &nbsp;Learning rate    
$p(s_t \vert s_{t-1}, a_{t-1})$ &nbsp;Transition model  
$p(o_t \vert s_t)$ &nbsp;Observation model  
$p(r_t \vert s_t)$ &nbsp;Reward model   
$q(s_t \vert o_{\leq t}, a_{< t})$ &nbsp;Encoder     
$p(\epsilon)$ &nbsp;Exploration noise  

Initialize dataset $\mathcal{D}$ with $S$ random seed episodes.  
Initialize model parameters $\theta$ randomly.  
**while** not converged **do**  
&emsp;// Model fitting  
&emsp;**for** update step $s=1..C$ **do**  
&emsp;&emsp;Draw sequence chunks ${(o_t, a_t, r_t)_{t=k}^{L+k}} _{i=1}^B \sim \mathcal{D}$ uniformly at random from the dataset.  
&emsp;&emsp;Compute loss $\mathcal{L}$ from [Equation 3].  
&emsp;&emsp;Update model parameters $\theta \gets \theta - \alpha \nabla _{\theta}\mathcal{L}(\theta)$.  
&emsp;// Data collection  
&emsp;$o_1 \gets$ env.reset()  
&emsp;**for** time step $t=1...\lceil \frac{T}{R} \rceil$ **do**  
&emsp;&emsp;Infer belief over current state $q(s _t \vert o _{\leq t}, a _{< t})$ from the history.  
&emsp;&emsp;$a _t \gets$ planner $(q(s _t \vert o _{\leq t}, a _{< t}), p)$, see [Algorithm 2] in the appendix for details.  
&emsp;&emsp;Add exploration noise $\epsilon \sim p(\epsilon)$ to the action.    
&emsp;&emsp;**for** action repeat $k=1..R$ **do**    
&emsp;&emsp;&emsp;$r _t^k, o _{t+1}^k \gets$ env.step($a _t$)     
&emsp;&emsp;$r _t, o _{t+1} \gets \sum _{k=1}^R r_t^k, o _{t+1}^R$    
&emsp;$\mathcal{D} \gets \mathcal{D} \cup \lbrace(o _t, a _t, r _t) _{t=1}^{T} \rbrace$  

**belief**는 Observation을 인코딩하여 얻은 Latent State $q(s _t \vert o _{\leq t}, a _{< t})$를 의미한다.

알고리즘의 목표는 expected sum of rewards $E_p \left[\sum_{t=1}^T r_t \right]$를 최대화 하는 policy $p(a_t \vert o_{\leq t}, a_{< t})$를 찾는 것이다. 

다음은 Planning Algorithm이다. 

**Algorithm 2:** Latent planning with CEM  
**Input :**  
$H$ &nbsp;Planning horizon distance  
$I$ &nbsp;Optimization iterations  
$J$ &nbsp;Candidates per iteration  
$K$ &nbsp;Number of top candidates to fit  
$q(s_t \vert o_{\leq t}, a_{< t})$ &nbsp;Current state belief  
$p(s_t \vert s_{t-1}, a_{t-1})$ &nbsp;Transition model  
$p(r_t \vert s_t)$ &nbsp;Reward model  

Initialize factorized belief over action sequences $q(a_{t:t+H}) \gets Normal(0, \mathbb{I})$.  
**for** optimization iteration $i=1..I$ **do**    
&emsp;// Evaluate $J$ action sequences from the current belief.    
&emsp;**for** candidate action sequence $j=1..J$ **do**    
&emsp;&emsp;$a_{t:t+H}^{(j)} \sim q(a_{t:t+H})$  
&emsp;&emsp;$a_{t:t+H+1}^{(j)} \sim q(s_t \vert o_{1:t}, a_{1:t-1}) \prod_{\tau=t+1}^{t+H+1} p(s_{\tau} \vert s_{\tau-1}, a_{\tau-1}^{(j)}$  
&emsp;&emsp;$R^{(j)} = \sum_{\tau=t+1}^{t+H+1} E \left[p(r_{\tau} \vert s_{\tau}^{(j)}) \right]$  
&emsp;// Re-fit belief to the K best action sequences.  
&emsp;$\mathcal{K} \gets argsort({R^{(j)}}_ {j=1}^J)_{1:K}$  
&emsp;$\mu _{t:t+H} = \frac{1}{k} \sum _{k \in \mathcal{K}} a _{t:t+H}^{(k)},  \sigma _{t:t+H} = \frac{1}{K-1} \sum _{k \in \mathcal{K}} \vert a _{t:t+H}^{(k)} - \mu _{t:t+H} \vert.$  
&emsp;$q(a _{t:t+H} \gets Normal(\mu _{t:t+H}, \sigma _{t:t+H}^2 \mathbb{I})$  
return first action mean $\mu _t$.     

### Model-based planning
PlaNet은 transition model($p(s_t \vert s_{t-1}, a_{t-1})$), observation model($p(o_t \vert s_t)$), reward model($p(r_t \vert s_t)$)을 학습하고, 현재 hidden state에 대한 belief를 근사시키기 위해 encoder $q(s_t \vert o_{\leq t}, a_{< t})$를 필터링을 통해 학습한다. 이렇게 얻어진 components를 통해 future action 시퀀스를 planning algorithm을 통해 구한다. model-free나 d hybrid reinforcement learning algorithms과는 다르게, 명시적인 policy를 직접 사용하지 않고, Planning을 통해 다음 action을 선택한다. 

### Experience collection
초기에는 모델이 학습되지 않은 상태이므로 random actions으로 수집된 seed episodes에서 시작한다. seed episodes를 따라 환경과 상호작용하며 수집된 데이터로 모델을 학습하고 학습된 모델을 사용하여 planning을 수행하여 더 나은 데이터를 수집한다. 이 과정에서 발생한 추가 episode를 data set에 넣는다. 이런식으로 data set에 episodes를 수집할 때, action에 small Gaussian exploration noise를 추가한다고 한다. 

$$a_t = \pi(z_t) + \epsilon_t,  \epsilon_t \sim \mathcal{N}(0, \sigma^2)$$

$\pi(z_t)$는 현재 latent state $z_t$를 기반으로 선택된 action이다. Gaussian noise를 추가함으로써 agent가 다양한 행동을 시도하고, 환경을 더 넓게 탐색하도록 하여 Exploration을 강화할 수 있다. 추가적으로 **planning horizon**을 줄이고 모델에 명확한 학습 신호를 보내기 위해 각 action을 $R$번 반복한다고 한다. planning horizon은 모델이 미래 상태를 예측하고 계획을 수립할 때 고려하는 시간 범위로 planning horizon이 길어질수록 예측해야 하는 미래 상태의 불확실성이 커지고, 예측 오차가 누적될 가능성이 높아진다. 


### Planning algorithm
Planning algorithm으로는 **cross entropy method(CEM)** 을 사용한다. Algorithm 2에 나와있듯이, 초기 Action Sampling Distribution은 Normal Distribution $a_{t:t+H} \sim Normal(\mu_{t:t+H}, \sigma_{t:t+H}^2 \mathbb{I})$으로 한다. $t$는 agent의 현재 time step이고, $H$는 planning horizon길이이다. 현재 시점 $t$에서 Planning을 통해 Action $a_t$를 결정하겠다는 것은 Latent State $s_t$에서 시작하여 만들어질 수 있는 무한한 State-Action Sequence 중에서 기대 누적 Reward가 가장 큰 경우에 따르겠다는 것을 뜻한다. PlaNet은 매 Optimization Iteration마다 $q(a_{t:t+H})$에서 Sampling하며, 이 중 Sequence의 기대 누적 Reward $R^{(j)}$가 큰 순서대로 $K$개를 뽑아 이들의 분포로 Action Sampling Distribution을 re-fit한다. 총 $I$번의 반복 후, planner는 $s_t$의 action으로 current time step에 대한 Action Sampling Distribution의 평균 $\mu_t$를 반환한다.


#### cem
https://towardsdatascience.com/cross-entropy-method-for-reinforcement-learning-2b6de2a4f3a0  
https://leekh7411.github.io/_build/html/Reinforcement_Learning_method_Cross_Entropy.html  
https://liger82.github.io/rl/rl/2021/06/05/DeepRLHandsOn-ch04-The_Cross-Entropy_Method.html  
https://wnthqmffhrm.tistory.com/13


# Reference 
POMDP : https://www.davidsilver.uk/teaching/        
https://ralasun.github.io/reinforcement%20learning/2020/07/12/mdp/  
https://benban.tistory.com/63  

논문: https://enfow.github.io/paper-review/reinforcement-learning/model-based-rl/2020/09/13/learning_latent_dynamics_for_planning_from_pixels/

https://planetrl.github.io/  
