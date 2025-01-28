---
title:  "Latent Dynamics"
excerpt: "Paper: Learning Latent Dynamics for Planning from Pixels"

categories:
  - Statistics
  - Autonomous Driving
tags:
  - Paper
  - Statistics
  - Autonomous Driving
last_modified_at: 2025-01-19T08:06:00-05:00
---

Abstract : 

여기서 말하는 dynamics는 environment 모델에 대한 확률모델을 말한다. 즉 논문 제목의 learning latent dynamics는 모델 p(s', r | s, a)에 대한 latent representation을 agent가 학습하는 것을 말한다. 본 논문은 이 latent code를 학습하고 이를 통해 planning을 하는 PlaNet을 제안한다.그리고 

multi-step으로 변분추론에 대한 기법인 latent overshooting를 제한한다.





INTRODUCTION

model을 직접적으로 학습하여 이를 agent의 학습에 이용하는 방법은 학습된 모델이 불안정하고 불완전하다는 점에서 상당한 리스크가 존재했다. 그리고 학습시간도 오래 필요한데, 모델을 학습하기 위해 충분한 데이터가 요구되고, replay buffer에 저장해야 할 pixel input의 크기도 상당하기 때문이다. 그리고 rnn을 이용해 multi-step 추론을 통한 학습을 한다면 gradient를 계산하는데 엄청난 시간과 메모리가 소요된다.



이러한 문제를 다루는 PlaNet은 다음과 같은 contribution을 가진다.

1. Planning in latent spaces: latent space에서 플래닝을 한다.

2. Recurrent state space model : rnn을 이용한다.

3. Latent overshooting : 멀티스텝 variational inference 방법론. 



Latent Space Planning


PlaNet은 우선 agent로 sample을 모으고(planning을 가미하여), 이를 가지고 dynamic model을 학습하는 단계를 거친다.


Problem setup


다음 네 가지 확률모델을 모델링한다.



Transition model : p(s' | s, a)

Observation model : p(o|s)

Reward model : p(r|s)

Encoder : q(s'|o, a)

policy : p(a|o..., a...)



POMDP라고 해서 o를 s로부터 추론한다. 희안한 점은 reward 모델에 a가 given이 안된다는거다.. 대신 encoder에 a가 given이다.. 아마 encoder에서 o, a로 s를 만들고 이를 reward model이 받기 때문인것 같다. 기존의 POMDP랑 좀 다르게 setting을 한다. policy는 rnn이라서, 현재의 행동을 결정할때 n스텝 이전의 o와 a를 이용한다.







Model-based planning



Policy model로 planning을 한다. 우리는 model-predictive-control(MPC)를 사용하는데, 이를통해 new observation을 기반으로 planning을 한다. each time step마다 replan이 일어난다.







##### Experience collection



랜덤액션으로 S개의 에피소드를 모은다. 그 후, C step동안 모델을 학습시킨다. C번째 스텝 후 부터 에이전트를 이용해 에피소드를 모은다. 여기서 에이전트의 행동에 noise가 추가된다. 여기서 에이전트는 학습하지 않는다. 환경에 따라 R번 frame동안 같은행동을 반복하게한다. reward는 R번동안 일어난 reward들의 합을 reward로 여기고, R번이 지난 후의 observation을 next observation으로 여긴다.







##### Planning algorithm



Cross entropy method(CEM)를 이용해 best action sequence를 탐색(search)한다.  이 방식은 강건하고 true dynamic model이 주어졌을때 대부분의 문제를 해결할 수 있다.(하지만 true dynamic model을 갖고있지 않잖아..?)







CEM은 population-based optimization 알고리즘인데, objective를 최대화하는 action sequence에 대한 분포를 추론하는 방법이다.







처음 time dependent한 action분포 $a_{t:t+H} \sim \mathcal{N}(\mu_{t:t+H}, \sigma_{t:t+T}^{2} I)$를 초기화한다. (H는 planning 길이)



표준정규분포로 시작해서, J개의 candidate action sequence들을 뽑는다. 예측모델을 이용해 이를 평가한다. top K action sequence를 이용해 re-fit한다.



I번째 iteration 후에, planner는 $\mu_t$를 반환한다.  그 후 다음 observation이 오면 다시 표준정규분포로 바꾼다. 



(무슨소리인지 이해가 안간다..)







액션 시퀀스들을 평가하기 위해, 액션 시퀀스를 예측모델로 돌려서 얻은 리워드의 평균으로 평가한다. 1개의 액션시퀀스를 평가하기(리워드의 평균) 위해 1번의 sample을 해도 충분했다. 







#### Latent overshooting



이는 latent space 위에서 multi-step prediction하는 모델의 general한 variational bound이다.







##### multi-step prediction



d step prediction



$$p_{d}(o_{1:T}, s_{1:T} \vert a_{1:T}) =  \prod_{t=1}^{T} p(s_t \vert s_{t-d}, a_{t-d:t-1}) p(o_t \vert s_t)$$


$$p_{d}(s_t \vert s_{t-d}, a_{t-d:t-1}) = \mathbb{E}_{p(s_{t-1} \vert s_{t-d}, a_{t-d:t-2})}[p(s_t \vert s_{t-1}, a_{t-1})]$$


$$= \int p(s_{t-d:t} \vert s_{t-d}, a_{t-d:t-1}) ds_{t-d+1:t-1}$$

$$= \int \prod_{k=t-d-1}^{t-1} p(s_{k+1} \vert s_{k}, a_{k}) ds_{t-d+1:t-1}$$

log likelihood형태는

$$\sum_{k=t-d-1}^{t-1} \int \log p(s_{k+1} \vert s_k, a_k) ds_{k}$$


1-step prediction 확률분포를 이용해 $p_{d}$분포를 계산할 수 있다. 이는 1-step $p$의 unbiased estimator를 만족한다고 한다.


$p_d$ 의 logliklihood는 $p$ 의 logliklihood의 하한이다.

$$I(s_t ; s_{t-d}) \le I(s_t;s_{t-1})$$

로부터 유도할 수 있다. 즉 state가 미래의 mutual information을  담을 수 있도록 한다. 이러한 방식은 policy에의해 상당히 bias될것으로 보인다. 그러나 본 논문의 모델인 planet은 특정한 policy 네트워크를 사용하지 않으므로, 이것이 큰 문제가 되지는 않는듯하다.


##### latent overshooting


d를 fixed하지말고 D개에 대해 일반화한다.

$$\log p_{1}(o_{1:T} \vert a_{1:T}) \ge \frac{1}{D} \sum_{d=1}^{D} \log p_{d}(o_{1:T} \vert a_{1:T})$$

$$\log p_{d}(o_{1:T} \vert a_{1:T}) = \int \log p_{d}(o_{1:T}, s_{1:T} \vert a_{1:T}) ds_{1:T}$$

q는  p의 s->o에서 o->s로 바뀌고 나머지는 같다.

$$\ge \int  q(s_{1:T} \vert o_{1:T}, a_{1:T}) \log \frac{p_{d}(o_{1:T}, s_{1:T} \vert a_{1:T}) }{q(s_{1:T} \vert o_{1:T}, a_{1:T})}  ds_{1:T}$$

$$= \sum_{t=1}^{T} \int q(s_t \vert o_{\le t}, a_{\lt t}) \log{\frac{p_{d}(s_t \vert s_{t-d}, a_{t-d:t-1})p(o_t \vert s_t)}{q(s_t \vert o_{\le t}, a_{\lt t})}}$$

KL 부분만 보자면,

$$\int q(s_{t} \vert o_{\le t}, a_{\lt t}) \log{\frac{p_{d}(s_t \vert s_{t-d}, a_{t-d:t-1})}{q(s_t \vert o_{\le t}, a_{\lt t})}} ds_t$$

$$= \int q(s_{t} \vert o_{\le t}, a_{\lt t}) \log{\frac{\prod_{k=1}^{d} p(s_{t-k+1} \vert s_{t-k}, a_{t-k})}{q(s_t \vert o_{\le t}, a_{\lt t})}} ds_t$$

$$= \mathcal{H}(q(s_t \vert o_{\le t}, a_{\lt t})) +  \int q(s_{t} \vert o_{\le t}, a_{\lt t}) \log{\prod_{k=1}^{d} p(s_{t-k+1} \vert s_{t-k}, a_{t-k})} ds_t$$

여기서 조심해야할 부분은 p(s_t)를 제외한 부분은 p로부터 샘프링된다는 점이다. 뒤부분만 다시 본다면

$$\int q(s_{t} \vert o_{\le t}, a_{\lt t}) \sum_{k=1}^{d} \log{ p(s_{t-k+1} \vert s_{t-k}, a_{t-k})} ds_t$$

여기서 p의 컨디션으로 오는 s는 모두 p로부터온다는 것 왜냐하면  $p_{d}(s_t \vert s_{t-d}, a_{t-d:t-1})$라는 것이 바로 그것을 의미하기 때문이다.


##### Planning algorithm


CEM :



$s_t$에서 plan을 시작한다고 하자. $q(a_{t:t+H} \vert s_{\le t})$


$q(a_{t:t+H} \vert s_{\le t})$을 local하게 학습한다. 처음 Normal 분포로 모델링한다. (본 논문에서는 time independent 모델을 이용했다.)


그리고 이건 neural network가 아니라 MPC라는 방법을 사용했다고 한다.

for i in range(epoch):


​ $a_{t:t+H}$를 J개 sampling한다. 


​ J개에 대한 Reward 계산


​ J개중에서 가장 리워드 높은거 K개 선택


​ 그 데이터에 대해서 $q(a_{t:t+H} \vert s_{\le t})$학습


학습이 끝나면 $q(a_{t})$의 mode를 action으로 반환
