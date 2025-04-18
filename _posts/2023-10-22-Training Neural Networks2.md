---
title:  "[CS231n] 5.Training Neural Networks2"
excerpt: "CS231n Lecture Note"

categories:
  - Computer Vision
tags:
  - CS231n
toc: true
toc_sticky: true
last_modified_at: 2023-10-22T08:06:00-05:00
---

# Optimization
Optimizer는 loss를 줄이기 위해 weight와 learning rate와 같은 neural network 속성을 변경하는데 사용하는 Algorithm이다. 
Optimization algorithms의 전략은 loss를 줄이고 정확한 결과를 제공하는 역할을한다. 

신경망을 학습시키는 과정을 다시 살펴보면 아래와 같다. 
```py
while True:
  data_batch = dataset.sample_data_batch()
  loss = network.forward(data_batch)
  dx = network.backward()
  x += - learning_rate * dx
```
우리가 관심을 가질 부분은 코드의 마지막 줄이다. 위 코드에서는 단순 경사 하강이지만 이러한 update 방법에는 여러가지가 있다. 

<p align="center"><img src="https://github.com/user-attachments/assets/249c9b4f-238d-4ed4-872e-e2d741f5a234" /></p>

### Stochastic Gradient Descent (SGD)
SGD같은 경우 매우 느려서 실제 사용하기가 쉽지 않다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/48054993-3795-4484-ad39-189b9f5f405f" height="50%" width="50%"></p>

SGD가 느린 이유는 이렇다. 위 사진의 경우 경사가 수직으로는 가파르고 수평으로는 완만한걸 알 수 있다. 따라서 경사가 급한 수직 방향으로는 빠르게 움직이고 수평으로는 느리게 움직여 
위 사진과 같이 지그제그형태로 update가 되고, 이 때문에 느리게 학습하는 것이다. 

### Momentum
```py
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```
Momentum은 위 코드와 같이 update가 된다. v는 속도가 되고 속도를 먼저 업데이트 하고 x를 속도로 업데이트 하는 방식이다. 
v를 구하는 식에 있는 뮤($u$)는 마찰계수로 점차 속도가 느려지게 만든다. 

Momentum은 영상을 보면 처음에 v를 빌드업해주는 과정이 있어서 overshooting이 발생하는 것을 볼 수 있다. 

### Nesterov Momentum
```py
v = mu * v - learning_rate * dx
x += v
```

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/39d6669c-7fa9-4b59-b892-f7dd5500eab0" height="70%" width="70%"></p>

기존 Momentum 업데이트의 v를 구하는 부분을 보면 뮤와 v를 곱한 `mu * v`를 momentum step이라 하고, `learning_rate * dx`를 gradient step이라고 한다. 
이에 따라 사진의 왼쪽은 기본 Momentum 업데이트를 하는 방식인데 Nesterov Momentum의 경우 gradient step을 원점에서 진행하지 않고 이미 momentum step이 진행되었을 것이라 예상되는 지점에서 한다. 

영상의 nag를 보면 방향을 예측하며 가기 때문에 일반 momentum보다 빠르게 update하는 것을 볼 수 있다. 

### AdaGrad 
```py
cache += dx ** 2
x += - learning_rate * dx / (np.sqrt(cache) + 1e-7)
```

AdaGrad는 cache라는 개념을 도입한다. 일반적인 경사하강에 cache를 나눈 값으로 계산하는 것을 볼 수 있다. 
cache는 계속해서 값이 커지며 우리가 가진 파라미터 벡터와 동일한 사이즈를 갖는 벡터라 생각할 수 있다. 
즉 파라미터 별로 다른 learning_rate를 제공하게 되는 것이다. 참고로 1e-7는 0으로 나누는 것을 방지하는 작은 값이다. 
AdaGrad가 update를 하는 방식은 SGD에서의 사진을 봤을 때 수직 축은 경사가 크므로 cache가 커져서 update 속도를 낮추게 되고 
수평의 경우 경사가 낮아 cache가 작고 update 속도는 빠르게 된다. 

하지만 AdaGrad도 문제가 있는데, 시간이 지남에 따라 cache가 계속 증가하므로 learning_rate가 0 이 되어 학습이 중단되는 경우가 발생할 수 있다.  


### RMSProp
AdaGrad의 학습 종료 현상을 개선하기 위해 만들어 졌다. 

```py
cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
x += - learning_rate * dx / (np.sqrt(cache) + 1e-7)
```

RMSProp은 decay_rate라는 하이퍼 파라미터를 도입하여 cache의 값이 서서히 줄어들도록 만들어 준다. 
이렇게 하므로서 AdaGrad의 장점인 경도에 따른 조정을 유지하면서 학습 종료 현상을 해결하게 된 것이다. 

### Adam
```py
m = beta1*m + (1 - beta1)*dx # update first moment
v = beta2*v + (1 - beta2)*(dx ** 2) # update second moment
x += - learning_rate * m / (np.sqrt(v) + 1e-7)
```
Adam은 RMSProp과 momentum을 결합한 형태이다. 코드의 1째 줄은 momentum과 유사하고 2~3번째 줄은 RMSProp과 유사하다. 
beta1, beta2는 하이퍼 파라미터로 보통은 0.9 0.99 등으로 설정한다. 

Adam의 최종적인 코드 형태는 아래와 같다. 

```py
m, v = # ... initialize caches to zeros
for t in xrange(1, big_number)L
  dx = # ... evaluate gradient
  m = beta1*m + (1 - beta1)*dx # update first moment
  v = beta2*v + (1 - beta2)*(dx ** 2) # update second moment
  mb = m / (1 - beta1**t) ## correct bias 
  vb = v / (1 - beta2**t) ## correct bias
  x +=  - learning_rate * mb / (np.sqrt(vb) + 1e-7)
```

추가된 부분은 bias correction으로 최초의 m과 v가 0으로 초기화 되었을 때, 즉 t가 작은 수일 때 m, v를 scaling up 해주는 역할을 한다. 


결론적으로 지금까지의 update 방식들은 모두 learning_rate을 하이퍼 파라미터로 갖게 되는데 어떤 learning_rate가 최선이냐는 질문에는 답이 없다. 
learning_rate은 결국 시간이 지남에 따라 decay 시키는 것이 가장 최적이 된다. 
step decay의 경우 가장 간단한 방법으로 epoch을 돌때마다 일정한 간격으로 learning_rate를 감소시키는 방법이다. 
epoch이란 모든 train data 한 바퀴 돌아 학습시키는 것을 말한다. 
이외에도 exponential decay, 1/t decay 등이 있다. 주로 쓰이는 것은 exponential decay라고 한다. 
그리고 update 방법으로는 Adam이 가장 많이 쓰인다. 

#### step decay:
decay learning rate by half every few epochs.

#### exponential decay:
$\alpha  = \alpha_0e^{-kt}$

#### 1/t decay:
$\alpha = \alpha_0 / (1 + kt)$

## Second order optimization methods
지금 까지 알아본 update 방식은 경사를 이용한 first order optimization methods 였고 Second order optimization methods의 경우에는
헤시안($H$)을 이용해 경사 뿐만 아니라 곡면에 대한 정보를 알아내어 학습할 필요없이 최저점으로 이동시키는 방법이다. 

$$
H(f) = 
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1\partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1\partial x_n} \\
\frac{\partial^2 f}{\partial x_2\partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \vdots  \\ 
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n\partial x_1} & \vdots & \vdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

second order Taylor expansion  

$$ J(\theta) \approx J(\theta_0) + (\theta - \theta_0)^T \nabla_{\theta}J(\theta_0) + \frac{1}{2}(\theta - \theta_0)^TH(\theta - \theta_0)$$

Solving for the critical point we obtain the Newton parameter update: 

$$\theta^* =  \theta_0 - H^{-1}\nabla_{\theta}J(\theta_0)$$

convergence가 매우 빠르고 하이퍼 파라미터가 필요 없다는 장점이 있다. 하지만 딥한 신경망에 쓰기에는 현실적이지 못하다. 
파라미터가 1억이라면 헤시안은 1억 x 1억 행렬이고 이 행렬의 역행렬을 구하는 것이 매우 많은 연산을 요한다. 

#### BGFS
위의 연산 단점을 극복하기 위해 헤시안의 역행렬 대신 rank1의 역행렬을 구해서 연산량을 줄이는 것인데, 여전히 메모리 저장을 하기 때문에 큰 신경망에서는 쓰기 어렵다. 

#### L-BFGS
L-BFGS같은 경우 메모리 저장을 요하지 않아서 가끔 사용되기는 한다. 
L-BFGS는 매우 무거운 함수라 랜덤 요소를 없애야 하고 full batch 에서는 잘 동작하지만 mini batch에서는 잘 동작하지 않는다. 

정리하면 대부분의 경우 Adam을 사용하면 되고 full batch update가 가능한 상황이라면 L-BFGS도 시도해 볼 수 있다. 

# Ensembles 
단일 모델을 학습시키는 대신 복수개의 독립모델을 학습시킨다. 이후 테스트 시 이들의 결과의 평균을 내주면 거의 매번 2%의 performance가 향상된다. 
단점은 트레이닝 시에 여러 모델을 관리해야 하고 test 시에도 평균을 내기 위해 그만큼 test 대상이 많아지므로 선형으로 속도가 느려진다. 

여러 모델이 아니라 단일 모델내에서 epoch의 체크포인트들 간에서의 Ensembles을 해도 성능 향상을 할 수 있다. 

더 흥미로운 점은 파라미터 간의 Ensembles도 성능 향상을 보인다는 것이다. 

```py 
while True:
  data_batch = dataset.sample_data_batch()
  loss = network.forward(data_batch)
  dx = network.backward()
  x += - learning_rate * dx
  x_test = 0.995 * x_test + 0.005 * x
```

x_test를 보면 Ensembles을 시키는 것을 볼 수 있다. 

Ensembles이 성능 향상을 가져오는 이유는 minimum에 가야하는데 이 값을 계속 지나친다면 step size가 큰 것이기 때문에 이 step의 평균을 내면 minimum에 가까워 질 수 있을 거라는 논리이다.

# Drop Out 
<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/5d24da12-39e6-47da-8487-dd4a25194389" height="70%" width="70%"></p>

Drop Out을 적용하면 일부 뉴런들을 랜덤하게 0으로 설정하게 된다. 

```py
p = 0.5
def train_step(X):
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = np.random.rand(*H1.shape) < p # first dropout mask 
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # second dropout mask 
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
```

코드를 보면 drop out 확률은 0.5이다. 이렇게 해서 만약 0.5보다 작으면 0 크면 1이 되도록 hidden layer에 곱해서 drop out을 진행한다. 
참고로 *H1, *H2에서 *가 붙은 이유는 튜블을 언팩하기 위함이다. 
이는 역전파에서도 마찬가지이다. drop 된 곳은 0이라는 것이 적용되어 경도가 죽은 상태로 역전파가 진행되고 w는 업데이트가 진행되지 않는다. 

drop out을 하는 이유는 노드가 중복성을 가지도록 하는 것이다. 고양이를 판별하는 신경망이 있을 때 어떤 노드는 귀, 어떤 노드는 입 이런식으로 판별할 때, 어떤 특정 노드가 사라지면 그 노드가 담당한 예를 들면 귀를 아에 인지하지 못할 수도 있기 때문에 drop out으로 학습시켜 다른 노드가 귀도 어느정도 인지하도록 하는 것이다. 

drop out을 하는 이유의 다른 해석은 drop out 또한 다른 하나의 Ensembles로서 모델의 값을 평균하여 성능향상을 가져온다는 것이다. 
drop out이 확률적이니까 매번 신경망의 형태가 달라지게 되고 이렇게 각각의 신경망 결과를 평균낸다는 것이다. 

train 때 drop out을 하여 노이즈를 만들고 이를 다시 통합하려고 할 때 Monte Carlo approximation을 사용할 수 있다. 
이는 test 시에도 drop out을 활용하여 각각의 drop out에 대한 평균을 내자는 것이다. 하지만 이는 매우 비효율 적인 방식이다. 

test 시에는 drop out을 하지 않고 모든 뉴런을 켜야한다.  
유의할 점이 있는데 test 시 x라는 output을 얻었다 할 떄, 만약 p=0.5 였다면 train 때 우리가 얻을 수 있는 결과 값의 기대치가 얼마일지를 생각해보면 아래 사진과 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/8b6b846d-d765-4f11-bf4e-90e77e3df87b" height="70%" width="70%"></p>

즉 p=0.5일 때 train에 사용했던 activation보다 2배가 더 inflate 되는 결과를 얻게 된다. 
즉 test시에 train 때 만큼 scaling 해줄 필요가 있다. 

정리하면 test 시에는 모든 뉴런이 살아있어야 하고 activation 값들을 train 때의 기대치만큼 scaling 해줘야 한다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/56e8ed2e-ebd8-4e7d-be93-b57d9f4d89ff" height="70%" width="70%"></p>

그래서 위 사진을 보면 test 시에 p를 곱해주는 것을 볼 수 있다. 

하지만 이 방법 말고 Inverted dropout이라고 해서 test는 그대로 두고 train 때 p를 나누어 scaling을 미리 처리해주는 것이 보다 일반적이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/b3163b8c-011d-40a8-9bd9-d98d6e7282e1" height="70%" width="70%"></p>




# Reference 
https://www.youtube.com/watch?v=5t1E3LZ3FDY&list=PL1Kb3QTCLIVtyOuMgyVgT-OeW0PYXl3j5&index=5    
https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=7

