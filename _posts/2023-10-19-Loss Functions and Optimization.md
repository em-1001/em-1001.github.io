---
title:  "[CS231n] 2.Loss Functions and Optimization"
excerpt: "CS231n Lecture Note"

categories:
  - Computer Vision
tags:
  - CS231n
toc: true
toc_sticky: true
last_modified_at: 2023-10-19T08:06:00-05:00
---

# Loss function
### Multiclass SVM loss
손실함수에도 여러종류가 있으나, 기본적이고 이미지분류에도 성능이 좋은 Multiclass SVM loss부터 살펴보자. 식은 다음과 같다. 

$$
\begin{aligned}
L_i &= \sum_{j \neq y_i} 
\begin{cases}
0 \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ if \ s_{y_i} \geq s_j + 1 \\  
s_j - s_{y_i} + 1 \ \ otherwise 
\end{cases} \\  
&= \sum_{j \neq y_i} max(0, s_j - s_{y_i} + 1)
\end{aligned}
$$

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/0159e5f0-3630-4cb3-92c9-b385ba7b534c" height="30%" width="30%"></p>

$S_j$ : Classifier로부터 나온 예측 값  
$S_{y_i}$ : True 값 

이 개념을 이용해 다음 예제를 풀어보자. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/3b8fd972-d322-482d-b988-7635826e2f86" height="40%" width="40%"></p>

cat의 Loss를 구해보면 $max(0, 5.1 - 3.2 + 1) + max(0, -1.7 - 3.2 + 1) = max(0, 2.9) + max(0, -3.9) = 2.9 + 0 = 2.9$로 2.9가 된다. 
마찬가지로 car는 $max(0, 1.3 - 4.9 + 1) + max(0, 2.0 - 4.9 + 1) = max(0, -2.6) + max(0, -1.9) = 0 + 0 = 0$으로 0, 
frog는 $max(0, 2.2 - (-3.1) + 1) + max(0, 2.5 - (-3.1) + 1) = max(0, 6.3) + max(0, 6.6) = 6.3 + 6.6 = 12.0$으로 12.9가 되어 
최종 Loss를 구하면 $L = (2.9 + 0 + 12.9)/3 = 5.27$이 된다. 

1. car 스코어가 조금 변하면 Loss에는 무슨일이 일어날까?  
Loss는 안바뀐다.
2. SVM Loss가 가질 수 있는 최대, 최소값은?   
최소 0, 최대 무한대
3. W가 매우 작아져서 모든 스코어 S가 0과 가깝고, 값이 서로 거의 비슷하다면 Loss는 어떻게 될까?  
비슷하니 그 차가 마진을 넘지 못하기에 마진값에 가까운 스코어를 얻게 됨. 이경우에서는 결국 (클래스의 수-1)
디버깅 전략으로 많이 사용한다. 트레이닝을 처음 시작할 때 Loss가 c-1이 아니면 버그가 있는 것  \
위 사진으로 예를 들면 $max(0, 0 - 0 + 1) + max(0, 0 - 0 + 1) = 2$ cat, car, frog -> $(2 + 2 + 2) / 3 = 2$ 가 된다.
이렇게 0으로 줬을 때 loss가 C-1로 나오는 지를 확인하는 것을 Sanity Check라고 한다. 
5. SVM Loss의 경우 정답인 클래스는 제외하고 더했다. 정답인 것도 같이 계산에 포함시킨다면 어떻게 될까?  
Loss 가 1 증가
6. Loss에서 전체 합이 아닌 평균을 쓴다면?  
영향없다. 단지 스케일만 변할 뿐.
7. 손실함수를 제곱항으로 바꾼다면?  
비 선형적으로 바뀜. 손실함수의 계산이 달라져 결과가 달라진다. squared hinge loss라 한다.

손실함수의 종류는 많다. 오차를 어떤식으로 해석할지는 우리의 몫이다. 가령 squared hinge loss는 큰 loss를 더 크게, 아주 작은 loss는 더 작게 해준다. 어떤 에러를 trade-off 할지 생각하여 손실함수를 정하게 된다.  

Multiclass SVM loss의 파이썬 코드는 다음과 같다. 
```py
def L_i_vectorized(x, y, w):
  scores = W.dot(x)
  margins = np.maximum(0, scores - scores[y] + 1)
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
```

# Regularization loss
loss가 0인 w를 찾았다고 할 때, 이 w는 유일할까? 그렇지 않다. 2w와 같은 값들도 loss는 0이 된다. 
이유는 2w로 하게 되면 정답 스코어의 차이역시 2배가 될 것이고, w에서 이미 차이가 1보다 크다면 2배를 해도 1보다 크고 loss는 여전히 0이 될 것이기 때문이다.  

loss가 작으면 작을 수록 즉 0이 되면 좋다고만은 할 수 없다. w가 0 이라는 것은 train data에서 완벽한 w라는 것인데 사실상 train set보다는 test 데이터 셋에서의 성능이 더 중요하기 때문이다. 이러한 현상을 Overfitting이라 한다. 

사실 함수가 단순해야 test 데이터를 맞출 가능성이 더 커지기 때문에 이를 위해 Regularization을 추가해준다. 
따라서 최종적인 Loss의 식은 다음과 같이 표현된다. Data Loss 와 Regularization loss의 합으로 변하고, 하이퍼파라미터인 람다로 두 항간의 트레이드오프를 조절할 수 있다. 

$$L(W) = \frac{1}{N} \sum_{i=1}^N L_i(f(x_i, W), y_i) + \lambda R(W)$$

Regularization에도 다음과 같이 많은 종류가 있다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/8fb22bec-c072-48fd-b629-e62ac3d666b6" height="60%" width="60%"></p>

그럼 이 Regularization는 모델이 복잡한지 아닌지, 자신이 원하는 모델인지 어떻게 파악할 수 있을까?

$$ 
\begin{align}
x &= [1,1,1,1]  \\     
w_1 &= [1,0,0,0]  \\    
w_2 &= [0.25,0.25,0.25,0.25]  \\  
\end{align}
$$

다음과 같이 x와 w1,w2가 주어졌고, Linear Classification(f=wx)의 관점으로 볼 때, 두 w는 같은 스코어를 제공한다. 앞서 배운 data loss는 같을 것이다. 
L2와 L1 regression의 관점에서 한 번 살펴보자. 

1. L2 regression의 경우 w2를 선호

$$R(W) = \sum_{k} \sum_{l} W^2_{k, l}$$

$W_2$가 norm이 작기 때문이다. coarse한 것을 고르고 모든 요소가 골고루 영향을 미치길 바란다. (parameter vector, Gaussian prior, MAP inference)

2. L1 regression의 경우 w1을 선호

$$R(W) = \sum_{k} \sum_{l} |W_{k,l}|$$

sparse 한 solution을 고르며 0이 많으면 좋다고 판단한다. 

- **L1 Regularization :** weight 업데이트 시 weight의 크기에 관계없이 상수값을 빼게 되므로(loss function 미분하면 확인 가능) 작은 weight 들은 0으로 수렴하고, 몇몇 중요한 weight 들만 남음. 몇 개의 의미있는 값을 산출하고 싶은 sparse model 같은 경우에 L1 Regularization이 효과적. 다만 미분 불가능한 지점이 있기 때문에 gradient-base learning 에서는 주의가 필요.

- **L2 Regularization :** weight 업데이트 시 weight의 크기가 직접적인 영향을 끼쳐 weight decay에 더욱 효과적
 

## Multinomial logistic regression(softmax)
딥러닝에서 자주 쓰이는 다른 유명한 손실함수로 softmax가 있다. 
앞서 Multi-class SVM loss에서는 스코어 자체에 대한 해석보다는 정답 클래스와 정답이 아닌 클래스들을 비교하는 형식으로 이루어졌다.
하지만 Multinomial logistic regression 경우 스코어 자체에 추가적인 의미를 부여한다. 
아래 수식을 가지고 클래스별 확률 분포를 계산하고, 이를 이용해서 Loss를 계산한다. 

$$Softmax \ Function = P(Y = k | X = x_i) = \frac{e^sk}{\sum_{j} e^{s_j}}$$

$$L_i = -log P(Y = y_i | X = x_i)$$

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/94ef5cf7-eab9-4675-8274-ef7f03714661" height="60%" width="60%"></p>

위 예제 사진을 보면 score 자체를 loss로 쓰는 것이 아니라 지수화 시킨다. 그리고 정규화, log 순으로 계산한다. 

SVM과 Softmax를 비교하면 아래와 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/1d41aac2-7a69-4c78-a9df-00277a89010a" height="60%" width="60%"></p>

1. softmax loss의 최대 최소는?   
최소 0, 최대 무한대
2. Loss가 0이 되려면 실제 스코어는 어떤 값이어야 하는가?   
정답 스코어는 극단적으로 높아야 한다. 지수화 정규화를 거치기 때문에 확률 1을 얻어 loss가 0이 되려면, 정답 클래스의 스코어는 +무한대가 되어야 하고, 나머지는 -무한대가 되어야 한다. 하지만 컴퓨터는 무한대 계산을 잘 못하기 때문에 loss가 0인 경우는 절대 없을 것이다. 유한 정밀도를 가지고는 최대 최소에 도달할 수 없다.
3. s가 모두 0 근처에 모여있는 작은 수 일때의 loss는?   
log C. 마찬가지로 디버깅 전략
4. SVM과 softmax의 차이?   
SVM의 경우에는 margin을 넘으면 성능 개선에 신경쓰지 않으나, softmax의 경우 1의 확률을 얻으려 할 것이다. 실제 딥러닝에서 성능 차이가 크진 않다.


# Optimization
지금까지 지도학습의 전반적인 개요를 살펴보았다. f를 정의해서 스코어를 구하고, 손실함수를 이용해서 w도 평가했다. 
어쨋든 최종 목적지는 최종 손실함수가 최소가 되게 하는 W를 구하는 것이다. 

최적화를 위해서 함수의 최소값을 찾는 방식을 사용할 수도 있겠지만, 이는 비효율적이기 때문에 임의의 지점에서 시작해서 점차 성능을 향상시키는 iterative한 방식을 사용한다. 

gradient는 벡터 x의 각 요소의 편도함수들의 집합이다. 그래서 gradient는 그쪽 방향으로 갈때 함수 f의 slope가 어떤지를 알려준다. 
gradient의 방향은 함수에서 가장 많이 올라가는 방향이므로, 이 반대방향으로 간다면 가장 많이 내려갈 수 있을 것이다. 
또한 우리가 원하는 방향의 경사를 알고 싶다면, 그 방향의 unit vec와 gradient를 내적하면 된다. 

$$
\nabla f(p) = 
\begin{bmatrix}
{\partial f \over \partial x_1}(p) \\
\vdots \\ 
{\partial f \over \partial x_n}(p) \\ 
\end{bmatrix}
$$

## Find new weight ($W_{i+1}$)

$$W_{i+1} \leq W_i - \alpha \frac{\partial E}{\partial W_i}$$

현재 가중치 $W_i$에서 $W_i$ 의 경사 값과 Learning rate값인 $\alpha$ 와 곱한 값을 빼서 다음 가중치 값을 구하게 된다. 

딥러닝에서의 경사하강은 아래와 같이 진행된다. 

```py
w = random value 
while(until total loss E on longer decreases):
  1. Calculate predicted value with W
  2. Calculate total loss E (predicted - true)
  3. Update W using gradient descent
```

Gradient descent보다 실제 학습시에 더 좋은 성능을 보이는 Updata Rule들이 있다. (momentum, Adam optimizer 등)

Gradient는 선형 연산자이기 때문에 실제 gradient를 계산하는 과정을 살펴보면, Loss는 각 데이터 Loss의 Gradient의 합이다. 즉, gradient를 한번 더 계산하려면, N개의 전체 트레이닝 셋을 한번 더 돌면서 계산해야하고, 이는 너무 많은 시간을 할애한다. 그래서 실제로는 stochastic gradient descent를 사용한다.


## stochastic gradient descent
전체 데이터 셋의 gradient와 loss를 계산하기보다, Minibatch라는 작은 샘플 집합으로 나누어서 학습을 진행한다. 보통 32,64,128을 쓴다.
minibatch를 이용해서 loss와 gradient의 추정치를 구한다. 이는 Monte Carlo Method와 유사

웹데모 사이트 Linear Classifier와 Gradient descent   
[Multiclass SVM optimization demo (stanford.edu)](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)

지금까지는 실제 Raw 이미지 픽셀을 입력으로 넣어주었다. 하지만 이건 Multi-Modality와 같은 것들 때문에 좋은 방식이 아니다. 
그래서 특징 벡터라는 것을 이용하기로 하고, 이 특징벡터가 Linear Classifier의 input으로 들어가게 된다. 아래와 같이 선형분리가 불가능한 것도, 극좌표계로 특징변환을 하면 분류가 가능해진다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/cc7cacd1-eba9-47c2-b62d-eb4e36f4ab05" height="60%" width="60%"></p>

하지만 이제 CNN,DNN쪽으로 넘어가면 이미 만들어 놓은 특징들을 쓰는 것이 아닌, 데이터로 부터 특징들을 직접 학습한다. raw 데이터가 그대로 들어가고, 여러 레이어들이 특징 벡터를 직접 만들어낸다.


# Reference 
https://wikidocs.net/35476  
https://oculus.tistory.com/8  
https://www.youtube.com/watch?v=h7iBpEHGVNc&t=2779s  
https://mvje.tistory.com/80  
