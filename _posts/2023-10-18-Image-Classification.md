---
title:  "[CS231n] 1.Image Classification"
excerpt: "CS231n Lecture Note"

categories:
  - Computer Vision
tags:
  - CS231n
toc: true
toc_sticky: true
last_modified_at: 2023-10-18T08:06:00-05:00
---

# Image Classification
<p align="center"><img src="https://github.com/user-attachments/assets/cc3ead32-4b98-4f31-b6e1-bcba4edb1e42"></p>

우리는 위 사진을 보고 바로 고양이 임을 알 수 있지만, 컴퓨터는 이를 어떻게 인식할 수 있을까? 컴퓨터에게 고양이 사진은 오른쪽 사진과 같이 
숫자의 집합체로 인식될 것이다. 이 이미지가 고양이라는 사실과 실제 컴퓨터가 보는 숫자값에는 격차가 있고, 이를 Semantic gap이라 한다. 

고양이를 좀 더 옆에서 찍거나, 조명이 달라지거나, 고양이의 자세가 달라지거나, 사물에 살짝 숨겨지거나 등 이러한 여러 픽셀값들의 변화에 컴퓨터는 모두 '고양이'라고 인식할 수 있어야겠고, 알고리즘이 이러한 변화에 Robust 해야한다.

```py
def classify_image(image):
  # some magic here?
  return class_label
```
이미지 분류에서 우리는 결국 위와 같은 함수가 필요하다. 

<p align="center"><img src="https://github.com/user-attachments/assets/27013236-3653-4d06-ab44-3ab166742002"></p>

이를 위해 이런 위 사진과 같은 방법이 있을 수 있다. 이미지에서 edges를 추출하고 귀모양, 코모양과 같은 고양에게 필요한 집합들을 하나하나 찾아서, 다 있으면 고양이로 분류하는 것이다. 
하지만 이런 방법은 잘 작동하지 않느다. 위에서 언급한 바 처럼 변화에 Robust하지 않기 때문에 결과가 잘 나오지 않는다. 
그리고 다른 객체(강아지, 집)들을 인식하려할때 그 클래스에 맞는 집합을 또 따로 하나하나 만들어야되서 굉장히 비효율적이다. 

그래서 생각한 다른 방식은 Data-Driven Approcach(데이터 중심 접근방법)이다. 
이제 고양이에게 필요한 규칙들, 강아지에게 필요한 규칙들을 만드는것이 아니라, 그냥 엄청나게 방대한 고양이 사진, 강아지 사진을 컴퓨터한테 제시하고, Machine Learning Classifier을 학습시키는 것이다. 

# Nearest Neighbor Classifier
```py
def train(images, labels):
  # Machine learning!
  return model
  
def predict(model, test_images):
  # Use model to predict labels
  return test_labels
```

데이터 중심 접근방법의 가장 기초인 Nearest neighbor을 알아보면, 이제는 이미지와 레이블을 input으로 주고 머신러닝을 시키는 train 함수와, train 함수에서 반환된 모델을 가지고 우리가 궁금한 테스트 이미지를 판단하는 predict 함수로 나눠지는 것이다. 

![image](https://github.com/user-attachments/assets/146bd8b1-3e4f-4039-a6f5-1da568017f6d)

CIFAR10 데이터셋을 가지고 더 자세히 살펴보면, 위 그림에서 오른쪽의 화살표 왼쪽에 있는 그림들은 우리가 궁금한 테스트 이미지이다. 
그리고 2열부터의 학습 데이터는 테스트 이미지와 유사한 순으로 정렬한 것이다. 눈으로 보기엔 유사하게 분류되었지만 자세히 보면 분류가 잘못된 것들도 있다.   

## Distance Metric - L1 distance
유사함을 판단하여 정렬하는 방법은 어떤 비교 함수를 사용하느냐에 따라 달려있다. 
이 예제에서는 L1 Distance(Manhattan distance)를 사용했다. 

![image](https://github.com/user-attachments/assets/f68eab34-fd6f-42b3-98e6-39f7b1b6ec07)

각 픽셀값을 비교하고, 그 차이의 절댓값을 모두 합하여 하나의 지표로 설정하는 것이다. 
위의 예제는 두 이미지간에 456만큼 차이가 나는 것을 계산할 수 있다. 

이런 식으로 각 테스트 이미지와 train 데이터를 비교해서 nearest한 것을 찾아낸다. 
위 알고리즘은 Train 함수의 시간복잡도보다 Predict에서가 더 시간이 많이 걸리게 된다. 
Train은 데이터를 그냥 넣기에 O(1)이지만 Predict는 오차가 가장 작은 항목을 일일이 찾아야 하기 때문에 O(N)이다.  
실제 사용시에, 학습시간은 오래걸려도 예측시간이 짧은 걸 선호할 것이다. 이것은 NN의 정말 큰 단점이고, 나중에 배울 parametic model들에서 해결되는 것을 확인할 수 잇다. 

<p align="center"><img height="40%" width="40%" src="https://github.com/user-attachments/assets/506be4c9-c5f2-4aca-b2d4-994f12c38ba9"></p>

어쨋든, NN을 이용하여 decision regions를 그려보면 위와 같다. 
점은 학습데이터, 점의 색은 클래스 레이블입니다. 여기서 눈여겨봐야 할 것은, 녹색 한 가운데 노란색 영역, 초록색 영역에서 파란색 영역 침범하는 구간 등이다. 
decision boundary가 Robust 하지 않음을 볼 수 있고 이는 NN의 두번째 한계이다. 

# K- Nearest Neighbor Classifier
좀 더 일반화를 위해 KNN(k-nearest neighbors)를 도입했다. 

<p align="center"><img height="40%" width="40%" src="https://github.com/user-attachments/assets/7edb5917-12be-473d-9dcc-39c6c7c1300c"></p>

위 사진을 보면 이해가 쉽다. 새로운 데이터가 주어졌을 때 (빨간 점) 이를 Class A로 분류할지, Class B로 분류할지 판단하는 문제인데, 
k=3일 때, 즉 안 쪽 원을 먼저 살펴보면, 빨간 점 주변에 노란색 점(Class A) 1개와 보라색 점(Class B) 2개가 있다. 
따라서 k=3 일 때는 해당 데이터가 Class B (보라색 점)으로 분류된다. 
k=6일 때를 보면, 원 안에 노란색 점 4개와 보라색 점 2개가 있으므로 노란색으로 분류된다. 

## Distance Metric - L2 distance
L2 Distance(Euclidean distance)는 유클리드 거리로 계산을 한다. 

<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/7f05c158-7a70-43e0-84f3-b0b0fef989bc"></p>

위 그림에서 두 그래프 모두 각 distance 방식을 이용했을 때 원점으로부터 같은 거리에 있는 점들을 나타낸 것이라 볼 수 있다. 
L1 distance의 경우 좌표계가 회전하면 변하고 L2는 좌표계랑 독립적이다. 
어떤 거리척도를 써야할지는 상황마다 의존적인데, L1 distance는 특징 벡터의 각 요소들이 개별적 의미를 가지고 있을 떄 유용하다. 

<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/ed460876-a257-4560-a376-99102f9f2e59"></p>

위 그림에서도 L1의 경우 region도 좌표축에 영향을 받고 있는 것을 볼 수 있는 반면, L2는 decision boundary가 좀 더 부드럽다. 

# Hyper Parameter
KNN을 사용할 때 우리가 정해주어야 하는 거리 척도, K값 같은 것들을 "하이퍼 파라미터" 라고 부른다. 
하이퍼 파라미터는 Train time에 학습되는 것이 아니라 직접 지정해줘야한다. 

### Idea #1 
<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/06b099ee-1cea-45dc-a056-55b3e59cf5cd"></p>

Train data에 대해선 완벽할 수 있으나 우리가 원하는건 학습 데이터가 아닌 한 번도 보지 못한 데이터를 얼마나 잘 예측하냐 이다. 

### Idea #2
<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/0b2b6071-bf75-45b5-b322-eccde5e676e7"></p>

전체 데이터를 train과 test셋으로 나눈다. train으로 여러 하이퍼파라미터 값들로 학습을 시키고, test 데이터에 적용해본 다음, 제일 좋은 하이퍼파라미터를 선택하는 방식이다. 하지만 이것도 여전히 한번도 보지 못한 데이터에 대한 방어책은 되지 못한다. 테스트 셋에서만 잘 동작하는 하이퍼파라미터를 고른 것일 수 있다.

### Idea #3
<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/a15f4a0e-f673-4988-9f99-a8535a6e31f9"></p>

train, validation, test로 데이터를 나눈다. 여러 하이퍼파라미터 값들로 train 데이터를 학습시키고, validation set으로 이를 검증한다. validation set에서 가장 좋았던 하이퍼파라미터를 선택. 테스트 셋은 가장 좋은 clasifier로 딱 한번만 수행한다.

### Idea #4
cross validation이라는 것도 있다. 데이터가 작을 때 많이 사용하고, 딥러닝에서는 많이 사용되지는 않는다. 

<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/60457aaf-6633-4724-adad-f0392637467b"></p>

일단 마지막에 딱 한번 사용할 테스트 데이터는 빼놓는다. 나머지 데이터들을 여러 부분으로 나누어주고, 그림처럼 번갈아 가면서 validation set을 바꾸어준다.
초록색 데이터로 하이퍼 파라미터를 학습시키고, 노란색에서 이를 평가 후 최적의 하이퍼파라미터를 결정한다. 딥러닝은 학습 계산량이 많아 이렇게 까지는 안한다. 

<p align="center"><img height="50%" width="50%" src="https://github.com/user-attachments/assets/fd62372f-6d6b-4b84-8c32-d73d7af17a83"></p>

k에 따른 정확도 그래프를 보면, 각 K마다 5번의 cross validation을 통해 알고리즘을 평가할 수 있다. 이 방법은 테스트셋이 알고리즘 성능에 미치는 영향을 알아볼 때 도움이 되고, 해당 하이퍼파라미터에서 성능의 분산값도 알 수 있다.

KNN은 실제로 이미지 분류에 잘 쓰지 않는다. 속도(train t < test t)도 느리고 L1, L2 distance가 이미지간 거리척도로써 적절하지 않다.

# Linear Classification
NN(Neural Network)과 CNN의 기반이 되는 알고리즘이다. parametric model의 기초가 된다. 

### parametric model이란?

<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/f3894898-5971-42c5-a60f-f57b27d4ca79"></p>

입력 이미지 : x  
파라미터(가중치) : W  

CIFAR10 데이터셋을 이용하고, 고양이 사진을 x라 할때, 함수 f는 x와 w를 가지고 10개의 숫자를 출력한다. 
이 10개의 숫자는 데이터셋의 각 클래스에 해당하는 스코어의 개념으로, "고양이" 스코어가 높다면 "고양이"일 확률이 큰 것으로 볼 수 있다.

KNN에서는 파라미터를 이용하지 않았고, 전체 트레이닝 셋을 Test time에서 다 비교하는 방식이었다면, parametric 접근법에서는 train 데이터의 정보를 요약해서 파라미터 w에 모아주는 것이라 생각할 수 있다.  
따라서 test time에 더 이상 트레이닝 데이터를 직접 비교하지 않고, W만 사용할 수 있게 된 것이다. 딥러닝에서는 이 함수 f를 잘 설계하는 것이 중요하다.  

Linear Classification은 $f = Wx$이다. 

<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/3edfa041-dcb9-435a-b498-9971bbb7df45"></p>

입력 이미지(32x32x3)을 하나의 열벡터로 피면 (3072x1)가 된다. 
이 x를 W와 행렬 곱했을 때 10개의 스코어가 나와야 되므로, W는 10x3072가 되어야하고, 결론적으로 10x1의 스코어를 가져다 줄 수 있다. 

종종 Bias도 더해주는데, Bias는 데이터와 무관하게 (x와 직접 연산되지 않음) 특정 클래스에 우선권을 부여할 수 있다. 
각 클래스에 scaling offsets를 추가해줄 수 있는 것이다. 

<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/4095c3b6-f3b0-43b4-a93e-8df0ccaa76de"></p>

input은 고양이 사진, output은 cat, dog, ship에 대한 스코어이다. 

Linear classifier는 템플릿 매칭의 관점에서도 볼 수 있다. 
가중치 행렬 W의 각 행은 각 이미지에 대한 템플릿으로 볼 수 있고, 결국 w의 각 행과 x는 내적이 되는데, 이는 결국 클래스의 탬플릿(w)과 인풋 이미지(x)의 유사도를 측정하는것으로 이해할 수 있다. 
아래 이미지는 가중치 행렬이 어떻게 학습되고 있는지를 보여준다. 각 클래스에 대해서 하나의 템플릿만을 학습한다는 것이 문제가 되는데, 나중에 신경망으로 해결된다. 

<p align="center"><img height="80%" width="80%" src="https://github.com/user-attachments/assets/9098ce34-383d-4796-be1c-8530ebc473f4"></p>

이미지를 고차원 공간의 한 점이라고 생각하면, Linear classifier은 아래와 같이 각 클래스를 구분시켜주는 선형 boundary 역할을 한다. 
아래 공간의 차원은 이미지의 픽셀 차원과 동일하다. 

<p align="center"><img height="50%" width="50%" src="https://github.com/user-attachments/assets/6b1f0866-8213-496c-8852-f52d51ef8383"></p>

하나 문제가 더 있는데, 아래와 같은 데이터 셋은 선형분류하기 힘들다. (parity problem, multimodal problem)

<p align="center"><img height="50%" width="50%" src="https://github.com/user-attachments/assets/1df8eb9f-0b4d-4377-b72f-a2cb4655ef1e"></p>

# Reference 
https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=2  
https://oculus.tistory.com/7  
https://cs231n.github.io/classification/    
Fully connected layer : https://dsbook.tistory.com/59  





