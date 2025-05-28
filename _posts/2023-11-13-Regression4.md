---
title:  "[Statistics] Regression IV"
excerpt: Regression

categories:
  - Statistics
tags:
  - Statistics
toc: true
toc_sticky: true
last_modified_at: 2025-01-19T08:06:00-05:00
---

> 데이터 사이언티스트 되기 책: https://recipesds.tistory.com/

## Information Value

로지스틱 회귀 예측 모형을 만들 때, 변수를 선택하는 방법론에 대해 알아보자. 이 선택방법은 회귀분석 결과를 설명 가능하게 하는 것이 목적이 아니라 철저하게 예측을 잘하기 위해 쓰는 방법이다. 

IV와 WoE라는 용어가 있는데, 먼저 IV는 Information Value이고, WoE는 Weight of Evidence이다. IV는 어떤 Feature가 얼마나 로지스틱 종속변수에 영향력과 가치 있는가를 따지고, WoE는 IV를 계산할 때의 가중치가 된다. 

그러면 다중 로지스틱 회귀를 할 때 어떤 Feature가 도움이 되는 Feature일지를 생각해보자. 로지스틱 회귀에 가장 도움이 되는 Feature는 0과 1에 대해서 관측치가 불균형 하면 판단에 도움이 된다고 생각하면 된다. 불균형을 따져보는 가장 손쉬운 방법으로는 각 구간마다 1과 0의 비율의 차이가 어떤지 보면 된다. 

따라서 Event의 비율 (Event %) - Non Event의 비율 (Non Event %)을 생각해 보자. 여기서 %는 100을 곱한 퍼센트가 아니라 비율의 의미이다. 
일단 비율의 차이를 계산하므로 비율이 불균형 할수록 값이 커진다. 즉, 어느 한쪽에 몰려 있을 수록 두 비율의 차이의 절댓값이 커진다. 하지만 이 값 만으로는 치우침의 불균형한 정도를 잘 표현하기 어렵다. 0~1사이의 작은 값이기 때문에 좀 더 과장되게 표현하기 위해 WoE가 존재한다. WoE는 비율의 비율에 log를 씌우는 건데 앞서 살펴본 Log(odds)와 동일한 Logit이다. 

logit변환의 특성을 보면 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/a8be5511-9ed9-4639-8fe9-3d1d7b6934a2" height="" width=""></p>

logit을 자세히 보면 0.5이상에서는 점점 늘어나다가 1이 가까워 지면 매우 커지고 반대의 방향으로는 점점 줄어들다가 0에 가까워지면 매우 작아진다. 

logit을 활용하면 불균형 할 수록 즉, Event가 월등히 커서 비율이 1에 가깝거나, Non Event가 월등히 커서 비율이 0에 가깝거나 하면, 이 값을 Logit 변환했을 때에는 플러스든 마이너스든 매우 큰 값이 되므로 Event와 Non Event의 비율이 비슷한 것들의 효과를 줄이고, 불균형이 큰 것들의 효과를 크게 한다. 

추가적으로 Event 비율 - Non Event 비율의 값이 Negative인 경우에는 WoE도 똑같이 Negative가 되어 서로 곱하면 Positive가 되고, 마찬가지로 서로 Positive인 경우에는 곱하면 Positive가 된다. 따라서 두 값을 곱하면 과장된 불균형 정도를 표현할 수 있다. 

정리해서 어떤 Feature의 "각각의 구간에 대한" 정보가치는 다음과 같이 표현된다. 

$$IV_i=(Event_i\%-NonEvent_i\%) \times WoE_i$$ 

어떤 Feature의 전체의 정보가치는 이걸 모두 더한 값이 된다. 

$$IV_{total}=\sum((Event_i\%-NonEvent_i\%) \times WoE_i)$$

결국 Event와 Non Event의 불균형을 과장해서 표현한 후에 전체적으로 불균형이 어느정도가 되는지를 측정하고 이를 Information Value라 부른다. 거꾸로 말하면 불균형이 없을수록 Information Value가 매우 작고, 이는 Logistic 회귀를 해 봐야 큰 의미가 없다고 주장하는 바와 같다. 

다음 예제를 살펴보자. 

<p align="center"><img src="https://github.com/user-attachments/assets/373dd9a7-1b02-4e03-955a-8d8b205a27bf" height="" width=""></p>

나이대에 대한 Event와 Non Event데이터 이다. IV를 정의대로 계산한다면, 20~60대까지 구간이 10살씩 나눠져 있고 이 구간에 대하여 Event비율, Non Event비율을 구한 후에 Event % - Non Event % = Delta를 각각 구하고 ln(Event % / Non Event %)로 WoE를 구한 후에 서로 곱한다. 그러면 각 나이대별 IV 정보 가치를 구하게 되고, IV 열을 모두 더하게 되면 결국 최종 IV가 되게 된다. 

최종 불균형 정도를 살펴보면 0.24인데, 이 수치에 대한 판단 기준은 아래와 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/f5e8828e-ec84-424e-bfe9-a1c4a2a9e8ed" height="" width=""></p>

참고: https://datapilots.tistory.com/92

## Logistic Classification

Logistic Regression은 확률 회귀선의 회귀식을 구한 후 그 결과를 이용하여 각 독립변수의 영향력을 분석하는 것이 목적이라면, Classification은 분류 예측에 가깝다. 이때, 이 모형에 어떤 Decision Rule을 적용한 후, Logistic Regression의 확률을 이용하여 분류를 할 수 있다. Decision Rule은 결정경계로 1, 0을 구분하는 Decision Boundary를 고려하는 걸 말한다. Logistic Classification은 1/0만 구분해서 Binary Classification이라 부르기도 한다. 

보통은 Decision Boundary을 0.5로 잡지만 상황에 따라 다르다. 

<p align="center"><img src="https://github.com/user-attachments/assets/5905dc17-34e0-46bf-a1f0-e8fd1a2f7bc6" height="" width=""></p>

Threshold를 0.5 즉, 1/2로 본다면 Logistic Regression 회귀식 $P=\frac{1}{1+e^{-(b_0+b_1x)}}$에서 $(b_0+b_1x)=0$이 되면, p=1/2가 된다. 따라서 $x=-\frac{b_0}{b_1}$보다 오른쪽에 있으면 1, 왼쪽에 있으면 0으로 판단한다. 

<p align="center"><img src="https://github.com/user-attachments/assets/83cf40a0-6953-4e6d-bff7-c3c3134119dd" height="" width=""></p>

이때, $-b_0+b_1x$를 0으로 만드는 $x=-\frac{b_0}{b_1}$직선을 Decision Boundary결정경계라고 한다. 

더 복잡한 경우를 다뤄보면, 여러개의 Feature가 있을 때 어떻게 해야 하는가도 표현할 수 있다. 

$P=\frac{1}{1+e^{-(b_0+b_1x_1+\cdots+b_ix_i)}}$에서 $b_0+b_1x_1+\cdots+b_ix_i=0$이 되면 threshold는 P=1/2가 된다. 

간단한 예시로 $x_1, x_2$ 분포가 다음과 같을 때, $b_0+b_1x_1+b_2x_2=0$을 생각해보자. 

<p align="center"><img src="https://github.com/user-attachments/assets/2f53f47a-30af-4f84-bf5c-2fa80fcd386d" height="" width=""></p>

이 분포에서 직선을 그어서 두 개의 class를 구분해야 한다. 마찬가지로 $b_0+b_1x_1+b_2x_2=0$인 선을 그으면 로짓이 0이므로 확률이 1/2인 선이 된다. $x_2$를 세로축으로 하여 정리하면  $x_2=-\frac{b_1}{b_2}x_1-\frac{b_0}{b_2}$ 직선이 된다. 

예를 들어 $b_0=-3, b_1=1, b_2=1$인 경우라면 $x_2=-x_1+3$을 경계로 하고 이 상태의 결정경계는 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/62280268-ec91-466b-9438-916ca4a35cbd" height="50%" width="50%"></p>

이 Decision Boundary에서 어떻게 Class를 판단하는 방식은 다음과 같다. 

$$b_0+b_1x_1+b_2x_2 \geq 0 \to P \geq 1/2 \to y=1$$

$$b_0+b_1x_1+b_2x_2 < 0 \to P < 1/2 \to y=0$$

이 같은 Decision Boundary를 설정하여 확률이 0.5를 기준으로 1, 0을 구분할 수 있다. 

$x_1, x_2$의 2차원에 대한 로지스틱 회귀는 실제로 아래와 같은 식으로 생겼다. 

<p align="center"><img src="https://github.com/user-attachments/assets/e0ffa0a8-341a-4a45-b956-8b4dd2a7ccf5" height="" width=""></p>

여기서 두 $x$에 대한 평면에 있는 $b_0+b_1x_1+b_2x_2=0$직선이 확률 세로축(z축)으로 그리면 다음과 같은 Decision Boudary 평면이 된다. 

<p align="center"><img src="https://github.com/user-attachments/assets/7621fbb5-666d-4a0b-91f6-25f923074d28" height="" width=""></p>

이런 경우에 Logit을 단순회귀로 풀 수도 있겠지만, Polynomial로 할 수도 있다. 이는 Logistic 함수에서의 Logit을 여러가지 형태로 둘 수 있다는 말인데, Logit(x)를 다음과 같이 변형할 수 있다. 

1. 단순 직선

$$\frac{1}{1+e^{-(b_0+b_1x_1+b_2x_2)}}$$

2. 2차식 곡선

$$\frac{1}{1+e^{-(b_0+b_1x_1+b_2x_2+b_3x_1^2+b_4x_2^2+b_5x_1x_2)}}$$

3. n차식 곡선

$$\frac{1}{1+e^{-(b_0+b_1x_1+b_2x_1^2+b_3x_1^2x_2+b_4x_1^2x_2^2 \cdots)}}$$

이런 식으로 Logit을 비선형으로 표현할 수도 있다. $S(z)=\frac{1}{1+e^{-z}}$라 정의하고 1/2인 z=0인 선을 그림으로 표현하면 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/96592d28-169a-4029-a778-9e3184093b6c" height="" width=""></p>

이런 식으로 Logit을 Polynomial로 표현할 수 있다. 심지어는 Decision Boudary가 원으로도 가능하다. 

<p align="center"><img src="https://github.com/user-attachments/assets/0deabc12-d3c7-4fac-a307-5ce5a9f3d9c3" height="" width=""></p>

$$y=\frac{1}{1+e^{-(x_1^2+x_2^2-1)}}$$

이런 식이면 반지름이 1인 원의 바깥과 안쪽으로 결정경계를 그릴 수도 있다.

이번에는 종속변수가 다중클래스인 경우에 Classification을 어떻게 할 수 있는지 살펴보자. 예를 들어 3개 class가 있다고 하자. 

<p align="center"><img src="https://github.com/user-attachments/assets/c29f74ef-b40c-4e71-bb1e-14ba9517db22" height="" width=""></p>

다중클래스인 경우 1개 클래스와 나머지 클래스를 묶어서 2개 클래스를 나누듯이 여러 번 나누면 된다. 

<p align="center"><img src="https://github.com/user-attachments/assets/ddb7cb59-f0ff-4413-a0c2-556c05668e3b" height="" width=""></p>

이런 경우라면 3번을 시행할 수 있다. 종속변수가 Multi Class인 경우에는 Logisitc 회귀를 두 개씩 짝지은 만큼 시행하는 것이다. 
판단 기준은 어떤 새로운 입력값 x가 있을 때 x에 대해 이 3번의 시행 후 나오는 경우 중 가장 높은 확률이 나온 클래스를 선택한다. 

추가적으로 단층신경망(로지스틱회귀)으로 분류 문제를 풀 수 있는가 없는가?를 판단할 때에는 시각화한 후에 Decision Boundary를 그릴 수 있는가 없는가로 판단할 수 있다. XOR문제 같은 경우에는 단순하게 Boundary를 그릴 수 없기 때문에 다층신경망을 이용해서 공간을 Non Linear 하게 뒤틀어 버린 후에 Classification을 하게 된다. 



