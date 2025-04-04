---
title:  "[Statistics] Regression I"
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

# Regression

회귀(Regression) 분석은 변수들 간의 관계를 파악함으로써 어떤 특정한 변수의 값을 다른 변수들로 설명하고 예측하는 통계적 기법이다. 
예측하고자 하는 변수의 값을 Dependent Variable 종속변수라고 하고, 종속변수들을 설명하는 변수들을 Independent Variable 독립변수라 하며, 이 둘 사이의 관계를 도출하는 것을 회귀분석이라 한다. 

회귀를 살펴보기 앞서 오차와 잔차의 차이를 설명하면, Error 오차는 '모집단'으로부터 만든 회귀식에서 예측값과 실제값의 차이이고, Residual 잔차는 '표본'으로부터 얻은 회귀식에서 예측값과 실제값의 차이를 말한다. 정확히 얘기하면 단순화된 회귀모형으로 설명하지 못하는 오차를 말하며, 보통 추정 오차(Estimation Error)와 거의 같은 의미로 사용된다. 결국 회귀식을 찾는다는 것은 Data로부터의 잔차의 제곱합을 최소로 만들어주는 식을 찾는다는 것과 같다. 

회귀는 인과관계를 밝혀내는 분석이 아니다. 앞서 살펴본 상관관계는 선형적 관계가 있는지를 알아내는 것이고, 회귀분석은 막상 선형적 관계를 갖는 변수 두 개가, 독립변수, 종속변수의 관계로 보았을 때 '수치적'으로 어떻게 관련되어 있는지를 보는 것이지 인과관계 분석이 아니다. 

인과관계의 성질은 다음과 같다.   
1. x가 y보다 시간적으로 먼저이고, (또는 논리적으로)  
2. x가 있으면 y가 있고, x가 없으면, y도 없고.  
3. 마지막으로 x와 y사이에 지금보다 더 정확한 영향을 끼치는 원인이 없다.

결국 인과관계는 위 1,2,3의 성질을 만족하는 상관관계라 볼 수 있다. 그리고 회귀는 상관관계를 가지고 있는 두 변수 간의 함수관계를 통계적인 방법으로 알아낼 수도 있는 분석 방법 이다. 

회귀분석에서의 독립변수와 종속변수를 흔히 쓰는 x, y로 설명하면 다음과 같다.   
$x$: 독립변수(예측 변수 or 설명변수) : 결과에 영향을 주는 변수들을 말한다.  
$y$: 종속변수(반응 변수) : 결과 값을 말한다.   

또한 상관분석과 회귀의 가장 큰 차이는 상관분석의 경우, x와 y의 관계나 y와 x의 관계가 비슷하다. 즉, 독립변수와 종속변수의 차이가 없이 얼마나 같이 변하는 가를 확인한다. 반대로 회귀의 경우 x와 y의 회귀와 y와 x의 회귀는 차이가 있고, arbitrary x에 대해서 y를 예측할 수 있다. 

## OLS Regression 

이제 실제 회귀를 하는 법에 대해 알아보자. 회귀는 다음의 회귀식을 구하는 것이 목표이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/7a781959-252e-4c43-81d7-a5acfb76a264" height="" width=""></p>

관측 데이터로부터 회귀식을 구하려면, y절편과 기울기만 구하면 된다. 실제 고차방정식까지의 확장성을 고려하면, $\hat{y} = b_0 + b_1x + b_2x^2 + b_3x^3 + \cdots$ 이런 식이 된다. 

회귀식을 구할 때는 기울기를 구하고, y절편을 구하는데, 이유는 y절편을 구하기 위해서는 $x=0$의 경우가 있어야 하는데, 보통 데이터는 $x=0$의 경우가 거의 없기 때문에 기울기를 먼저 구한다. 기울기를 구할 때 사용하는 것은 최소제곱법이며 결론부터 말하면 $b_1$은 다음과 같이 구한다. 

$$b_1 = \frac{\sum(x_i - \bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}$$

이렇게 $b_1$을 구하면 y절편 $b_0$는 $x=0$에 대한 y값이 없는 경우가 많기 때문에 $(\bar{x}, \bar{y})$의 데이터를 이용한다. 이유는 우리가 구하는 회귀선은 $(\bar{x}, \bar{y})$를 무조건 지나기 때문이다. 

$b_1$이 구해지는 과정에 대해 설명하면, Error 최소화 관점에서 구해지며 이를 (에러)최소제곱법이라 한다. 이렇게 계수를 구하는 방법을 OLS, Ordinary Least Squares라고 부르며 Ordinary인 이유는 최소제곱법 회귀가 가장 기본이 되는 회귀이기 때문이다. 

오차 Error를 다음과 같이 정의하면, 

$$E = \sum(y_i - \hat{y}_i)^2 = \sum(y_i - a -bx_i)^2 = \sum(y_i - b_0 - b_1x_i)^2$$

위와 같고, 최소값을 구하기 위해 오차제곱의 합을 y절편 $b_0$와 기울기 $b_1$에 대해 각각 편미분 하면 다음과 같다. 

$$\frac{\partial E}{\partial b_0} = \sum 2 \cdot (y_i - b_0 - b_1x_i)(-1) = 0 \\  
\sum y_i = \sum b_0 + b_1 \sum x_i \\  
\sum y_i = nb_0 + b_1 \sum x_i \cdots (1)$$

$$\frac{\partial E}{\partial b_1} = \sum 2 \cdot (y_i - b_0 - b_1x_i)(-x_i) = 0 \\  
\sum (y_i - b_0 - b_1x_i)(x_i) = 0 \\  
\sum x_iy_i = b_0\sum x_i + b_1 \sum x_i^2 \cdots (2)$$

위 처럼 2개의 연립방정식이 나오고 두 식 모두 계수에 대해서 아래로 볼록한 convex 함수로부터 미분한 것이므로 두 식을 0으로 만드는 계수 값들이 최소값이다. 결국 (1), (2) 연립방정식을 풀면 $b_0, b_1$에 대한 오차제곱합의 최소값을 구할 수 있다. 

$$\begin{align} 
b_1 &= \frac{n\sum x_iy_i - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2} \\  
&= \frac{\sum(x_i - \bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}
\end{align}$$

$$b_0 = \frac{\sum y \sum x^2 - \sum x \sum xy}{n\sum x^2 - (\sum x)^2} = \bar{y} - b_1\bar{x}$$

$b_1$은 식의 형태가 Covariance를 Variance로 나눈 듯한 형태인데 Regression Coefficent, 즉 기울기는 $x$에 대한 $x, y$의 공동 변화량 느낌으로 생각하면 된다. 

<p align="center"><img src="https://github.com/user-attachments/assets/55950b06-62bc-4fa9-b696-f1b8d4373610" height="" width=""></p>

원래 회귀선은 각각의 $x_i$에서의 관측치 $y_i$가 정규 분포를 갖고, 그때 각각의 $x_i$에 대한 $y$ 예측값을 $\hat{y_i}$라 볼 수 있다. 

다음으로 잔차와 오차의 정의는 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/8ecd13ef-fc15-4e72-8f91-702826ba3f51" height="" width=""></p>

우리가 회귀를 할 때 관심있는건 회귀 모델과 관측값의 차이이고, 그 차이가 잔차(Residual)이다. 

$$E = \sum(y_i - \hat{y}_i)^2 = \sum(y_i - b_0 - b_1x_i)^2$$

잔차는 오차 Error를 정의해서 편미분 한 위 식에서 

$$(y_i - \hat{y}_i)$$

이 값이 잔차이다. 따라서 다시 정리하면 단순 선형회귀는 잔차의 제곱의 합을 최소화하는 계수를 찾아내는 것이라고 정리할 수 있다. 

$$OLS = minimize(\sum(Residual)^2) = minimize(\sum(y_i - \hat{y}_i)^2)$$











