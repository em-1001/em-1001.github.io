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

사실 회귀분석을 할 때 고려해야할 사항이 몇가지 있다.   
정규성: 잔차의 분포가 정규분포를 따르는지 확인  
등분산성: 잔차의 분포가 등분산성인지 확인. 즉, 잔차가 서로 같은 분포를 가진 iid인지 확인한다. 

$b_1$을 구할 때 미분을 통해 나온 공식으로 구할 수도 있지만 공분산으로도 구할 수 있다. 

$$b_1 = \frac{\sum(x_i - \bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2} = \frac{Cov(x, y)}{Var(x)}$$

x, y의 공분산은 다음과 같이 표현할 수 있다.   
> 공분산의 성질: https://wikidocs.net/256585

$$Cov(x, y) = Cov(x, b_0 + b_1x + \epsilon) = b_0 Cov(x, 1) + b_1 Cov(x, x) + Cov(x, \epsilon)$$

이때, $Cov(x, 1) = 0$이고, $x$와 $\epsilon$은 독립이므로 둘의 공분산도 0이 된다. 따라서 다음과 같이 된다. 

$$Cov(x, y) = 0 + b_1 Cov(x, x) + 0 = b_1 Cov(x, x) = b_1 Var(x)$$

결국 $Cov(x, y) = b_1 Var(x)$ 이므로 $b_1 = \frac{Cov(x, y)}{Var(x)}$가 된다. 

이런 방식 외에도 원래 $b_1$식 $\frac{\sum(x_i - \bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}$에 대해 분모 분자를 각각 자유도인 $n-1$로 나누어도 똑같이 공분산/분산의 형태가 되는 것을 알 수 있다. 

실제 예를 들어 아래와 같은 데이터가 있다고 하자. 

배달거리 x: [100 200 300 400 500]  
배달시간 y: [30 56 76 140 197]

파이썬의 sklearn을 통해 회귀를 할 수 있다. 

```py
from sklearn.linear_model import LinearRegression # Package를 import하고요,
x = [[100], [200], [300], [400], [500]] # x데이터
y = [28, 56, 76, 142, 198] # y데이터
line_fitter = LinearRegression() 
line_fitter.fit(x, y) # Regression 실행
print (line_fitter.coef_) # Slope 출력
print (line_fitter.intercept_) # Intercept 출력
>
[0.426]
-27.79999999999994
```

<p align="center"><img src="https://github.com/user-attachments/assets/ea359a78-9b94-4388-922b-eda7270bf243" height="" width=""></p>

이 방법말고도 더 자세한 데이터 회귀 모델의 분석 결과를 얻을 수도 있고, pandas Dataframe을 이용할 수 있는 방식이 있다. 

```py
from statsmodels.formula.api import ols
import pandas as pd
x = [100, 200, 300, 400, 500] # x데이터
y = [28, 56, 76, 142, 198] # y데이터
intercept = [0, 0, 0, 0, 0]
df = pd.DataFrame({'x':x, 'y':y, 'intercept':intercept})
res = ols('y ~ x', data=df).fit() # y~x는 y=... x라는 뜻
res.summary()
```

```py
R-squared:	0.952 ...(1)
Adj. R-squared:	0.936 ...(1)
F-statistic:	59.41 ...(2)
Prob (F-statistic):	0.00454 ...(2)


(3)      	coef	std err	t	P>|t|	[0.025	0.975]
Intercept	-27.8	18.331	-1.517	0.227	-86.136	30.536
X		0.426	0.055	7.708	0.005	0.25	0.602
```

summary를 확인해보면 값이 굉장히 많이 나오는데, 일단 기울기가 0.426, y절편이 -27.8라는 것은 알 수 있다.   

값들을 순서대로 보면 다음과 같다.  
(1): 모형이 얼마나 설명력을 갖는지? $\to$ 결정계수 R_squared($R^2$)를 확인한다.   
(2): 모형이 통계적으로 유의한지? $\to$ F검정과 유의확률(p value)로 확인한다.   
(3): 회귀계수가 유의한지? $\to$ 회귀계수의 t값과 유의확률(p value)로 확인한다. 

하나하나 자세히 살펴보면 다음과 같다. 

**(1)** 결과를 보면 값이 매우 높다. 이 값은 모형 적합도, 설명력이라 하며 결정계수라 부른다. 0.952의 뜻은 y의 분산을 95.2% 설명한다는 뜻으로 전체 변동의 95.2%를 회귀 결과가 커버한다는 의미이다. 여기에 $R^2$adjusted 도 93.6%로 매우 높은 것을 확인할 수 있는데, 이는 독립변수의 개수와 표본의 크기를 고려하여 R-squared를 보정한 값이다. $R^2$값은 독립변수의 개수와 표본의 크기가 커짐에 따라 과장되는 경향이 있는데 이를 보정한 값이다. 

**(2)(3)** F 검정통계량도 매우 크고, p value도 0.00454로 매우 작으며 유의하다. 이 의미는 관측한 표본뿐 아니라 모집단에서도 통하는 의미있는 모델이라는 의미이다. t 검정도 이후에 확인하겠지만 유의하다. 

결과를 살펴보니 전반적으로 괜찮은 회귀 결과인 것 같은데 어떤 원리로 이런 해석이 나올 수 있는지 알아봐야 할 것 같다. 우선 R_squared값 부터 보면, R_squared는 간단하게 기울기가 0이고, 회귀분석이라고 하기엔 부족한 경우의 기본 회귀선인 $y=\bar{y}$의 직선인 경우의 Residual 제곱의 합과, 실제 회귀선을 찾은 후의 Residual 제곱의 합을 비교하는 것이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/40a4f90c-28cf-457f-a339-1605d965482c" height="" width=""></p>

여기서 T, R, E라는 notation이 사용되는데, T(Total)는 전체 변동, R(Regresssion)는 기본회귀선으로부터 회귀분석을 통해 찾아낸 회귀선 까지의 변동, E(Error)는 실제 관측 데이터와 회귀와의 변동(회귀로 커버칠 수 없는 여전히 존재하는 변동)인 잔차와 관련된 값이다. 여기서 가장 중요한 것은 회귀를 통해 기본회귀선을 R만큼 개선했다는 점이다. 참고로 여기서 R은 R_squared의 R과 다른 R이다. 

이때 설명력 R_squared($R^2$)은 전체오류중에 회귀를 함으로써 얼마나 개선되었는가를 따지는 값이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/f59ddb7c-82e3-4fe4-a559-65eea39312b0" height="" width=""></p>

즉, $R^2$은 전체 편차 중에서 회귀분석을 통해 찾아낸 회귀선이 기본회귀선으로부터 변동을 얼마나 개선했는가 또는 찾아낸 회귀선이 얼마나 Residual을 줄였냐를 의미한다.  

실제 아래 데이터로 $R^2$을 구해보자. 

<p align="center"><img src="https://github.com/user-attachments/assets/9f54926c-ad4b-4caa-9ef8-6bdbabfb116a" height="" width=""></p>

먼저 전체편차 T는 엉망 회귀선의 잔차 제곱의 합이므로 $T = \sum(y - \bar{y})^2$가 되고 결과는 19064이다. 분자 R은 회귀를 통해 얻은 $\hat{y} = -27.8 + 0.426x$ 회귀선과 엉망 회귀선과의 차이 이므로 $R = \sum(\hat{y} - \bar{y})^2$이고 결과는 18147.6이다. 

최종 $R^2$을 구해보면 $\frac{R}{T} = \frac{18147.6}{19064} = 0.9519 = R^2$가 된다. 1에 가까울수록 회귀선이 편차의 대부분을 커버친다고 할 수 있다. 

T, R, E의 $T = R + E$관계를 일반적으로 사용하는 notation으로 정리하면 다음과 같다. 

$$\underbrace{\sum _{i=1}^n(y_i - \bar{y})^2} _{SST} = \underbrace{\sum _{i=1}^n(\hat{y}_i - \bar{y})^2} _{SSR} + \underbrace{\sum _{i=1}^n(y_i - \hat{y}_i)^2} _{SSE}$$

$T$: $\sum(y_i - \bar{y})^2$: 전체 편차를 나타내는 분산 느낌 (기울기 0의 엉망 회귀선으로부터), : SST (Sum of Square Total)  
$R$: $\sum(\hat{y}_i - \bar{y})^2$: 회귀선과 엉망 회귀선과의 변동에 대한 분산 느낌  : SSR (Sum of Square Regression)  
$E$: $\sum(y_i - \hat{y}_i)^2$: 잔차의 제곱의 합. 실제 관찰 값과 회귀선 사이의 편차에 대한 분산 느낌 : SSE (Sum of Square Error)

$R^2$의 값이 1에 가까울수록 추정 회귀식이 표본 자료를 더 잘 설명한다고 할 수 있다. 즉, R²가 클수록 관측치들이 추정 회귀식에 가까이 집중되어 있다는 뜻이다. 









