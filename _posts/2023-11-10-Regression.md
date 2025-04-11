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

최소제곱법(OLS)에서는 다음 조건을 만족하는 $b_0, b_1$을 찾는 게 목적이다. 
1. 잔차들의 합이 0이어야 한다. $\to \sum_{i=1}^n (y_i - \hat{y}_i) = 0
2. 직선과 데이터 간의 거리(오차 제곱합)가 최소가 되야 한다.

앞서 회귀선은 $(\bar{x}, \bar{y})$를 무조건 지난다고 했는데 이는 1번 대문이다. 잔차합이 0이면 다음과 같다. 

$$\sum_{i=1}^n (y_i - \hat{y}_i) = 0 \to \sum y_i = \sum \hat{y}_i \to \bar{y} = \bar{\hat{y}}$$

즉, 관측값의 평균과 예측값의 평균이 같게 되고, 예측값의 평균 $\bar{\hat{y}}$은 다음과 같다. 

$$\bar{\hat{y}} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i) = \frac{1}{n} \sum _{i=1}^n (b_0 + b_1x_i) = b_0 + b_1\bar{x}$$

이를 다시 정리하면 $\bar{y} = b_0 + b_1\bar{x} \to b_0 = \bar{y} - b_1\bar{x}$가 되므로 $(\bar{x}, \bar{y})$를 무조건 지난다. 

이제 2번 오차 제곱합이 최소가 되야한다는 조건을 이용해 계수를 구해보자. 오차 Error를 정의하면, 

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

예를 들어,  
$\hat{y}_i = 10 + 20x_i$ (회귀)  
$y_i = 10 + 20x_i + \epsilon_i$ (관측)  
일 때, 실제 관측값이 (2, 55)라면 회귀식에 의한 $y_i$값은 $10 + 20 \times 2=50$ 실제 관측값은 55니까 $\epsilon_i = 5$가 된다. 

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

추가로 $R_{adjusted}^2 = 1 - \frac{(1-R^2)(n-1)}{n-k-1}$인데, $n$은 표본의 수, $k$는 독립변수의 수이다. 표본수가 많을 수록, 독립변수가 많을수록 결정계수를 조정해줘야 한다는 것이다. 이유는 Least Square 방법이 $\min(SSE) = \min(\sum\epsilon^2) = \min_{\beta} \sum(y_i - \hat{y})^2$가 되는데, 다중 회귀에서는 $\min_{\beta} \left( \sum_{i=1}^n(y_i - (\beta_0 + \beta_1x_{i,1} + \beta_2x_{i,2} \cdots + \beta_kx_{i,k}))^2 \right)$ 가 된다. $n$은 관측수, $k$는 독립변수 수이다. 

여기서 $R^2 = \frac{R}{T} = 1 - \frac{E}{T} (\because T = R + E / \therefore R = T - E)$ 이고, SSE를 최소화하는 것이 $R^2$을 최대화 하는 것이 된다. 

여기서 만약 1개의 독립변수가 추가된다면,   

$$\min_{\beta} \left( \sum_{i=1}^n(y_i - (\beta_0 + \beta_1x_{i,1} + \beta_2x_{i,2} \cdots + \beta_kx_{i,k} + \beta_{k+1}x_{i+1,k+1}))^2 \right)$$

이런 식으로 괄호안의 빼기가 늘어나게되고, 이는 $k+1$번째가 $x$가 $y$와 전혀 상관관계가 없다면 0이고, 그 외에는 무조건 더 빼질 수밖에 없으니까 SSE는 줄어들 수 밖에 없고, $R^2$는 최소한 그대로 유지하거나 늘어날 수 밖에 없다. 이를 non-decreasing property of R square라 하는데, 이 때문에 결정계수는 계속 커질 수 밖에 없으므로 조정을 해주는 것이다. 

회귀를 할 때  Least Sqaure 방법을 쓴다고 하는데, Least Square와 Mean Square, Root Mean Square는 다른 개념이다.  Least Square는 잔차의 제곱의 합을 최소화하여 회귀선을 구하는 방법이고, Mean Square, Root Mean Square는 모델에 대한 성능평가를 위한 Error 측정방법이다. 

회귀분석은 결국 각각의 데이터의 잔차(residual)의 제곱의 합이 최소화되는 공식을 도출하는  방법이다. 따라서 $R^2$값이 1에 가까울수록 설명력이 좋다고하는데 그렇다는 것은 $T = R + E$에서 $R$이 $T$에 가깝다는 뜻이므로 $E$가 작다 즉, 잔차가 작다는 뜻이다. 이는 결국 데이터가 회귀직선에 가까이 있다는 뜻과 같다. 상관계수를 다룰 때는 상관계수가 크다고 더 촘촘하다고 할 수는 없었는데, 결정계수의 경우는 1에 가까울수록 관측 데이터가 더 촘촘하다고 할 수 있다. 

<p align="center"><img src="https://github.com/user-attachments/assets/e0408600-8888-4c00-883c-3a360efb6069" height="" width=""></p>

추가적으로 OLS 선형 회귀에서는 $R^2$가 크면 좋은 모델로 판단하는 것이 일반적인데, 결정계수 $R^2$이 높다는 것이 꼭 모델의 쓸모 있음을 판단하는 의미는 아니다. 대강 결정계수가 0.25이면 상관계수는 0.5로 매우 크다. 결정계수가 0.09면 상관계수는 0.3으로 엄청 나쁜정도는 아니다. 결국 설명력이 낮다는 것은 직선이 나쁘다는 의미가 아니라 거꾸로 데이터가 회귀에 적합하지 않다고 볼 수도 있다. 

OLS의 $R^2$가 상관계수의 제곱인데, 서술해보면 다음과 같다. 

$$R^2 = \frac{R}{T} = \frac{\sum(\hat{y}_i - \bar{y})^2}{\sum(y_i - \bar{y})^2} = \frac{\sum((b_0 - b_1x_i) - (b_0 + b_1\bar{x}))^2}{\sum(y_i - \bar{y})^2} = b_1^2 \frac{\sum(x_i - \bar{x})^2}{\sum(y_i - \bar{y})^2}$$

이겨서 $b_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$ 이므로 전개하면 다음과 같다. 

$$R^2 = \left( \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} \right)^2 \cdot \frac{\sum(x_i - \bar{x})^2}{\sum(y_i - \bar{y})^2} = \left( \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2} \right)^2 = r^2$$ 

## Regression Analysis Test


```py
R-squared:	0.952 ...(1)
Adj. R-squared:	0.936 ...(1)
F-statistic:	59.41 ...(2)
Prob (F-statistic):	0.00454 ...(2)


(3)      	coef	std err	t	P>|t|	[0.025	0.975]
Intercept	-27.8	18.331	-1.517	0.227	-86.136	30.536
X		0.426	0.055	7.708	0.005	0.25	0.602
```

앞서 회귀분석 결과에서 확인했듯이 회귀분석에도 **(2)** F검정과 **(3)** t검정이 있다. 

### $t$ test

먼저 t 검정부터 살펴보면 가설은 $y = \beta_0 + \beta_1x $가 회귀식이라 할 때 Null Hypothesis은 $\beta_1=0$이고, Alternative Hypothesis은 $\beta_1 \neq 0 $이다. 

가설의 의미는 모 기울기가 0이면 즉, 저번에 다룬 엉망 회귀선과 같은경우 귀무가설을 기각하지 못하면 회귀분석을 통해 회귀선을 찾았다 해도 엉망이라는 뜻이다. 기울기가 0이 아니면 모집단의 독립변수와 종송변수가 관계가 있긴 하고, 회귀분석이 어느정도 의미가 있다는 뜻이다. 

t검정은 이전에 살펴보았던 One Sample t-Test와 동일하고 검정통계량은 다음과 같다. 

$$t_{stat} = \frac{\bar{X} - \mu}{\frac{s}{\sqrt{n}}} = \frac{\bar{x}- \mu}{SE} \ where \ SE = \frac{s}{\sqrt{n}}$$

회귀분석에서의 검정통계량을 쓰면 다음과 같다. 

$$t_{stat} = \frac{b_1 - \beta_1}{SE(b_1)} = \frac{b_1 - 0}{SE(b_1)}$$

$SE(b_1)$는 회귀계수 $b_1$의 표준오차인데 구하는 방법을 살펴보면 우선 $SE$는 분산에 루트를 씌운 값이다. 분산은 $Var(b_1) = Var \left( \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} \right)$이다. 여기서 $(x_i - \bar{x})$를 $c_i$로 치환하면 $= Var \left( \frac{\sum c_i(y_i - \bar{y})}{\sum c_i^2} \right) = \sum \left( Var \left( \frac{c_i(y_i - \bar{y})}{c_i^2} \right)\right)$   
$c_i$를 밖으로 빼면 제곱이 되고, 상수의 Var은 0이므로 $\sum \left( \frac{1}{c_i^2}Var(y_i - \bar{y}) \right) = \sum \left( 
\frac{Var(y_i)}{c_i^2} \right)$ 가 된다.   
$Var(y_i) = Var(b_0 + b_1x_i + \epsilon_i) = Var(\epsilon_i)$ 이다(Var(상수) = 0 이므로). 여기에서 모든 $i$에 대해 $Var(\epsilon_i)$는 가우시안을 따르는 오차항으로써 $\sum$과 관계없는 $Var(y_i) \triangleq \sigma^2$ (상수)로 정의할 수 있다. 
결국 $\sum \left( \frac{\sigma^2}{c_i^2} \right) = \sum \left( \frac{\sigma^2}{(x_i - \bar{x})^2} \right) = \frac{\sigma^2}{\sum(x_i-\bar{x})^2}$ 가 된다. 

여기서 기울기는 y/x의 비(ratio)인데, $x$는 상수이고, $y_i$는 개별 오차항 $\epsilon$이 가우시안이기 때문에 회귀계수는 다음의 가우시안 분포를 따른다. 

$$b_1 \sim \mathcal{N} \left( \beta_1, \frac{\sigma^2}{\sum(x_i - \bar{x})^2} \right)$$

여기서 우리는 가우시안을 따르는 모분산의 오차항의 분산을 모르고 표본의 분산밖에 모르기 때문에 $\sigma$ 대신 $s$를 구해야 한다. 
$s$는 $s^2=Var(y_i)=Var(b_0+b_1x_i+\epsilon_i) = Var(\epsilon_i)=\frac{\sum(\epsilon_i - 0)^2}{dof} = \frac{\sum(\epsilon_i)^2}{dof}$ 가 된다. $Var(\epsilon_i)$ 에서 $(\epsilon_i - 0)$가 나오는 이유는 오차항 $\epsilon$의 평균이기 때문에 0을 기준으로 $\pm$ 오차라 0 인 것 같다. 자유도 dof는 $n-2$인데 이유는 이후에 설명할 것이다. 

정리하면 $b_1 \sim t_{n-2} \left( \beta_1, \frac{s^2}{\sum(x_i - \bar{x})^2} \right) \ where \ s^2=\frac{\sum \epsilon_i^2}{n-2}$ 가 된다. 추가적으로 T, R, E에서 $\sum \epsilon_i^2=SSE$이다. 따라서 SE는 $\sqrt{\frac{s^2}{\sum(x_i - \bar{x})^2}} = \sqrt{\frac{SSE/(n-2)}{\sum(x_i - \bar{x})^2}}$ 이다. 

### $F$ test 

이제 F 검정을 다뤄볼 것이다. 회귀분석에서의 F 검정은 회귀모델이 유의한가, 유의하지 않은가를 따지는데 목적이 있다. 독립변수 ($x_i$)들이 종속변수 ($y_i$)를 설명하는데 있어서 계수가 의미가 있는지, 아니면 다중 회귀의 경우 한 개 이상의 독립변수가 종속변수를 설명하는 유의한 설명력을 가지는가를 본다. 

이러한 의미로 귀무가설은 t 검정과 마찬가지로 $\beta_1 = 0$이다. 마찬가지로 종속변수가 영향을 끼치지 않는다는 의미이고, 이후에 $\hat{y} = b_0 + b_1x_1 + b_2x_2 + \cdots + b_ix_i$의 다중회귀에도 적용할 수 있다. 대립가설은 $\beta_1 \neq 0$이다. 

ANOVA F 검정을 다시 살펴보면 F = 설명가능한 변량의 평균 / 설명하지 못하는 변량의 평균 = $\alpha / \beta$이다. 

이를 다시 해석하면 $\alpha$는 효과의 분산 $\beta$는 오차의 분산 이라 할 수 있고, 회귀에서 다음과 같다. 
 
$\alpha$: 회귀선을 찾았으므로 회귀선과 평균(엉망 회귀선)과의 차이의 평균 (R)    
$\beta$: 회귀선을 찾았지만 여전히 있는 관측치와 회귀선과의 차이의 평균 (E)  

ANOVA F 검정에 따라 $R^2$에서는 R/T를 보았다면, F 검정에서는 R/E 를 보는 느낌이다. F 검정을 위해서는 분산을 다뤄야 하는데, SSE, SSR는 평균을 내지 않고 합이므로, 차이 제곱의 합의 평균을 내어 분산으로 만든 뒤, 분산의 비로 $F = \frac{Mean(SSR)}{Mean(SSE)} = \frac{MSR}{MSE}$ 를 통해 검정한다. 분산의 비로 통제불능의 Residual에 비해 얼마나 Regression이 개선되었는가를 분산으로 측정한다. 
비율을 보면 이 값이 클 때 Residual 제곱합(E)은 작거나, 회귀 개선의 제곱합(R)이 크다는 것인데, Residual(E)이 작다면 Regression 근처에 Data가 몰려있다는 뜻이고, 몰려 있다는 것은 Regression이 의미가 있다고 할 수 있다. 그러면 자연스럽게 $R^2$도 커진다. 

분산의 비를 봐야 하기 때문에 SSR, SSE 각각을 평균내야 하는데, 평균을 내기 위해 각각의 자유도로 나누게 된다. T, R, E 각각의 자유도를 알아보면 T의 자유도는 n-1, R의 자유도는 1, E의 자유도는 n-2이다. T의 자유도부터 보면 관측과 평균의 차이인 표본분산의 자유도이므로 평균을 구하는데 쓰인 1을 빼서 n-1이다. R의 자유도는 회귀계수 $b_0, b_1$를 추정하는데 2개의 파라미터가 필요한데 $b_0$ 절편은 $\bar{y}$를 통해 조정되므로(b_0 = \bar{y} - b_1\bar{x}) 1을 빼서 설명변수의 수 2-1=1이 자유도가 된다. E의 자유도는 $n$개 중 $b_1, b_2$를 추정했으므로 2를 빼서 $n-2$가 된다.   

<p align="center"><img src="https://github.com/user-attachments/assets/99c0e582-c387-48fc-8fc4-195aea4c4580" height="" width=""></p>

표로 정리하면 위와 같고 $F_0 > F(1, n-2; \alpha)$이면 귀무가설을 기각한다. 

실제로 계산해보면 F 검정의 p value와 t 검정의 p value가 같은 것을 확인할 수 있다. t 검정의 검정 통계량을 제곱하면 F 검정 통계량이 되고, p value는 같다. 결국 단순 회귀 분석에서 F와 t의 검정이 같은데 이유는 t(dof=k)의 제곱은 F(dof1 = 1,dof2 = k)이기 때문이다. t 제곱이 F이기 때문에 p value가 동일하고 단순회귀분석 즉, 독립변수가 한 개 일때 그 계수에 대한 t 검정의 회귀모델이 유의한지 검정하는 F 검정과 동일하다. 실제 검정통계량도 t 검정 통계량을 제곱하면 F 검정 통계량과 동일하다. 

마지막으로 앞서 ANOVA가 언급됐었는데, Regression의 F검정은 사실 ANOVA이다. 회귀를 위해 각 y의 분포를 3차원으로 표현하면 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/604fed93-a58d-4787-a9ca-ef60a4b970b8" height="" width=""></p>

이걸 오른쪽의 시선으로 바라보면 다음과 같이 ANOVA의 형태가 된다. 

<p align="center"><img src="https://github.com/user-attachments/assets/7a706358-4776-4952-93c2-6af384ec860f" height="" width=""></p>

결국 집단간의 차이를 보는 것과 똑같다. ANOVA의 입장에서 보면 집단내 분산 Within은 MSE가 되고, 집단간 분산 Between은 MSR이 된다. 따라서 ANOVA F 검정의 $F = \frac{Var_{Between}}{Var_{Within}} = \frac{MSR}{MSE}$가 된다. 결국 Regression의 F검정은 연속형 집단에서 집단간 최소한 1개의 집단은 차이가 있는가? 와 같다. 옆에서 바라본 것에 대하여 차이가 나면 기울기가 0이 아니라는 것과 같다. 














