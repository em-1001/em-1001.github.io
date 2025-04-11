---
title:  "[Statistics] Regression II"
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

## Multiple Regression

지금까지 단순 회귀분석에 대해 알아보았다면 이제는 2개 이상의 독립변수에 의한 다중회귀를 알아볼 것이다. 
다중회귀도 단순 회귀와 마찬가지로 LS(Least Square) 방법을 사용해 계수들을 찾아낸다. 

먼저 지금까지 보았던 독립변수가 1개인 단순 선형 회귀분석의 경우 $\hat{y} = b_0 + b_1x$의 계수를 $\frac{\partial E}{\partial b_0} = 0, \frac{\partial E}{\partial b_1} = 0, \ where \ E = \sum(y_i - \hat{y}_i)^2 = \sum e_i^2$ 로 편미분하여 구했다. 

독립변수가 여러 개인 다중 선현 회귀분석의 경우 $\hat{y} = b_0 + b_1x_1 + b_2x_2 + b_3x_3 \cdots$의 계수들을 구해야 한다. 다중회귀 분석도 마찬가지로 LS(Least Sqaure) 방법을 이용해 최적화된 회귀선을 찾는데, 단순 회귀와 마찬가지로 E를 각 계수로 편미분하여 구한다. 

$E = \sum _{i=1}^n (y_i - \hat{y}_i)^2 = \sum _{i=1}^n e _i^2 = e _1^2 + e _2^2 + e _3^2 \cdots$ 를 $b_1, b_2, \cdots$ 로 각각 편미분 한 결과가 0이되는 방정식들을 모아($\frac{\partial E}{\partial b_1} = \frac{\partial E}{\partial b_2} = \cdots = \frac{\partial E}{\partial b_n} = 0$) 연립방정식을 풀어서 E를 최소화하는 계수들을 구한다. 

연립방정식은 Linear Algebra를 이용해서 푼다. 관측값들을 행렬로 표현하면 다음과 같다. 

$$\begin{align}
y_1 &= b_0 + b_1x_{11} + b_2x_{12} + \cdots + b_kx_{1k} + \epsilon_1 \\  
y_2 &= b_0 + b_1x_{21} + b_2x_{22} + \cdots + b_kx_{2k} + \epsilon_1 \\ 
\vdots \\  
y_n &= b_0 + b_1x_{n1} + b_2x_{n2} + \cdots + b_kx_{nk} + \epsilon_n
\end{align}$$

$$\begin{pmatrix}y_1 \\ y_2 \\ \vdots \\ y_n\end{pmatrix} = 
\begin{pmatrix}
1 & x_{11} & x_{12} \cdots x_{1k} \\ 
1 & x_{21} & x_{22} \cdots x_{2k} \\  
\vdots & \vdots & \vdots \\ 
1 & x_{n1} & x_{n2} \cdots x_{nk}
\end{pmatrix} 
\begin{pmatrix}b_0 \\ b_1 \\ \vdots \\ b_k\end{pmatrix} + 
\begin{pmatrix}\epsilon_1 \\ \epsilon_2 \\ \vdots \\\epsilon_n\end{pmatrix}$$

$y = Xb + \epsilon$이 되고, $E$를 구하면 다음과 같다. 

> 행렬미분: https://datascienceschool.net/02%20mathematics/04.04%20%ED%96%89%EB%A0%AC%EC%9D%98%20%EB%AF%B8%EB%B6%84.html

$$\begin{align}
E = \sum_{i=1}^n \epsilon_i^2 &= \epsilon^T \epsilon = (y - Xb)^T(y - Xb) \\  
&= y^Ty + b^TX^TXb - 2b^TX^Ty \\  
\end{align}$$

$$\begin{align}
\frac{\partial E}{\partial b} = 2X^TXb - 2X^Ty = 0 \\  
X^TXb = X^Ty \\  
b = (X^TX)^{-1}X^Ty
\end{align}$$

결과를 보면 $X^TX$의 역행렬이 존재해야 한다. 혹시라도 Full Rank가 안되어 행렬식이 존재하지 않는다면 b의 해가 무수히 많아져 특정할 수 없다. 이는 후에 설명할 다중공선성과 연관된다. 

다중회귀의 검정은 각각의 계수에 대해 t검정을 하고 가설은 다음과 같다. 

Null Hypothesis :  "해당" 회귀계수는 0이다.   
Alternative Hypothesis : "해당" 회귀계수가 0은 아니다.   

검정 통계량은 단순 회귀에서 보았듯이 각각의 계수에 대해서 $t_i = \frac{b_i}{SE(b_i)}$가 되고, 모든 회귀계수에 대해서 이 t 통계량을 이용해 p value가 계산된다. 

F 검정의 경우 정의인 MSR/MSE 대로 구하면 된다. 실제 예를 들어서 아래와 같은 데이터가 있다고 하자. 

<p align="center"><img src="https://github.com/user-attachments/assets/55e61c56-7a2b-4fde-a519-3def4cda5593" height="" width=""></p>

음식 준비시간, 배달 숙련도, 배달 거리를 독립변수로, 배달 시간을 종속변수로 한다. 

```py
import statsmodels.formula.api as smf
 
sales = smf.ols("배달시간 ~ 음식준비시간 + 배달숙련도 + 배달거리", data=df).fit()
```

<p align="center"><img src="https://github.com/user-attachments/assets/e3503b94-f320-412b-b3b5-689aaca54a25" height="" width=""></p>

회귀분석 결과 위와 같다.

➊: 각각의 독립변수에 대한 계수(b)값들이다. 회귀선으로 표현하면 다음과 같다. 

y(배달시간)=-12.648223 + 0.7700365 음식준비시간 + 0.2802135 배달숙련도 + 1.105769 배달거리 

배달거리가 단위 변화에 대해 가장 변화율이 크다는 것을 확인할 수 있다. 

➋: R_squared값이 매우 큰 것으로 보아 회귀가 잘 된 것 같다. 

➋: F검정에 의한 결과를 보면 p value가 0.00458로 유의한 결과가 나왔다. 

➍: Double tail t 검정에 의한 각각의 계수에 대한 p value인데, 배달거리 빼고는 나머지가 유의하지 않다. 앞서 회귀결과가 잘 나온것으로 보아 나머지 계수들에 대해서도 유의한 값이 나와야 할 거 같은데 그렇지 않다. 이유는 이후에 다룰 것이다. 

결국, 전체적인 해석을 해보자면, "배달시간은 음식준비시간이 0.77정도의 영향을 미치고, 배달숙련도가 0.28정도의 영향을, 그리고 배달거리가 1.10 정도의 영향을 미친다고 해석할 수 있고, 이 결과는 꽤나 괜찮은 R²적합도와 유의한 F검정의 결과를 보았을 때, 쓸만한 결과라 할 수 있다. 

$b=(X^TX)^{-1}X^Ty$에 따라 계수들을 직접 구해보면 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/4f060a15-f78b-4492-a9e8-e8973457a528" height="" width=""></p>

SST, SSR, SSE도 계산해보면 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/a0c70bff-1c63-48a3-96e2-f9c185c6f360" height="" width=""></p>

$R^2 = \frac{SSR}{SST}$이므로, 계산해보면 SSR/SST =  2370.07/2377.33=0.9969가 된다. 또한 이전에 알아보았듯이 $y, \hat{y}$ 의 상관계수가 $r _{y\hat{y}} = 0.9984703$ 가 되는데, 이 값을 제곱하면 0.9969427로 $R^2$와 동일하다. 

SST, SSR, SSE를 구한 것을 기반으로 F 검정까지 해보면 가설은 다음과 같다. 

$H_0$: $b_1=b_2=b_3=\cdots=b_k=0$   
$H_1$: $b_j \neq 0, \ for \ some \ j$  

$F=\frac{\frac{SSR}{df_{SSR}}}{\frac{SSE}{df_{SSE}}} = \frac{MSR}{MSE}$이므로 $\frac{2370.07/3}{7.27/2} = 217.392$가 된다. 

자유도를 보면 우선 SST는 평균 $\bar{y}$를 구하는데 하나가 쓰이므로 $n-1$이다. SSR은 회귀선 $\hat{y}$를 만들 때 회귀계수 $b_0, b_1, \cdots, b_p$를 추정해야 한다. 이때 $p+1$개의 파라미터가 필요하지만 $b_0$ 절편은 $\bar{y}$를 통해 조정되므로($b_0 = \bar{y} - b_1\bar{x}$) 설명변수의 개수 $p$ 만큼만 자유도로 사용한다. SSE는 관측치 $n$개 중 $p+1$개가 모델 추정에 쓰였으므로 남은 자유도 $n-p-1$이다. 

따라서 $df_{SST} = n-1, \ df_{SSR} = p, \ df_{SSE} = n-p-1$ 이므로 $df_{SST} = df_{SSR} + df_{SSE}$의 관계가 성립한다. 

추가적으로 $\frac{SSR}{SSE} = \frac{SSR}{SST - SSR} = \frac{\frac{SSR}{SST}}{1 - \frac{SSR}{SST}} = \frac{R^2}{1 - R^2} = F \frac{df_{SSR}}{df_{SSE}}$ 이므로 $F = \frac{R^2}{1 - R^2} \times \frac{df_{SSE}}{df_{SSR}}$가 된다. 즉, F 검정통계량을 $R^2$결정계수로도 구할 수 있다. 

이제 앞서 결과에서 보았던 것 처럼 R_squared나 F 검정의 결과는 괜찮은데, 각각의 독립변수 계수의 t검정 통계량이 유의하지 않은 것에 대해 알아보면 다음과 같은 원인 때문일 수 있다. 

1. 표본 수가 너무 적다.
2. Non linear Correlation이 변수 사이에 존재한다.
3. 어떤 독립변수가 collinearity (공선성)을 발생시킨다.

1번의 경우 표본이 너무 적으므로 더 뽑으면 된다. 권장 표본 크기는 적어도 한 개의 독립변수 당 10 ~ 15개이다. 예를 들어 두 개의 독립변수에 대하여 다중회귀를 한다면 적어도 30 ~ 45 정도가 적당하다. 

2번의 경우 Non linear Polynomial Regression을 시도해 볼만 하다. (2차 이상의 다항식). 

3번의 경우 공선성은 독립변수들끼리 상관관계가 있어서 계수의 분산이 커져 불안정한 것을 말한다. 이럴 때 전체적으로는 변수들을 더하거나 빼서 문제를 해결하는데,  제대로 된 회귀분석을 위해서 유의하지 않거나 필요 없는 독립변수를 제외하는 방법을 많이 사용한다. 하지만 마구잡이로 독립변수를 제거하는 것은 자료의 다양성을 망가트리고, 원래의 분석 목적에 영향을 미칠 수 있기 때문에 조심해서 빼야한다. 예를 들어 유의성이 없더라도 분석에 꼭 필요한 것은 남겨야 한다. 이렇게 변수들을 제외하거나 PCA 같은 것을 이용해서 제외할 변수들을 하나로 만드는 방법으로도 접근 가능하다. 

이전에 독립변수를 추가하면 독리변수가 유의하든 유의하지 않든 $R^2$는 최소한 같거나 커진다고 한 적이 있다. T = R + E 인데, 여기서 T는 고정이다. 따라서 R이 커지면, E가 줄고, R이 줄어들면, E가 커진다. SST는 관측과 엉망회귀의 차이이므로 독립변수가 추가되었다고 해서 바뀔일이 없다. $R^2 = \frac{SSR}{SST} = \frac{SST - SSE}{SST} = 1 - \frac{SSE}{SST}$ 이고, 독립변수가 추가되면, $R^2$이 변할 텐데, SST는 그대로인데 SSE는 줄어들거나 그대로가 된다. 이유는 회귀식이 $\hat{y}_ i = b_ 0 + b _ 1x{i1} + \cdots + b_ px_ {ip}$라고 할 때, 설명변수 $x_{p+1}$를 추가한다고 하면, 만약 해당 설명변수의 계수 $b_{p+1}$이 0이라 해도 기존 모델과 똑같이 예측하게 된다. 즉, 최악의 경우에도 성능은 그대로이고, 조금이라도 유용한 정보가 있다면 성능은 향상되어 SSE는 줄어들거나 그대로가 되고, $R^2$은 같거나 커지게 된다. 결국 정보가 추가되면 Error (Residual)은 작아지고, 회귀 개선 정도는 늘어난다라고 생각할 수 있다. 

이에 대한 proof는 linear algebra - Prove that $R^2$ cannot decrease when adding a variable - Mathematics Stack Exchange https://math.stackexchange.com/questions/1976747/prove-that-r2-cannot-decrease-when-adding-a-variable 를 참고하면 된다. 

$R^2$이 너무 크면 과적합이라고 하는데, Rsquared와 Radjsqaured로도 과적합을 판단할 수 있다. 둘의 차이가 많이 나면, 너무 많은 독립변수가 추가된 것이므로 과적합이라 할 수 있다. 

$X^TX$에 대해 몇가지 사실이 있는데, 예를 들어 설명변수가 2개($x_1, x_2$)이고, 샘플 수가 $n$개라고 하면, 절편항을 포함한 $X$는 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/75309733-f296-4b10-966e-236e6673276a" height="" width=""></p>

여기서 $X^TX$를 계산해보면 Sum of square가 대각선에 있고, 나머지는 cross product가 된다. 

<p align="center"><img src="https://github.com/user-attachments/assets/608b464e-ba10-47eb-899e-b2d802c00e9e" height="" width=""> <img src="https://github.com/user-attachments/assets/92176c49-e286-48e3-b549-cfa6e57cc8cd" height="" width=""></p>

이것을 시작으로 다음의 사실들을 알 수 있다.

1.$X$자체를 centering (평균을 0으로)해서 곱하고 n-1로 나누면 covariance matrix가 된다. 

centering을 위한 평균값은 $\bar{X}_{ij} = x_ij - \bar{x}_j$라 할 수 있다. $X$의 각 요소에서 평균을 뺀 뒤 $X^TX$를 구하고 n-1로 나누면 다음과 같고 공분산 행렬이 된다. 

$$Cov(X) = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$$

<p align="center"><img src="https://github.com/user-attachments/assets/430cdfd6-a05a-4e37-a221-0fe2dfc4e168" height="" width=""></p>

2.$X$컬럼들을 표준화하고 곱한 후 n-1로 나누면 Pearson correlation matrix가 된다. 

각 변수 열을 다음과 같이 바꿔준다. 

$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$$

- $\bar{x}_j$: 변수 $j$의 평균  
- $s_j$: 변수 $j$의 표준편차

이렇게 만든 행렬 $Z$는 $\to$ **centered + scaled**된 형태이다. 

$$\frac{1}{n-1}(X_{standardized}^T \cdot X_{standardized}) = \mathsf{Pearson \ Correlation \ Matrix}$$

3.$X$ 컬럼들을 벡터로 보고 유닛 스케일링을 한 후에 곱하면(정규화와 같음) 코사인 유사도 matrix가 된다.

이정도만 알아도 matrix를 좀 더 유용하게 쓸 수 있다. 

어떤 행렬 $X$에 대하여 ...







