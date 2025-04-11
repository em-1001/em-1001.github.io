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







