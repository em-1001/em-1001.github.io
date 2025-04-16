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

### $T$, $F$ test

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

### $X^TX$ matrix

$X^TX$에 대해 몇가지 사실이 있는데, 예를 들어 설명변수가 2개($x_1, x_2$)이고, 샘플 수가 $n$개라고 하면, 절편항을 포함한 $X$는 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/75309733-f296-4b10-966e-236e6673276a" height="" width=""></p>

여기서 $X^TX$를 계산해보면 Sum of square가 대각선에 있고, 나머지는 cross product가 된다. 

<p align="center"><img src="https://github.com/user-attachments/assets/608b464e-ba10-47eb-899e-b2d802c00e9e" height="" width=""> <img src="https://github.com/user-attachments/assets/92176c49-e286-48e3-b549-cfa6e57cc8cd" height="" width=""></p>

이것을 시작으로 다음의 사실들을 알 수 있다.

**1. $X$자체를 centering (평균을 0으로)해서 곱하고 n-1로 나누면 covariance matrix가 된다.**   

centering을 위한 평균값은 $\bar{X}_{ij} = x_ij - \bar{x}_j$라 할 수 있다. $X$의 각 요소에서 평균을 뺀 뒤 $X^TX$를 구하고 n-1로 나누면 다음과 같고 공분산 행렬이 된다. 

$$Cov(X) = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$$

<p align="center"><img src="https://github.com/user-attachments/assets/430cdfd6-a05a-4e37-a221-0fe2dfc4e168" height="" width=""></p>

**2. $X$ 컬럼들을 표준화하고 곱한 후 n-1로 나누면 Pearson correlation matrix가 된다.**   

각 변수 열을 다음과 같이 바꿔준다. 

$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$$

- $\bar{x}_j$: 변수 $j$의 평균  
- $s_j$: 변수 $j$의 표준편차

이렇게 만든 행렬 $Z$는 $\to$ **centered + scaled**된 형태이다. 

$$\frac{1}{n-1}(X_{standardized}^T \cdot X_{standardized}) = Pearson \ Correlation \ Matrix$$

**3. $X$ 컬럼들을 벡터로 보고 유닛 스케일링을 한 후에 곱하면(정규화와 같음) 코사인 유사도 matrix가 된다.**   

이정도만 알아도 matrix를 좀 더 유용하게 쓸 수 있다. 

추가적으로 $X^TX$는 $X$의 각 column에 대한 Covariance 느낌이고, $XX^T$는 $X$의 row에 대한 Covariance 느낌이다. 느낌이라 표현한 이유는 Expection을 취하지 않았기 때문에 직접적으로 Covariance는 아니지만 실제로 Covariance와는 비례관계가 성립한다. 

<p align="center"><img src="https://github.com/user-attachments/assets/a6a19d21-d374-4cb3-8791-8782c419fe03" height="" width=""> 　　<img src="https://github.com/user-attachments/assets/34709429-c657-461d-9c5a-9f65f596be40" height="" width=""></p>

### Pseudo Inverse matrix

미분을 하지 않고도 회귀분석을 할 수 있다. 

<p align="center"><img src="https://github.com/user-attachments/assets/1342431d-f65e-4243-81d9-9e4014d87315" height="" width=""></p>

➊ 원래의 관측 벡터 y와 ➋ X의 컬럼으로 이루어진 평면 위에 ➌ 회귀를 통해 구한 y 즉, 벡터 (Xb) ➍  이 두 벡터의 차이, 즉 (Xb - y) 이것이 X의 컬럼으로 이루어진 평면과 직교하면 (X컬럼의 평면에 대한 정사영) 그 차이가 가장 작아지므로 관측치 y와 회귀로 구한 y가 가장 비슷한 관계가 된다는 아이디어를 이용한다. 

따라서 $X$와 $Xb - y$의 내적이 0이 되면 되므로, $X^T(Xb - y) = 0$이 되면 된다. 이걸 풀면 다음과 같다. 

$$\begin{align}
X^TXb &= X^Ty \\  
b &= (X^TX)^{-1}X^Ty 
\end{align}$$

결과를 보면 행렬로 연립방정식을 풀었을 때와 결과가 동일하다. 그리고 $(X^TX)^{-1}X^T$를 묶어서 의사 역행렬이라 한다. 
또한 $\hat{y} = Xb$ 이므로 여기에 구해진 $b$를 넣으면 $\hat{y} = X(X^TX)^{-1}X^Ty$가 되고, 이는 $X(X^TX)^{-1}X^T$를 $y$에 내적하면 $X$평면의 $y$ 추정치가 된다는 말이라, $X(X^TX)^{-1}X^T$를 $X$ 평면으로의 투영 행렬이라고도 부른다. 

## Multicollinearity

앞서 다중회귀를 하면서 $R^2$이 적당히 크거나, F 검정이 유의한데도 불구하고, 계수 중 t검정 결과가 유의하지 않은 경우가 있었는데, 그 원인 중 **"어떤 독립변수가 collinearity (공선성)이 라는 것을 발생시킨다."** 에 대해 좀 더 자세히 알아볼 것이다. 

다중회귀에서 회귀 계수 $b_i$는 $x_i$를 제외한 나머지 독립변수들을 고정시킨 상태에서 $x_i$의 한 단위 증가에 따른 $y$의 변화를 구한 것인데, 만약 독립변수들끼리 상관관계가 없으면 각각의 계수들이 종속변수를 잘 설명하는데, 상관계가 있으면 종속변수를 설명할 때 각각의 계수가 자신이 설명해야할 것을 잘 설명하지 못하게 된다. 즉, 각 독립변수가 종속변수에 얼마나 영향을 미치는지가 애매해지고, 이걸 다중공선성(Multicollinearity)이라 한다. 

예를 들면 독립변수간의 상관관게 때문에, 어떤 한 변수가 다른 변수와 비슷하니까 그 변수의 효과가 거의 나타나지 않을 수도 있어서 해당 변수가 중요하지 않은 것 처럼 보일 수도 있다. 

계수에 대한 t 검정통계량은 이전에 보았듯이 $t_{stat, i} = \frac{b_i}{SE(b_i)}$ 인데, 분모에 있는 분산의 크기에 상관관계가 영향을 주기 때문에 회귀계수의 분산의 크기가 커지면 회귀계수 추정량에 대한 t 검정 통계량의 값이 작아져 p value가 크게 나오게 된다. 다중공선성으로 발생하는 문제는 계수의 높은 분산값과 이로 인해 얻게되는 낮은 t 통계량, 그리고 이 때문에 생기는 회귀계수 추정치의 불안정성이 있다. 

그렇다면 계수들간의 상관관계가 크면 계수의 분산이 커지는 이유는 무엇일까? 수식으로 살펴보면 $b = (X^TX)^{-1}X^Ty$ 이므로, $Var(b) = Var((X^TX)^{-1}X^Ty)$가 된다. 여기서 $Var(Ay) = AVar(y)A^T$가 됨을 이용해서 정리하면 다음과 같다. 

$$\begin{align}
Var(b) &= (X^TX)^{-1}X^TVar(y)((X^TX)^{-1}X^T)^T \\  
&= (X^TX)^{-1}X^TVar(y)X(X^TX)^{-1} \\  
&= \sigma^2(X^TX)^{-1}X^TIX(X^TX)^{-1} \\  
&= \sigma^2(X^TX)^{-1}X^TX(X^TX)^{-1} \\  
&= \sigma^2((X^TX)^{-1}X^TX)(X^TX)^{-1} \\  
&= \sigma^2I(X^TX)^{-1} \\  
&= \sigma^2(X^TX)^{-1}
\end{align}$$

모분산을 표본분산으로 대치하면 $s^2(X^TX)^{-1}$가 된다. 이것은 절편을 제외한다 했을 때 독립변수인 Predictor 개수 × Predictor 개수 행렬로 pxp 이고, 대각 성분이 Variance인 Covariance행렬의 역행렬이다. 행렬의 각 성분은 $(X^TX)^{-1}_{ij} = \frac{1}{det(X^TX)} adj(X^TX) _{ij}$ 로 계산하는데, det가 0에 가까워질수록 이 값이 커진다. 그 의미는 행렬의 Column들이 선형종속이 될수록 det가 0에 가까워진다. determinant는 PCA에서 살펴보았듯이 컬럼 벡터들이 같은 방향으로 있으면 det가 0이 된다. 즉, 공분산 행렬의 컬럼들이 서로 선형종속이 될 수록 계수의 분산이 커진다는 의미이다. 

좀 더 자세히 살펴보면, 

$$X^TX = \begin{pmatrix} \sigma_{11}^2 & \sigma_{12} \\ \sigma_{21} & \sigma_{22}^2 \end{pmatrix}$$

의 공분산 행렬이 있다고 하면 다음과 같이 구할 수 있다. 

$$(X^TX)^{-1} = \frac{1}{det(X^TX)} adj(X^TX) = \frac{1}{\sigma_{11}^2\sigma_{22}^2 - \sigma_{12}\sigma_{21}} \begin{pmatrix} \sigma_{22}^2 & -\sigma_{21} \\ -\sigma_{12} & \sigma_{11}^2 \end{pmatrix}$$

이 중에 행렬의 (2,2)를 살펴본다고 하면, (2,2)는 $\hat{y} = b_1x_1 + b_2x_2$에서 $x_2$에 대한 분산이고 역행렬 상에서 아래와 같다. 

$$\frac{1}{det(X^TX)} adj(X^TX)_ {22} = \frac{1}{\sigma_{11}^2\sigma_{22}^2 - \sigma_{12}\sigma_{21}} \sigma_{11}^2 = \frac{\sigma_{11}^2}{\sigma_{11}^2\sigma_{22}^2 - \sigma_{12}^2}$$

공분산 배열은 Symmetry 대칭이므로 위와 같고, det부분이 분산을 크게 만드는 부분이다. 

정리해서 보면 다음과 같다. 

$$= \frac{1}{\sigma_{22}^2 - \frac{\sigma_{12}^2}{\sigma_{11}^2}} = \frac{1}{\sigma_{22}^2} \frac{1}{1 - \frac{\sigma_{12}^2}{\sigma_{11}^2\sigma_{22}^2}}$$

여기서 결정계수의 식을 다시 살펴보면 $R^2 = \left( \frac{Cov(x, y)}{Var(x)Var(y)} \right)^2$ 이고, 이는 y를 종속변수, x를 독립변수로 놓았을 때의 결정계수이다. 이것을 $x_2$를 종속변수로, $x_1$을 독립변수로 두고 다시 생각해보면 다음과 같다.

$$R_2^2 = \left( \frac{Cov(x_1, x_2)}{Var(x_1)Var(x_2)} \right)^2 = \frac{\sigma_{12}^2}{\sigma_{11}^2\sigma_{22}^2}$$

그러면 $(X^TX)^{-1}_{22} = \frac{1}{Var^2(x_2)} \frac{1}{1 - R_2^2}$ 이므로 $b_2$의 분산은 다음과 같다. 

$$Var(b_2) = \frac{s^2}{\sigma_{22}^2} \frac{1}{1 - \frac{\sigma_{12}^2}{\sigma_{11}^2\sigma_{22}^2}} = \frac{s^2}{Var^2(x_2)} \frac{1}{1 - R_2^2}$$

일반화하면 다음과 같다. 

$$Var(b_j) = \frac{s^2}{Var^2(x_j)} \frac{1}{1 - R_j^2}$$ 

이제 우리가 주의깊게 볼 부분은 분산을 크게 만드는 부분인 $\frac{1}{1 - R_j^2}$이다. $Var(b_j) \propto \frac{1}{1 - R_j^2}$ 이므로 이 부분을 VIF(Variance Inflation Factor)라 부른다. 여기서 $R_j$의 의미는 j번째 x를 y로 두고 나머지 x들로 회귀를 했을 때 결정계수이다. 이때 독립변수 간의 비슷한 정도를 측정하므로 진짜 종속변수 y는 판단에서 제외한다. 

$$x_j = \beta_0x_0 + \beta_1x_1 + \cdots + \beta_nx_n \ : \ omit \ x_j \ term$$

위와 같은 회귀식의 결정계수이고, j번째 x에 대하여 다른 x들에 의한 회귀 적합성을 판단한 것이니까 다른 x들과 선형적 관련이 클수록 큰 값이 나와서 계수 b의 분산을 크게 즉, VIF를 크게 만든다. 결정계수가 크다는 것은 다른 변수들과 선형적으로 잘 맞는다는 얘기이고, 다른 변수들이 변수j를 잘 설명할 수 있다는 얘기이다. 

VIF를 이용하면 다중공선성의 심각성 정도를 측정할 수도 있고, 이걸 지표로 해서 중요한 독립변수를 선정하는 데에도 사용할 수 있다. 참고로 VIF의 역수를 tolerance(공차한계)라고도 부르는데, 대상 x의 결정계수가 클 수록 VIF는 커지고, tolerance는 작아진다. 또한 tolerance가 작을수록 해당 계수의 분산이 커진다. 

그리고 , Acceptable VIF에 대한 현실적인 기준으로는 모형 자체의 결정계수와 비교하는 방법이 실무적으로 제안되기도 한다. 

$$VIF_{j \ acceptable} < MAX \left( 15, \frac{1}{1 - R_{model}^2} \right)$$

모형자체의 결정계수와 15중 더 큰 값이 어떤 j번째 변수의 Acceptable VIF보다 크다면 이 모형은 다중공선성이 있다고 판단할 수 있다. 

나아가 Condition Number라고 VIF와 관계없이 모형자체에 다중공선성이 있는지 판단하는 기준이 있다. 이는 어떤 행렬의 가장 큰 Eigen Value와 가장 작은 Eigen Value의 비율을 말한다. 

$$Condition \ Number = \frac{\lambda_{\max}}{\lambda_{\min}}$$ 

회귀분석에서는 XᵀX (공분산행렬)의 최대 고유값 / 최소 고유값의 비율을 의미한다. 

공분산행렬의 Condition Number가 클 때는 어떤 경우냐 하면, 다중공선성이 있는 경우인데, 다중공선성이 있을 때 아래와 같이 보인다. 

<p align="center"><img src="https://github.com/user-attachments/assets/cdbc59a7-79b3-46bb-823c-1f6683941e5b" height="" width=""></p>

위와 같이 얇게 보이며, 이렇게 얇아지면 공분산 행렬의 Determinant가 작아진다. 즉, 분산이 커지게 된다. 
그래서 모형 자체의 Condition Number를 보면 VIF를 따지기 전에 다중공선성이 있는지 확인할 수 있는데, statsmodels의 ols는 이런 이상 시그널에 대해서 Condition Number가 30 이상이면 다중공선성으로 인한 문제가 있을 수 있다고 알려준다.

그리고 이 현상과 비슷한 경우가, 변수끼리의 스케일 차이가 많이나는 경우 역시  Condition Number가 크게 나온다. 

<p align="center"><img src="https://github.com/user-attachments/assets/706659a4-b159-47d6-bf59-cd30701728f6" height="" width=""></p>

이런 경우는 Determinant가 작기도 하고, 사실상 컴퓨터의 floating 연산에 문제가 생겨서 계산에 문제가 있을 것이라는 시그널을 준다. 이런 경우는 다중공선성의 문제는 아니므로 정규화를 하면 문제가 해결된다. 

모형에 다중공선성이 있다고 의심할 수 있는 경우를 정리하면 다음과 같다.

1. Conditional Number가 30 이상인 경우 (사실상 30에 의미가 있다기보다는 꽤 큰 경우라는 것을 의미하는 것이니까 30에 너무 매몰되면 안 된다.)
2.  각 변수들의 VIF를 보았더니, 15 이상으로 큰 변수가 있는 경우









