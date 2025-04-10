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

$$\begin{pmatrix}y_1 \\ y_2 \\ \vdots \\y_n\end{pmatrix} = 
\begin{pmatrix}
1 & x_{11} & x_{12} \cdots x_{1k} \\ 
1 & x_{21} & x_{22} \cdots x_{2k} \\  
\vdots & \vdots & \vdots \\ 
1 & x_{n1} & x_{n2} \cdots x_{nk}
\end{pmatrix} 
\begin{pmatrix}b_0 \\ b_1 \\ \vdots \\b_k\end{pmatrix} + 
\begin{pmatrix}\epsilon_1 \\ \epsilon_2 \\ \vdots \\\epsilon_n\end{pmatrix}$$

$y = Xb + \epsilon$이 되고, $E$를 구하면 다음과 같다. 

$$\begin{align}
E = \sum_{i=1}^n \epsilon_i^2 &= \epsilon^T \epsilon = (y - Xb)^T(y - Xb) \\  
&= y^Ty + b^TX^TXb - 2b^TX^Ty \\  
\end{align}$$

$$\begin{align}
\frac{\partial E}{\partial b} = 2
\end{align}$$





