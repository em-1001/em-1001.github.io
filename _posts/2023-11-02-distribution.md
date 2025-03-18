---
title:  "[Statistics] t, chi-squared, F distribution"
excerpt: "t, chi-squared, F distribution"

categories:
  - Statistics
tags:
  - Statistics
toc: true
toc_sticky: true
last_modified_at: 2025-01-19T08:06:00-05:00
---

> 데이터 사이언티스트 되기 책: https://recipesds.tistory.com/

# t-distribution

t-분포는 표본 평균에 관련되어 있다. t-분포는 가운데를 중심으로 양쪽으로 펼쳐져 있고, 평균을 0으로 두는 좌우가 동일한 분포이다. 
평균에 관련되어 있다보니, 평균을 추정하거나, 두 집단의 평균이 같은지 확인할 때 검정 통계량으로 사용한다. 
즉, $\sigma^2$를 모를 때, 표본분산 $s^2$을 대신 사용하여 $\mu$를 추정하고, 평균 검정에 사용한다. 

<p align="center"><img src="https://github.com/user-attachments/assets/78df45b5-e3ff-4001-b8cd-97b65e413889" height="" width=""></p>

t-분포는 표본의 수에 해당하는 자유도 $n$이 30을 초과하는 순간 표준정규분포($z$ 분포)에 가까워지고, 무한($\infty$) 자유도는 $z$-분포에 수렴한다. 

아래는 자유도와 유의수준 $\alpha$에 따른 t 분포표이다. 

> https://ko.wikipedia.org/wiki/T%EB%B6%84%ED%8F%AC%ED%91%9C


# $\boldsymbol{\chi}^2$ (chi squared) distribution

$\boldsymbol{\chi}^2$ 분포는 표본분산에 관련되어 있다. $\boldsymbol{\chi}^2$ 분포는 표준정규분포 확률변수의 제곱합으로, $\sigma^2$를 추정할 때 사용하며 적합도 검정, 독립성검정(동질성 검정) 등에도 사용된다. 
표본 크기 $n$일 때, 표본분산 $s^2$의 표본분포가 $n-1$자유도를 갖는 카이제곱 분포를 따르기 때문에, $\sigma^2$를 추정할 때는 표본분산에 차이가 있는데, 자유도 $n-1$을 이용해서 추정한다. 

카이제곱 분포는 자유도에 따라 모양이 달라진다. 자유도 1에서는 확률변수 $X=Z^2$가 카이제곱 분포를 따르고, 일반화하면 $X=Z_1^2+Z_2^2+\cdots+Z_n^2$이다. 
이때, $Z$는 가우시안이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/040f9b5b-df82-4cf2-9184-75ab43c45715" height="" width=""></p>

카이제곱 분포의 모양은 자유도가 낮으면 0주변에 몰려있고, 자유도가 높을수록 가우시안에 가까워진다. 
카이제곱의 정의는 다음과 같이 가우시안 확률변수의 합으로 정의된다. 

$$\sum_{i=1}^n Z_i^2 = \sum_{i=1}^n \lbrack \frac{X_i - \mu}{\sigma} \rbrack^2 = \boldsymbol{\chi}^2$$

이때, 표본분산과 모분산의 비율을 $\frac{(n-1)s^2}{\sigma^2} \sim \boldsymbol{\chi}^2_{(n-1)}$로 정의할 수 있는데, 표본분산이 모분산과 비슷하다면, 카이제곱 분포는 자유도와 같아진다. 

좀 더 자세히 설명하면, 원래 정규화된 가우시안은 $\frac{(X-\mu)}{\sigma}$이다. 이 값들의 제곱의 합은 다음과 같다. 

$$\frac{(x_1 - \mu)^2}{\sigma^2} + \frac{(x_2 - \mu)^2}{\sigma^2} + \cdots + \frac{(x_n - \mu)^2}{\sigma^2}$$

위 합이 카이제곱 분포를 따른다. 

하지만 우리는 모평균 $\mu$를 모르기 때문에, 표본평균을 이용한다. 아래는 $\sigma$를 제외하고, 우리가 구할 수 있는 표본에 관련된 값으로 표현한 것이다. 

$$\frac{(x_1 - \bar{x})^2}{\sigma^2} + \frac{(x_2 - \bar{x})^2}{\sigma^2} + \cdots + \frac{(x_n - \bar{x})^2}{\sigma^2}$$

이때, 표본의 분산은 $s^2 = \frac{\sum_{i=1}^n(x_i-\bar{x})^2}{n-1}$ 이므로 $(n-1)s^2 = \sum_{i=1}^n(x_i - \bar{x})^2$이니까, 양변을 $\sigma^2$로 나누면, 결국 $\frac{s^2}{\sigma^2} \times (n-1)$로 
표본의 평균과 표본의 분산을 이용해서, 표본분산과 모분산의 비로 나타낼 수 있고, 카이제곱을 따른다. 

보통 $\frac{(n-1)s^2}{\sigma^2}$와 같이 표현하며, 이것이 표준정규분포의 합이 되며, 이 통계량이 $\boldsymbol{\chi}^2_{(n-1)}$를 따른다. 

정리하면 Normal Distribution은 $X \sim \frac{X-\mu}{\sigma} \sim N(0, 1) \sim Z$분포가 되고, $\sum X^2 = \sum Z^2 \sim \boldsymbol{\chi}^2_{n-1}$가 되며, 표본 수-1인 $n-1$이 카이제곱 분포의 자유도가 된다. 

따라서 카이제곱 분포는 정규분포를 따르는 변수의 분산을 분석할 때 사용된다. 

추가적으로 카이제곱은 t분포와 다음과 같은 관계를 갖는다. 

$$T = \frac{Z}{\sqrt{\boldsymbol{\chi}^2_{(dof)}}} \sim t_{(dof)}$$

dof는 자유도로 $T = \frac{X-\mu}{s}$의 형태니까, t분포는 Normal Distribution을 따르는 변수와 카이제곱을 따르는 변수의 비율 형태로, 모분산을 모르는 경우에 표본분산을 이용해서 분석할 때 사용된다. 


# F-distribution

F-분포는 두 개 집단의 분산의 비를 통해 집단간 평균을 비교하는데, 그 평균이 얼마나 퍼져있는지에 관한 분포이다. 
즉, $\frac{\sigma_1^2}{\sigma_2^2}$을 구할 때 사용되는데, 카이제곱을 따르는 두 확률 변수의 비에 쓰인다. 정확히는 분산 추정 치의 비율이다. 카이제곱을 따르는 두 확률분포에 대한 확률분포의 비율이니까 분산의 비율을 비교하는 느낌이고, 분산비 검정, 분산 분석, 회귀 분석 등에 사용된다. 

분자, 분모가 모두 제곱 합으로 표현되는 검정 통계량은 보통 F-분포를 따른다. 보통 통계에서 사용되는 제곱의 합은 카이제곱 분포의 비율의 형태로써, 서로 다른 카이제곱 분포의 비율이다. 

결국 $F = \frac{\boldsymbol{\chi}^2_ {(dof1)}}{\boldsymbol{\chi}^2_ {(dof2)}} \sim F(dof1, dof2)$의 형태가 되어, Normal Distribution을 따르는 두 개의 데이터에 대한 분산의 비율에 대해 분석할 때 사용한다. F-분포는 신뢰구간, 가설검정에 사용하는 분포이고, 다집단의 평균이 같은지(ANOVA)도 확인할 수 있다. 

<p align="center"><img src="https://github.com/user-attachments/assets/28def51e-7621-4f3e-a46d-9566e32ad6d7" height="" width=""></p>

참고로 t 분포를 따르는 변수를 제곱하면 $T^2 = \frac{Z^2}{\boldsymbol{\chi}^2_{(dof)}} \sim F(1, dof)$가 된고 이는 나중에 ANOVA를 다룰 때, t검정이 ANOVA의 특수형임을 알 수 있다. 

검정에서 t-분포를 활용한 평균의 차이를 분석할 때는 귀무가설 $H_0$으로 $\mu=0$이나, $\mu_1=\mu_2$를 사용하고 대립가설 $H_1$은 $\mu \neq 0$이나, $\mu_1 \neq \mu_2$를 사용한다. 

카이제곱 분포를 이용한 모분산 분석에서는 귀무가설 $H_0$는 $\sigma^2=1$이고, 대립가설 $H_1$은 $\sigma^2 \neq 1$이다. 

F-분포를 이용한 ANOVA 분산분석에서는 귀무가설 $H_0$로 $\mu_1=\mu_2=\mu_3$ 와 같은 것이고, 대립가설 $H_1$은 "$H_0$이 아니다."와 같은 것이다. 

회귀분석에서의 귀무가설 $H_0$은 "기울기 $\beta_1=0$이다."와 같은 것이고, 대립가설 $H_1$은 "기울기 $\beta_1 \neq 0$이다."와 같은 것이다. 


