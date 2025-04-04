---
title:  "[Statistics] Statistical Estimation"
excerpt: "Statistical Estimation"

categories:
  - Statistics
tags:
  - Statistics
toc: true
toc_sticky: true
last_modified_at: 2025-01-19T08:06:00-05:00
---

> 데이터 사이언티스트 되기 책: https://recipesds.tistory.com/

# Estimation

통계적 추론은 표본을 통해 모집단에 대한 추론을 하는 과정을 말한다. 추정(Estimation)은 표본을 이용해 모수(Parameter, $\theta$)를 
어떠한 값으로 추측하는 과정으로서 추정값(점추정) 또는 추정 값의 범위를 이용해 모수를 추정하는데, 이때 추정 값의 범위를 오차범위(표본오차, 신뢰구간) 등으로 제시한다.   
가설검정은(Hypothesis Testing)은 표본을 이용해 모집단에 대한 주장을 하거나, 주장하는 가설이 모집단에 대해 옳거나 그르다고 판단하는 과정으로 
이 과정에서 가설이 귀무가설(Null Hypothesis), 대립가설(Alternative Hypothesis)로 제시된다. 
통계에서는 추정을 하기 때문에 검정을 할 수 있고, 검정이 가능하기에 통계적 결론을 낼 수 있다. 

주어진 신뢰도로 표본들이 어디에 있을지 추정하는 통계 추정의 간단한 예를 들어보면, $\mu=200, \sigma=1$인 가우시안 분포의 모집단이 있다고 할 때, 
무작위로 표본을 뽑으면 관측될 표본의 95%가 어느 구간에 있을지 예측해보자. 가우시안에서 95%의 데이터가 들어있는 구간은 $\mu$로부터 $-1.96\sigma ~ +1.96\sigma$로서
 1.96은 95%에 대한 critical value로, 보통 한쪽 구역에서 2.5%를 의미하니까, $z_{2.5\%}$라 표현한다. (2.5%가 양쪽에 있으니, 합쳐서 신뢰도 95%이다.)
결과적으로 $\sigma=1$이니까, $200 - 1.96 \cdot 1 ~ 200 + 1.96 \cdot 1$사이에 95%의 표본이 관측된다. 참고로 신뢰도 99%일 때는 임계값 $z_{0.5\%}$가 2.58이고, 
임계값을 일반형으로 표기할 때는 $z_{critical}$로 표현한다. 

이제 본격적인 모평균 추정에 앞서 평균, 분산에 대한 notation을 정리하면 다음과 같다. 

|집단|평균|분산|표준편차|
|-|-|-|-|
|모집단|$m$|$\sigma^2$|$\sigma$|
|표본집단|$\bar{X}$|$s^2$|$s$|
|표본평균의 집단|$E(\bar{X})$|$\frac{\sigma^2}{n}$|$\frac{\sigma}{\sqrt{n}}$|

마지막 표본평균의 집단은 중심극한정리에 의한 값이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/fc3ceffa-4cde-4016-a165-d58160dab99d"></p>

어떤 $\bar{X}$를 관측했는데, 그 값이 중심극한정리에 의해서 가우시안 분포에서 나온 값이고, 해당 값이 가우시안 분포의 어디에 위치하는지 계산하기 위해 관측값을 $Z$값으로 표준화 한다. 

예를 들어 온도를 측정하는데 온도계로 30번 측정했더니($n=30, N=1$), 평균이 20도 였다고 하면, 95%의 신뢰도로 모평균 온도 구간을 추정해보자. 이때, 표본평균은 가우시안 분포를 하고, 표준오차가 5라고 가정하자. 우리는 표본으로 20도를 관측했으므로, 20도가 신뢰구간 안에 포함되어야 한다. 따라서 upper bound는 20도 보다 커야하고, lower bound는 20도 보다 작아야 한다. 

$$\mu + 1.96 \cdot 5 \geq 20$$

$$\mu - 1.96 \cdot 5 \leq 20$$

위 두 부등식을 만족하는 구간은 $10.2 \leq \mu \leq 29.8$이고, 따라서 이 구간에 평균이 있을 확률이 95%가 된다. 

예시를 한가지 더 들어서 4개의 데이터를 관측했을 때, 이 4개의 데이터의 평균은 200, 표준편차는 10이라 하자. 모평균을 95% 신뢰도로 추정한다고 하면, $\bar{X}=200, \sigma=10, n=4$이므로, 중심극한정리에 의해 모평균은 $\mu$를 중심으로 $\pm1.96\frac{\sigma}{\sqrt{n}}$이다. 따라서 표본에서 관측한 평균 200은 이 구간 안에서 관측되면 된다. 

$$\mu + z_{2.5\%} \frac{10}{\sqrt{4}} \geq 200$$

$$\mu - z_{2.5\%} \frac{10}{\sqrt{4}} \leq 200$$

$z_{2.5\%}$는 앞서 말했듯이 95% 신뢰도를 위한 가우시안 오른쪽 2.5%에서의 값으로 1.96이다. 

이상한 점이 있는데, 이 예시에서 우리는 관측한 표본의 표준편차만 알고, 모집단의 분산을 알지 못한다. 
따라서 위와 같은 계산은 표본 평균이 완벽한 정규분포를 이룬다는 가정 하에 구할 수 있는 신뢰구간 추정량이 된다. 

일반적으로 우리는 모평균을 추정할 때, 모표준편차를 알지 못한다. 따라서 $s$(표본표준편차)로 대치하면 $t$분포가 된다는 점을 이용한다. 
분산의 불편 추정량(표본분산 $s^2$)을 사용하는 표본평균의 분포는 가우시안과 닯은 $t$분포가 된다. 
이때는 통계량 $\frac{\bar{X} - \mu}{\frac{s}{\sqrt{n}}}$이 $t$분포를 따른다. 이는 표본 평균의 표준편차가 $\frac{s}{\sqrt{n}}$이기 때문에, $z$분포에서 표준편차 $\sigma$만 표본표준편차 $s$로 바꾼 것이다. 

결과적으로 위 예시의 경우  $t$분포에서 분산의 불편추정량을 이용해 평균을 추정하고, 자유도는 당연히 $n-1$이다. 

불편추정량을 이용해 모분산을 대신하게 되면 $\frac{\sigma^2}{n}$가 $\frac{s^2}{n}$가 된다. 
여기서 $s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$로, $n-1$자유도로 표본분산을 계산한다. 

따라서 $t$분포를 이용하면 모평균으로부터 $\pm t_{2.5\%, dof}\frac{s}{\sqrt{n}}$ 사이에 관측값이 있으면 된다. 

예를 들어, 어떤 모집단에서 표본 500개를 추출해서 표본 평균이 100,000이 나왔고, 표본 분산이 2,000,000이 나왔다고 한다면 

$$\mu + t_{2.5\%, 499} \frac{\sqrt{2000000}}{\sqrt{500}} \geq 100000$$

$$\mu - t_{2.5\%, 499} \frac{\sqrt{2000000}}{\sqrt{500}} \leq 100000$$

을 만족하는 범위를 찾으면 된다. 여기서 $t_{2.5\%, 499}$는 자유도($n-1$)이 499, 확률이 2.5%인 $t$값을 의미하고, 자유도와 유의수준에 따른 해당 값은 t분포표에서 찾을 수 있다.

정리하면 모집단 평균의 구간 추정을 할 때, $\sigma$가 알려지지 않은 경우(현실세계)  $s$를 $\sigma$의 추정치로 사용하고 $\mu$의 구간 추정은 $t$분포에 기초한다고 할 수 있다. 이때 자유도가 30을 초과하면, $t$값은 정규분포 $z$의 값과 거의 같아지기 시작한다. 따라서 정규분포 $z$는 $t$ 테이블의 무한 자유도 값으로 대치할 수 있다. 

##  Estimating Proportions

앞서 평균의 구간 추정을 해봤다면, 이젠 모비율의 구간 추정을 해볼 것이다. 모비율을 추정한다는 것은 표본비율을 가지고 모비율을 추정한다는 것으로, 비율인 $p$에 대해 $p$와 $1-p$로 어떤 현상을 설명하는 Binomial분포를 이용한다. Binomial는 $n$개의 확률변수가 어떤 확률 분포를 따르는지에 상관없이 $n$이 충분히 크다면 그 합은 가우시안을 따른다는 중심극한정리에 의해 Bernoulli의 성공횟수의 합을 확률변수로 가지므로 가우시안으로 근사된다. 

$$X \to P(X=x) \sim B(n, p) \approx N(np, np(1-p))$$

이때 주의할 점은 $p$는 구하고자 하는 모수이기 때문에 조사한 표본의 비율로 바꿀 수 없지만, 분산은 $n$이 충분히 크기 때문에 표본분산을 모분산으로 대체할 수 있다. 

예를 들어 600가구를 대상으로 현재 시청하는 채널을 조사했더니, 99가구가 KBS를 보고 있었다고 할 때, 진짜 KBS의 시청율을 95%의 신뢰구간으로 추정해보자. $\hat{p} = \frac{99}{600}$이다. 여기서 $\hat{p}$는 관측했다는 뜻이다. 따라서 가우시안으로 근사하면 다음과 같다.

$$B(n, p) \approx N(np, n\hat{p}\hat{q}) = N(600p, 600 \cdot \frac{99}{600} \cdot \frac{501}{600})$$

따라서 $600p \pm 1.96 \cdot \sqrt{600 \cdot \frac{99}{600} \cdot \frac{501}{600}}$ 사이를 95% 구간으로 볼 수 있다. 
이제 평균 추정과 마찬가지로 관측한 99가 위 구간에 포함되야 하므로, 

$$600p + 1.96 \cdot \sqrt{600 \cdot \frac{99}{600} \cdot \frac{501}{600}} \geq 99$$

$$600p - 1.96 \cdot \sqrt{600 \cdot \frac{99}{600} \cdot \frac{501}{600}} \leq 99$$

이다. 따라서 $p$는 $\frac{99}{600} \pm 1.96 \frac{\sqrt{600 \cdot \frac{99}{600} \cdot \frac{501}{600}}}{600}$ 구간이 95% 신뢰구간이 된다. $0.165 \pm 0.0297$

이러한 모비율 추정을 공식화 하면 원래 $X \sim B(n, p)$인 것을, $X$를 $n$으로 나눠서 비율로 다시 표현하면 $\hat{p} = \frac{X}{n}$이 된다. 표본비율이 $n$개의 표본 중 $X$개를 차지한다고 보면, $X \sim B(n, p) \approx N(np, n\hat{p}\hat{q})$ 이므로, 

$$\hat{p} = \frac{X}{n} \sim N \left( p, \frac{\hat{p}\hat{q}}{n} \right)$$

가 된다. 이유는 $E(\hat{p}) = E\left( \frac{X}{n} \right) = \frac{np}{n} = p$ 이고, $Var(\hat{p}) = Var \left(\frac{X}{n} \right) = \frac{n \hat{p} \hat{q}}{n^2} = \frac{\hat{p} \hat{q}}{n}$ 이기 때문이다. 

따라서 $\hat{p}$는 $N(p, \frac{\hat{p} \hat{q}}{n})$를 따르고, 정규화 하면 $\frac{\hat{p} - p}{\sqrt{\frac{\hat{p} \hat{q}}{n}}}$이 근사적으로 표준 정규분포를 따른다. 

최종적으로 모비율 추정을 공식화하면 다음과 같다. 

$$P \left( -z_{critical} \leq \frac{\hat{p} - p}{\sqrt{\frac{\hat{p} \hat{q}}{n}}} \leq z_{critical} \right) \to P \left( \hat{p}-z_{critical}\sqrt{\frac{\hat{p} \hat{q}}{n}} \leq p \leq \hat{p}+z_{critical}\sqrt{\frac{\hat{p} \hat{q}}{n}}  \right)$$ 

## Estimating Population Variance

이번엔 모분산을 추정할 것이다. 모분산을 추정할 때는 카이제곱 분포를 이용한다. 앞서 $\frac{(n-1)s^2}{\sigma^2}$ 이 통계량이 $\chi_{(n-1)}^2$ 카이스퀘어 분포를 따른다고 했었다. 이를 이용해서 평균 추정과 마찬가지로 upper bound는 관측치보다 크게 하고, lower bound는 관측치보다 작게 하면, 다음이 성립한다. 

<p align="center"><img src="https://github.com/user-attachments/assets/f7bdd02b-7f74-4efc-ad7e-c925f23c0f92"></p>

카이스퀘어의 분포를 따른다 할 때, 95% 신뢰구간으로 추정한다고 하면, 어떤 관측지가 있다고 할 때, 95% 신뢰구간의 상한인 $\chi_{(n-1), 97.5\%}^2$는 관측치보다 커야하고, 하한인 $\chi_{(n-1), 2.5\%}^2$는 관측치보다 작아야 한다. 이를 일반화하면 다음과 같다. 

$$\frac{(n-1)s^2}{\chi_{\alpha/2, n-1}^2} \leq \sigma^2 \leq \frac{(n-1)s^2}{\chi_{1-\alpha/2, n-1}^2}$$

예를 들어 78m, 85m, 82m, 79m, 77m의 데이터가 있다고 하자. 이때 모분산 $\sigma^2$를 추정한다고 하면, 먼저 표본분산을 계산한다. 
$s^2 = 10.75 \approx 10.8$. 그리고, 위 카이스퀘어 통계량을 계산하면, $\frac{(n-1)s^2}{\sigma^2} = \frac{10.8}{\sigma^2} \times 4 = \frac{43.2}{\sigma^2}$ 가 된다. 카이제곱분포의 95% 구간은 0.4844 ~ 11.1433($n-1 \ dof$)이므로

$$0.4844 \leq \frac{43.2}{\sigma^2} \leq 11.1433$$

가 된다. 이 부등식을 풀면 

$$3.877 \leq \sigma^2 \leq 89.182$$

3.877~89.182의 구간에서 신뢰도 95%로 모분산을 추정할 수 있다. 


# Maximum Likelihood Estimation(MLE)

확률(Probability)은 모수 $\theta$가 정해진 상태에서 Data가 목격될 가능성을 의미하고, 우도(Likelihood)는 Data가 관측된 상태에서 특정 확률분포의 모수 $\theta$에 대한 믿음의 강도(strength)를 나타낸다. Likelihood와 확률의 관계는 다음과 같다. 

$$\mathcal{L}(\theta; Observation) = P(Observation; \theta) \cdots (*)$$

Likelihood function은 $\mathcal{L}(\theta \vert x)$로 $x$는 이미 정해진 data이다. $\theta$를 특정 값으로 가정하면, Likelihood function은 함수가 아니라 상수가 되어 특정 모수 $\theta$의 분포에 대한 data $x$의 강도를 나타낸다. 가정한 분포에서 관측한 데이터에 대한 높이 $y$가 강도가 되고, 이 강도가 Likelihood이다. 따라서 강도가 높을수록 해당 데이터를 잘 표현하는 모수라 할 수 있다. 

Likelihood는 $\theta \to \mu, \sigma, p$를 Parameter라 할 때, 다음과 같다. 

$$
\mathcal{L}(\theta \vert x) = 
\begin{cases}
P(X=x \vert \theta) & when \ Discrete \ pmf \\  
f(X=x \vert \theta) & when \ Continuous \ pdf   
\end{cases}
$$

<p align="center"><img src="https://github.com/user-attachments/assets/948993e3-b18d-4e71-a5a7-7d63c289612d"></p>

위 Binomial 분포의 그림을 보면 확률은 확률변수인 event가 $x$축인데 반해, Likelihood는 모수 $p$가 변수가 된다. 특적한 event를 fix하고, 모수가 변함에 따라 관측한 데이터가 얼마나 모수에 잘 맞는지에 대한 강도인 Likelihood의 변화를 나타낸다. 

최대우도추정(Maximum Likelihood Estimation)은 관측한 데이터를 통해 Parameter모수를 추정하는 방법이다. 
MLE의 특징은 가장 적절한 분포를 먼저 가정하고, 그 가정된 분포에 대한 가장 적절한 모수 $\theta$를 찾는 것인데, $\theta$를 찾는 근거가 Likelihood이다. 따라서 $\theta$에 대한 Likelihood의 최대값을 찾아내면 그때의 $\theta$가 가장 데이터를 잘 설명하는 분포라 할 수 있다. Likelihood의 최대는 미분해서 0이 되는 지점을 찾으면 된다. $\left( \frac{\partial \mathcal{L}}{\partial \theta} = 0 \right)$ 

MLE를 할 때 모든 Observation은 **i.i.d**이다. **i.i.d**는 Independent and Identical Distribution의 약자로 각 데이터를 뽑을 때, 같은 분포에서 서로 독립인 경우를 말한다. 따라서 **i.i.d**이기 때문에 각각의 Observation에 대해 곱형태의 joint probability를 사용할 수 있다. 

주머니 안에 검은 구슬과 흰색 구슬이 총 100개 있다고 하자. 10번을 뽑아 보았더니 검은 구슬이 7번, 흰색 구슬이 3번 나왔다. 이때 MLE로 검은 구슬은 전체 구슬 중 몇 개가 있을지 추정해보자. 확률 분포는 Binomial로 정하고, 모수 $\theta$는 $p$가 된다. 검은 구슬이 나오는 사건이 $A$, $n=10$, $p$는 검은 구슬이 나올 확률이라 했을 때, $P(A)$는 다음과 같다.

$$P(A) = \left. \binom{n}{k} \cdot p^k q^{n-k} = C \cdot p^k q^{n-k} \right|_{k=7, n=10} = C \cdot p^7 (1-p)^3$$

$C$는 constant이고, 우리가 추정하는 모수 $\hat{p}$는 아래처럼 Likelihood $P(A)$를 최대화하는 $p$를 구하면 된다. 

$$\hat{p} = arg \max_p(C \cdot p^7 (1-p)^3)$$

Linear function에 log를 취해보면 monotonically increasing한다. 이는 원래 함수의 최대값을 갖는 점과 log를 취한 함수의 최댓값을 갖는 점이 같다는 것으로, Binomial Likelihood에 log를 씌워도 마찬가지 이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/2bb7a43c-49bf-4120-82fb-acce114a2273"></p>

따라서 log를 씌운 형태에서 미분을 하여 최댓값을 구할 수 있다. 이에 따라 추정 값 $\hat{p}$를 구해보면 다음과 같다. 

$$\begin{align}
\hat{p} &= \ arg \max_p(C \cdot p^7 (1-p)^3) \\  
&= \to arg \max_p(\log (C \cdot p^7 (1-p)^3)) = arg \max_p(\log(C) + 7 \cdot \log(p) + 3 \cdot \log(1 - p))
\end{align}$$

$p$에 대해 미분하여 0이 되는 지점을 찾으면 되므로 

$$\frac{d(\log(C) + 7 \cdot \log(p) + 3 \cdot \log(1 - p))}{dp} = 0$$ 

$\frac{7}{p} - \frac{3}{1-p} = 0$을 풀면 된다. 

다른 예로 Gaussian Distribution의 모수 $\mu$를 MLE로 추정해보자. Gaussian Distribution의 pdf는 다음과 같다. 

$$P(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)}$$

Gaussian분포에서 9, 10, 11을 관측했다고 하자. 이 경우 어떤 모수를 갖는 가우시안 분포가 관측한 데이터를 가장 잘 설명하는지 MLE로 추정한다. MLE로 모수를 추정하기 위한 joint probability를 표현하면 다음과 같다. 

$$\begin{align}
P(9,10,11;\mu, \sigma) &= \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(9-\mu)^2}{2\sigma^2} \right) \\ 
&\times \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(10-\mu)^2}{2\sigma^2} \right) \\ 
&\times \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(11-\mu)^2}{2\sigma^2} \right)
\end{align}$$

i.i.d를 만족하기 때문에 위와 같이 곱형태로 나타낼 수 있다. 여기에 log를 씌으면 다음과 같다. 

$$\begin{align}
\ln(P(x; \mu, \sigma)) &= \ln \left( \frac{1}{\sqrt{2\pi\sigma^2}} \right) - \frac{(9-\mu)^2}{2\sigma^2} \\  
&+ \ln \left( \frac{1}{\sqrt{2\pi\sigma^2}} \right) - \frac{(10-\mu)^2}{2\sigma^2} \\  
&+ \ln \left( \frac{1}{\sqrt{2\pi\sigma^2}} \right) - \frac{(11-\mu)^2}{2\sigma^2} \\  
&= 3 \cdot \ln \left( \frac{1}{\sqrt{2\pi\sigma^2}} \right) - \frac{1}{2\sigma^2}\lbrack (9-\mu)^2 + (10-\mu)^2 + (11-\mu)^2 \rbrack
\end{align}$$

$\mu$에 대해 미분하여 max를 찾으면 다음과 같다. 

$$\frac{\partial \ln(P(x; \mu, \sigma))}{\partial \mu} = \frac{1}{\sigma^2} \lbrack 9 + 10 + 11 -3\mu \rbrack = 0$$

$$\mu = \frac{9 + 10 + 11}{3} = 10$$









