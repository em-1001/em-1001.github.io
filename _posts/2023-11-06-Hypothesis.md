---
title:  "[Statistics] Hypothesis Testing I"
excerpt: "statistical hypothesis test"

categories:
  - Statistics
tags:
  - Statistics
toc: true
toc_sticky: true
last_modified_at: 2025-01-19T08:06:00-05:00
---

> 데이터 사이언티스트 되기 책: https://recipesds.tistory.com/


# Hypothesis Testing

가설검정(Hypothesis Testing)은 관심을 가지고 있는 모집단에 대한 가설을 세우고, 표본을 뽑아서 표본정보로부터 그 가설들을 검증하는 것을 의미한다. 

신뢰수준(구간)은 모수가 포함된 구간의 확률이고, 유의수준은 신뢰수준을 제외한 나머지 Extreme 영역을 말한다. 유의수준은 보통 $\alpha%$로 
표기하는데, 만약 신뢰수준이 95%라면, 유의수준은 5%가 된다. 유의수준(Significant Level)을 양쪽 구간으로 따질 때는 유의수준을 둘로 나눠서 
양측 검정($\alpha/2$)을하고, 한쪽만 따질 때는 나누지 않고 단측검정을 한다. 

<p align="center"><img src="https://github.com/user-attachments/assets/12416bbb-c296-46ab-a725-34733ba4ed1e" height="50%" width="50%"></p>

## Null Hypothesis & Alternative Hypothesis

가설검정을 위해서 Null Hypothesis(귀무가설 $H_0$), Alternative Hypothesis(대립가설, $H_1$)을 설정해야 한다. 
Null Hypothesis는 검정의 대상이며 어떤 확률분포(모수)를 가정하고, 그 조건에서 해당 확률분포에 따라 어떤 관측이 무작위로 관측될 것이라는 가설이다. 
Alternative Hypothesis는 실체가 없이, 관찰해보니 가정한 확률분포가 아닌 것 같을 때 Null Hypothesis를 기각하고 채택되며 검정의 대상이 아니다. 
Null Hypothesis는 확률분포가 어떻다고 가정했을 뿐이지 이 가설이 무조건 참인 것은 아니다. 

예를 들어 다음과 같이 두 가설이 있다고 하자. 

$$\begin{align}
\mu &= 100 \cdots (1) \\  
\mu &\neq 100 \cdots (2)
\end{align}$$

$\sigma$를 안다는 가정하에 평균 $\mu$로 분포를 가정할 수 있는 가설은 (1)이다. (2)로는 분포를 표현할 수 없으며 실체가 없고, (1)을 부정할 뿐이다. 따라서 (1)이 귀무가설(Null Hypothesis)이 되고, (2)가 대립가설(Alternative Hypothesis)이 된다. 이러한 경우 검정을 할 때는 양측검정을 한다. 

다른 예로 다음과 같은 두 가설이 있다고 하자. 

$$\begin{align}
\mu &\ge 100 \cdots (3) \\  
\mu &< 100 \cdots (4)
\end{align}$$

이 경우 (3)은 $=$을 이용해 분포를 가정하고, 그 가정한 분포에서 $\mu > 1000$을 기본 채택역으로 정하여 $\mu < 100$에 기각역이 있도록 할 수 있지만, (4)로는 분포를 표현할 수 없다. 따라서 (3)이 Null Hypothesis이 된다. 이러한 가설의 경우 Null Hypothesis를 검정할 때는 단측검정을 한다. 

(1)(2) 가설의 경우 양측검정을 하므로 채택역이 95%라면 기각역은 5%를 반으로 나눠 2.5%씩 기각역이 되고 (3)(4)가설의 경우 단측검정이므로 한쪽의 5%가 기각역이 된다. 

다음의 정의는 귀무가설과 대립가설에 대하여 공부할 때 헷갈리게 하는 정의들이다. 

1. 귀무가설은 검정전과 검정 후가 다르지 않은 것을 귀무가설로 한다. 그러니까, 의미 없는 행위를 하는 것이므로 귀무가설은 기각되는 것이 좋다. (X)
2. 귀무가설이란 관습적이고 보수적인 주장, 차이가 없다, 0이다 등의  연구자가 타파하고자 하는 주장을 말하고, 대립가설이란 우리가 적극적으로 입증하려는 주장, 차이가 있음을 통계적 근거를 통해 입증하고자 하는 주장을 말한다. (X)
3. 귀무가설은 무죄추정의 원칙을 의미한다. 그러니까, 연구자가 유죄를 주장하여 본인의 주장이 맞다고 주장하는 가설이 대립가설인 것이다. (X)
4. 귀무가설이란 직접 검증의 대상이 되는 가설로 연구자가 부정하고자 하는 가설이고, 대립가설이란 ​귀무가설에 반대되는 사실로  연구자가 주장하고자 하는 가설이다. (X)
5. 연구자는 귀무가설을 기각하고 싶어한다. (X)
6. 귀무가설은 우리가 증명하고자 하는 가설의 반대되는 가설, 효과와 차이가 없는 가설을 의미하며 우리가 증명 또는 입증하고자 하는 가설, 효과와 차이가 있는 가설을 대립가설이라고 한다. (X)
7. 일반적으로 믿어지는 사실을 귀무가설로 설정하고, 그것을 부정하는 가설을 대립가설로 설정한다. (X)

위와 같은 정의를 따르면, 연구를 할 때 모두 대립가설을 증명하려고 하는 꼴이 되어 버린다. 하지만 귀무가설이 옳다는 것을 증명하고 싶을 때도 있으므로 우리의 의도와 상관없이 귀무가설과 대립가설이 정해져야 한다. 
다시 말해 '증명하려는 명제'와 귀무가설, 대립가설을 정하는 것은 전혀 관계가 없다. 

귀무가설이 맞을 것 같으면 귀무가설을 기각하지 못한다 라고 하고, 귀무가설이 맞지 않을 것 같으면 귀무가설을 기각한다라고 검정한다. 
이 표현은 모든 가설검정은 귀무가설 중심으로 이루어진다는 사실을 보이며,  주의할 점은 검정의 과정에서 귀무가설을 기각하지 못한 경우에도 확률적으로 귀무가설이 틀렸다는 확실한 증거를 찾지 못했다 정도로 해석해야지 귀무가설이 참이라고 확정적인 결론을 내릴 수 없다. 

대립가설은 귀무가설이 기각되었을 때 채택되는 가설이기 때문에 귀무가설과 수학적으로 exclusive하다. 즉, 두 가설의 교집합은 없고 두 가설의 합집합은 전체집합이다. 

## p value 

귀무가설과 대립가설을 세우고 나면, 통계적인 유의성(Significance)과 p value를 고려한다. 예를 들어 어떤 실험을 통해 얻은 그룹 간의 차이가 무작위로 발생할 수 있는 합리적인 수준보다 더 극단적으로 다르다면, 두 그룹의 차이가 우연히 나온 것이 아니라고 접근할 수 있고, 이때 판단 기준이 유의 수준과 p value가 된다. 

p value 유의확률은 귀무가설이 맞다는 가정 아래, 우리가 표본을 통해 관측한 통계 값 또는 관심 있는 통계 값을 포함해서 더 극단적인 값을 관측할 확률을 말한다. 가설검정에서 Significance Probability라고하고 유의확률이라는 표현을 사용한다. 

실제로 p value가 어떻게 계산하는지 예를 들어보자. 회사의 동료중 1명이 너무나 지각을 많이 해서 조금은 주의를 주고 싶었는데, 막상 조사해 보니, 지각하는 시간이 평균 20분이라 하자. 그런데  이 동료가 억울한 나머지, 아무리 생각해도 20분보다는 일찍 오고 있다고 주장하고, 임의의 10일간의 지각 시간을 평균을 내보니 15분을 지각했으니 20분 지각은 아니라고 주장한다. 

아래 데이터는 사장님이 몰래 회사 동료가 입사한 이래 임의의 10일간의 **평균 지각 시간**들을 40개의 데이터로 모아 둔 것이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/9dd0ed41-6e45-49fb-8287-9da0f13dcd71" height="50%" width="50%"></p>

이때 귀무가설 Null Hypothesis는 $\mu \ge 20$이고, 대립가설 Alternative Hypothesis는 $\mu < 20$이다. 따라서 우리가 가정한 분포는 지각 시간이 평균 20이라는 것이다. 

관측된 15분 이하인 경우를 따져보면 전체 40개 중 15분 4개, 13분 1개로 5개가 15분을 포함하면서 더 극단적인 값을 의미한다. 

$\frac{4+1}{40} = 0.125$이므로 이때의 p value는 0.125이다. 다시 말해 12.5% 정도가 15분 이하의 값이다. 

만약 유의수준이 5%라고 하면, 0.125는 0.05보다 크기 때문에 귀무가설의 채택역에 들어가고, 평균이 20분이라는 귀무가설을 기각할 수 없게 된다. 

위의 예를 조금 변형해서 만약 귀무가설이 $\mu = 20$이고, 대립가설이 $\mu \neq 20$로 양측 검정이 되면 p value를 구할 때, 양쪽으로 같은 정도의 관측치와 더 극단적인 값으로 보면 된다. 따라서 왼족으로 15이하, 오른쪽으로 25이상을 보면 40중 5+3으로 p value는 0.2가 된다. 

동전의 예를 한 가지 더 들어보자. 동전을 던졌을 때, 2번이 모두 앞면이 나왔다고 하자. 우리의 귀무가설은 동전이 fair하므로 앞면, 뒷면이 나올 확률이 모두 $\frac{1}{2}$로 같다고 할 수 있다. 정확한 표현은 앞면이 나올 확률 $P(H) = \frac{1}{2}$이 귀무가설이 된다. 따라서 대립가설은 $P(H) \neq \frac{1}{2}$이다. 귀무가설이 참이라 가정하고, 동전을 2번 던졌더니 두 번 모두 앞면으로 관측되었다 하자. 귀무가설이 참이라는 가정아래, 2번 모두 앞면이 나올 확률은 $0.5 \times 0.5 = 0.25$이다. p value를 구하기 위해서 같거나, 더 극단적인 경우를 구해야 하므로 2번 모두 뒷면이 나올 확률도 고려한다. 이외에 더 극단치가 나오는 경우는 없으므로 0이 된다. 
따라서 최종 p value를 구해보면 앞면이 두 번 나올 확률 + 같은 정도의 극단치(뒷면이 두 번 나올 확률) + 앞의 두 값보다 더 극단치가 나올 확률 = $0.25 + 0.25 + 0 = 0.5$가 된다. p value 0.5라는 값은 귀무가설이 참이라는 가정하에 발생할 확률이 매우 큰 경우에 해당되므로, 이 값으로 귀무가설을 기각하기는 어렵다. 

## False Positive & False Negative

p value는 정리하면 관측한 값인 검정 통계량에 근거한 확률을 알아내고 이 값이 크면 귀무가설이 사실일 때 흔히 일어나는 일이니까, 귀무가설을 기각할 근거가 부족해 귀무가설을 채택하고, 거꾸로 p value가 아주 작다는 것은 귀무가설이 사실이라고 했을 때 그만큼 드문 일이라는 뜻으로, 단순한 우연으론 보기 어려우니까, 귀무가설을 받아들이기 어려워서 귀무가설을 기각한다는 의미이다. 

그렇다면 p value를 계산할 때, 관측한 값만으로 계산하는 것이 아니라 더 극단적인 값들을 포함하는 이유가 무엇일까? 
검정만을 위한다면 관측값만 유의수준 $\alpha$와 비교해서 귀무가설을 기각할지 말지 결정할 수 있다. 
가우시안을 예로들어 유의수준이 5%라면 $\alpha$는 1.645.. 일 것이므로 표준화된 관측값이 1.645보다 큰지 작은지만 확인하면 검정이 가능하다. 

p value를 구하는데 극단치를 더 더해서 구하는 이유는, 예를 들어 $\alpha$를 5%로 관리한다고 하면, 가정했던 분포가 무엇이든 앞서 1.645와 같이 해당 분포에서 5%에 해당하는 확률변수값을 구할 필요가 없이 그저 최극단으로부터 5%까지의 **영역**으로 생각할 수 있고, 이 5%와 비교하기 위해 최극단으로부터 관측값까지의 확률값을 구하면 전체의 몇%에 해당하는 확률인지 알 수 있기 때문에 바로 $\alpha$와 비교할 수 있다. 

<p align="center"><img src="https://github.com/user-attachments/assets/55a5de72-baa3-4171-86be-6d7f92bd5330"></p>

추가적으로 $\alpha$와 P value를 확률로 관리하면 또 다른 관점에서의 해석이 가능하다. False Positive, False Negative인데, 
False Positive는 잘못된 긍정으로 없는 것을 있다고 하는 것(잘못된 양성 판정)이다. False Negative는 잘못된 부정으로 있는것을 없다고 하는 것(잘못된 음성 판정)이다. 통계에서는 False Positiv를 더 위험한 것으로 판단한다. 

|Null Hypothesis|$H_0$: True|$H_0$: False| 
|-|-|-|
|Accept the $H_0$|True Negative|Type II Error<br/>$\beta$ Error<br/>False Negative|
|Reject the $H_0$|Type I Error<br/>$\alpha$ Error<br/>False Positive|True Positive|

귀무가설을 잘못 기각하면 1종 오류(Type I Error), 귀무가설을 기각하지 않아서 생기는 오류를 2종 오류(Type II Error)라 한다. 

유의수준이 5%일 때 양측 가설 검정을 수행한다고 하면, 귀무가설이 옳다고 할 때, 양쪽의 2.5% 유의수준보다 바깥쪽의 데이터가 관측되어 귀무가설을 잘못 기각하는 경우(False Positive)는 5%의 확률로 발생한다. 사실상 유의수준 $\alpha$는 False Positive에 의한 위험도의 확률로서의 최댓값이라 할 수 있다. 

$\alpha$가 False Positive 위험도의 확률 최댓값이라 하면, 실제 얼마나 위험한지 알고싶을 때 p value를 사용한다. 검정을 극단치를 더하는 것인 정의인 p value를 통해 검정을 하면, 관측된 검정 통계량에 의한 p value가 5%보다 작을 때, False Positive에 의한 오류의 확률이 5%보다 작아진다. 따라서  p value는 관측된 데이터에 의한 False Positive 위험률의 최댓값이 되는데, p value와 $\alpha$를 비교해서 p value가 작으면 귀무가설을 기각하는데 확률적으로 부담이 적게된다. 

## $z$ test & One-Sample $t$ test

검정의 예를 $z$검정과 $t$검정을 예로 들어보자. $t$분포는 모분산을 모를 때, 가우시안을 대용해 실제 표본으로 통계적인 접근을 할 수 있는 분포이다. 다음과 같은 경우 $t$검정을 하지 않고, $z$검정을 할 수 있다. 

- 평균을 내기 위해 한번에 추출하는 표본의 크기가 30보다 크다.
- 데이터가 서로 독립적이고, 각각의 데이터는 모집단에서 동일한 확률로 선택된다.(i.i.d)
- 혹은 모분표가 정규분포이다.(다만, 평균을 내기 위한 표본의 크기가 30보다 큰 경우 상관없다.)

위와 같은 조건을 만족하면 중심극한정리에 의해 $z$검정을 할 수 있다. 

과수원에서 매년 100개 포도의 알 수를 적어 두었는데, 전체적으로 평균 75알 표준편차는 15알 이었다고 하자. 올해 과수원에서 100개의 포도를 따서 알 수를 확인했더니 평균 알 수 가 79.5알 이었다고하면, 평균 포도알 수를 75로 하는 것이 타당한지 유의수준 5%에서 검정을 해보자. 

Null Hypothesis는 $\mu=75$, Alternative Hypothesis는 $\mu \neq 75$가 된다. 이때 확률 분포는 중심극한정리에 의해 가우시안이고, 표준편차(표준오차)는 $\frac{\sigma}{\sqrt{n}} = \frac{15}{\sqrt{100}}$이다. Null Hypothesis에 의해 구성된 확률분포는 가우시안이다. Null Hypothesis의 95% 신뢰구간을 추정해 보면, $75 \pm 1.95 \cdot \frac{15}{\sqrt{100}} \to (72.06, 77.94)$이다. 

검정의 관점에서 유의수준 $\alpha$ 5%에 속하는 기각역에 있는지 확인해보자. 79.5알을 목격한 경우의 표준화된 $z$값은 $\frac{79.5-75}{\frac{15}{\sqrt{100}}}=3$이고, 이때의 p value는 0.00135이다. 양측검정이므로 양쪽에 기각역을 각각 2.5%로 설정하고 79.5알을 표준화한 $z$값은 3이므로 p value로 0.00135 = 0.135%이다. 양측검정이므로 $z=-3$ 부터 $-\infty$까지의 값도 0.135%이고, 합은 0.27%이다. 
따라서 귀무가설이 참이라는 가정하에 79.5알이라는 것이 관측(목격)될 확률이 매우 낮음에도 불구하고 관측되었기 때문에, 귀무가설을 기각할 수 있다. 

이제 이 문제를 $t$검정으로 풀어보자. $t$검정을 해야하는 경우는, 모분산으로 몰라서 모분산 대신에 표본분산을 이용하는 경우, 모분포가 가우시안이거나, 중심극한정리를 적용할 수 있는 $n \geq 30$인 경우이다. 

과수원에서 포도의 평균 알 수를 적었는데, 평균 75알이었다. 올해 과수원에서 20개의 포도를 따서 알 수를 확인한 결과 평균 알 수가 79.5였고, 표본의 표준편차는 7알이었다. 이때 표본표준편차 7알은 $n-1$의 자유도=19로 계산된 것이다. 평균 알 수가 75알이어도 될 지 유의수준 5%에서 검정해보자. 

똑같이 Null Hypothesis는 $\mu=75$, Alternative Hypothesis는 $\mu \neq 75$이다. 
중심극한정리에 의한 가우시안이라 하면, $z = \frac{\bar{X} - \mu}{SE} = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{\mu}}}$이고, $t$분포는 닮은꼴로 unknown $\sigma$를 $s$로 치환하면 따르는 분포이므로, $t = \frac{\bar{X} - \mu}{SE} = \frac{\bar{X} - \mu}{\frac{s}{\sqrt{\mu}}}$이다. 이때 표준오차는 $\frac{s}{\sqrt{\mu}} = \frac{7}{\sqrt{20}} = 1.56$이다. 

따라서 목격한 79.5알은 $t_{stat} = \frac{\bar{X} - \mu}{SE} = \frac{\bar{X} - \mu}{\frac{s}{\sqrt{\mu}}} = \frac{79.5-75}{\frac{7}{\sqrt{20}}} = 2.885$, 자유도 $df=20-1=19$이다. $t$분포에서 $t$값이 2.885일 때의 p value를 구해보면 0.0047이다. 0.47%로 간단하게 오른쪽만 보면 2.5%에 비해 매우 작은 값이므로, 75알이 평균이라는 귀무가설을 기각할 수 있다. 

## Independent Samples $t$-test

이번엔 $t$-test를 이용해 집단 간 차이가 있는지를 검정해 볼 것이다. 통계에서는 두 집단의 평균을 비교해서 평균이 차이가 나면 두 집단이 차이가 있다고 결론낸다. 

어떤 집단의 평균을 추정하면 검정을 통해 해당 집단의 평균이 진짜 그런지, 어떤 일이 벌어지기 전과 후의 평균이 같은지 다른지, 서로 이질적인 집단 2개의 평균이 같은지 다른지, 세 개 이상의 집단의 평균이 같은지 다른지 등을 확률적으로 검사할 수 있다. 

추정에서 보았듯이 표본평균의 분포는 중심극한정리(또는 모분포가 정규분포인 경우)에 의해 가우시안이므로, 평균을 추정할 수 있고, 서로 다른 두 집단의 평균 역시 중심극한정리를 이용해 비교할 수 있다. 

예를 들어 남자와 여자의 평균 손톱 길이를 비교해 본다고 하자. 일단 남자의 손톱 길이 분포와 여자의 손톱 길이 분포를 바로 비교할 수는 없다. 각 분포의 표본평균의 분포를 구하면 다 분포 모두 가우시안이 되므로 비교가 가능하다. 이때 당연히 표본이 달라질 때마다 표본 평균의 차이도 달라질 것이다. 따라서 남자와 여자 각각의 표본평균의 분포도 가우시안이지만, 표본평균의 차이도 가우시안이 된다. 즉, 평균의 차이가 확률분포인 가우시안이 된다. 참고로 가우시안은  가우시안끼리의 합도 가우시안이 되고, 차도 가우시안이 된다.  

이제 $\mu_1 - \mu_2$ 평균의 차이가 0이라 가정하고, 평균의 차이 분포인 가우시안 분포를 통해 실제로 차이가 나는지 검정을 한다. 
$X, Y$ 두 집단이 있다고 하면, 두 집단의 평균이 차이가 나는지 확인할 때, 귀무가설은 $H_0: \mu_X = \mu_Y \leftrightarrow \mu_X - \mu_Y = 0$ 가 되고, 대립가설은 $H_1: \mu_X \neq \mu_Y \leftrightarrow \mu_X - \mu_Y \neq 0$ 가 된다. 따라서 귀무가설에 따라 차이의 분포의 평균은 0으로 가정한다. 

그러면 각각의 표본평균의 평균은 중심극한정리에 의해 $\bar{X}$는 $\mathcal{N} \left( \mu_X, \frac{\sigma_X^2}{n_X} \right)$, $\bar{Y}$는 $\mathcal{N} \left( \mu_Y, \frac{\sigma_Y^2}{n_Y} \right)$의 분포를 따른다. 

이때 $\bar{X}$와 $\bar{Y}$가 독립이라면, 평균은 두 평균의 차이가 되고, 분산은 합이 된다. 분산이 합이되는 이유는 분산은 항상 0이상의 값이고, 각각의 변화가 합쳐져 더 커진다. 

<p align="center"><img src="https://github.com/user-attachments/assets/69f77ffc-504b-4c93-bc28-fae788294571" height="70%" width="70%"></p>

결국, 표준오차가 합이되는 특성을 갖는다. 

$$\bar{X} - \bar{Y} \sim \mathcal{N} \left( \mu_X - \mu_Y, \frac{\sigma_X^2}{n_X} + \frac{\sigma_Y^2}{n_Y} \right)$$

그래서 이 분포를 Normalization하면 다음과 같이 된다. 


$$Z = \frac{(\bar{X} - \bar{Y}) - (\mu_X - \mu_Y)}{\sqrt{\left( \frac{\sigma_X^2}{n_X} + \frac{\sigma_Y^2}{n_Y} \right)}} \sim \mathcal{N}(0, 1)$$

Independet Samples $t$-test는 이전 평균추정과 매우 유사하다. 

<p align="center"><img src="https://github.com/user-attachments/assets/0762ae4e-1163-48dc-ac7b-7f050452099c"></p>

평균의 차이에 대해서 95% 신뢰구간을 추정한 후에 5% 유의수준에 의하여 Null Hypothesis를 판단하는 것과 똑같다. 

이제 남은 과정은 분산의 수식을 간단하게 만드는 것이다. 먼저 등분산일 경우는 매우 간단한데, $\sigma_X = \sigma_Y = \sigma$이므로, 아래와 같이 간략화 된다. 

$$Z = \frac{(\bar{X} - \bar{Y}) - (\mu_X - \mu_Y)}{\sqrt{\left( \frac{1}{n_X} + \frac{1}{n_Y} \right)\sigma^2}} \sim \mathcal{N}(0, 1)$$

이제 여기서 우리는 모분산을 모르므로 표본분산을 사용하면 된다. 표본분산으로 대체하면 $((n_X-1)+(n_Y-1)) = n_X + n_Y - 2$ 자유도의 $t$분포를 따르게 된다. 최종적으로 $t$분포를 사용하면 다음과 같이 된다. 

$$Z \to T = \frac{(\bar{X} - \bar{Y}) - (\mu_X - \mu_Y)}{\sqrt{\left( \frac{1}{n_X} + \frac{1}{n_Y} \right)s^2}} \sim t(n_X + n_Y - 2)$$

이때, 두 분포의 차이에 대해 통합된 표본분산(불편분산)은 다음과 같이 계산된다. 

$$S_{unbiased-pooled}^2 = \frac{\sum_i^{n_X}(X_i - \bar{X})^2 + \sum_i^{n_Y}(Y_i - \bar{Y})^2}{(n_X - 1) + (n_Y - 1)} = \frac{(n_X - 1)S_X + (n_Y - 1)S_Y}{n_X + n_Y - 2}$$

불편분산은 각 그룹의 평균으로부터의 변화량의 합을 자유도로 나눈다. 

최종 검정을 위한 식은 다음과 같다. 

$$T = \frac{(\bar{X} - \bar{Y}) - (\mu_X - \mu_Y)}{\sqrt{\left( \frac{1}{n_X} + \frac{1}{n_Y} \right)S_{unbiased-pooled}^2}} \sim t(n_X + n_Y - 2)$$

여기에 Null Hypotesis인 $\mu_X - \mu_Y = 0$을 고려하여 0으로 대치하면, 검정 통계량이 된다. 

만약 등분산이 아니고 분산이 다른 경우는 불편분산을 그대로 계산하는 Welch's t test라는 것을 사용하여 계산한다. 

실제 예를 들어 앞서 남자와 여자의 손톱 길이에 대한 데이터가 다음과 같다고 하자. 

```py
boy = [4.7 5.1 4.8 5.5 4.6 4.9 5. 4.6 5.1 5. 5.1 5.8 5.1 4.4 5. 5.5 5.4 4.4
 4.7 5.3 5.1 5.4 5.2 4.6 4.8 5. 4.3 4.8 4.9 5.7 5. 5.1 5.7 5.1 4.8 5.
 5.4 5.2 5. 4.4 4.9 5.4 5.4 4.9 4.5 5.1 5. 4.8 4.6 5.2]
girl = [6.1 5. 6.4 6.3 6.2 5.5 6.3 5.1 6.1 6.5 5.7 5.4 6.3 5.9 5.7 5.9 5.6 6.7
 6.4 6. 5.6 6.7 6.9 6.6 5.8 5.6 5.2 5.8 6.7 6.1 5.7 5.6 6. 5.6 5. 5.5
 6.8 7. 6.1 5.5 6. 5.5 6. 4.9 6.2 5.7 5.5 6.6 5.7 5.8]
```

귀무가설은 "남자와 여자의 손톱 길이 **평균값**의 차이가 없다."이고, 대립가설은 "남자와 여자의 손톱 길이 **평균값**의 차이가 있다."이다. 

먼저 등분산성을 확인해보면, Null Hypothesis를 "두 개의 분산이 같다."로 설정하여 다음과 같이 확인할 수 있다. 

```py
from scipy.stats import bartlett
bartlett(boy, girl)
 
BartlettResult(statistic=6.891726740802407, pvalue=0.008659557933880048)
```

결과를 확인해보면 p value가 0.05보다 작으므로 귀무가설이 기각되어 등분산성을 만족하지 못한다.

등분산이 아니므로 Welch's t-test를 이용해 구해야 한다. 

```py
from scipy.stats import ttest_ind
ttest_ind(boy, girl, equal_var=False)
```
equal_var 옵션을 False로 주면된다. 등분산일 경우에는 True로 주거나, 아예 아무것도 주지 않아도 된다. 

```py
Ttest_indResult(statistic=-10.52098626754912, pvalue=3.746742613983681e-17)
```
최종 결과를 확인해보면 p value가 유의수준 5%라 할 때, 매우 작으므로 귀무가설이 기각되어 남자와 여자의 손톱 길이의 평균은 차이가 있다고 할 수 있다. 

등분산이 아닌경우의 Welch's t-test의 검정통계량은 다음과 같다. 

$$T = \frac{(\bar{X} - \bar{Y})}{\sqrt{\left( \frac{\sigma_X^2}{n_X} + \frac{\sigma_Y^2}{n_Y} \right)}} \sim t(dof)$$

$$dof = \frac{\left( \frac{\sigma_X^2}{n_X} + \frac{\sigma_Y^2}{n_Y} \right)^2}{\frac{S_X^4}{n_X^2(n_X-1)}+\frac{S_Y^4}{n_Y^2(n_Y-1)}}$$

위 자유도는 그냥 이렇게 나온다라고 생각하면 되고, 이 식을 통해 자유도가 항상 정수인 것은 아니라는 사실을 알 수 있다. 

## A/B test

비율을 검정할 때는 $np \geq 5, nq \geq 5$인 경우 가우시안 근사 $X_{n, p} \sim B(n, p) \sim N(np, npq)$ 를 활용해 $z$ 검정을 한다. $X_{n,p}$는 성공 횟수 이므로, 전체 시행 횟수와의 비율로 나타내어 가우시안으로 근사하면 다음과 같다. 

$$\frac{X_{n,p}}{n} = p \sim N \left( p, \frac{pq}{n} \right)$$

$$\because E\left( \frac{X_{n,p}}{n} \right) = \frac{np}{n}=p, \ Var \left( \frac{X_{n,p}}{n} \right) = \frac{npq}{n^2}=\frac{pq}{n}$$

비율에 관한 통계적 접근은 3가지가 있는데, 이 세 가지에서 분산을 구하는 방법이 조금씩 다르다. 

1. 모비율에 대한 신뢰구간 구하기
2. 모비율에 대한 검정
3. 두 집단의 비율의 차이 검정

1,3의 경우와 2의 경우 분산이 다르다. 1의 경우 표본을 통해 모비율의 신뢰구간을 추정하기 때문에, 표본의 비율이 분산으로 사용되어 $\frac{\hat{p}\hat{q}}{n}$ 이다. 2의 경우 Null Hypothesis가 맞다는 가정으로 표본이 어떻게 관측되었는지 접근하기 때문에 검정하고자 하는 비율(모비율) $\frac{pq}{n}$ 이 정해지고 그에 대한 분산이 계산된다. 이때의 검정통계량은 $Z = \frac{\bar{X} - \mu}{\sigma}$ 와 같은 원리로 $\frac{\hat{p}-p}{\sqrt{\frac{pq}{n}}}$ 이다. 3의 경우 표본을 통해 두 집단의 비율 차이를 확률적으로 계산하려는 것이기 때문에 표본의 비율이 분산으로 사용되어 $\frac{\hat{p}\hat{q}}{n}$ 이다. 

1의 경우는 앞서 추정에서 살펴보았으므로, 2와 3의 검정에 대해 알아볼 것이다. 

먼저 2의 경우 모비율을 $p$라 가정하고, 어떤 비율을 관측했을 때, 그 비율이 맞는지 검정할 수 있다. 이때의 검정통계량은 $Z_{stat} = \frac{\hat{p}-p}{\sqrt{\frac{pq}{n}}}$ 이다. 

예를 들어 1000명 중 87명이 광고를 보고 상품을 구매했다고 할 때, 상품을 구입한 사람의 비율이 국가별 평균 6.5%와 다른지 확인한다고 하자. 이 경우 $p=6.5%$가 귀무가설이 된다. 대립가설은 $p \neq 6.5%$이고, 양측검정이다. 

귀무가설에 대한 $p$의 분포는 평균 $p$, 분산 $\frac{pq}{n}$인 가우시안으로 검정통계량은 $Z_{stat} = \frac{\hat{p}-p}{\sqrt{\frac{pq}{n}}} = \frac{87/1000 - 0.065}{\sqrt{\frac{0.065(1-0.065)}{1000}}} = 2.82$ 이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/45d99950-4d91-437f-b012-dcbf5ee5dc58" height="60%" width="60%"></p>

계산해 보면 관측치가 신뢰구간 2.5%의 바깥쪽에 있고, p value가 0.477%로 유의확률보다 작아서 귀무가설이 기각된다. 

statsmodel 패키지의 proportions_ztest를 이용해서 구해보면 다음과 같다. 

```py
import scipy
from statsmodels.stats.proportion import proportions_ztest
import math
p = 0.065   
q = 1-p
n = 1000
nobs = 87  # Number of OBservationS
p_hat = nobs/n
q_hat = 1-p_hat
z_stat = ((p_hat-p)/(math.sqrt(p*q/n)))
pstat, ppval = proportions_ztest(nobs, n, p, "two-sided", prop_var=p)
print("z_stat : %f"%(z_stat))
print("p value two-sided : %f" %((1-scipy.stats.norm(0, 1).cdf(z_stat))*2))
print("z_stat statsmodel : %f"%(pstat))
print("p value two-sided statsmodel : %f"%(ppval))
 
z_stat : 2.822021
p value two-sided : 0.004772
z_stat statsmodel : 2.822021
p value sstatsmodel : 0.004772
```

z_stat, p value two-sided는 통계량과 p value를 직접 계산한 결과이고, pstat, ppval은 statsmodel의 패키지의 검정을 이용한 것으로 결과가 같음을 확인할 수 있다. 

이제 3번의 경우인 모비율 차이에 대한 검정을 보면 모비율 차이 검정을 가장 잘 쓸 수 있는 예시가 A/B 테스트이다. 모비율 차이에 대한 검정은 모비율의 차이를 0으로 한 Null Hypothesis를 설정하고 검정하게 되는데,  Independent Samples t test에서 했던 평균의 차가 0 인 조건의 분포와 똑같이 취급할 수 있다. 

$$T = \frac{(\bar{X} - \bar{Y}) - (\mu_X - \mu_Y)}{\sqrt{\left( \frac{\sigma_X^2}{n_X} + \frac{\sigma_Y^2}{n_Y} \right)}} \sim N(0, 1)$$

평균을 비율로 대치하면 다음과 같다. 

$$Z = \frac{(\hat{p_X} - \hat{p_Y}) - (p_X - p_Y)}{\sqrt{\left( \frac{\hat{p_X}\hat{q_X}}{n_X} + \frac{\hat{p_Y}\hat{q_Y}}{n_Y} \right)}} \sim N(0, 1)$$

코끼리 밥 주기 앱을 서비스하는 우리 회사는 이번에 하는 앱 설치 마케팅 캠페인의 효과를 A/B테스트를 통해 측정하려고 한다. 새로운 캠페인 중 A 안은 50명이 랜딩페이지에 왔다가 20명이 앱을 설치했고, B 안은 추적을 해 보니, 200명이 랜딩페이지에 도착 후 120명이 앱을 설치했다고 하자. 이런 경우에 B 캠페인이 앱 설치도가 더 좋은지 확인햅 보자. 

A/B 테스트에서는 분산이 표본의 분산으로 계산된다는 점을 기억해야 한다. 관측치 $\hat{p}$를 이용해 분산을 계산한다. 이걸 간략화 하는 경우가 있는데, $\hat{p}_ A, \hat{p}_ B \sim \hat{p} _0$ 형태로 공통표본비율(합동표본비율)로 한번에 간략화하는 경우로, 이 경우 $p _0 = \frac{count_A + count_B}{n_A + n_B}$ 로 전체 발생건수 / 전체 표본수로 간략화해서 계산하기도 한다. 이렇게 하면 통계량은 다음과 같이 된다. 

$$Z = \frac{(\hat{p_B} - \hat{p_A}) - (p_B - p_A)}{\sqrt{\hat{p} _0 \hat{q} _0 \left( \frac{1}{n_A} + \frac{1}{n_B} \right)}} \sim N(0, 1)$$

이제 가설을 세워보면, $\hat{p_A} = 20/50 = 0.4, n_A = 50, \hat{p_B} = 120/200 = 0.6, n_B = 200$이다. 
귀무가설을 $p_B - p_A \leq 0$이라 하면, 대립가설은 $p_B - p_A > 0$이다. 귀무가설이 참일 때의 검정통계량은 $p_B - p_A = 0$으로 설정한 분포에서의 관측값으로 공통 비율 $p_0 = \frac{20+120}{50+200}=0.56$이다. 공통 표준편차는 $\sqrt{0.56 \times (1-0.56)\left( 
\frac{1}{50} + \frac{1}{200} \right)} = 0.0784..$이다. 따라서 $Z_{stat} = \frac{0.6 - 0.4}{0.0784..} = 2.54$가 된다. 

statsmodels를 이용해서 검정하면 다음과 같다. 

```py
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
 
# 이거는 직접 계산 
count_b = 120
count_a = 20
n_b = 200
n_a = 50
p_a = count_a/n_a
p_b = count_b/n_b
p_0 = (count_b+count_a)/(n_b+n_a)
prop_var = math.sqrt((p_0*(1-p_0)/n_a  + p_0*(1-p_0)/n_b))
z_val = (0.6-0.4)/prop_var
print("z_stat : %f"%(z_val)) # 직접 계산
print ("p_val : %f"%(1-scipy.stats.norm(0, 1).cdf(z_val))) # 직접 계산
 
# 이거는 statsmodel 이용해서 검정
count = np.array([count_b, count_a])
nobs = np.array([n_b, n_a])
stat, pval = proportions_ztest(count, nobs, alternative="larger")
print("z_stat statsmodels : %f" %(stat)) # 라이브러리 이용
print("p_val statsmodels : %f " %(pval)) # 라이브러리 이용 
 
> 직접 계산값
z_stat : 2.548236
p_val : 0.005413
 
> statsmodel 결과값
z_stat statsmodels : 2.548236
p_val statsmodels : 0.005413
```

p value를 확이해보면 0.05보다 작다. 따라서 귀무가설이 기각되므로, B안이 더 효과적이라 할 수 있다. 

## Paired t test 

대응표본 차이 검정(Paired t test)은 한 집단에 어떤 변화를 주었을 때, 그에 대한 변화가 있는지를 알아보는 검정이다. Paired t test를 위해선 동일한 개체의 사전, 사후의 쌍(pair) 데이터가 있어야 하고, 각 개체의 변화 결과는 다른 개체의 결과와 독립이어야 한다. 
결국 사전 사후의 차이가 0인진를 통해 변화를 확인하므로 One-Sample t-Test와 논리가 비슷하다. 

예를 들어 어떤 정신과 의사가 환자들을 상대로 스트레스 호르몬인 코르티솔을 줄이기 위해 음악을 듣게 했다고 하자. 
이 환자들(각 개체)의 전후 상태는 다음과 같다. 

|환자|실험 전|실험 후|차이(후-전) diff|
|-|-|-|-|
|1|201|200|-1|
|2|231|236|5|
|3|221|216|-5|
|4|260|233|-27|
|5|228|224|-4|
|6|237|216|-21|
|7|326|296|-30|
|8|235|195|-40|
|9|240|207|-33|
|10|267|247|-20|
|11|284|210|-74|
|12|201|209|8|

Null Hypothesis는 "after가 더 크거나 같다.", Alternative Hypothesis는 "after가 더 작다."로 설정하면 귀무가설에 따른 확률분포를 전과 후가 같은걸로 확실하게 모수를 가정할 수 있고, 음악이 치료에 별 효과가 없다고 주장할 수 있다. 또한 만약 단측검정의 결과가 Significant하다면 귀무가설을 기각하여 치료에 효과가 있다고 주장할 수 있다. 

Null Hypothesis : $\mu_{after} - \mu_{before} \geq 0$  
Alternative Hypothesis : $\mu_{after} - \mu_{before} \leq 0$   

평균의 차이를 따져야 하는데 표준편차를 표본의 표준편차를 사용할 것이므로, 모집단이 정규분포라는 가정아래 Paired t test에서의 실험 전, 실험 후의 차이의 분포가 t 분포를 따르게 된다. 따라서 $\sigma$를 $\frac{s_{diff}}{\sqrt{n}}$로 대치하면, t 검정통계량은 $t = frac{\bar{diff} - \mu_{diff}}{\frac{s_{diff}}{\sqrt{n}}}$가 된다. 한 개의 집단에서 pair를 이루는 대상에 대한 데이터의 차이를 표본으로 두면 1 Sample t test와 같은 형태이다. 

통계량은 계산해보면 다음과 같다. 

$$t_{stat} = \left.frac{\bar{diff} - \mu_{diff}}{\frac{s_{diff}}{\sqrt{n}}} \right\vert_{\bar{diff}=-20.07, \mu_{diff}=0, s_{diff}=538.06, n=12} = -3.02$$

t가 -3.02가 나왔다. t 분포에서 자유도 11의 5%유의수준인 경우, one-sided t 값은 -1.80이므로 값이 더 작아 귀무가설을 기각할 수 있다. 즉 음악이 스트레스 완화에 효과가 있다고 결론지을 수 있다. 

statmodel을 이용하면 다음과 같다. 

```py
paired_sample = stats.ttest_rel(df_raw['실험후'], df_raw['실험전'] )
print('t검정 통계량 = %.3f, p_value = %.3f'%paired_sample)
```

t검정 통계량 = -3.020, p_value = 0.012가 나오고, p value가 5%보다 작다. 

diff를 1개의 변수로 보았으니, 1 Sample t test로 검정해보면 다음과 같다. 

```py
import numpy as np
 
x = np.subtract(np.array(after), np.array(before))
ttest_1samp(x, 0)
```

결과는 t검정 통계량 = -3.020, p_value = 0.012로 같다. 














