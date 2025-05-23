---
title:  "[Statistics] Regression IV"
excerpt: Regression

categories:
  - Statistics
tags:
  - Statistics
toc: true
toc_sticky: true
last_modified_at: 2025-01-19T08:06:00-05:00
---

## Information Value

로지스틱 회귀 예측 모형을 만들 때, 변수를 선택하는 방법론에 대해 알아보자. 이 선택방법은 회귀분석 결과를 설명 가능하게 하는 것이 목적이 아니라 철저하게 예측을 잘하기 위해 쓰는 방법이다. 

IV와 WoE라는 용어가 있는데, 먼저 IV는 Information Value이고, WoE는 Weight of Evidence이다. IV는 어떤 Feature가 얼마나 로지스틱 종속변수에 영향력과 가치 있는가를 따지고, WoE는 IV를 계산할 때의 가중치가 된다. 

그러면 다중 로지스틱 회귀를 할 때 어떤 Feature가 도움이 되는 Feature일지를 생각해보자. 로지스틱 회귀에 가장 도움이 되는 Feature는 0과 1에 대해서 관측치가 불균형 하면 판단에 도움이 된다고 생각하면 된다. 불균형을 따져보는 가장 손쉬운 방법으로는 각 구간마다 1과 0의 비율의 차이가 어떤지 보면 된다. 

따라서 Event의 비율 (Event %) - Non Event의 비율 (Non Event %)을 생각해 보자. 여기서 %는 100을 곱한 퍼센트가 아니라 비율의 의미이다. 
일단 비율의 차이를 계산하므로 비율이 불균형 할수록 값이 커진다. 즉, 어느 한쪽에 몰려 있을 수록 두 비율의 차이의 절댓값이 커진다. 하지만 이 값 만으로는 치우침의 불균형한 정도를 잘 표현하기 어렵다. 0~1사이의 작은 값이기 때문에 좀 더 과장되게 표현하기 위해 WoE가 존재한다. WoE는 비율의 비율에 log를 씌우는 건데 앞서 살펴본 Log(odds)와 동일한 Logit이다. 

logit변환의 특성을 보면 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/a8be5511-9ed9-4639-8fe9-3d1d7b6934a2" height="" width=""></p>

logit을 자세히 보면 0.5이상에서는 점점 늘어나다가 1이 가까워 지면 매우 커지고 반대의 방향으로는 점점 줄어들다가 0에 가까워지면 매우 작아진다. 

logit을 활용하면 불균형 할 수록 즉, Event가 월등히 커서 비율이 1에 가깝거나, Non Event가 월등히 커서 비율이 0에 가깝거나 하면, 이 값을 Logit 변환했을 때에는 플러스든 마이너스든 매우 큰 값이 되므로 Event와 Non Event의 비율이 비슷한 것들의 효과를 줄이고, 불균형이 큰 것들의 효과를 크게 한다. 

추가적으로 Event 비율 - Non Event 비율의 값이 Negative인 경우에는 WoE도 똑같이 Negative가 되어 서로 곱하면 Positive가 되고, 마찬가지로 서로 Positive인 경우에는 곱하면 Positive가 된다. 따라서 두 값을 곱하면 과장된 불균형 정도를 표현할 수 있다. 

정리해서 어떤 Feature의 "각각의 구간에 대한" 정보가치는 다음과 같이 표현된다. 

$$IV_i=(Event_i\%-NonEvent_i\%) \times WoE_i$$ 

어떤 Feature의 전체의 정보가치는 이걸 모두 더한 값이 된다. 

$$IV_{total}=\sum((Event_i\%-NonEvent_i\%) \times WoE_i)$$

결국 Event와 Non Event의 불균형을 과장해서 표현한 후에 전체적으로 불균형이 어느정도가 되는지를 측정하고 이를 Information Value라 부른다. 거꾸로 말하면 불균형이 없을수록 Information Value가 매우 작고, 이는 Logistic 회귀를 해 봐야 큰 의미가 없다고 주장하는 바와 같다. 

다음 예제를 살펴보자. 

<p align="center"><img src="https://github.com/user-attachments/assets/373dd9a7-1b02-4e03-955a-8d8b205a27bf" height="" width=""></p>

나이대에 대한 Event와 Non Event데이터 이다. IV를 정의대로 계산한다면, 20~60대까지 구간이 10살씩 나눠져 있고 이 구간에 대하여 Event비율, Non Event비율을 구한 후에 Event % - Non Event % = Delta를 각각 구하고 ln(Event % / Non Event %)로 WoE를 구한 후에 서로 곱한다. 그러면 각 나이대별 IV 정보 가치를 구하게 되고, IV 열을 모두 더하게 되면 결국 최종 IV가 되게 된다. 

최종 불균형 정도를 살펴보면 0.24인데, 이 수치에 대한 판단 기준은 아래와 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/f5e8828e-ec84-424e-bfe9-a1c4a2a9e8ed" height="" width=""></p>











