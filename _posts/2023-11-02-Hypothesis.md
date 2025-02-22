---
title:  "[Statistics] Null Hypothesis & Alternative Hypothesis"
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


가설검정은 관심을 가지고 있는 모집단에 대한 가설을 세우고, 표본을 뽑아서 표본정보로부터 그 가설들을 검증하는 것을 의미한다. 

신뢰수준(구간)은 모수가 포함된 구간의 확률이고, 유의수준은 신뢰수준을 제외한 나머지 Extreme 영역을 말한다. 유의수준은 보통 $\alpha%$로 
표기하는데, 만약 신뢰수준이 95%라면, 유의수준은 5%가 된다. 유의수준(Significant Level)을 양쪽 구간으로 따질 때는 유의수준을 둘로 나눠서 
양측 검정($\alpha/2$)을하고, 한쪽만 따질 때는 나누지 않고 단측검정을 한다. 

가설검정을 위해서 Null Hypothesis(귀무가설 $H_0$), Alternative Hypothesis(대립가설, $H_1$)을 설정해야 한다. 
Null Hypothesis는 어떤 확률분포를 가정하고, 그 조건에서 해당 확률분포에 따라 어떤 관측이 무작위로 관측될 것이라는 가설이다. 
Alternative Hypothesis는 관찰해보니 가정한 확률분포가 아닌 것 같을 때 Null Hypothesis를 기각하고 채택한다. 
Null Hypothesis는 확률분포가 어떻다고 가정했을 뿐이지 이 가설이 무조건 참인 것은 아니다. 

두 가설을 세우고 나면, 통계적인 유의성(Significance)과 p value를 고려한다. 예를 들어 어떤 실험을 통해 얻은 그룹 간의 차이가 무작위로 발생할 수 있는 합리적인 수준보다 더 극단적으로 다르다면, 두 그룹의 차이가 우연히 나온 것이 아니라고 접근할 수 있고, 이때 판단 기준이 유의 수준과 p value가 된다. 


