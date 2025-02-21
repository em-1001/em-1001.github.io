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


가설검정은 가설을 세우고, 그것이 올바른 가설인지 확률로 판단하는 것을 말한다. 자세히 말하면 "관측된 표본으로부터 무엇인가를 판단할 때,
이 판단에 오류가 있을 수 있는지에 대해서 오류의 가능성을 미리 정해진 수준에서 관리하고자 하는 것"이다. 

신뢰수준(구간)은 모수가 포함된 구간의 확률이고, 유의수준은 신뢰수준을 제외한 나머지 Extreme 영역을 말한다. 유의수준은 보통 $\alpha%$로 
표기하는데, 만약 신뢰수준이 95%라면, 유의수준은 5%가 된다. 유의수준(Significant Level)을 양쪽 구간으로 따질 때는 유의수준을 둘로 나눠서 
양측 검정($\alpha/2$)을하고, 한쪽만 따질 때는 나누지 않고 단측검정을 한다. 



 
