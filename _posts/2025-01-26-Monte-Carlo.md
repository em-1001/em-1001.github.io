---
title:  "Monte Carlo method"
excerpt: "Law of Large Numbers & Monte Carlo method"

categories:
  - Statistics
tags:
  - Statistics
last_modified_at: 2025-01-19T08:06:00-05:00
---

> [Untitled님의 블로그](https://untitledtblog.tistory.com/190)를 보고 정리한 글입니다.

# Law of Large Numbers(LLN)
큰 수의 법칙은 표본 크기(샘플 수)가 커질수록 표본 평균이 기대값에 점점 가까워진다는 통계학의 기본 원리이다. 몬테카를로 방법은 이 원리를 활용하여 확률적 시뮬레이션을 통해 적분, 기대값, 또는 복잡한 문제를 근사적으로 해결한다. 큰 수의 (약한) 법칙은 동일한 확률분포에서 독립적으로 추출된 (i.i.d.) 확률변수에 대해 아래와 같이 정의된다. 

$$\lim_{N \to \inf} \frac{1}{N} \sum_{i=1}^N f(x_i) = \mathbb{E}[f(x)] = \mu$$

$x_i$는 확률변수이고, $f(x_i)$는 우리가 계산하려는 함수이다. 

# Monte Carlo Integration
몬테카를로 방법 (Monte Carlo method)은 무작위로 추출된 샘플을 바탕으로 함수의 값을 수리적으로 근사하는 알고리즘이다. 몬테카를로 방법의 한 응용으로써 몬테카를로 적분 (Monte Carlo integration)이 있으며, 몬테카를로 적분은 수학, 물리, 인공지능 등에서 매우 다양하게 활용되고 있다. 몬테카를로 적분은 독립적으로 추출된 샘플을 이용하여 적분값을 계산하기 위한 방법이다.

어떠한 함수 $f(x)$의 적분값을 계산하기 위한 몬테카를로 적분은 아래와 같이 유도된다.

$$\begin{align}
\int_{\Omega} f(x)dx &= \int_{\Omega} \frac{f(x)}{\phi(x)}\phi(x)dx \\ 
&= E_{\phi(x)} \left[ \frac{f(x)}{\phi(x)} \right]  
\end{align}$$

$\phi(x)$는 $x$의 확률밀도함수이며, $\Omega$는 적분 공간이다. 

샘플의 수 $n$이 충분히 크면 큰 수의 법칙에 의해 아래와 같이 근사된다. 

$$E_{\phi(x)} \left[ \frac{f(x)}{\phi(x)} \right] \approx \frac{1}{n}\sum_{i=1}^n \frac{f(x_i)}{\phi(x_i)}$$ 

최종적으로 함수 $f(x)$의 적분값은 아래와 같이 근사되며, 이를 일반화된 몬테카를로 적분(general Monte Carlo integration)이라 한다.

$$\int_{\Omega} f(x)dx \approx \frac{1}{n}\sum_{i=1}^n \frac{f(x_i)}{\phi(x_i)}$$

몬테카를로 적분의 정확한 계산을 위해서는 $x$의 확률밀도함수인 $\phi(x)$에 대한 정의가 필요하다. 몬테카를로 방법에서 $\phi(x)$ 를 정의하기 위해 일반적으로 이용되는 것은 균등분포 (uniform distribution)이다.

만약 $x \in [a, b]^d$ 가 균등분포를 따라 존재하는 변수라고 가정하면, $\phi(x) = 1/(b-a)^d$가 되며 몬테카를로 적분은 아래와 같다. 

$$\int_{\Omega} f(x)dx \approx \frac{(b-a)^d}{n}\sum_{i=1}^n f(x_i)$$

따라서, 우리는 $f(x)$의 역도함수 (antiderivative)인 $F(x)$를 계산할 수 없는 경우에도 샘플 $x_1, x_2, \cdots, x_n$를 기반으로 몬테카를로 적분을 이용하여 함수의 적분값을 근사할 수 있는 것이다.

## Monte Carlo integration example: Area of ​​a circle

<p align="center"><img src="https://github.com/user-attachments/assets/bf96cced-54ec-4b68-94b8-132c7cccfd9d"></p>

몬테카를로 적분의 예제로써 적분 계산 없이 원의 넓이를 구하는 방법을 소개한다. 실제로 적분계산을 통해 나온 결과와 몬테카를로 적분를 활용했을 때의 결과가 어떻게 나오는지 확인해보자.

$$\begin{align}
S &= 4 \int_0^r \sqrt{r^2 - x^2}dx \\   
&= 4 \int_{0}^{\frac{\pi}{2}} r^2 \sin^2 \theta d\theta \because \ x \triangleq r \cos \theta, dx = -r \sin \theta d\theta \\ 
&= 4 \cdot r^2 \cdot \frac{1}{2} \cdot \frac{\pi}{2} \because \ {\scriptstyle\text{Wallis  formula}} \\  
&= \pi r^2
\end{align}$$

반지름이 3인경우($r=3$) 원의 넓이는 약 28.274가 나온다. 

$[0, 3]$의 범위에서 균등분포에 의해 무작위로 생성된 $n$개의 샘플 $x_1, x_2, \cdots, x_n$에 대해 몬테카를로 적분을 기반으로 원의 넓이를 근사하면 아래와 같다.

$$4 \int_0^r \sqrt{9 - x^2}dx \approx \frac{12}{n}\sum_{i=1}^n \sqrt{9 - x_i^2}$$

위 근사에 따라 결과를 확인하면 다음과 같다. 

```py
import numpy as np
import matplotlib.pyplot as plt

# 파라미터 설정
radius = 3.0               # 원의 반지름
max_samples = 100000        # 최대 샘플 수
step = 500                 # 샘플 수 증가 단위
true_area = np.pi * radius**2  # 원의 실제 넓이 (9π)

# 샘플 수와 추정된 넓이를 저장할 리스트
sample_sizes = []
estimated_areas = []

# 몬테카를로 시뮬레이션
for num_samples in range(1, max_samples + 1, step):
    # 균등분포로 점 생성
    x = np.random.uniform(0, radius, num_samples)
    
    area_estimate = 12/num_samples * np.sum((9-x**2)**0.5)
    
    sample_sizes.append(num_samples)
    estimated_areas.append(area_estimate)

# 그래프 그리기
plt.figure(figsize=(8, 4))
plt.plot(sample_sizes, estimated_areas, label='Monte Carlo Estimate', color='blue')
plt.axhline(true_area, color='red', linestyle='--', label='True Area (9π)')
plt.xlabel('Number of Samples')
plt.ylabel('Estimated Area')
plt.title(f'Monte Carlo Estimation of Circle Area (Radius = {radius})')
plt.legend()
plt.grid(True)
plt.show()
```

<img src="https://github.com/user-attachments/assets/f055cf9c-e7f0-4088-8927-29c4a104679f">


## Monte Carlo integration for stochastic models
확률모델의 학습이나 추론 과정에서 가장 빈번하게 마주치는 문제는 아래 식과 같이 어떠한 확률분포를 따르는 변수 $x \in X$와 관련된 기댓값을 계산하는 것이다.

$$E_{x \sim p} [f(x)] = \int_X f(x)p(x)dx$$

머신러닝에서는 일반적으로 $p(x)$를 따르는 i.i.d. 샘플 $x_1, x_2, \cdots, x_n$에 대한 몬테카를로 적분을 기반으로 아래와 같이 $f(x)$의 기댓값을 계산한다. 

$$\int_X f(x)p(x)dx \approx \frac{1}{n}\sum_{i=1}^N f(x_i)$$

대부분의 머신러닝 응용에서는 데이터 (샘플)이 주어지기 때문에 가능도나 사후확률과 같은 어떠한 함수 $f(x)$의 기댓값을 계산해야하는 문제에서 우리는 몬테카를로 적분을 이용하여 이를 해결할 수 있는 것이다.







