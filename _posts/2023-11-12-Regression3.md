---
title:  "[Statistics] Regression III"
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

# Prediction
지금까지는 통계분석을 위한 회귀분석을 했다면, 이제는 예측을 목적으로 둘 것이다. 분석이 아니라 예측이 목적이면 계수의 크기, 계수의 유의성, 상관관계, VIF 등이 크게 중요하지 않을 수 있다. 계수의 분산이 크더라도, 다중공선성이 있더라도 각각의 독립변수들의 기여량을 합하면 종속변수를 예측할 수 있게 된다. 하지만 이런 경우는 예측 성능이 좋게 나오더라도 왜 이렇게 나오는지 설명하지는 못한다. 

<p align="center"><img src="https://github.com/user-attachments/assets/31e3bbed-3fa5-49f0-91be-3d1d8902999c" height="" width=""></p>

VIF등이 상관 없을 수도 있다곤 했지만, 설명 가능한 회귀 예측 모형을 만들려면 주요 Feature의 계수가 유의하도록 모형을 만들어야 예측도 가능하면서 설명 가능한 회귀 예측 모형이 된다. 

회귀모형이 설명과 관련없이 예측만을 목적으로 한다면 예측한 값과 실제값의 차이를 Root Mean Square Error(RMSE)로 측정한다. 즉, 모형이 실제값과 얼마나 틀리고 있냐만 관심을 갖는다. 그래서 회귀분석을 할 때는 관심 Feature의 계수를 유의하게 만드는 것이 목적이므로 지금까지의 과정을 거친 것이라면, 예측을 할 때는 Feature 중에 RMSE를 안 좋게 하는 Feature만을 제거하기도 한다. 

참고로 회귀 예측모형에서 학습(Training) 데이터와 시험(Test) 데이터의 공분산구조가 비슷한 것이 중요하다. 왜냐하면 학습데이터로부터 학습한 계수는 공분산행렬과 밀접하게 관련이 있는데, 막상 시험데이터에서 그게 많이 다르다면 예측이 엉망이 될 가능성이 크다. 그래서 학습데이터와 시험데이터를 잘 섞어서 예측 모형을 만드는 것이 중요하다. 

## Categorical independent variables
지금까지 연속형 독립변수를 이용해 회귀를 했다면, 이제는 남/여, 학년, 혈액형과 같은 범주형 독립변수에 대한 회귀를 알아볼 것이다. 방법은 범주형 변수를 숫자형 변수로 만들어 회귀를 하는 것인데, 데이터의 범주를 1, 0으로 feature를 늘려서 보는 것이다. 예를 들어 다음과 같은 데이터가 있다고 하자. 

<p align="center"><img src="https://github.com/user-attachments/assets/007aa6d4-9268-4d23-a233-bdc4a8c3c342" height="" width=""></p>

위 데이터의 경우 성별과 혈액형이 범주형 데이터이다. 우선 성별 Feature를 Dummy Variable로 만들면, 남/여인 성별을 1과 0으로 표현한다. 

<p align="center"><img src="https://github.com/user-attachments/assets/68641788-cc11-4d3d-8a68-c6c28faf5385" height="" width=""></p>

이런 식으로 성별이 남자인 경우에는 성별_남자만을  1로 나머지는 0, 여자인 경우에는 성별_여자만 1로 나머지는 0으로 표현한다. 이런 식으로 숫자로 나타낼 수 있으니까, 결국 회귀분석을 할 수 있고, 이렇게 범주형 변수를 1/0 형식으로 만들어 주는 행위를 One Hot Encoding이라고 한다. 또한 이런  One Hot Encoding 한 변수를 Dummy 변수라 한다. 

그런데, 여기서 조심해야 할 부분이 있는데, 이렇게 모든 범주를 고려해 Dummy 변수를 만들게 되면, 다중공선성이 발생하게 된다. 

예를 들어 성별_남자, 성별_여자를 모두 감안해 회귀를 하면 다음과 같은 회귀식을 구하게 된다. 

$$y=b_0+b_1x_{man}+b_2x_{woman}$$

여기서 남자와 여자의 관계는 $x_{man}+x_{woman}=1$이다. 따라서 이를 원래 회귀식에 넣으면 다음과 같다. 

$$\begin{align}
y &= b_0 + b_1x_{man}+b_2x_{woman} \\ 
&= b_0 + b_1(1-x_{woman})+b_2x_{woman} \\ 
&= (b_0+b_1)+(b_2-b_1)x_{woman}
\end{align}$$

이렇게 되면 성별 중 여자 하나로 회귀식이 풀리게 되어 원래 분석하려던 계수들이 섞여 분석할 수 없게 되고, 행렬을 이용해 회귀를 풀게 되면 det가 0이 된다. 따라서 $x_{woman}=1-x_{man}$과 같이 독립변수가 서로 선형관계를 맺으면 안된다. 
이렇게 Dummy Variable이 다중공선성을 띄여서 문제가 발생하는 것을 Dummy Trap이라 한다. 

이러한 문제를 해결하기 위한 방법은 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/df2c33f8-8db2-4536-8a14-34d3651d310d" height="" width=""></p>

이런 식으로 여자냐 아니냐 1개로 구분하면 Dummy 변수로 인한 다중공선성을 해결할 수 있다. 그렇다면 혈액형은 어떻게 해야 할까?

<p align="center"><img src="https://github.com/user-attachments/assets/c99a1e2b-2751-4bc3-bc9c-5a612b300001" height="" width=""></p>

이때도 마찬가지로 혈액형 A일 때를 제외하고 만들면 된다. 

그런데 다중공선성을 제거하기 위해 이렇게 하는 대신 A형 1, B형 2, O형 3, AB형 4 이런 식으로 정하면 어떻게 될까? 이렇게 하면 값에 대해서 크기에 따라 순서가 생기게 되고, 순서가 생기게 되면 회귀를 할 때 이 크기가 의미를 갖게 된다. 범주라는 것은 그런 의미가 아니니까, 모두 따로따로 만들어서 그런 순서가 생기지 않도록 해야한다. 

결국에 Dummy 변수라는 건 모든 값이 0인 경우를 기준으로 하면 다른 범주들과 기준범주를 비교할 수 있다. 
이제 범주형 데이터를 Dummy Variable로 만드는 방법에 대하여 알았으니까 해석방법을 통해 의미를 확인해보자. 

가장 중요한 점은 Dummy는 범주형 데이터이기 때문에 회귀식의 기울기에는 관여하지 않고, y절편에 관계한다. 

<p align="center"><img src="https://github.com/user-attachments/assets/4dfc1ec8-4103-4ff1-985c-eb4b5a384c5f" height="" width=""></p>

이럴 때 $x_1$이 dummy인 경우라 생각하면, $x_1$이 1이면 $x_1$의 계수 $b_1$만큼, $x_1$이 0이라면 $x_1$의 영향력이 없어지므로 $x_1$이 0일 때와 1일 때의 차이만큼 shift가 되기만 하고 기울기에는 영향이 없다. 위 예시는 1개의 Dummy변수에 관련하여 보았는데, 이게 혈액형의 경우처럼 여러 개라면 그 개수만큼 y절편의 shifting이 생긴다. $x_1$이 성별이라고 생각한다면 남성과 여성이 $b_1$만큼 차이가 난다고 해석하면 된다. 결국 기준 더미 변수로부터 0이 아닌 Dummy의 계수만큼 차이가 난다는 의미로 해석을 하면 된다. 

그러면, 이 Dummy 변수라는 것을 알게 되었으니까, Dummy 변수가 포함된 회귀를 한 후에 그 결과를 해석할 수 있어야 한다. 
예시 분석 결과는 다음과 같다. 

<p align="center"><img src="https://github.com/user-attachments/assets/d57bd02d-b964-48d1-8c53-653957de13d4" height="" width=""></p>

이 결과는 종속변수가 자신의 행복도, 연령 Dummy에서 기준 Dummy변수는 20대인 회귀분석의 결과이다. 회귀식으로는 다음과 같다. 

$$y_{happy}=b_0+b_1x_{woman}+b_2x_{age30}+b_3x_{age40}+b_4x_{dating}+b_5x_{friendship}+b_6x_{salary}+b_7x_{exercise}$$

1. 결정계수와 수정 결정계수를 보면 0.535, 0.522로 회귀식 설명력이 꽤 높다.
2. F검정 결과가 유의하다. F통계량이 12.121로 꽤 크므로 확률적으로 의미 있는 회귀 분석이라 할 수 있다.
3. VIF를 보면 문제가 될 만큼 큰 크기의 변수는 없다.  그렇지만 참고로 Dummy 변수의 VIF가 큰 경우에도 Dummy 변수가 우리가 꼭 분석해야 하는 범주형 변수라면 그대로 두고 해석해야한다.
4. 계수의 유의성을 참고해서 해석하면, 더미(독립)변수는 여성, 40대 연령이 유의했고, 나머지 독립변수 중 유의한 것들은 연애기간, 연봉이다. 그렇다는 이야기는, 여성과 남성의 행복도의 차이가 나고요,  20대(기준)와 40대의 행복도가 차이가 난다. 30대의 경우에는 계수가 유의하지 않아 기준연령인 20대와 차이가 난다는 증거를 발견하지 못했다고 생각하면 된다.
5. 그러면 이 분석결과를 영향력을 보려면 표준화된 계수와 함께 보면 상대적인 영향력을 잘 설명할 수 있다. 40대가 -0.036으로 부정적으로 큰 영향을 주었으며, 다음으로 여자 인가 와 연애기간이 0.02, 연봉이 0.02의 순으로 영향을 준다. 종합해서 해석하면 여자이고, 연애기간과 연봉이 클수록 행복도가 높아지고, 나이로 본다면 40대라는 나이는 20대에 비해 행복감에 부정적인 영향을 준다고 해석할 수 있다.
6. 구체적으로 실제값의 변화를 해석해 보면, 여성이 남성에 비해 4.808 정도 더 행복했고, 20대에 비해 40대가 7.885 정도 덜 행복하다. 연애기간은 1 단위 (1년) 늘때마다 0.721 행복해 졌고, 연봉은 1단위 (100만원)이 늘 때 마다 0.530 만큼 더 행복해졌다. 

참고로 pandas Dataframe에는 이런 Dummy Variable을 자동으로 만들 수 있는 방법이 제공되는데, `pd.get_dummies(data=df, columns=['성별', '혈액형']`이런 식으로 해 주면 자동으로 컬럼을 분석해서 Dummy를 만들어 준다. 이때, `drop_first=True`를 추가해서 넣어주면 아예 첫번째 값을 0으로 해서 전체범주-1 개의 Dummy 컬럼을 만들어주긴 하는데, 이게 first의 reference값이 무엇인지를 사용자 마음대로 지정할 수가 없는 불편함이 있긴하다. 그래서 그냥 first drop을 하지 않고, dummy를 만든 후에 reference 컬럼을 drop 해 버리는 방식을 사용한다. 

## Logistic Regression

### log-transformation

데이터를 로그 변환하는 경우는 분포가 치우쳐져 있는 경우 매우 유용하다. 

<p align="center"><img src="https://github.com/user-attachments/assets/27bff698-d6cc-4f40-82cf-68ed96f2f439" height="" width=""></p>

위 이미지의 경우 왼쪽으로 치우친 분포에 로그를 씌우면 운이 좋은 경우 정규성을 보일 수도 있다. 

<p align="center"><img src="https://github.com/user-attachments/assets/1088adae-33f1-41dc-bfc1-fcacc4bc7188" height="" width=""></p>

반대로 오른쪽으로 몰린 경우 로그의 역함수인 지수함수를 이용할 수 있다. 

<p align="center"><img src="https://github.com/user-attachments/assets/835ab894-8a29-442d-bac5-725efc4d8b68" height="" width=""></p>

이렇게 하면 로그나 지수함수를 썼으니 해석이 달라지는데, 다음 3가지 경우의 변환이 있다. (로그가 취해진 변수는 %단위로 변화한다고 생각한다.)

1. 독립변수에만 로그가 취해진 경우
2. 종속변수에만 로그가 취해진 경우
3. 둘 다 로그가 취해진 경우

구체적인 예를 들어보면 독립변수에만 로그가 취해진 경우     
회귀) y = 상수항 + 1234*ln(x)     
해석) x가 1% 변화할 때 y의 변화량은 12.34이다.  

종속변수에만 로그가 취해진 경우     
회귀) ln(y) = 상수항 + 0.1234*(x)    
해석) x가 1 증가할 때 y는 12.34%증가한다.   

둘 다 로그가 취해진 경우    
회귀) ln(y) = 상수항 + 0.413*ln(x)  
해석) x가 1%증가할 때 y는 0.4%증가한다. 

이렇게 해석되는 이유를 살펴보면, 

**1. 독립변수에 로그를 취하게 되면**, $y=Constant+bln(x)$ 이므로, 양변을 미분하면  

$$dy=b\frac{dx}{x}$$ 

위와 같이 되고, 우변은 x에 대하여 dx만큼 변화하는 변화율이 된다. 이걸 퍼센트로 표시하기 위해 100을 곱하면 y의 변화는 b/100이 된다. 

$$dy=\frac{b}{100} \cdot \left(100 \frac{dx}{x} \% \right)$$

따라서 x가 1%변할 때 y는 b/100만큼 변한다. 

**2. 종속변수에 로그를 취하면**, $ln(y)=Constant+bx$ 이므로, 같은 식으로 미분하면

$$\frac{dy}{y}=bdx$$

위와 같이 되고, 이것도 y변화율 퍼센트로 나타내면 

$$\left( \frac{dy}{y} \cdot 100\% \right) = bdx \cdot 100$$ 

x가 1변할 때 $100 \cdot b\%$만큼 변한다. 

**3. 둘 다에 로그가 취해졌을 때**, $ln(y)=Constant+bln(x)$이므로 미분하여 퍼센트로 바꾸면 다음과 같다. 

$$100 \frac{dy}{y}=100b\frac{dx}{x}$$

따라서 x가 1%변할 때, y는 b%변하게 된다. 

매번 이렇게 미분해서 생각하기 쉽지 않으므로 쉽게 생각하는 방법은 로그가 붙은 변수는 퍼센트로 생각하고 $ln=\frac{1}{100}$으로 생각하면 빠르게 읽을 수 있다. 

로그를 씌우는 것은 정리하면 큰 값을 매우 작게, 작은 값은 조금 작게 바꾸는 것이라 생각하면 된다. 

### Logistic Regression & Sigmoid

종속 변수가 연속인 경우의 회귀를 해 보았다면, 종속 변수가 1, 0만 갖는 binary의 경우를 살펴보자. 
종속 변수가 binary라면 회귀 모형 출력이 1, 0으로 나올 것 같지만 실제 결과는 확률이다. 

종속 변수가 1, 0으로 이루어진 로지스틱 회귀를 할 때는 시그모이드를 사용한다. 시그모이드는 S자 곡선을 그리면서 a~b사이의 한정된 값을 갖고, 항상 Monotonically 증가하는 함수를 통칭한다. 시그모이드에 해당하는 함수는 Logistic함수, tanh(x), Smoothstep, algebraic 등이 있다. 지금은 logistic함수를 시그모이드로 하여 예를 들어보자. 

<p align="center"><img src="https://github.com/user-attachments/assets/551d2b33-f611-4fdf-b42d-2515e9092e56" height="" width=""></p>

<p align="center"><img src="https://github.com/user-attachments/assets/2aa9fa35-34b9-492a-b176-c58604dca889" height="" width=""></p>

x를 구획을 나눠서 각각의 구획에서 $\frac{\#1}{\#1+\#0}$을 이용해 각 구획의 1의 비율을 구한다. 여기서 #은 개수라는 뜻이다. 결국 1이 나올 확률이고 따라서 실제로는 y가 1,0값을 갖는 것에 대해 회귀하는 것이 아니라, y가 갖는 확률값을 회귀하는 것이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/a6fcff3f-3c38-4305-b08e-f3d4eb5aa479" height="" width=""></p>

이렇게 하면 실제로 확률값에 대한 회귀식을 구하는 것이기 때문에 Sigmoid의 출력값은 확률이 된다. 

이제 Sigmoid를 Logistic함수로 특정하여 예를 들어보자. Logistic함수는 다음과 같다.

$$f(x)=\frac{1}{1+e^{-x}}$$

<p align="center"><img src="https://github.com/user-attachments/assets/74085ba9-1b81-462d-9e8c-2fa8eb0d4707" height="" width=""></p>

이제 P(x)를 Logistic함수를 이용해 회귀 분석하여 표현할 수 있다. 이때 Logistic 함수의 가파른 정도와 꺾이는 시점이 여러가지 있다. 이를 감안하여 분모의 $e^{-1}$를 조금 더 여러 모양새의 시그모이드 형태가 가능하도록 $f(x)=b_0+b_1x$같은 식으로 대체를 하면 $b_1$에 의해 가파른 정도를 결정하고, $b_0$에 의해 어디서 부터 꺾이기 시작할지의 모양새를 결정할 수 있다. $b_1$이 클수록 Logistic함수가 가파르게 되고, $b_0$값이 양수로 클수록 왼쪽으로 shift된다. 이렇게 하면 종속변수가 1,0인 것에 대해서는 어떤 모양이라도 종속변수가 확률인 것에 대한 회귀를 할 수 있다. 

최종적인 식을 정리하면 우선 회귀식을 우변에 두고 $P(x)=\frac{1}{1+e^{-(b_0+b_1x)}}$를 정리하면 다음과 같다. 

$$\frac{P(x)}{1-P(x)}=e^{b_0+b_1x}$$

여기서 Odds가 나오는데 Odds는 1을 성공, 0을 실패라 했을 때, 성공과 실패의 비율 즉, P(1)/P(0)를 말한다. 실패에 비해서 얼마나 성공 확률이 큰가를 따지는 것이다. 

다음으로 Exponential이 나왔으므로 양변에 로그를 취하면 우변에 선형회귀식과 비슷한 꼴이 남는다. 

$$ln\left(\frac{P(x)}{1-P(x)}\right)=b_0+b_1x$$

따라서 ln(Odds)가 우변의 선형회귀처럼 된다. 이렇게 함으로써, Logistic함수의 모양새를 결정하는 선형회귀식을 구할 수 있게된다. 

<p align="center"><img src="https://github.com/user-attachments/assets/0244e84e-e74a-483c-a3ae-bcbeaa6be11c" height="" width=""></p>

결국 log(Odds)가 오른쪽 처럼 직선이 되고, 이렇게 되면 지금까지 했던 선형회귀를 그대로 적용해서 분석할 수 있다. 또한 이 log(Odds)를 **Logit**이라 부르고 Logit을 선형회귀하는 것이 로지스틱회귀이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/2b206bda-4f49-460a-a7ee-1bb4e32660b3" height="" width=""></p>

참고로 Logit은 Log + Probit이라는 말의 합성어인데, probit은 Probability Unit의 합성어로 나중에 다룰 것이다. 

<p align="center"><img src="https://github.com/user-attachments/assets/4200ab01-3c05-4382-bf17-d87cbe8f2c17" height="" width=""></p>

이제 종속변수를 p/(1-p)로 생각하면 이전에 살펴보았던 로그를 적용한 종속변수의 해석 그대로 독립변수 x가 1단위 변하면 P/(1-P) 즉, 성공/실패 활률의 값이 $(100 \cdot b_1)%$ 만큼 변하게 된다. 

로지스틱 회귀도 다중회귀가 가능한데 일반화 하면 다음과 같다. 

$$ln \left( \frac{P(y=1 \vert \boldsymbol{x})}{1-P(y=1 \vert \boldsymbol{x})} \right) = b_0 + b_1x_1 + \cdots + b_ix_i$$

자세히 보면 x가 여러 종류 즉, $x_1, x_2, x_3, \cdots, x_i$라는 의미에서 bold이다. 

$$\therefore \frac{P(y=1 \vert \boldsymbol{x})}{1-P(y=1 \vert \boldsymbol{x})} = e^{(b_0 + b_1x_1 + \cdots + b_ix_i)}$$

만약 j번째 x를 1만큼 증가시키면 다중회귀에서 j번째 계수가 얼마나 영향을 미치는지 알 수 있다. 

$$\frac{e^{(b_0 + b_1x_1 + \cdots + b_j(x_j+1) + \cdots + b_ix_i)}}{e^{(b_0 + b_1x_1 + \cdots + b_j(x_j) + \cdots + b_ix_i)}} = e^{b_j}$$

이것은 $odds_{new}=odds_{old} \cdot e^{b_j}$라 해석할 수 있다. 원래 odds에 비해 $e^{b_j}$배가 된다는 의미이다. 예를 들어 원래 odds ratio가 1이었고, $b_j=0.3 \to e^{0.3} \approx 1.34$일 때, $x_j$를 1만큼 올리면 odd는 1.34배가 된다는 뜻이다. 이를 로그를 씌운 변수에 대해서 대략적으로 읽어도 j번째 x가 1변할 때, 계수가 0.3이라면 종속변수 odds는 30%정도 늘어난다고 읽을 수 있다. 















