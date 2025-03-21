---
title:  "[CS231n] 3.Back Propagation"
excerpt: "CS231n Lecture Note"

categories:
  - Computer Vision
tags:
  - CS231n
toc: true
toc_sticky: true
last_modified_at: 2023-10-20T08:06:00-05:00
---

# Back Propagation

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/4ad56091-b9f7-4a61-9891-2eb21ab559ac" height="40%" width="40%"></p>

활성함수로는 sigmoid함수가 쓰였다고 하자. z는 이전 layer에서 각각의 weight와 곱해진 값들의 합이다. h, o는 각각의 z에 sigmoid를 적용한 값이다. 

역전파는 weight를 output layer에서 input으로 거꾸로 진행하며 update를 한다. 
output layer에서 N개의 layer로의 역전파를 step1이라 하고 N개의 layer에서의 역전파를 step2로 하고 설명을 하겠다. 

### step1
처음으로 update될 weight들은 $w_5, w_6, w_7, w_8$이다. 
먼저 $w_5$를 update하려면 $\frac{\partial E_{total}}{\partial w_5}$를 계산해야 한다. 
이를 계산하기 위해선 아래와 같이 chain rule을 적용한다. 

$$\frac{\partial E_{total}}{\partial w_5} = \frac{\partial E_{total}}{\partial o_1} \times \frac{\partial o_1}{\partial z_3} \times \frac{\partial z_3}{\partial w_5}$$

첫 번째 term부터 계산해보자. 

$$E_{total} = \frac{1}{2}(target_{o_1} - output_{o_1})^2 + \frac{1}{2}(target_{o_2} - output_{o_2})^2$$

$$\frac{\partial E_{total}}{\partial o_1} = 2 \times \frac{1}{2}(target_{o_1} - output_{o_1})^{2-1} \times (-1) + 0$$

다음으로 두 번째 term으로 가서 $o_1$은 sigmoid 함수의 결과이다. sigmoid를 미분하면 아래와 같은 결과가 나온다. 

$$f(x)  = \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x}$$

$$\frac{d}{dx}f(x) = \frac{e^x(1+e^x)-e^xe^x}{(1+e^x)^2} = \frac{e^x}{(1+e^x)^2} = f(x)(1 - f(x))$$

그러므로 

$$\frac{\partial o_1}{\partial z_3} = o_1 \times (1 - o_1)$$

마지막으로 세 번째 term은 $h_1$이 된다. 

$$\frac{\partial z_3}{\partial w_5} = h_1$$

이제 이렇게 우리는 right side의 term을 구했고 아래와 같이 경사하강을 적용하여 w를 update할 수 있다. 

$$w_5^+ = w_5 - \alpha \frac{\partial E_{total}}{\partial w_5}$$

이러한 방법으로 나머지 $w_6^+, w_7^+, w_8^+$도 구한다. 

### step2
이번에 update할 weight는 $w_1, w_2, w_3, w_4$이다. 
$w_1$을 update하기 위해서는 $\frac{\partial E_{total}}{\partial w_1}$을 계산해야 한다. 

$$\frac{\partial E_{total}}{\partial w_1} = \frac{\partial E_{total}}{\partial h_1} \times \frac{\partial h_1}{\partial z_1} \times \frac{\partial z_1}{\partial w_1}$$

여기서 right side의 first term은 아래로 다시 쓸 수 있다. 

$$\frac{\partial E_{total}}{\partial h_1} = \frac{\partial E_{o_1}}{\partial h_1} + \frac{\partial E_{o_2}}{\partial h_1}$$

그리고 right side의 2개의 term은 아래와 같다. 

$$ 
\begin{aligned}
\frac{\partial E_{o_1}}{\partial h_1} &= \frac{\partial E_{o_1}}{\partial z_3} \times \frac{\partial z_3}{\partial h_1} = \frac{\partial E_{o_1}}{\partial o_1} \times \frac{\partial o_1}{\partial z_3} \times \frac{\partial z_3}{\partial h_1} \\ 
\\  
&= -(target_{o_1} - output_{o_1}) \times o_1 \times (1 - o_1) \times w_5 \\   
\\   
\frac{\partial E_{o_2}}{\partial h_1} &= \frac{\partial E_{o_2}}{\partial z_4} \times \frac{\partial z_4}{\partial h_1} = \frac{\partial E_{o_2}}{\partial o_2} \times \frac{\partial o_2}{\partial z_4} \times \frac{\partial z_4}{\partial h_1} \\     
\\  
&= -(target_{o_2} - output_{o_2}) \times o_2 \times (1 - o_2) \times w_7 \\   
\end{aligned}
$$

이렇게 우리는 $\frac{\partial E_{total}}{\partial w_1}$을 계산하기 위한 first term을 구했고 나머지 두 개의 term에 대해서는 아래와 같다. 

$$\frac{\partial h_1}{\partial z_1} = h_1 \times (1 - h_1)$$

$$\frac{\partial z_1}{\partial w_1} = x_1$$

이제 같은 방식으로 경사하강을 이용하여 w를 update 해주면 된다. 

$$w_1^+ = w_1 - \alpha \frac{\partial E_{total}}{\partial w_1}$$

이러한 방법으로 나머지 $w_2^+, w_3^+, w_4^+$도 구한다. 

## Vectorized operations
이제 인풋이 벡터인 경우를 생각해보자. 과정은 다 같으나 gradient이 아닌 Jacobian을 계산해주면 된다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/13f81118-aa6e-4399-bb0d-14335fc52808" height="50%" width="50%"></p>

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/c94cb1b8-4253-4e55-9c5c-28cf764e0dae" height="20%" width="20%"></p>

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/85e41c27-7629-418f-914b-47c4a65feb0f" height="70%" width="70%"></p>

$$ 
\begin{aligned}
i) \ &f(q) = q_1^2 + q_2^2 + ... + q_n^2 = ||q||^2 \\  
&\triangledown_g f = 2q
\\      
ii) \ &q = Wx \\
&\triangledown_w q = x^T \\ 
&\therefore \triangledown_w f = 2q \cdot x^T \\
&\triangledown_x q = W^T \\ 
&\therefore \triangledown_x f = 2W^T \cdot q \\
\end{aligned}
$$

$$
\begin{aligned}
&\therefore \triangledown_w f = 2 \cdot 
\begin{bmatrix} 
0.22\\
0.26\\ 
\end{bmatrix}
\cdot
\begin{bmatrix} 
0.2&0.4\\ 
\end{bmatrix} =
\begin{bmatrix} 
0.088&0.176\\
0.104&0.208\\
\end{bmatrix}
\\    
&\therefore \triangledown_x f = 2 \cdot 
\begin{bmatrix} 
0.1&-0.3\\
0.5&0.8\\
\end{bmatrix}
\cdot
\begin{bmatrix} 
0.22\\
0.26\\
\end{bmatrix} = 
\begin{bmatrix} 
-0.112\\
0.636\\
\end{bmatrix}
\end{aligned}
$$

벡터의 gradient는 항상 원본 벡터의 사이즈와 같다.

딥러닝 프레임워크 라이브러리 : https://github.com/BVLC/caffe

# Neural Network
지금까지는 layer가 하나짜리인 linear function만을 이용했다면, 이제는 Single 변환이 아닌 그 이상의 레이어들을 쌓아보자. 
쌓는 방법은 아래와 같다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/b874e790-c7db-43ec-bc1a-b78c48532294" height="50%" width="50%"></p>

max와 같은 비선형 레이어를 추가할 수 있다.

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/753a81f8-e12e-410e-9cb5-0c3a42eca7c1" height="40%" width="40%"></p>

중간에 있는 레이어들을 hidden layer라 부른다. 

아래는 나중에 더 다룰 여러 Activation functions이다. 

<p align="center"><img src="https://github.com/em-1001/AI/assets/80628552/8680366a-6240-4fcd-8b66-3f3b6c67a477" height="60%" width="60%"></p>

# Reference 
https://oculus.tistory.com/9    
https://www.youtube.com/watch?v=d14TUNcbn1k&t=2430s
