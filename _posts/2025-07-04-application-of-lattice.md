---
title:  "Application of Lattice Reduction"
excerpt: "Application of Lattice Reduction"

categories:
  - Cryptography
tags:
  - Cryptography
  - Lattice
last_modified_at: 2025-04-18T08:06:00-05:00
---

# An Application of Lattice Reduction

Lattice Reduction을 암호학에 적용하기 전에 이를 이용한 문제 해결 예시 몇 가지를 살펴보자. 지금 살펴볼 문제는 유리수 계수의 다항식의 근(해)인 수(algebraic number)가 주어졌을 때, 그 수를 만족하는 유일한 차수가 가장 낮은 정수계수 다항식인 최소 다항식(minimal polynomial) 을 복원하는 문제이다. 이 문제는 격자 축소로 해결이 가능하며 문제를 살펴보면 다음과 같다. 

**Definition (Minimal Polynomial)**. Let $\alpha \in F$ where $F$ is a field. The minimal polynomial of $\alpha$ is the monic polynomial of lowest degree in $F[x]$ such that $\alpha$ is a root. 

우리는 $\alpha = 7 + \sqrt[3]{5}$ 일 때 minimal polynomial $f(x)$를 구하기 위해 lattice reduction을 이용할 것이다. 계산할 때 $\alpha$는 $\beta = 7 + \sqrt[3]{5} \approx 8.70997594$로 근사하여 구할 것이다. 우선 $f$의 degree가 3차라고 가정해보자. $f$의 형태는 $f(x) = a_0 + a_1x + a_2x^2 + a_3x^3 \ a_i \in \mathbb{R}$ 가 된다. $\alpha$가 root이므로 $f(\alpha) = 0$ 즉, $f(\beta) \approx 0$ 이다. lattice reduction을 통해 우리는 $a_0, a_1, a_2, \cdot$을 구해야 한다. 정수상에서 계산해야 하므로 $\beta$에 $10^8$을 곱해여 다음과 같이 계산한다. 

$$10^8a_0 +  + \lfloor 10^8\beta \rfloor a_1 + \lfloor 10^8\beta^2 \rfloor a_2 + \lfloor 10^8\beta^3 \rfloor a_3 \approx 0$$ 

이제 이를 바탕으로 격자를 구성하면 다음과 같다. 

$$
\mathbf{B} =
\begin{bmatrix}
10^8 & 1 & 0 & 0 \\
\lfloor 10^8 \beta \rfloor & 0 & 1 & 0 \\
\lfloor 10^8 \beta^2 \rfloor & 0 & 0 & 1 \\
\lfloor 10^8 \beta^3 \rfloor & 0 & 0 & 0
\end{bmatrix}
= \begin{bmatrix}
100000000 & 1 & 0 & 0 \\
870997594 & 0 & 1 & 0 \\
758636087 & 0 & 0 & 1 \\
6607703514 & 0 & 0 & 0
\end{bmatrix}
$$

이 격자에 대해 $t=(a_0, a_1, a_2, 1)$ 라 할 때 $tB$를 계산하면 다음과 같다. 

$$tB = (10^8a_0 +  + \lfloor 10^8\beta \rfloor a_1 + \lfloor 10^8\beta^2 \rfloor a_2 + \lfloor 10^8\beta^3 \rfloor, a_0, a_1, a_2) = (c, a_0, a_1, a_2)$$ 

$c$는 0에 매우 가깝다. 즉, 이 벡터가 우리가 Lattice Reduction을 통해 찾고자 하는 작은 벡터임을 알 수 있고 이를 통해 $a_0, a_1, a_2$를 구할 수 있다. 

이 문제에서 LLL이 성공적으로 minimal polynomial의 coefficients를 찾을 수 있음을 정당화해보자. 먼저 coefficients의 upper bound $M=400$이라고 하자. 이는 우리의 target vector에 대한 rough upper bound $\lVert (0, M, M, M) \rVert \approx 629$를 준다. 이전에 살펴보았던 LLL-reduced basis의 first vector bounds에 대한 정리를 이용하면 다음을 얻는다. 

$$\lVert b_1 \rVert \leq \left( \frac{2}{\sqrt{4 \delta -1}} \right)^{n-1} \sqrt{n} \cdot \vert \det(\mathcal{L}) \vert^{1/n} \approx 2868$$

즉, LLL 알고리즘이 $\lVert b_1 \rVert \leq 2868$을 보장하는데, 이는 우리가 설정한 target vector의 upper bound $629$보다 크므로 LLL이 target vector를 찾아줄 가능성이 있다. 

위 식은 이론적인 upper bound일 뿐이고, 실제 LLL을 적용하면 이보다 훨씬 짧은 벡터가 나올 때도 많다. [LLL on the Average](https://link.springer.com/chapter/10.1007/11792086_18) 논문에서는 LLL-reduced basis의 first vector에 대한 heuristic 값을 제안하는데 random bases에 대해 $\frac{\lVert b_1 \rVert}{\vert det(\mathcal{L})\vert^{1/n}} \approx 1.02^n$를 만족한다고 한다. 

다시 돌아와서 앞서 세운 lattice에 대해 LLL algorithm을 돌리면 다음의 basis $B^{\prime}$을 얻는다. 

$$
\mathbf{B}^{\prime} = 
\begin{bmatrix}
5 & -348 & 147 & -21 \\
438 & -75 & 116 & 188 \\
109 & -214 & -563 & -159 \\
357 & 136 & 220 & -419
\end{bmatrix}
$$

첫 행으로 shortest vector를 주고 있고, first entry가 5로 0에 가까운 것으로 보아 우리가 찾는 해임을 알 수 있다. 즉, 이를 통해 coefficients $a_0=-384, a_1=147, a_2=-21$을 얻을 수 있고 최종 $f$는 다음과 같다. 

$$f(x) = x^3 - 21x^2 + 147x - 348$$

$7 + \sqrt[3]{5} \approx 8.70997594$로 직접 계산해보면 $f(8.70997594) \approx -0.000000058568341059939416$로 $0$에 매우 가까운 것을 확인할 수 있다. 

아래 Lattice Reduction 적용 예시 몇 가지를 더 살펴보자. 

## Subset Sum Problem

집합 $A ∈ ℤ$와 자연수 $M$이 주어져 있다고 하자. 이 때, $∑S = M$을 만족하는 $A$의 부분집합 $S$를 찾는 문제가 있다고 하자. 
$A = \lbrace a_1, ⋯, a_n \rbrace$이라 하면 이 문제를 다음과 같은 방정식으로 변형할 수 있다. 

$$a_1x_1 + ⋯ + a_nx_n - M = 0, x_i = 1 \vert 0$$

이제 $S = \lbrace a_i \vert x_i = 1 \rbrace$가 Subset Sum Problem의 해가 되고, $x_1, \cdots, x_n$은 선형이고 작은 해 이므로 LLL을 통해 해결할 수 있다. 

$$\mathcal{L} =
\begin{bmatrix}
1 & 0 & \cdots & 0 & -a_1 \\
0 & 1 & \cdots & 0 & -a_2 \\
\vdots  & \vdots  & \ddots & \vdots  \\
0 & 0 & \cdots & 1 & -a_n \\  
0 & 0 & \cdots & 0 & M
\end{bmatrix}$$

각 row들을 $b_i$라고 하면, 

$$
\begin{aligned}
x_1b_1 + ⋯ + x_nb_n &= (x_1, ⋯, x_n, M - \sum x_ia_i) \\  
&= (x_1, ⋯, x_n, 0)
\end{aligned}$$

위 벡터는 $\mathcal{L}$위의 작은 벡터이고, 따라서 SVP의 답이 될 가능성이 크다. 

```py
arr = [1562, 1283, 1381, 1272, 1540]
target = 4483

L = []
N = len(arr)

for i, a in enumerate(arr):
    row = [0] * (N+1)
    row[i] = 1
    row[-1] = -a
    L.append(row)
L.append([0] * N + [target])

L = Matrix(L)

L = L.LLL()
print(L)
# [ 1  0  1  0  1  0]
# [-1  2  0 -2  1  0]
# [-3 -1  2 -1  0 -4]
# [ 2 -2 -2 -3  1 -3]
# [ 1  1 -4  2  3 -2]
# [-4 -2  1  1  4  1]
```
`L[0]` 이 우리가 정확히 원하는 형식으로 나왔음을 알 수 있다. (맨 끝의 값은 0이고, 나머지 값은 0 또는 1이다.)

## Polynomial Coefficient Estimation

이번엔 아래 합동식에서 다항식의 $x$와 $N$이 주어지고, 계수 $c_i$가 작다고 가정할 때 이 값들을 찾는 문제를 살펴보자.

$$\sum_{i=1}^n c_ix^i \equiv 0 \pmod{N}$$

역시 구하려는 $c_i$가 작은 값이므로 short vector 문제로 변환해서 LLL로 해결이 가능하다. 

$$\sum_{i=1}^n c_ix^i - N = 0$$

위 관계를 이용해 아래 Lattice를 만들 수 있다. 

$$\mathcal{L} =
 \begin{bmatrix}
  1 & 0 & \cdots & 0 & x^n \\
  0 & 1 & \cdots & 0 & x^{n-1} \\
  \vdots  & \vdots  & \ddots & \vdots & \vdots  \\
  0 & 0 & \cdots & 1 & x \\  
  0 & 0 & \cdots & 0 & -N   
 \end{bmatrix}$$

각 row를 basis로 생각하면 $(c_n,c_{n-1} ⋯,c_1, 0)$ 가 결과로 나오게 된다. 

```py
N = 10007
mark = 7

L = matrix(
    [
        [1, 0, mark^2],
        [0, 1, mark^1],
        [0, 0,     -N]
    ]
)

L = L.LLL()

# print(L)
# [  0   1   7]
# [  1  -7   0]
# [200  29  -4]

L = L[2] 

R.<x> = PolynomialRing(ZZ)
f = 0
for j in range(2):
    f += L[j] * x^(2 -j)
f -= L[2]

# print(f)
# 200*x^2 + 29*x + 4
```
`L[2]`이 우리가 찾는 해 임을 알 수 있다. 

## Bytes Recovery

CTF에서 바이트로 이루어진 플래그의 길이를 안다면 LLL로 복구할 수도 있다. 플래그도 결국 바이트 배열 $x=[x_0, x_1, \cdots, x_n]$ 이고, 바이트 값이 작기(ASCII) 때문에 아래와 같이 하나의 정수로 표현하고, 이를 LLL 격자로 만들면 된다. 

$$X=x_0\cdot 256^{n} + x_1 \cdot 256^{n-1} + \cdots + x_n \cdot 256^0$$

결국 flag 정수값을 $X$라 했을 때, 아래의 합동식을 해결하는 문제이므로 앞선 문제들과 비슷해진다. 

$$X \equiv 0 \pmod{N}$$

이때 현재 바이트에서 다음 바이트로의 관계는 다음과 같다. 

$$x_{i+1}-256 \cdot x_i =0 \to x_{i+1}=256 \cdot x_i$$

따라서 위 관계를 이용해 격자를 구성하면 다음과 같다. 

$$\mathcal{L} =
\begin{bmatrix}
N & 0 & 0 & \cdots & 0 & 0 \\
-256 & 1 & 0 & \cdots & 0 & 0 \\
0 & -256 & 1 & \cdots & 0 & 0 \\
\vdots  & \vdots & \vdots  & \ddots & \vdots & \vdots  \\
0 & 0 & 0 & \cdots & 1 & 0 \\  
0 & 0 & 0 & \cdots & -256 & 1   
\end{bmatrix}$$

이외에도 앞서 살펴본 다항식의 계수 문제처럼 $x$를 256이라 생각하여 격자를 아래와 같이 구성할 수도 있다.  

$$\mathcal{L} =
\begin{bmatrix}
1 & 0 & \cdots & 0 & 256^n \\
0 & 1 & \cdots & 0 & 256^{n-1} \\
\vdots  & \vdots  & \ddots & \vdots & \vdots  \\
0 & 0 & \cdots & 1 & 256 \\  
0 & 0 & \cdots & 0 & -N   
\end{bmatrix}$$

다음은 이와 관련된 문제인 SeeTF 2023 Onelinecrypto Writeup으로 LLL을 통한 Bytes Recovery의 예를 보여준다. 

```py
# "https://web.archive.org/web/20240412075022/https://demo.hedgedoc.org/s/DnzmwnCd7"
# "https://nush.app/blog/2023/06/21/see-tf-2023/"

from sage.all import *
from cpmpy import *
import re

n = 13**37
empty = b'SEE{' + bytes(23) + b'}'
target = int.from_bytes(empty, 'big') * pow(256, -1, n)

m = matrix(24, 24)
m[0,0] = n
for i in range(22):
    m[i+1,i:i+2] = [[-256, 1]]
m[-1,0] = -target
m[-1,-1] = 2**256 # some arbitrarily large number

def disp():
    flag = bytes(x.value())[-8::-8].decode()
    print(flag, '<--- WIN' if re.fullmatch(r'\w+', flag) else '')

x = cpm_array(list(intvar(-99999, 99999, 23)) + [1]) @ m.LLL()[:,:-1]
Model([x >= 48, x <= 122]).solveAll(display=disp)
```
