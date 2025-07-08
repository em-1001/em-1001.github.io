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

# Lattices configuration by type

LLL을 사용해 해결할 수 있는 문제들과 각 상황에 따른 격자를 어떻게 구성해야 하는지 예시를 몇가지 살펴보자. 

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

이번엔 반대로 아래 합동식에서 다항식의 $x$와 $N$이 주어지고, 계수 $c_i$가 작다고 가정할 때 이 값들을 찾는 문제를 살펴보자.

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
