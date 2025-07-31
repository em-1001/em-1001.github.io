---
title:  "Coppersmith’s Method"
excerpt: "Finding Small Roots"

categories:
  - Cryptography
tags:
  - Cryptography
  - Lattice
last_modified_at: 2025-04-18T08:06:00-05:00
---


# Coppersmith’s Method

$N$이 합성수(composite integer), $f(x)= x^d + \sum_{i=0}^{d-1} a_i x^i \in \mathbb{Z} [x]$는 $d$ 차수의 일변수 다항식(monic polynomial)이라 하자. Coppersmith’s Method는  $f(x_0) = 0 \pmod{N}$의 small integer roots인 $x_0$를 찾는 문제를 해결할 수 있다. 이때 $\vert x_0 \vert < B$를 만족해야 하는데 $B$는 $N$과 $d$에 의해 결정되는 Bound로 $N$에 대해 작은 값이다. Coppersmith’s Method는 $\vert x_0 \vert < N^{1/d}$인 $x_0$가 존재한다면, $N$의 factor를 모르더라도 root를 효율적으로 찾을 수 있음을 보장한다. 이 점이 중요한 이유는 보통 합성수 $N=pq$에 대해서 $\mathbb{Z}_N$는 유한체가 아니기 때문에 $N$의 factor를 알지 못하면 $\bmod N$에 대한 임의의 다항식의 해를 찾는 것은 어렵기 때문이다. 이러한 특징은 결국 소인수 분해 문제로 귀결되기 때문에 RSA의 보안성과도 연결되어 있다. 

## From Modular Root to Integer Root via Small Coefficients

Coppersmith’s method의 핵심 아이디어는 modular root-finding problem을 integer root-finding problem으로 바꾸는데 있다. 이 과정에서 다음을 만족하는 small coefficients를 갖는 다항식 $h(x)$를 만드는 것을 목표로 한다. 

1. $h(x_0) = 0 \pmod{N}$ ($x_0$ is a root of $h$ modulo $N$, just like it is for $f$)
2. $h(x_0) = 0$ as an integer (i.e. the value of $h(x_0)$ is literally zero, not just divisible by $N$)

만약 $x_0$를 actual root로 갖는 $h(x)$를 찾을 수 있다면, $h(x)$는 small coefficients에 대부분 small degree이므로 $h(x)=0$의 해 $x_0$는 이미 알려진 다항식의 해 알고리즘(factoring the polynomial, using rational root tests) 등을 통해 쉽게 찾을 수 있다. 중요한건 $h(x)$의 해 $x_0$가 곧 원래 찾고자 했던 $f(x) = 0 \pmod{N}$의 해와 동일하다는 것이다. 

$h(x_0)=0$의 해 $x_0$가 modular root에서 integer root가 되도록 하기 위해선 $h(x_0)$가 $N$보다 작도록 하면 된다. $x_0$는 $\bmod N$ 상에서의 root이므로 $h(x_0)$의 값은 $N$의 배수이다. 여기서 $\vert h(x_0) \vert < N$라면 $N$의 integer multiple이 될 수 있는 값은 0밖에 없고, 결국 $h(x_0) = 0$인 integer root문제가 된다. 

Coppersmith’s method는 $h(x)$를 만들기 위해 lattice-based techniques을 이용한다. $x_0 \bmod N$에서 모든 다항식의 근이 되도록 격자를 구성하고, 이 다항식들의 short (small-coefficient) combination을 찾음으로써 $\vert h(x_0) \vert < N$를 만족시킬 수 있다. 즉, $x_0 \bmod N$를 근으로 갖는 다항식들을 많이 만들어 격자를 구성하는 것인데, 이렇게 하는 이유는 $f(x)$ 하나만으로는 격자상에서 short vector를 찾기 어렵기 때문이다. 만약 $x_0$가 $N$에 비해 충분히 작다면, $h(x)$를 만드는 combination은 반드시 존재하며 lattice reduction을 통해 효율적으로 찾아낼 수 있다. 

## Constructing Polynomials that Share the Root Mod N

앞서 언급했듯이 $h(x)$를 만들기 위해선 $x_0 \bmod N$를 근으로 갖는 다항식들로 격자를 구성해야 한다. $x_0 \bmod N$를 근으로 갖는 다항식들을 생각해보면 다음의 방법들이 있다. 

- Multiplying by Powers of $x$ : $f(x_0) \equiv 0 \pmod{N}$ 이므로 $i \geq 0$에 대해 $x_0^i \cdot f(x_0) \equiv 0 \pmod{N}$ 역시 성립한다. 즉, $x_if(x)$ 도 $x_0 \bmod N$를 해로 갖는 다항식이 될 수 있다.
- Multiplying by Powers of $N$ : $N$의 배수는 modulo $N$ 에서 0 이다. 따라서 임의의 다항식 $g(x)$에 대해 $N \cdot g(x)$ 역시 $N$을 factor로 갖기 때문에 $N \cdot g(x_0) \equiv 0 \pmod{N}$가 성립한다. $N \cdot x^i$와 같은 단순한 monomials도 $x_0 \bmod N$를 근으로 갖고 이러한 monomials은 격자의 기저 다항식(basis polynomials)으로 사용될 수 있다. 

위 아이디어를 통해 다음의 basis polynomials을 고려할 수 있다. 

$$g_i(x) = N \cdot x^i, \text{ for } i = 0, 1, 2, \cdots, t$$ 

$f(x)$와 마찬가지로 각각의 $g_i(x)$역시 $g_i(x_0) = Nx_0^i \equiv 0 \pmod{N}$를 만족한다. 이렇게 만들어진 basis polynomials의 linear combination역시 $x_0 \bmod N$를 해로 갖는다. 왜냐하면 $p(x)$와 $q(x)$가 모두 $x_0 \bmod N$를 해로 가지면, $ap(x) + bq(x)$ 역시 그렇기 때문이다. 

그렇다면 $x_0 \bmod N$을 해로 갖는 다항식들로 격자를 구성하는 것이 목적이라면 왜 단순히 $f(x)$의 multiples of $x^i$로 격자를 구성하지 않는 것일까? 이유는 격자를 통해 얻고자 하는 궁극적인 목적이 $x_0 \bmod N$을 해로 갖는 것과 동시에 $\vert h(x_0) \vert < N$를 만족할 정도로 작은 다항식 $h(x)$를 찾아야 하기 때문이다. 만약 $f(x)$가 large coefficients를 갖고 있다면 integer combination을 통해 large coefficients을 cancellation할 수 있어야 우리가 원하는 small coefficients를 갖는 다항식을 얻을 수 있다. 이러한 점에서 monomials $Nx^i$는 $f(x)$의 large term을 cancel out시키는데 도움을 준다. 

참고로 이 방법에서 좀 더 확장하여 Coppersmith’s method를 일반화하면, $f(x)$ 뿐만 아니라 $f(x)^2, f(x)^3, \cdots$도 함께 고려해 볼 수 있다. 지금까지의 아이디어가 $f(x)$와 $N \cdot x^i$의 단항식들이 모두 $x_0 \bmod N$를 근으로 공유한다는 점을 이용해 격자를 세웠다면, $f(x)$ 뿐만 아니라 $f(x)^j \cdot x^i$와 같은 다항식들을 이용해 격자차원을 늘리고 보다 넓은 해 공간을 커버하도록 할 수 있다. 

$f(x_0) \equiv 0 \pmod{N}$이면 $f(x_0)^j \equiv 0 \pmod{N^j}$ 역시 만족한다. 
따라서 다항식 $f(x_0)^j \cdot x^i$는 $x_0$을 $\pmod{N^j}$에서의 근으로 공유하기 때문에 격자 구성에 포함시킬 수 있다. 
결과적으로 $f(x)$의 degree가 $d$라 할 때, $\bmod N$에서 $f(x)$의 roots를 $\bmod N^m$ 상에서 모두 포함하는 아래의 다항식 $g_{u,v}(x)$를 고려할 수 있다. 

$$g_{u, v}(x) = N^{m-v} f(x)^vx^u, \text{ for } u \in \lbrace 0, \cdots, d-1 \rbrace, v \in \lbrace 0, \cdots, m \rbrace$$

$f(x_0) \equiv 0 \pmod{N}$이면 $N^{m-v} f(x_0)^v$가 $N^m$에 의해 나눠지기 때문에 모든 $u, v$에 대해서 $g_{u,v}(x_0) \equiv 0 \pmod{N^m}$이 성립한다. 이러한 확장은 격자의 차원을 늘려 LLL의 정확도와 안정성을 향상시킬 수 있다. Coppersmith’s method의 확장에 대해서는 여기까지만 설명하고 이 글에서는 one power of $f(x)$인 case에 대해 자세히 살펴볼 것이다.  

## Lattice Construction with the $x = Bx^{\prime}$ Scaling

우리가 찾고자 하는 small-coefficient를 갖는 $h(x)$를 얻기 위해, 위에서 구성한 다항식들의 coefficient vectors로 이루어진 lattice basis를 사용한다. 각 다항식은 그 coefficient들을 같은 크기로 padding하여 vector로 표현되고, 그 coefficient vectors의 linear combination은 다항식들의 정수 계수 선형 결합과 동일하다. 즉, 이러한 coefficient vector들이 span하는 lattice 내의 integer vector 하나가 다항식을 의미하게 된다. 따라서 이 lattice에서 short vector를 찾는다는 것은 곧 small-coefficient를 갖는 다항식을 찾는 것과 같은 의미이다. 

**Scaling by** $B$ : Coppersmith’s method에서는 상한값(bound)로 $\vert x_0 \vert$를 갖는 scaling factor $B$를 이용한다. 우리가 찾는 해 $x_0$가 $\vert x_0 \vert < B$를 만족한다고 가정하며 $B$는 $N^{1/d}$이거나 좀 더 작은 값으로 설정된다. 그리고 우리가 이용할 다항식 변수를 $x=Bx^{\prime}$ 치환한다. 이는 $x/B$인 새로운 변수 $x^{\prime}$로 다항식을 다룬다는 의미가 된다. 예를 들어 $f(x)$는 $f(Bx^{\prime})$이 되고, $Nx^i$는 $N(Bx^{\prime})^i = NB^ix^{{\prime}^i}$가 된다. 이렇게 치환하는 데에는 다음의 두 가지 이유가 있다. 

- root의 크기를 정규화할 수 있다. 우리가 찾고자 하는 근 $x_0$는 $\vert x_0 \vert < B$라는 제한된 범위를 가지므로 $x = Bx^{\prime}$이라 치환하면, $x_0 = Bx_0^{\prime} \to x_0^{\prime} = \frac{x_0}{B} < 1$이 되어 모든 해가 $[-1, 1]$구간에 있도록 한다. 이는 수치적으로 안정적이며 LLL reduction을 할 때 각 항의 기여도와 계수 크기 분석이 쉬워진다.
- coefficients에 powers of $B$를 곱하는 것은 integer lattice를 구성하는데 있어 각 엔트리들이 적절히 scaled되도록 한다. $x=Bx^{\prime}$ 치환을 하게되면 각 단항식 $x^i$는 $B^ix^{{\prime}^i}$가 된다. 따라서 $Nx^i$ polynomial은 $NB^ix^{{\prime}^i}$가 되고 즉, $x^{{\prime}^i}$의 coefficient는 $NB^i$가 된다. 이 치환을 통해 basis vector들이 powers of $B$에 대해 계층적 구조를 갖게 된다. higher-degree terms은 coefficient로 larger power of $B$를 갖게 되는데, 이게 갖는 의미는 LLL에서 고차항들을 억제(베제)하는 것이다. higher-degree terms에 larger power of $B$인 $B^i$를 곱하게 되면, LLL은 short-vector를 찾기 위해 즉, norm을 줄이기 위해 larger power of $B$로 크기가 커져버린 고차항 계수 $h_i$들을 작게 하려 할 것이고 결국 고차항의 계수가 거의 0이 되어 저차항 중심의 짧은 벡터가 나오게 된다. 



$$
\mathbf{B} = 
\begin{bmatrix}
N & & & & & \\
 & BN & & & & \\
 & & B^2N & & & \\
 & & & \ddots & & \\
 & & & & B^{d-1}N & \\
a_0 & a_1 B & a_2 B^2 & \cdots & a_{d-1}B^{d-1} & B^d \\ 
\end{bmatrix}
$$





