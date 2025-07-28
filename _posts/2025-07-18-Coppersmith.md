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

$N$이 합성수(composite integer), $f(x)= x^d + \sum_{i=0}^{d-1} a_i x^i \in \mathbb{Z} [x]$는 $d$ 차수의 일변수 다항식(monic polynomial)이라 하자. Coppersmith’s Method는  $f(x_0) = 0 (\mod N)$의 small integer roots인 $x_0$를 찾는 문제를 해결할 수 있다. 이때 $\vert x_0 \vert < B$를 만족해야 하는데 $B$는 $N$과 $d$에 의해 결정되는 Bound로 $N$에 대해 작은 값이다. Coppersmith’s Method는 $\vert x_0 \vert < N^{1/d}$인 $x_0$가 존재한다면, $N$의 factor를 모르더라도 root를 효율적으로 찾을 수 있음을 보장한다. 이 점이 중요한 이유는 보통 합성수 $N=pq$에 대해서 $\mathbb{Z}_N$는 유한체가 아니기 때문에 $N$의 factor를 알지 못하면 $\mod N$에 대한 임의의 다항식의 해를 찾는 것은 어렵기 때문이다. 이러한 특징은 결국 소인수 분해 문제로 귀결되기 때문에 RSA의 보안성과도 연결되어 있다. 

## From Modular Root to Integer Root via Small Coefficients

Coppersmith’s method의 핵심 아이디어는 modular root-finding problem을 integer root-finding problem으로 바꾸는데 있다. 이 과정에서 다음을 만족하는 small coefficients를 갖는 다항식 $h(x)$를 만드는 것을 목표로 한다. 

1. $h(x_0) = 0 (\mod N)$ ($x_0$ is a root of $h$ modulo $N$, just like it is for $f$)
2. $h(x_0) = 0$ as an integer (i.e. the value of h(x_0)$ is literally zero, not just divisible by $N$)

만약 $x_0$를 actual root로 갖는 $h(x)$를 찾을 수 있다면, $h(x)$는 small coefficients에 대부분 small degree이므로 $h(x)=0$의 해 $x_0$는 이미 알려진 다항식의 해 알고리즘(factoring the polynomial, using rational root tests) 등을 통해 쉽게 찾을 수 있다. 중요한 건 $h(x)$의 해 $x_0$가 곧 원래 찾고자 했던 $f(x) = 0 (\mod N)$의 해와 동일하다는 것이다. 

$h(x_0)=0$의 해 $x_0$가 modular root에서 integer root가 되도록 하기 위해선 $h(x_0)$가 $N$보다 작도록 하면 된다. $x_0$는 $\mod N$ 상에서의 root이므로 $h(x_0)$의 값은 $N$의 배수이다. 여기서 $\vert h(x_0) \vert < N$라면 $N$의 integer multiple이 될 수 있는 값은 0밖에 없고, 결국 $h(x_0) = 0$인 integer root문제가 된다. 

Coppersmith’s method는 $h(x)$를 만들기 위해 lattice-based techniques을 이용한다. $x_0 \mod N$에서 모든 다항식의 근이 되도록 격자를 구성하고, 이 다항식들의 short (small-coefficient) combination을 찾음으로 써 $\vert h(x_0) \vert < N$를 만족시킬 수 있다. 만약 $x_0$가 $N$에 비해 충분히 작다면, 이러한 combination은 반드시 존재하며 lattice reduction을 통해 효율적으로 찾아낼 수 있다. 

## Constructing Polynomials that Share the Root Mod N











