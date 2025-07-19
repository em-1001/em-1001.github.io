---
title:  " Coppersmith’s Method"
excerpt: " Finding Small Roots"

categories:
  - Cryptography
tags:
  - Cryptography
  - Lattice
last_modified_at: 2025-04-18T08:06:00-05:00
---


# Coppersmith’s Method for Small Roots of Modular Univariate Polynomials

## Problem Setting: Finding Small Roots Modulo N

Coppersmith’s method addresses the problem of finding *small integer roots* of a univariate polynomial congruence. Formally, given a monic polynomial $f(x) \in \mathbb{Z}[x]$ of degree $d$ and a composite integer $N$ (whose factorization is unknown), the task is to find all solutions $x_0$ such that:

$f(x_0) \equiv 0 \pmod{N}, \qquad \text{with } |x_0| < B,$

for some bound $B$ that is small relative to $N$. In particular, Coppersmith’s theorem guarantees that if a root $x_0$ exists with $|x_0| < N^{1/d}$ (roughly the $d$th root of $N$), then that root can be found efficiently (in time polynomial in the size of $N$ and $d$). This is remarkable because solving arbitrary polynomial congruences mod $N$ is generally hard without knowing the factors of $N$. By restricting to *small* roots, Coppersmith’s method turns the problem into one that can be solved without factoring $N$.

## Importance in Cryptography

Finding small roots of modular equations has critical applications in cryptanalysis, especially of RSA. In RSA, one often encounters polynomials such as $F(x) = x^e - c$ (coming from the equation $x^e \equiv c \pmod{N}$) whose roots correspond to secret values (e.g. the plaintext message $m$ when given a ciphertext). Normally, recovering $m$ from $c = m^e \bmod N$ is believed to be hard without the private key. However, if $m$ is small enough or has a certain known structure, it can be recovered by finding a small root of $x^e - c \equiv 0 \pmod{N}$. Coppersmith’s method provides a rigorous way to do this for low-exponent RSA:

* **Low Public Exponent (e.g. e = 3) Attacks:** If an RSA ciphertext was produced with $e=3$ and the message $m$ is small (say $m < N^{1/3}$), then $m^3 < N$ and in fact $m^3 \equiv c \pmod{N}$ implies $m^3 = c$ as integers (since $c < N$). In this trivial case, $m$ can be found by taking the cube root of $c$. More generally, even if $m^3$ is slightly larger, Coppersmith’s method can recover $m$ as long as $m < N^{1/3}$ (up to small polynomial factors). This was extended by Coppersmith to scenarios where parts of the message are known. For example, Coppersmith showed that given an RSA modulus $N=pq$ and $e=3$, if the attacker knows all but (about) the least significant 1/3 of the bits of the message, the remaining unknown bits can be found in polynomial time. The problem is reduced to finding a small root of a polynomial derived from the partially known message, and Coppersmith’s technique efficiently finds that root. This result has profound implications: it means that if an RSA encryption with small exponent leaks a large portion of the message (or if the message has a predictable structure), the encryption can be broken without factoring $N$.

* **Partial Key Exposure:** Another application is attacking RSA when parts of the secret key (the primes or private exponent) are known. For instance, knowing half of the bits of one prime $p$ of RSA allows setting up a polynomial whose root is the unknown half of $p$. Coppersmith’s method can then recover the rest of $p$, leading to factorization of $N$. This is known as a *partial key exposure attack*, and Coppersmith’s lattice method is at its core.

These examples underscore why finding small roots modulo $N$ is important: it enables cryptanalytic attacks on RSA and related cryptosystems in cases that would otherwise be infeasible. Coppersmith’s method provides a *general strategy* to exploit any scenario that can be cast as a polynomial equation with a small solution modulo an unknown-factorization integer.

## Core Idea: From Modular Root to Integer Root via Small Coefficients

At the heart of Coppersmith’s method is a clever reduction of the modular root-finding problem to an *integer* root-finding problem. The goal is to construct a new polynomial $h(x)$ with **small coefficients** such that:

1. $h(x_0) \equiv 0 \pmod{N}$ (so $x_0$ is a root of $h$ modulo $N$, just like it is for $f$), **and**
2. $h(x_0) = 0$ as an integer (i.e. the value of $h(x_0)$ is literally zero, not just divisible by $N$).

If we can find such an $h(x)$, then $x_0$ is an *actual root* of $h(x)$ over the integers. Since $h(x)$ will have small coefficients and (hopefully) small degree, we can find all its integer roots using standard algorithms (e.g. factoring the polynomial or using rational root tests, etc.). In particular, once we have $h(x)$ with integer root $x_0$, we have *found* our small solution to the original congruence.

How can we ensure $h(x_0) = 0$ as an integer? A sufficient condition is that the *magnitude* of $h(x_0)$ is smaller than $N$. If we know $h(x_0)$ is a multiple of $N$ (because $x_0$ is a root mod $N$), and also $|h(x_0)| < N$, the only integer multiple of $N$ in that range is 0. Thus, $h(x_0) = 0$. In summary, we seek an $h(x)$ such that:

* $h(x)$ has the same root $x_0$ modulo $N$ as our original polynomial $f(x)$.
* The coefficients of $h(x)$ (and hence the value $h(x_0)$) are sufficiently small that $|h(x_0)| < N$.

Coppersmith’s method uses lattice-based techniques to **construct** this polynomial $h(x)$. By designing a lattice of polynomials that all vanish at $x_0$ mod $N$, and then finding a short (small-coefficient) combination of these polynomials, the method guarantees the conditions above. If $x_0$ is small enough (relative to $N$), such a combination exists and can be efficiently found using lattice reduction.

## Constructing Polynomials that Share the Root Mod N

The first step is to generate a set of polynomials that are known to have $x_0$ as a root modulo $N$. Obviously, the original polynomial $f(x)$ satisfies $f(x_0) \equiv 0 \pmod{N}$. We can create additional polynomials by multiplying by powers of $x$ or by factors of $N$ so that the root modulo $N$ is preserved:

* **Multiplying by Powers of $x$:** If $f(x_0) \equiv 0 \pmod{N}$, then for any exponent $i \ge 0$, we also have $x_0^i \cdot f(x_0) \equiv 0 \pmod{N}$. In other words, $x^i f(x)$ vanishes at $x_0$ mod $N$. However, $x^i f(x)$ alone will generally have large coefficients (because of the product by $x^i$), so we will use these with other combinations.

* **Multiplying by Powers of $N$:** Trivially, any multiple of $N$ is 0 modulo $N$. So for any polynomial $g(x)$, $N \cdot g(x)$ has the property that $N \cdot g(x_0) \equiv 0 \pmod{N}$ (because of the factor $N$). In particular, simple monomials like $N \cdot x^i$ vanish mod $N$ at *any* $x_0$. These will serve as basis polynomials that have the desired root mod $N$ built in.

Using these ideas, a natural basis set of polynomials to consider is:

$g_i(x) = N \cdot x^i, \qquad \text{for } i = 0,1,2,\dots, t,$

along with the original $f(x)$. Each $g_i(x)$ clearly satisfies $g_i(x_0) = N x_0^i \equiv 0 \pmod{N}$ (since it has an explicit $N$ factor). And $f(x)$ satisfies $f(x_0) \equiv 0 \pmod{N}$ by assumption. By construction, **any** integer linear combination of these polynomials will also have $x_0$ as a root modulo $N$, because if $p(x)$ and $q(x)$ both vanish mod $N$ at $x_0$, then so does $a\,p(x) + b\,q(x)$ for any integers $a,b$. This closure under addition is crucial – it means we can form new polynomials that still have the desired root mod $N$.

*Why include multiples of $x^i$ and not just constant multiples of $f(x)$?* Because our ultimate goal is to find a combination that not only vanishes mod $N$ at $x_0$ but is also *small*. Simply using $f(x)$ itself may not be enough – if $f(x)$ has large coefficients, any integer combination that yields small coefficients might require cancellation between different polynomials. The monomials $N x^i$ help provide flexibility: they have the root mod $N$ trivially, and by adjusting their coefficients we can cancel out large terms in $f(x)$.

\*Extension to higher powers of $$f$ and $N$:* In the full version of Coppersmith’s method, one often considers not only \(f(x)$$ but also higher powers like $f(x)^2, f(x)^3, \ldots$ combined with appropriate powers of $N$. For example, if $f(x_0) \equiv 0 \pmod{N}$, then $f(x_0)^2 \equiv 0 \pmod{N^2}$, etc. One can construct polynomials that vanish to higher modulus (like $N^m$) at $x_0$. This is used to push the method to its theoretical limits by considering a root of $f(x)$ modulo $N$ as also a root of these new polynomials modulo higher powers of $N$. However, for clarity, we will focus on the basic case with just one power of $f(x)$ and simple multiples by $N$, which already illustrates the method.

In summary, we prepare a collection of polynomials such that each one has $x_0$ as a root mod $N$. A simple choice (sufficient for many cases) is:
$\{f(x),\ N\cdot 1,\ N \cdot x,\ N \cdot x^2,\ \ldots,\ N \cdot x^t\},$
for some degree range $t$ we will determine. The next task is to find an **integer combination** of these polynomials that yields a new polynomial $h(x)$ with small coefficients.

## Lattice Construction with the $x = Bx'$ Scaling

To systematically find a small-coefficient combination, Coppersmith’s method uses a *lattice* basis composed of coefficient vectors of the above polynomials. The idea is to represent each polynomial by the vector of its coefficients (padded to the same size). Any integer combination of the polynomials corresponds to an integer linear combination of these coefficient vectors – i.e., an integer vector in the lattice they generate. Thus, finding a short (small-norm) vector in this lattice will yield a polynomial with small coefficients.

**Scaling by $B$:** We introduce a scaling factor $B$ which is an upper bound on $|x_0|$. We assume $x_0$ satisfies $|x_0| < B$ and choose $B$ accordingly (in practice, $B$ might be something like $N^{1/d}$ or a bit smaller). We perform a change of variables $x = Bx'$ in our polynomials. This means we will consider polynomials in the new variable $x'$, where $x'$ is effectively $x/B$. For example, $f(x)$ becomes $f(Bx')$, and $N x^i$ becomes $N (B x')^i = N B^i {x'}^i$.

This substitution serves two purposes:

* It normalizes the size of the root. If $|x_0| < B$, then $x'_0 = x_0/B < 1$. The actual root $x'_0$ in the new variable is a small *fraction* (less than 1 in absolute value). We don’t know $x'_0$ yet, but this hypothetical scaling helps to reason about the size of terms.
* More concretely, it incorporates powers of $B$ into the coefficients of the polynomial, which will be important for building an integer lattice with properly scaled entries. After substituting $x = Bx'$, each monomial $x^i$ turns into $B^i {x'}^i$. So a polynomial like $N x^i$ becomes $N B^i {x'}^i$; the coefficient in front of \${x'}^i\$ is now $N B^i$. Using this trick, we ensure that the *basis vectors have a hierarchical structure* with powers of $B$ scaling the coefficients. Intuitively, higher-degree terms get a larger power of $B$ in their coefficient, reflecting the fact that if $x_0 < B$, then $x_0^i$ (for large $i$) will make the term $h_i x_0^i$ small if $h_i B^i$ is bounded appropriately.

We now form a lattice basis $B$ (not to be confused with the bound $B$; here $B$ will denote the basis matrix) consisting of the coefficient vectors of the scaled polynomials. For illustration, suppose $f(x) = x^d + a_{d-1}x^{d-1} + \cdots + a_1 x + a_0$ is monic of degree $d$. We take the following $d+1$ polynomials (scaled by $B$) as rows of our basis matrix:

* For each $i = 0,1,\dots,d-1$: the polynomial $g_i(Bx') = N \cdot (Bx')^i = N B^i {x'}^i$. This polynomial has coefficient vector with a single nonzero entry $N B^i$ in the position corresponding to $x'^i$.
* The polynomial $f(Bx') = (Bx')^d + a_{d-1}(Bx')^{d-1} + \cdots + a_1 (Bx') + a_0$. This has coefficient vector $[a_0,\, a_1 B,\, a_2 B^2,\, \dots,\, a_{d-1} B^{d-1},\, B^d]$.

&#x20;*Lattice basis matrix for Coppersmith’s method (monic $d$th-degree $f(x)$). The first $d$ rows are the coefficient vectors of $g_i(Bx)=N B^i x'^i$ for $0 \le i < d$, and the last row is the coefficient vector of $f(Bx)$. All these polynomials share the same root $x_0$ modulo $N$. The basis matrix is nearly triangular, with large entries $N B^i$ or $B^d$ on the diagonal.*【10†】

In the matrix above, each row corresponds to one polynomial’s coefficients (arranged from constant term up to $x'^d$ term). Notice the structure:

* The first row has $[N, 0, 0, \dots, 0]$ corresponding to $N$.
* The second row corresponds to $N B x'$ with coefficients $[0, N B, 0, \dots, 0]$.
* Similarly, the $i$th row (for $i=0$ indexing) has $N B^i$ in the $x'^i$ column.
* The last row (degree $d$ of $f(Bx')$) has entries $[a_0, a_1 B, a_2 B^2, \dots, a_{d-1}B^{d-1}, B^d]$.

This matrix is lower-triangular and full rank. Importantly, the large powers $B^i$ and the factor $N$ appear on the diagonal. The lattice $\mathcal{L}$ generated by the rows consists of all integer combinations of these rows. Any vector in $\mathcal{L}$ corresponds to the coefficient vector of **some** polynomial $h(Bx')$ which is an integer combination of the basis polynomials. By construction, such an $h(x)$ (being a combination of the $g_i$ and $f$) will still satisfy $h(x_0) \equiv 0 \pmod{N}$.

The virtue of setting up this lattice is that we have transformed the problem of finding a good combination of polynomials into a problem of finding a *short lattice vector*. A “short” vector here means one with small Euclidean norm (and typically that implies its entries, which are coefficients of the combination, are relatively small).

## Using LLL to Find a Short Vector (Polynomial) in the Lattice

The lattice basis $B$ we constructed is deliberately designed so that it has a very large determinant (because of those big diagonal entries like $N B^i$ and $B^d$). As a result, by Minkowski’s theorem or intuition from geometry, we expect there to be unusually short vectors in this lattice compared to the lengths of the basis vectors. The Lenstra–Lenstra–Lovász (LLL) algorithm is a polynomial-time lattice basis reduction algorithm that can find a reasonably short vector in any lattice. LLL doesn’t necessarily find the shortest vector, but it finds one that is guaranteed to be not too long (within an exponential factor in the lattice dimension of the shortest). In practice, and in Coppersmith’s theoretical analysis, this is sufficient to find the desired combination.

We run LLL on the basis matrix $B$ to obtain a reduced basis. Let $\mathbf{b}_1 = (b_0, b_1, \dots, b_d)$ be the *first vector* of the LLL-reduced basis (which should be a short vector in the lattice). This $\mathbf{b}_1$ is the coefficient vector of some combination polynomial $h(Bx')$. Concretely, we interpret $\mathbf{b}_1$ as the coefficients $[h_0, h_1, \dots, h_d]$ of a polynomial:

$h(Bx') = h_0 + h_1 x' + h_2 {x'}^2 + \cdots + h_d {x'}^d.$

But recall $x' = x/B$, so equivalently:

$h(x) = h_0 + h_1 \frac{x}{B} + h_2 \frac{x^2}{B^2} + \cdots + h_d \frac{x^d}{B^d}.$

Or multiplying through by a common denominator $B^d$:

$h(x) = \frac{b_0}{B^d} + \frac{b_1}{B^{d-1}} x + \frac{b_2}{B^{d-2}} x^2 + \cdots + \frac{b_{d-1}}{B} x^{d-1} + b_d\, x^d.$

Since $f(x)$ was monic and included in the lattice, one combination will yield a monic (leading coefficient $b_d = 1$) polynomial $h(x)$ of degree $d$. In essence, we expect $h(Bx')$ to share the leading term with $f(Bx')$ (because the lattice basis was triangular with that $B^d$ term for $x'^d$). So we can take $h(x)$ to be monic of degree $d$ as well, without loss of generality (if not, one can divide by any constant leading coefficient).

The critical property now is that LLL has made $\mathbf{b}_1$ short, meaning the coefficients $b_i$ are small. In fact, theoretical analysis shows that if $B$ was chosen correctly, all these coefficients satisfy a bound like:

$|b_i| \text{ (which equals } |h_i B^i| \text{)} < \frac{N}{d+1},$

for each $i=0,1,\dots,d$【11†】. This condition is crafted to ensure that when we evaluate $h(x)$ at the root $x_0$, each term $h_i x_0^i$ is small. In particular, if $|h_i B^i| < N/(d+1)$ for all $i$, then since $|x_0|<B$ we have $|h_i x_0^i| \le |h_i| B^i < N/(d+1)$. Summing up the $d+1$ terms of $h(x_0)$ by the triangle inequality:

$|h(x_0)| = \Big|\sum_{i=0}^{d} h_i x_0^i \Big| \le \sum_{i=0}^{d} |h_i x_0^i| < (d+1) \cdot \frac{N}{d+1} = N.$

Thus $|h(x_0)| < N$. Meanwhile, because $h(x)$ is in the lattice, it is a combination of the basis polynomials, so it shares the root mod $N$: $h(x_0) \equiv 0 \pmod{N}$. The only way these two facts can simultaneously hold is if **$h(x_0) = 0$ exactly (as an integer)**. In other words, we have succeeded – $x_0$ is now an integer root of the polynomial $h(x)$.

Finally, we can use any standard method to find the roots of $h(x)$ over the integers (or rationals). Since $d$ is the degree of $h$, we can factor $h(x)$ or use root-finding algorithms to obtain all potential roots. Among those, we extract the ones that are less than $B$ in magnitude and test which ones indeed satisfy $f(x) \equiv 0 \pmod{N}$. The theorem guarantees that the actual root $x_0$ will be found in this process (as long as $x_0 < B$ and $B$ was chosen within the method’s allowable range).

## Selecting the Bound $B$ and Success Conditions

A key aspect of Coppersmith’s method is choosing the bound $B$ (and other parameters, like the number of basis polynomials) such that the method provably works. The larger the allowed $B$, the more powerful the method (since it can find bigger roots), but if $B$ is too large relative to $N$, the constructed lattice might not produce the desired small polynomial.

Through lattice determinant arguments and LLL’s guarantees, one can derive an upper bound on $B$ for which the method succeeds. In the simple setup described (using $d+1$ polynomials: $f$ and $N x^i$ for $i=0,\dots,d-1$), a sufficient condition is:

$B < \frac{N^{\frac{2}{d(d+1)}}}{\big(2(d+1)\big)^{3/d}}【11†】.$

This rather inscrutable formula arises from the LLL approximation factor (with δ≈0.75) and ensuring each coefficient bound $|h_i B^i| < N/(d+1)$ as discussed. It essentially says $B$ can be almost $N^{1/d}$, up to some small multiplicative factors that depend polynomially on $d$. In fact, as $d$ grows, those extra factors approach 1, and one can achieve roughly $B < N^{1/d}$ for large $N$. Coppersmith’s original result showed that one can find all roots $x_0$ with $|x_0| < N^{1/d - \epsilon}$ for any small $\epsilon > 0$ in polynomial time, and with more refined analysis (using more polynomials, i.e. considering powers of $f$ and a larger lattice) the bound can be pushed essentially to $B < N^{1/d}$ exactly (for asymptotically large $N$). In simpler terms, **Coppersmith’s method can find any root up to size about $N^{1/d}$.** This is a surprisingly large bound – for example, for a quadratic polynomial ($d=2$), roots up to size about $N^{1/2}$ (roughly the square root of $N$) can be found. For a cubic, up to $N^{1/3}$, etc. In terms of bit-length, if $N$ is an $k$-bit number, the method can find roots of size roughly $k/d$ bits. Scanning a space of size $N^{1/d}$ by brute force would be exponential in $k$, but Coppersmith’s algorithm does it in polynomial time – a remarkable feat achieved by lattice reduction.

It’s important to note that if a root is larger than this threshold (say $x_0 \approx N^{1/d}$ or bigger), Coppersmith’s method **does not guarantee** finding it. There is a sharp phase transition: just below $N^{1/d}$, all roots can be found in polynomial time; at or above $N^{1/d}$, the problem is believed to be as hard as general modular root-finding (which includes, for example, finding non-trivial square roots mod $N$ that would factor $N$). This is why Coppersmith’s method doesn’t outright break RSA in general – it only works when the solution happens to be unusually small compared to the modulus.

## Example Scenario and Visualization

To tie everything together, consider a concrete (simplified) example. Suppose $N$ is a large composite and we have a quadratic polynomial $f(x) = x^2 + bx + c$ (so $d=2$) that we suspect has a small root mod $N$. For instance, maybe $f(x) = x^2 - C$ corresponds to trying to recover a small secret $x$ from $x^2 \equiv C \pmod{N}$. If $|x_0| < N^{1/2}$, Coppersmith’s method should find $x_0$.

We would construct the basis polynomials: $g_0(x) = N$, $g_1(x) = N x$, and include $f(x)$. Using a bound $B$ slightly below $N^{1/2}$, we form the lattice basis matrix as described (it will be a 3×3 matrix in this case). After reducing the basis with LLL, we obtain a short vector that might correspond to a combination like:

$h(x) = c_2 f(x) + c_1 (N x) + c_0 (N),$

for some integers $c_2, c_1, c_0$. This $h(x)$ will also be a quadratic (degree ≤ 2). The LLL algorithm adjusts $c_2, c_1, c_0$ such that the coefficients of $h(x)$ are much smaller than those of either $f$ or the trivial combinations. For example, one possible outcome is that LLL finds $h(x) = x^2 - D$ for some very small $D$, which has $x_0$ as a root mod $N$. If $x_0$ is small, perhaps $h(x_0) = 0$ in fact (because $x_0^2 = D < N$). Factoring $h(x) = x^2 - D$ over the integers is trivial (roots are $\pm \sqrt{D}$). In this way, the small root is revealed.

Another application example is **RSA with partially known message**: say $N$ is a large RSA modulus, $e=3$, and the message $m$ is known to have a certain structure (for instance, the high-order bits of $m$ are known or fixed, and only a small portion of $m$ is unknown). We can write $m = m_{\text{known}} + x_0$, where $x_0$ represents the unknown part (which we hope is small). The RSA encryption gives $c = m^3 \mod N$. We can plug $m = m_{\text{known}} + x$ into that and expand, obtaining a polynomial congruence in $x$. This polynomial $F(x) = (m_{\text{known}} + x)^3 - c$ will have $x_0$ (the unknown part of the message) as a root mod $N$. If $x_0$ (the unknown part) is small enough, we can then apply Coppersmith’s method to find it. In practice, this strategy has been used to show that if, say, at least 2/3 of the message bits are known, the remaining 1/3 can be found. The method sets up a lattice with polynomials derived from expanding $(m_{\text{known}}+x)^3 - c$ along with multiples of $N$, and finds the small root $x_0$.

In summary, Coppersmith’s method provides a powerful lattice-based algorithm to find small solutions of $f(x)\equiv0 \pmod{N}$ without factoring $N$. It works by constructing a tailored lattice of polynomial coefficients such that any point in the lattice corresponds to a polynomial vanishing at the desired root mod $N$. Lattice reduction (LLL) then finds a short vector in this lattice, yielding a polynomial $h(x)$ with tiny coefficients. Provided the root is within the size bound, this polynomial will vanish at the root *over the integers*, allowing us to efficiently recover the root by standard algebraic techniques. The method has proven invaluable in cryptanalysis, especially for attacking RSA in scenarios with low encryption exponents or partially known secrets, scanning an exponential-size search space of potential roots in only polynomial time. Its limitations are well-defined by the size of the root relative to $N$, and it stands as a beautiful application of lattice basis reduction in the field of cryptography.

**References:** Coppersmith’s original papers introduced the method in 1996, and further improvements and applications have been explored by many researchers. Detailed expositions can be found in cryptography textbooks and surveys, and the method is implemented in computer algebra systems (e.g., SageMath’s `small_roots` function).
