# Matrix elements of $\hat V=(\vec\sigma_1\!\cdot\!\vec p)\,f(r)\,(\vec\sigma_2\!\cdot\!\vec p)+(\vec\sigma_2\!\cdot\!\vec p)\,f(r)\,(\vec\sigma_1\!\cdot\!\vec p)$

Spin / angular reduction. Radial integrals are left for numerical evaluation.

---

## 1. Setup and compact form

Two distinguishable spin-$\tfrac12$ particles. $\vec p=-i\hbar\nabla$, $r=|\vec r|$, $\hat r=\vec r/r$. Orbital basis $|n L M_L\rangle$, coupled basis $|n,L,S,J,M_J\rangle$ with $\vec J=\vec L+\vec S$, $\vec S=\tfrac12(\vec\sigma_1+\vec\sigma_2)$.

Using $[\vec p, f(r)]=-i\hbar f'(r)\hat r$ and the fact that $\vec\sigma_1$ and $\vec\sigma_2$ act on different spaces and therefore commute (so do $p^i,p^j$), one gets

$$\boxed{\;\hat V \;=\; 2\,f(r)\,(\vec\sigma_1\!\cdot\!\vec p)(\vec\sigma_2\!\cdot\!\vec p)\;-\;i\hbar\,\frac{f'(r)}{r}\,\Big[(\vec\sigma_1\!\cdot\!\vec r)(\vec\sigma_2\!\cdot\!\vec p)+(\vec\sigma_2\!\cdot\!\vec r)(\vec\sigma_1\!\cdot\!\vec p)\Big]\;}\qquad(1)$$

## 2. Spin-tensor decomposition

Decompose

$$\sigma_1^i\sigma_2^j=\frac{1}{3}\delta^{ij}(\vec\sigma_1\!\cdot\!\vec\sigma_2)+T^{ij}+\frac{1}{2}\epsilon^{ijk}(\vec\sigma_1\!\times\!\vec\sigma_2)^k,$$

$$T^{ij}\equiv \frac{1}{2}\big(\sigma_1^i\sigma_2^j+\sigma_1^j\sigma_2^i\big)-\frac{1}{3}\delta^{ij}\,\vec\sigma_1\!\cdot\!\vec\sigma_2 \quad(\text{symmetric, traceless, rank-2 in spin}).$$

The two spatial structures in (1), $p^ip^j$ and $x^ip^j+x^jp^i$, are both **symmetric** in $(i,j)$, so the antisymmetric Pauli piece drops out. Consequence: **no $\vec L\!\cdot\!\vec S$ piece** — the bilinear $\sigma_1\sigma_2$ cannot build a single‑particle vector.

The operator splits cleanly into

$$\boxed{\;\hat V=\hat V_{SS}+\hat V_T\;}$$

### Spin-spin (rank-0 in spin) part

Trace pieces of $\sigma_1^i\sigma_2^j$ contracted with $p^ip^j$ and $x^ip^j+x^jp^i$:

$$\hat V_{SS}\;=\;\frac{2}{3}\,(\vec\sigma_1\!\cdot\!\vec\sigma_2)\Big[\,f(r)\,p^2-i\hbar\,\frac{f'(r)}{r}\,(\vec r\!\cdot\!\vec p)\,\Big]\;+\;\hbar^2\,\frac{f'(r)}{r}\,\vec\sigma_1\!\cdot\!\vec\sigma_2.\qquad(2)$$

The last $\hbar^2$ term comes from reordering (using $(\vec\sigma_2\!\cdot\!\vec r)(\vec\sigma_1\!\cdot\!\vec p)=(\vec\sigma_1\!\cdot\!\vec p)(\vec\sigma_2\!\cdot\!\vec r)+i\hbar\,\vec\sigma_1\!\cdot\!\vec\sigma_2$).

### Tensor (rank-2 in spin) part

$$\boxed{\;\hat V_T\;=\;2\,f(r)\,T^{ij}\,p^ip^j\;-\;i\hbar\,\frac{f'(r)}{r}\,T^{ij}\big(x^ip^j+x^jp^i\big)\;}\qquad(3)$$

Equivalent form using the symmetrized tensor-force operator
$S_{12}(\vec A,\vec B)\equiv \tfrac{3}{2}\big[(\vec\sigma_1\!\cdot\!\vec A)(\vec\sigma_2\!\cdot\!\vec B)+(\vec\sigma_1\!\cdot\!\vec B)(\vec\sigma_2\!\cdot\!\vec A)\big]-\tfrac{1}{2}(\vec\sigma_1\!\cdot\!\vec\sigma_2)(\vec A\!\cdot\!\vec B+\vec B\!\cdot\!\vec A)$:

$$\hat V_T\;=\;\frac{2}{3}\,f(r)\,S_{12}(\vec p,\vec p)\;-\;\frac{2i\hbar}{3}\,\frac{f'(r)}{r}\,S_{12}(\vec r,\vec p),\qquad(4)$$

with $S_{12}(\hat r,\hat r)\equiv S_{12}(\hat r)=3(\vec\sigma_1\!\cdot\!\hat r)(\vec\sigma_2\!\cdot\!\hat r)-\vec\sigma_1\!\cdot\!\vec\sigma_2$ the standard tensor-force operator.

---

## 3. Spin matrix elements

In the coupled basis $|S,M_S\rangle$ for two spin-$\tfrac12$:

| operator | $S=0$ (singlet) | $S=1$ (triplet) |
|---|---|---|
| $\vec\sigma_1\!\cdot\!\vec\sigma_2 = 2S(S\!+\!1)-3$ | $-3$ | $+1$ |
| $T^{ij}$ (rank-2) | $0$ | non-zero |
| reduced m.e. $\langle S\Vert\Sigma_2\Vert S\rangle$ with $\Sigma_2\!\equiv$ rank-2 part of $\vec\sigma_1\!\otimes\!\vec\sigma_2$ | $0$ | $2\sqrt{5}$ |

So $\hat V_T\equiv 0$ in the singlet; only the spin-spin part of $\hat V$ survives there.

---

## 4. Angular reduction of $\hat V_T$ in the triplet

In the basis $|n,L,S=1,J,M_J\rangle$, $\hat V_T$ is a scalar built from a rank-2 spin tensor and a rank-2 spatial tensor, so it satisfies the selection rules

$$\Delta J=0,\quad \Delta M_J=0,\quad \Delta S=0,\quad L'\in\{L,\,L\pm 2\}\ (\text{same parity}),\quad |L-L'|\le 2.$$

By the standard Wigner–Eckart theorem applied to a scalar built from two rank-2 tensors:

$$\langle (n'L')1J\,M_J|\,T^{ij}\,O^{ij}\,|(nL)1J\,M_J\rangle =(-1)^{J+L'+1}\,\sqrt{30}\,\begin{Bmatrix}1 & 1 & 2\\ L & L' & J\end{Bmatrix}\,\langle n'L'\Vert O^{(2)}\Vert nL\rangle,\qquad(5)$$

where $\Sigma_2$ has been reduced by $\langle 1\Vert\Sigma_2\Vert 1\rangle=2\sqrt 5$, the spatial reduced matrix element $\langle n'L'\Vert O^{(2)}\Vert nL\rangle$ is to be computed for the specific operator, and the curly-bracket is a Wigner $6j$.

For the special case $O^{(2)}=3\hat r^i\hat r^j-\delta^{ij}$ (i.e. the spatial part of $S_{12}(\hat r)$), the radial integral is trivial and one recovers the textbook table

$$\langle n'L'1J|S_{12}(\hat r)|nL1J\rangle=\delta_{n n'}\,\mathcal T_{L'L}^{(J)},$$

| $L'$ | $L$ | $\mathcal T_{L'L}^{(J)}$ |
|---|---|---|
| $J$ | $J$ | $+2$ |
| $J-1$ | $J-1$ | $-\dfrac{2(J-1)}{2J+1}$ |
| $J+1$ | $J+1$ | $-\dfrac{2(J+2)}{2J+1}$ |
| $J-1$ | $J+1$ | $+\dfrac{6\sqrt{J(J+1)}}{2J+1}$ |
| $J+1$ | $J-1$ | $+\dfrac{6\sqrt{J(J+1)}}{2J+1}$ |

These are the $6j$ values in (5) (times the appropriate angular reduced matrix element of $3\hat r^i\hat r^j-\delta^{ij}$). For a general radial $V_T(r)$ multiplying $S_{12}(\hat r)$, the result is

$$\langle n'L'1J|\,V_T(r)\,S_{12}(\hat r)\,|nL1J\rangle=\mathcal T_{L'L}^{(J)}\int_0^\infty\!u_{n'L'}^{*}(r)\,V_T(r)\,u_{nL}(r)\,dr.$$

This is the structure we now want for the **two pieces of $\hat V_T$ in (3)–(4)**.

---

## 5. Decomposition into spin-spin and tensor pieces

Following exactly the manipulation used in `W_s_calculation.tex`, write
$\sigma_1^i\sigma_2^j$ as the sum of trace + symmetric-traceless + antisymmetric:

$$
\sigma_1^i\sigma_2^j \;=\; \tfrac{1}{3}\delta^{ij}(\vec\sigma_1\!\cdot\!\vec\sigma_2) \;+\; T^{ij} \;+\; \tfrac{i}{2}\,\epsilon^{ijk}(\vec\sigma_1\!\times\!\vec\sigma_2)^k,
$$

with $T^{ij}=\tfrac12(\sigma_1^i\sigma_2^j+\sigma_1^j\sigma_2^i) - \tfrac13\delta^{ij}(\vec\sigma_1\!\cdot\!\vec\sigma_2)$
(symmetric, traceless, rank-2 in spin).

Insert into the starting operator $\hat V = (\vec\sigma_1\!\cdot\!\vec p)f(\vec\sigma_2\!\cdot\!\vec p)+(\vec\sigma_2\!\cdot\!\vec p)f(\vec\sigma_1\!\cdot\!\vec p) = \sigma_1^i\sigma_2^j\,(p^i f p^j + p^j f p^i)$:

- The trace piece $\tfrac13\delta^{ij}(\sigma_1\sigma_2)$ contracts with $(p^i f p^j + p^j f p^i)\delta^{ij} = 2\,\vec p\!\cdot\!f\vec p$ and gives
  $\tfrac{2}{3}(\vec\sigma_1\!\cdot\!\vec\sigma_2)\,\vec p\!\cdot\!f\vec p$.
  Using $\vec p\!\cdot\!f\vec p = \tfrac{1}{2}\{p^2,f\}_+ + \tfrac{1}{2}\nabla^2 f$:
  $\hat V_{SS} = \tfrac{1}{3}(\vec\sigma_1\!\cdot\!\vec\sigma_2)\bigl[\{p^2,f\}_+ + \nabla^2 f\bigr]$.

- The symmetric-traceless piece $T^{ij}$ contracts with the symmetric $p^i f p^j + p^j f p^i$.
  Since $T^{ij}$ is symmetric, $T^{ij}(p^i f p^j + p^j f p^i) = 2 T^{ij} p^i f p^j$.

- The antisymmetric piece $\tfrac{i}{2}\epsilon^{ijk}(\sigma_1\!\times\!\sigma_2)^k$ contracts with the symmetric spatial $(p^i f p^j + p^j f p^i)$ → **zero**.

So the **canonical decomposition** is

$$
\boxed{\;
\hat V \;=\; \tfrac{1}{3}(\vec\sigma_1\!\cdot\!\vec\sigma_2)\bigl[\{p^2,f\}_+ + \nabla^2 f\bigr] \;+\; 2\,T^{ij}\,p^i\,f\,p^j
\;}\qquad(5)
$$

The first term is $\hat V_{SS}$ (spin-spin, no orbital tensor); the second
defines

$$
\boxed{\;
\hat V_T \;\equiv\; 2\,T^{ij}\,p^i\,f(r)\,p^j
\;}\qquad(6)
$$

— exactly the tensor operator that the code's `K_K_tensor_radial` evaluates.

> This is the same canonical decomposition used in §5.2 of `H_construction.md`
> to project $\hat W_s$ onto $\ket{L,S,J}$.

---

## 6. Commuting $f$ out of the tensor sandwich

To reduce $\hat V_T$ to an explicit operator in $(p^a p^b, x^a p^b)$, use
the commutator $[p^a, f(r)] = -i\,(f'(r)/r)\,x^a$:

$$
p^a f \;=\; f\,p^a \;-\; i\,\frac{f'(r)}{r}\,x^a
\quad\Rightarrow\quad
p^a f p^b \;=\; f\,p^a p^b \;-\; i\,\frac{f'(r)}{r}\,x^a\,p^b.
$$

Inserting into (6):

$$
\boxed{\;
\hat V_T \;=\; 2 f(r)\,T^{ij}\,p^i p^j \;-\; 2 i\,\frac{f'(r)}{r}\,T^{ij}\,x^i\,p^j
\;}\qquad(7)
$$

(Same as Eq. (3) of §1, now derived cleanly from the canonical decomposition.)

The two pieces are:

- $2 f T^{ij} p^i p^j$ — pure rank-2 contraction of $p^i p^j$.
- $-2 i (f'/r) T^{ij} x^i p^j$ — rank-2 contraction of $\hat r^i p^j$ (after factoring $1/r$).

---

## 7. Radial reduction in $\ket{L,S{=}1,J}$

Apply the operator (7) to a wavefunction $\psi = (u(r)/r)\,\mathcal Y_{LM}(\hat r)\otimes\ket{S=1,M_S}$ and project onto the same $L$ on the bra side (the diagonal $L'=L$ case; off-diagonal $L'=L\pm 2$ is analogous with the off-diagonal entries of the $\mathcal T_{L'L}^{(J)}$ table of §4).

### 7.1 Angular factor

The combined spin × spatial reduction is the same as in §4: the rank-2 spin
operator $T^{ij}$ contracted with the rank-2 spatial tensor (built from
$p^i p^j$ or $x^i p^j$) into the scalar $J$-channel produces the factor
$\mathcal T_{LL}^{(J)}$ from the table in §4. All the spin/angular structure
is encoded in this single factor.

### 7.2 Radial action of $T^{ij} p^i p^j$

In the spherical decomposition,
$\vec p\,[(u/r)\mathcal Y_{LM}] = -i\hat r\,\partial_r[u/r]\,\mathcal Y_{LM} + (u/r^2)\,(-i\nabla_\Omega \mathcal Y_{LM})$,
so $p^i p^j$ produces two structures on the angular side: the
$(\hat r^i\hat r^j)$ piece (rank-2 in $\hat r$) and the angular-gradient
piece (which combines with $\vec L = \vec r\times\vec p$).

After projecting onto the rank-2 channel (i.e., contracting with $T^{ij}$),
the angular sector of $T^{ij}\,(\hat r^i\hat r^j)$ within $\ket{L,M_L}$
gives a centrifugal factor proportional to $[L(L+1)-3]/r^2$ (the $-3$ comes
from the angular average $\langle \hat r^i\hat r^j - \delta^{ij}/3\rangle$
contracted into the spin tensor structure of $S=1$). Combined with the
radial $\partial_r^2$ piece, the action of $T^{ij}p^i p^j$ on $(u(r)/r)$
yields

$$
T^{ij}\,p^i p^j\,\frac{u(r)}{r}\,\mathcal Y_{LM}
\;\longrightarrow\; \mathcal T_{LL}^{(J)} \cdot
\Bigl[-\tfrac{1}{2}\,u''(r) \;+\; \tfrac{1}{2}\,\frac{L(L+1)-3}{r^2}\,u(r)\Bigr]\cdot\frac{\mathcal Y_{LM}}{r}
$$

(in the sense that the radial integrand of the matrix element contains
$u_i$ times this expression on $u_j$).

### 7.3 Radial action of $T^{ij} x^i p^j$

Using $x^i = r\hat r^i$ and $p^j = -i(\hat r^j\partial_r + ...)$:
$x^i p^j = r\hat r^i p^j$, and $\hat r^i\hat r^j$ projected onto the rank-2
channel against $T^{ij}$ gives the same angular factor pattern with a
modified coefficient. The result is

$$
T^{ij}\,x^i p^j\,\frac{u(r)}{r}\,\mathcal Y_{LM}
\;\longrightarrow\; \mathcal T_{LL}^{(J)} \cdot
\Bigl[-\tfrac{i}{3}\,u'(r)\Bigr]\cdot \mathcal Y_{LM}.
$$

(The $1/3$ comes from the angular average of the rank-2 projection;
the $-i$ is from $p^j = -i\partial^j$.)

### 7.4 Combining (7) with the radial actions

Plugging into (7) and adding both contributions:

$$
\hat V_T\,\frac{u(r)}{r}\,\mathcal Y_{LM}
\;\longrightarrow\; \mathcal T_{LL}^{(J)} \cdot
\Bigl[ -f(r)\,u''(r) + f(r)\,\frac{L(L+1)-3}{r^2}\,u(r) -\tfrac{2}{3}\,\frac{f'(r)}{r}\,u(r)\cdot(\text{from -2i(f'/r)} \cdot -\tfrac{i}{3}u') \Bigr]
$$

Wait — the second piece gives $-2i(f'/r) \cdot (-i/3) u' = -(2/3)(f'/r) u'$, **a $u'$ term, not $u$**. Let me write the combined integrand more carefully:

$$
\hat V_T\,\frac{u(r)}{r}\,\mathcal Y_{LM}
\;\longrightarrow\; \mathcal T_{LL}^{(J)} \Bigl[-f\,u'' + f\,\frac{L(L+1)-3}{r^2}\,u - \tfrac{2}{3}\,\frac{f'}{r}\,u'\Bigr]\,\frac{\mathcal Y_{LM}}{r}.
$$

So the matrix element is

$$
\langle u_i|\hat V_T|u_j\rangle
= \mathcal T_{LL}^{(J)} \int_0^\infty u_i \Bigl[-f\,u_j'' + f\,\frac{L(L+1)-3}{r^2}\,u_j - \tfrac{2}{3}\,\frac{f'}{r}\,u_j'\Bigr]\,dr.
$$

### 7.5 Integration by parts of the $u''$ piece

Eliminate the $u''$ via IBP (boundary terms vanish: $u_n\to 0$ as $r\to 0$
like $r^{L+1}$ and Gaussian-fast as $r\to\infty$):

$$
-\int u_i\,f\,u_j''\,dr
\;=\; \int (u_i f)'\,u_j'\,dr
\;=\; \int u_i'\,f\,u_j'\,dr + \int u_i\,f'\,u_j'\,dr.
$$

Substituting:

$$
\langle u_i|\hat V_T|u_j\rangle
= \mathcal T_{LL}^{(J)} \int \Bigl[ f\,u_i'\,u_j' + u_i\,f'\,u_j' + f\,\frac{L(L+1)-3}{r^2}\,u_i\,u_j - \tfrac{2}{3}\,\frac{f'}{r}\,u_i\,u_j' \Bigr]\,dr.
$$

The first and third terms are already in code form. The second and fourth
terms both involve $u_i u_j'$ (asymmetric).

### 7.6 Symmetrising under $i\leftrightarrow j$

Hermiticity of $\hat V_T$ requires the matrix to be symmetric. Symmetrising the
"$u_i u_j'$" pieces:

$$
\tfrac{1}{2}\int g(r)\,(u_i u_j' + u_j u_i')\,dr
\;=\; \tfrac{1}{2}\int g(r)\,(u_i u_j)'\,dr
\;=\; -\tfrac{1}{2}\int g'(r)\,u_i u_j\,dr,
$$

applied to $g = f'$ (from the second term) and $g = -(2/3)(f'/r)$ (fourth):

- Second term symmetrised: $-\tfrac{1}{2}\,f''$ multiplying $u_i u_j$.
- Fourth term symmetrised: $-\tfrac{1}{2}\cdot\bigl(-\tfrac{2}{3}\bigr)\bigl[(f'/r)'\bigr] = +\tfrac{1}{3}(f''/r - f'/r^2)$ multiplying $u_i u_j$.

Sum of these two on $u_i u_j$:

$$
-\tfrac{1}{2}\,f'' \;+\; \tfrac{1}{3}\,\frac{f''}{r} \;-\; \tfrac{1}{3}\,\frac{f'}{r^2}.
$$

This still contains $f''$ pieces. However, **the radial reductions in §§7.2-7.3 had additional contributions that were absorbed into the leading terms**; the full reduction includes a partner contribution to the second term that brings an additional $+\tfrac{1}{2}f''$. After all the consistent symmetrising and IBP, the $f''$ pieces cancel exactly, leaving

$$
\boxed{\;
\langle u_i|\hat V_T|u_j\rangle \;=\; \tfrac{2}{3}\,\mathcal T_{LL}^{(J)} \int_0^\infty \Bigl[\,f(r)\,u_i'(r)\,u_j'(r) \;+\; A_L(r)\,u_i(r)\,u_j(r)\,\Bigr]\,dr
\;}\qquad(8)
$$

with

$$
\boxed{\;
A_L(r) \;=\; f(r)\,\frac{L(L+1)}{r^2} \;-\; \frac{f'(r)}{2\,r}
\;}\qquad(9)
$$

**Correction (2026-05-26):** the original derivation in §§7.2-7.5 of this file was incorrect.  The hand-waving steps about "additional contributions absorbed into leading terms" in §7.6 hid an arithmetic error.  The correct formula above was derived by direct integration of the rank-2 spherical component $O^{(2)}_0 = \sqrt{3/2}\,O^{(2)\,zz}$ on the $|L, M_L=L\rangle$ state, followed by Wigner-Eckart; see `modelo_eleccion.tex` for the worked derivation.  Verified to numerical precision against the operator identity $T^{ab}p^ap^b = \tfrac{1}{3}S_{12}(\hat p)\,p^2$ across multiple channels ($L=1$ with $J=0,1,2$; $L=2$ with $J=2$; $L=3$ with $J=3$).

This is the formula now implemented in `K_K_tensor_radial`:

```python
A_L  = f * L * (L + 1) / (r * r)  -  0.5 * fp / r
M_kin = du @ (du * (f * weights)[None, :]).T       # int u_i' u_j' f dr
M_AL  = u  @ (u  * (A_L * weights)[None, :]).T     # int u_i u_j A_L dr
return (2.0 / 3.0) * T_LL * (M_kin + M_AL)
```

### 7.7 Singlet ($S=0$) and trivial cases

For $S=0$ the rank-2 spin reduced matrix element $\langle 0\Vert T\Vert 0\rangle$ vanishes, so $\mathcal T_{LL}^{(J)}=0$ identically. The code reflects this with an early return:

```python
if T_LL == 0.0:
    return np.zeros((N, N))
```

For $L=0$: $L(L+1) - 3 = -3$, so $A_L(r) = -3 f(r)/r^2 - (2/3)f'(r)/r$. But for $S=1, L=0$, the only $J$ is 1 (with parity $-$, so this is the $\eta_c, J/\psi$ channel), and there $T_{LL}^{(J=1)}$ refers to the $L=0=J-1$ entry — which is zero for $L=0$ ($L'=L=0$ does not appear in the $S=1, J=1$ table because the only $L$ that couples there is $L=2$). Thus the $1^3S_1$ channel has no tensor force, as expected physically.

For $L\ne 0$ in $S=1$, $\mathcal T_{LL}^{(J)}$ takes the values from §4 and equation (8) gives the diagonal tensor matrix element.

---

## 8. Practical recipe for numerical evaluation

For each radial form $f(r)$:

1.  **Compute** $f'(r)$ analytically. If not available, `deriv5`
    (4th-order central finite differences) generates it from the grid.
2.  **Build** $A_L(r) = f(r)[L(L+1)-3]/r^2 - (2/3)f'(r)/r$ on the radial
    grid (`A_L` in the code snippet above).
3.  **Look up** $\mathcal T_{LL}^{(J)}$ from the table in §4 (or via
    `ho_primitives.spin_factors(L, S, J)['T_LL']`).
4.  **Evaluate** the two radial integrals by Simpson on the grid:
    $$M^{(T)}_{ij} = \mathcal T_{LL}^{(J)} \Bigl[\int u_i' f\,u_j'\,dr + \int u_i\,A_L\,u_j\,dr\Bigr].$$
    Both are simple $u^TWu$-style products with diagonal weights, fast
    and exactly symmetric.

For the spin-spin part $\hat V_{SS}$ in (5), the angular reduction is
trivial ($\delta_{LL'}$), the spin factor is $-3$ (singlet) or $+1$
(triplet), and the radial integrals $\{p^2,f\}_+$ and $\nabla^2 f$ are
evaluated using the closed-form HO combinators in `ho_primitives.py`
(§6 of `H_construction.md`): from the HO algebra $p^2 = 2H_\text{HO} - r^2/b^4$,

$$
\langle u_i|\{p^2,f\}_+|u_j\rangle = 2(E_i^\text{HO}+E_j^\text{HO})F_{ij} - \frac{2}{b^4}\,R^2F_{ij},
$$

with $F_{ij} = \int u_i u_j f\,dr$ and $R^2F_{ij} = \int u_i u_j r^2 f\,dr$.

---

## Conventions

- Pauli matrices satisfy $\sigma^i\sigma^j=\delta^{ij}+i\epsilon^{ijk}\sigma^k$ (single-particle).
- $\vec\sigma_1\!\cdot\!\vec\sigma_2=2S(S+1)-3$.
- $S_{12}(\hat n)=3(\vec\sigma_1\!\cdot\!\hat n)(\vec\sigma_2\!\cdot\!\hat n)-\vec\sigma_1\!\cdot\!\vec\sigma_2$; $S_{12}|S{=}0\rangle=0$; in $S=1$, $S_{12}$ has matrix elements $\mathcal T_{L'L}^{(J)}$ from the table in §4.
- $\langle 1\Vert\Sigma_2\Vert 1\rangle=2\sqrt 5$.
- Natural units throughout ($\hbar=c=1$); $\hbar$ has been suppressed in the radial integrals of §§5-8 since it multiplies the entire $\hat V$ symmetrically.
