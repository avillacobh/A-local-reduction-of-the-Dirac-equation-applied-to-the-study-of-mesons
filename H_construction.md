# Construcción del Hamiltoniano efectivo en `H_full_matrix.py`

Documento de referencia que explica, de extremo a extremo, cómo se ensambla
la matriz de Salpeter efectiva que aparece en `H_full_matrix.py`,
`salpeter_solver.py`, y los `meson_potential*.py`. El énfasis está en la parte
del potencial: cómo entran $V_v$ y $V_s$, qué papel juegan las combinaciones
$V_L,V_U$, y por qué cada pieza del Hamiltoniano depende de $E_T$.

Convención de unidades: $\hbar=c=1$, GeV en todo el módulo de fit.

---

## 1. Ecuación de partida y reducción $K^\dagger(\cdot)K$

Se parte de la ecuación de Breit en el marco CM para un sistema
quark–antiquark $(p_1=-p,\;p_2=+p)$. El núcleo es

$$
H_\text{Breit} \;=\; D_1 + D_2 + W^{0} + W_s,
$$

con $D_i=\beta_i m+\vec\alpha_i\!\cdot\!\vec p_i$ (Dirac libre por partícula),

$$
W^{0} \;=\; V_v(r) \;+\; \beta_1\beta_2\,V_s(r),
\qquad
W_s \;=\; \vec\alpha_1\!\cdot\!\vec\alpha_2\,V_v^{\,s}(r).
$$

Al proyectar sobre el subespacio "Salpeter" $(\Lambda_+\Lambda_+\!+\!\Lambda_-\Lambda_-)$
y diagonalizar la parte cinética mediante una transformación
$K\!=\!K_1\otimes K_2$ (un Foldy–Wouthuysen acoplado a dos cuerpos), se
obtiene una representación 2-componentes con un kernel **no lineal en $V_X$**
a través de los factores

$$
A_X(r) \;=\; \frac{1}{m + \tfrac12 E_T \;-\; \tfrac12 V_X(r)}\,,\qquad X\in\{L,U\}.
$$

Las dos potenciales que aparecen son combinaciones lineales del input físico
$(V_v,V_s)$:

$$
\boxed{\;V_L \;=\; V_v - V_s\,,\qquad V_U \;=\; V_v + V_s\;}
$$

La elección de signos surge de la estructura $V_v + \beta_1\beta_2 V_s$ al actuar
sobre los sectores del espinor: $\beta_1\beta_2 = +1$ en $UU$ y $LL$, que ven
$V_U = V_v + V_s$, y $\beta_1\beta_2 = -1$ en los mixtos $LU$, $UL$, que ven
$V_L = V_v - V_s$.

> **Corrección 2026-08-27.** Este documento escribía antes
> $W^{0} = \gamma_1^0\gamma_2^0 V_v + V_s$, es decir con $\beta_1\beta_2$ sobre
> la pieza vectorial en vez de la escalar.  Esa forma NO reproduce
> $V_L = V_v - V_s$ en los sectores mixtos (da $-V_L$) y es inconsistente con
> los conjuntos acoplados del trabajo de grado y con lo que construye
> `build_radial_blocks`.  La asignación correcta la fija la estructura de
> Lorentz del intercambio: el acoplamiento escalar es
> $\bar\psi\psi = \psi^\dagger\beta\psi$ y por tanto lleva un $\beta$ por
> partícula, mientras que la componente temporal del vectorial es
> $\bar\psi\gamma^0\psi = \psi^\dagger\psi$ y lleva la identidad.
> Sólo afecta a esta nota: el código nunca usó la forma incorrecta.

El resultado de la reducción, **antes** de truncar a una base, es la fórmula
maestra que aparece como docstring de `H_full_matrix.py`:

$$
\boxed{\;
H(E_T) \;=\; 2m \;+\; V_U \;+\; 2m\,(K^\dagger K) \;+\;
2\,K_{LL}^\dagger\!\left(m + \tfrac{E_T}{2} - \tfrac{V_U}{2}\right)\!K_{LL}
\;+\; W_s \;}
$$

con $m+\tfrac{E_T}{2}-\tfrac{V_U}{2}\equiv 1/A_U$. El primer sumando $2m$ es
la energía en reposo de la pareja; el segundo $V_U=V_v+V_s$ aporta el
potencial vector y escalar como término diagonal en $r$; el tercero y cuarto
son las dos contribuciones cinéticas de la reducción $K$; y $W_s$ es la
pieza espacial-vectorial (OGE) que se trata aparte porque no admite la
misma forma compacta.

### 1.1 Condición de auto-consistencia en $E_T$

Como $A_X$ contiene $E_T$, el Hamiltoniano matricial depende
**no-trivialmente** de la propia energía total que se busca. La condición
física para el nivel $n$ es

$$
f_n(E_T) \;=\; \lambda_n\bigl[H(E_T)\bigr] - E_T \;=\; 0,
$$

con autovalores ordenados de menor a mayor. En `salpeter_solver.py` esto se
resuelve por Brent (con bisección como respaldo) dentro de un *bracket*
$[E_T^\text{lo},E_T^\text{hi}]$.

---

## 2. La pieza puramente potencial: $V_U$ en la diagonal

El sumando $V_U=V_v+V_s$ en la fórmula maestra es una función de $r$ y, en
la base HO 3D isotrópica, sus elementos de matriz son la integral radial

$$
\bigl[V_U\bigr]_{ij} \;=\; \int_0^\infty u_i(r)\,u_j(r)\,V_U(r)\,dr,
$$

con $u_n(r)=rR_{nL}(r)$ las funciones radiales del oscilador. Este término
es **independiente de $E_T$** y por eso se cachea de una vez al construir
`MesonHamiltonian` (línea 522 de `H_full_matrix.py`: `self.F_VU = ...`).

---

## 3. Forma funcional del potencial: cuatro variantes

Las funciones $V_v(r)$ y $V_s(r)$ se eligen en los módulos
`meson_potential.py` (v1), `meson_potential_v2.py` (v2), `meson_potential_v3.py`
(v3) y `meson_potential_coulomb.py` (coulomb). Cada uno expone
`make_potential_funcs(...)` y devuelve un diccionario con
$V_L,V_U,V_v$ y sus derivadas primeras y segundas (analíticas siempre
que sea factible).

| Variante | $V_v(r)$ | $V_s(r)$ | $N_\text{params}$ |
|---|---|---|---|
| v1 | $\bar V_v - \tfrac{4\alpha}{3}\,\dfrac{\mathrm{erf}(r/d_v)}{r}$ | $\tfrac12\bar V_s\!\left[\mathrm{erf}\!\left(\tfrac{r-r_s}{d_s}\right)-1\right]$ | 6 |
| v2 | $\bar V_v - \tfrac{4\alpha}{3}\,\dfrac{\mathrm{erf}(r/d)}{r}$ | $-\bar V_s\,e^{-r^2/r_s^2}$ | 5 |
| v3 | (similar a v2, con $\bar V_v$ derivado) | $-\bar V_s\,e^{-r^2/r_s^2}$ | 4 |
| coulomb | $\bar V_v - \tfrac{4\alpha}{3r}$ (singular en $0$) | $-\bar V_s\,e^{-r^2/r_s^2}$ | 4 |

Observaciones que conviene tener presentes:

- **`bar_V_v` no es trivial**. En el límite no-relativista una constante
  añadida a $V_v$ sería absorbible en $2m_q$. Pero en Salpeter
  $A_X(r)$ es no-lineal en $V_X$, así que `bar_V_v` cambia el espectro.
  Sin `bar_V_v` el root-finder de $E_T$ pierde sign change en casi todos
  los canales.
- **El `erf` regulariza la singularidad** de $-1/r$ en $r=0$.
  La variante `coulomb` se incluye sólo por completitud y comparación;
  introduce dependencia del corte numérico en $r_\text{min}$.
- **$V_s\to 0$ cuando $r\to\infty$** en todas las variantes; **$V_v\to\bar V_v$**.
  Esto fija las asintóticas
  $V_L(\infty)=V_U(\infty)=\bar V_v$ y
  $A_L(\infty)=A_U(\infty)=1/(m+E_T/2-\bar V_v/2)$.

---

## 4. Las dos piezas cinéticas $K^\dagger K$ y $K_{LL}^\dagger(\cdot)K_{LL}$

Estos son los términos que contienen *todas* las derivadas radiales del
potencial. La estructura general es:

$$
K^\dagger K \;=\; \{p^2,\,A_L^2\}_+ \;+\; \nabla^2 A_L^2 \;+\;
\kappa_{LS}\,\frac{(A_L^2)^{\prime}}{r},
$$

con $\vec\sigma_T=\vec\sigma_1+\vec\sigma_2$ el espín total y, al proyectar
sobre $\ket{L,S,J}$, $\vec\ell\!\cdot\!\vec\sigma_T\to\kappa_{LS}$ con

$$
\kappa_{LS} \;=\; J(J+1)-L(L+1)-S(S+1).
$$

Esto es exactamente lo que devuelve `K_dagK_matrix` (línea 246), donde
cada uno de los tres sumandos se evalúa **como integral radial cerrada**
mediante las primitivas de `ho_primitives.py` (ver §6).

La pieza larga $K_{LL}^\dagger(1/A_U)K_{LL}$ se ensambla en
`KLL_dagger_KLL_matrix` (línea 261) y se organiza en **ocho líneas** que
corresponden a la expansión completa del producto de operadores. Con
$X\equiv A_L^2 A_U$, sus roles físicos son:

- **Línea 2** ($\{p^4,X\}$, $\nabla^4 X$, $p^2 X p^2$, $\{p^2,\nabla^2X\}$ y
  un sandwich $(\vec p\!\cdot\!\vec r)f(\vec r\!\cdot\!\vec p)$):
  pieza cinética dominante (cuarto orden en $p$).
- **Línea 3**: correcciones $\nabla^2$ de derivadas mixtas
  $(A_L A_L^{\prime} A_U)'$ y $A_L(A_L^{\prime}A_U)'$.
- **Líneas 4–5**: estructura $\kappa_{LS}\,\vec\ell\!\cdot\!\vec\sigma_T$
  (espín-órbita).
- **Línea 6**: pieza $\{\vec\ell\!\cdot\!\vec\sigma_1,\,\vec\ell\!\cdot\!\vec\sigma_2\}_+$
  con coeficiente $\kappa_{LL12}$.
- **Línea 7**: pieza $\vec\sigma_1\!\cdot\!\vec\sigma_2$ multiplicada por
  $\nabla^2(A_L^2A_U'/r)$ y $\{p^2,A_L^2A_U'/r\}$. Aporta hiperfino.
- **Línea 8**: $-\tfrac12[(\vec\sigma\!\cdot\!p)f(\vec\sigma\!\cdot\!p)+\text{h.c.}]$
  con $f=A_L^2A_U'/r$. Se descompone canónicamente como

  $$
  (\vec\sigma_1\!\cdot\!\vec p)f(\vec\sigma_2\!\cdot\!\vec p)+\text{h.c.}
   \;=\;
  \tfrac13(\vec\sigma_1\!\cdot\!\vec\sigma_2)\bigl[\{p^2,f\}+\nabla^2 f\bigr]
   \;+\; 2\,T^{ab}\,p^a\,f\,p^b,
  $$

  con $T^{ab}=\sigma_1^a\sigma_2^b+\sigma_1^b\sigma_2^a-\tfrac{2}{3}\delta^{ab}\,\vec\sigma_1\!\cdot\!\vec\sigma_2$
  el operador tensor. Al proyectar sobre $(L,S,J)$ el escalar pasa a
  $\kappa_{SS}=2S(S+1)-3$ y el tensor a $T_{LL}(L,S,J)$.

---

## 5. La pieza espacial-vectorial $W_s = \vec\alpha_1\!\cdot\!\vec\alpha_2\,V_v^{\,s}$

Es la contribución del intercambio de un gluón en su parte espacial
(la temporal ya está en $V_v$ dentro de $W^0$). Se trata aparte porque
la reducción $K^\dagger(\vec\alpha_1\!\cdot\!\vec\alpha_2\,V_v^s)K$ no se
absorbe en las ocho líneas anteriores.

Convención: $V_v^{\,s}=V_v$ (consistencia OGE: la misma función de
intercambio en tiempo y espacio). El signo OGE relativo entre $W^0_v$ y
$W_s$ se controla con el flag `--ws-sign {+1,-1}` (default $-1$, que
corresponde a $V_v^{\,s} = -V_v$ — la parte espacial del propagador del
gluón entra con signo opuesto a la temporal en $\gamma^\mu\otimes\gamma_\mu
= \gamma^0\gamma_0 - \vec\gamma\cdot\vec\gamma$).

### 5.1 Forma operacional, antes de proyectar

A partir de la derivación manual en `W_s_calculation.tex` y usando
$\sigma_i^a\sigma_i^b=\delta^{ab}+i\,\epsilon^{abc}\sigma_i^c$ para cada
*sándwich* $(\vec\sigma_1\!\cdot\!\vec p)(\vec\sigma_1\!\cdot\!\vec\sigma_2)(\vec\sigma_2\!\cdot\!\vec p)$
y sus permutaciones, se obtiene

$$
W_s \;=\; -\bigl\{F_{VU},\,(1-\vec\sigma_1\!\cdot\!\vec\sigma_2)\,\vec p\!\cdot\!A_L\vec p
 + \tfrac12 S_{A_L}\bigr\}_+
$$
$$
\quad-\;2(1+\vec\sigma_1\!\cdot\!\vec\sigma_2)\,\vec p\!\cdot\!F_{VLL}\vec p
\;-\; 2\frac{F_{VLL}^{\,\prime}}{r}\,\vec\ell\!\cdot\!\vec\sigma_T
\;+\; S_{F_{VLL}},
$$

> **Corrección 2026-09-01.** El último término llevaba antes signo menos.
> El signo correcto es $+S_{F_{VLL}}$: es el único que reproduce los
> coeficientes $-(1+2\kappa_{SS}/3)$ y $+2T^{ab}p^a g_L p^b$ de §5.2 (que es
> lo que implementa `W_s_matrix`), y el único que en el límite no relativista
> da el término de contacto de Fermi con el coeficiente estándar
> $32\pi\alpha_s/9m^2$ y la combinación espín-órbita $3V_v' - V_s'$.

con $F_{VU}=A_U V_v$, $F_{VLL}=A_L^2 V_v$ y
$S_f\equiv(\vec\sigma_1\!\cdot\!\vec p)f(\vec\sigma_2\!\cdot\!\vec p)
       +(\vec\sigma_2\!\cdot\!\vec p)f(\vec\sigma_1\!\cdot\!\vec p)$.

### 5.2 Proyección a $\ket{L,S,J}$

Usando $S_f\to(\kappa_{SS}/3)\bigl[\{p^2,f\}+\nabla^2 f\bigr]+2T^{ab}p^afp^b$,
$\vec p\!\cdot\!f\vec p=\tfrac12\{p^2,f\}_++\tfrac12\nabla^2 f$,
y la identidad útil $\{F_{VU},\{p^2,A_L\}_+\}_+ = 2\{p^2,F\}_++2A_L'f_U'$
con $F=f_UA_L$ y $f_U=V_v A_U$, resulta

$$
\boxed{\;
W_s\bigr|_{\ket{L,S,J}} \;=\;
-\!\left(1-\tfrac{2\kappa_{SS}}{3}\right)\!\bigl[\{p^2,F\}_+\!+e(r)\bigr]
-\!\left(1+\tfrac{2\kappa_{SS}}{3}\right)\!\bigl[\{p^2,g_L\}_+\!+\nabla^2 g_L\bigr]
}
$$
$$
\quad-\;2\,T^{ab}p^a F p^b
\;+\;2\,T^{ab}p^a g_L p^b
\;-\;2\,\kappa_{LS}\frac{g_L^{\prime}}{r}
\;+\;\frac{T_{LL}}{3}\,h(r),
$$

con

$$
F = f_U\,A_L,\qquad g_L = V_v\,A_L^2,\qquad f_U = V_v\,A_U,
$$
$$
e(r) = A_L^{\prime}\,f_U^{\prime} + f_U\,\nabla^2 A_L,\qquad
h(r) = A_L^{\prime}\,f_U^{\prime} + A_L\!\left(f_U^{\prime\prime} - \tfrac{f_U^{\prime}}{r}\right).
$$

Esto es lo que implementa `W_s_matrix` (`H_full_matrix.py` líneas 327–407),
donde

- los dos prefactores `cF, cgL` codifican $-(1\mp 2\kappa_{SS}/3)$;
- los dos tensores se calculan con `K_K_tensor_radial` y entran con signos opuestos;
- la pieza espín-órbita aparece con coeficiente $-2\kappa_{LS}$ sobre $g_L^{\prime}/r$;
- la corrección multiplicativa $h(r)$ entra con $T_{LL}/3$.  Esta pieza
  proviene del *leftover* tensor del anticonmutador $\{F_{UV},T^{ab}p^a A_L p^b\}_+$:
  la identidad correcta es $\{F_{UV},T^{ab}p^aA_Lp^b\}_+ = 2T^{ab}p^aFp^b - (h/3)S_{12}(\hat r)$,
  no simplemente $\to 2T^{ab}p^aFp^b$ (que sería una flecha aproximada).
  El $h(r)$ codifica las correcciones de conmutador $[p,F_{UV}]$ que sobreviven
  a la contracción con $T^{ab}$ vía $T^{ab}x^ax^b = (r^2/3)S_{12}$.

---

## 6. De operadores con momento a integrales radiales

Esta sección es el corazón numérico del módulo: cómo cada operador que
aparece en la fórmula maestra (con $p^2$, $p^4$, $\vec\sigma\!\cdot\!\vec p$,
etc.) se reduce a **una o más integrales radiales** sin necesidad de
formar matrices de momento truncadas. Todo está implementado en
`ho_primitives.py`.

### 6.1 La base HO 3D isotrópica y su derivada analítica

Las funciones radiales del oscilador son

$$
u_{nL}(r) \;=\; r\,R_{nL}(r),\qquad
R_{nL}(r) \;=\; \sqrt{\frac{2\,n!}{b^3\,\Gamma(n+L+\tfrac32)}}\;
\Bigl(\frac{r}{b}\Bigr)^L\,e^{-r^2/2b^2}\,L_n^{L+1/2}\!\Bigl(\frac{r^2}{b^2}\Bigr),
$$

con $L_n^{\alpha}$ los polinomios de Laguerre generalizados y $b$ la escala
del oscilador. Se normalizan por $\int_0^\infty |u_{nL}|^2\,dr=1$.

La derivada $u_{nL}^{\prime}(r)$ se obtiene **analíticamente** vía la
recurrencia de Laguerre $\frac{d}{dx}L_n^{\alpha}(x) = -L_{n-1}^{\alpha+1}(x)$,
sin recurrir a diferencias finitas. Esto es importante: cualquier operador
con un solo factor de momento ($\vec\sigma\!\cdot\!\vec p$,
$\vec r\!\cdot\!\vec p$) se evaluará usando $u_{nL}^{\prime}(r)$
analítica, manteniendo precisión $\sim 10^{-12}$.

### 6.2 La identidad central: $p^2 = 2H_\text{HO} - r^2/b^4$

El Hamiltoniano del oscilador armónico isotrópico se escribe

$$
H_\text{HO} \;=\; \frac{p^2}{2} + \frac{r^2}{2b^4},
$$

con autovalores $E_n^\text{HO} = (2n+L+\tfrac32)/b^2$. De aquí

$$
\boxed{\;p^2 \;=\; 2H_\text{HO} - \frac{r^2}{b^4}\;}
$$

Esta identidad operacional es la clave: actuando sobre un estado HO,
$p^2$ se reduce a una *combinación de funciones multiplicativas* ($r^2/b^4$)
y un autovalor numérico ($2E_n^\text{HO}$). En particular, los elementos
de matriz de cualquier polinomio en $p^2$ entre estados HO se expresan
como combinaciones lineales de integrales radiales del tipo

$$
F_{ij} \;=\; \int_0^\infty u_i\,u_j\,f(r)\,dr,\qquad
R^k F_{ij} \;=\; \int_0^\infty u_i\,u_j\,r^k f(r)\,dr,
$$

con $k=2,4,\ldots$, **sin necesidad de formar la matriz de $p^2$** y
multiplicarla por la matriz de $f$.

### 6.3 Anticonmutadores y sandwiches en forma cerrada

Aplicando $p^2 = 2H_\text{HO} - r^2/b^4$ a derecha o izquierda y usando
hermiticidad:

$$
\langle u_i|\,p^2 f\,|u_j\rangle
\;=\;
2E_i^\text{HO}\,F_{ij} \;-\; \frac{1}{b^4}\,R^2F_{ij},
$$

y de aquí, simetrizando,

$$
\boxed{\;
\langle u_i|\{p^2, f\}_+|u_j\rangle
\;=\; 2\bigl(E_i^\text{HO}+E_j^\text{HO}\bigr)F_{ij}
\;-\; \frac{2}{b^4}\,R^2F_{ij}
\;}
$$

Este es `acomm_p2_f_closed(F, R2F, E_arr, b)` en `ho_primitives.py`.

Análogamente, expandiendo $p^4 = (2H_\text{HO} - r^2/b^4)^2$ y teniendo
cuidado con los conmutadores $[H_\text{HO}, r^2]$, se obtiene una fórmula
cerrada para $\{p^4, f\}_+$ que sólo requiere

$$
F_{ij},\quad R^2F_{ij},\quad R^4F_{ij},\quad rF^{\prime}_{ij},
$$

donde la última es $rF^{\prime}_{ij} = \int u_i u_j\,r\,f^{\prime}(r)\,dr$.
La función `acomm_p4_f_closed(F, R2F, R4F, rFp, E_arr, b)` la calcula.

Para el sándwich $p^2 f p^2$, una expansión similar da
`p2_f_p2_closed(F, R2F, R4F, E_arr, b)`. **Nunca** aparece el producto
$P^2_{ij}\cdot F_{ij}$ de matrices truncadas — siempre evaluamos los
operadores en forma analítica usando la identidad HO.

### 6.4 El sándwich $(\vec p\!\cdot\!\vec r)\,f(r)\,(\vec r\!\cdot\!\vec p)$

Esta es la única integral de §4 que involucra UN solo factor de momento
en cada lado. Usando $\vec r\!\cdot\!\vec p = -i(r\partial_r) = -i(r\partial_r)$
sobre una onda parcial $u(r)/r$, se obtiene

$$
\bigl[\vec r\!\cdot\!\vec p\bigr]\,u_n(r) \;=\; -i\bigl[r\,u_n^{\prime}(r) - u_n(r)\bigr].
$$

Por hermiticidad,

$$
\boxed{\;
\langle u_i|(\vec p\!\cdot\!\vec r)\,f(r)\,(\vec r\!\cdot\!\vec p)|u_j\rangle
\;=\; \int_0^\infty
\bigl(r\,u_i^{\prime}-u_i\bigr)\bigl(r\,u_j^{\prime}-u_j\bigr)\,f(r)\,dr
\;}
$$

Una sola integral radial 1D. La función `pdotr_f_rdotp_radial(u, du, f, r, weights)`
la evalúa, con `du` calculada vía la recurrencia de Laguerre (§6.1).

### 6.5 La pieza tensor $T^{ab}\,p^a\,f(r)\,p^b$

Tras la reducción espín-angular descrita en `V_T_matrix_elements.md`, el
elemento de matriz $\langle n'L'1J|T^{ab}p^a f(r) p^b|nL1J\rangle$ se reduce
a una **única integral radial** sobre $u_{n'L'}^{*}$, $u_{nL}$, sus
derivadas $u^{\prime}$, $u^{\prime\prime}$, y los coeficientes radiales
$\mathcal A(r), \mathcal B(r), \mathcal C(r)$ que dependen de $f$ y $f^{\prime}$.
La función `K_K_tensor_radial(L, J, S, f, fp, u, du, r, weights)` empaqueta
esta integral, incluyendo el factor angular $\mathcal T_{L'L}^{(J)}$
(tabla en §4 de `V_T_matrix_elements.md`).

### 6.6 Resumen del flujo

Cada llamada a `MesonHamiltonian.matrix(E_T)` ejecuta:

1. Construye los bloques radiales $A_L(r), A_U(r)$ y combinaciones
   $X = A_L^2 A_U$, $g_L = V_v A_L^2$, etc., sobre la grilla uniforme.
2. Para cada función $f(r)$ relevante, calcula las integrales
   $F_{ij} = \int u_i u_j f\,dr$ por cuadratura de Simpson (vectorizada).
   Análogamente para $R^2F$, $R^4F$, $rF^{\prime}$.
3. Combina linealmente $F, R^2F, R^4F, rF^{\prime}$ con los autovalores
   $E_n^\text{HO}$ usando las fórmulas cerradas de §6.3 para obtener los
   $\{p^2,f\}_+, \{p^4,f\}_+, p^2 f p^2$, etc.
4. Evalúa el sándwich $(\vec p\!\cdot\!\vec r)f(\vec r\!\cdot\!\vec p)$
   y los tensores $T^{ab} p^a f p^b$ como integrales radiales con $u^{\prime}$
   analítica.
5. Suma todas las contribuciones según la fórmula maestra de §1 y devuelve
   la matriz $H(E_T)$ en la base HO.

**En ningún momento se hace un producto matricial entre operadores**. Esto
es decisivo: en una base truncada, un producto $P^2_{ij}\cdot F_{ij}$
generaría errores de truncado que escalan como $1/n_\text{states}^2$ y que
se acumulan en cada potencia de momento. Las fórmulas cerradas, en cambio,
son **exactas dentro de la base** truncada.

---

## 7. Aspectos numéricos

1. **Grilla radial uniforme**. `build_grid` produce $r_k$ uniforme en
   $(r_\text{max}/N_\text{grid},\,r_\text{max})$ con $r_\text{max}$
   determinado adaptativamente por el tamaño de la base. $N_\text{grid}=8000$
   por defecto. Aumentar $N_\text{grid}$ elimina artefactos de hiperfino
   producidos por escalas $r_s\!\sim\!r_\text{min}$ (esto es lo que
   diagnostica `diagnose_grid.py`).

2. **Cuadratura adaptativa (`--quadrature scipy_quad`)**. Reemplaza
   Simpson por `scipy.integrate.quad` con límite superior $\infty$
   (sin truncado real). Es $\sim 30$-$100\times$ más lento pero da $\sim 10^{-10}$
   de precisión absoluta; útil para validación.

3. **Derivadas analíticas de $V_X$**. Si la variante del potencial expone
   `V_X_prime` y `V_X_pp` analíticas, se pasan directamente. Si no,
   `deriv5` (diferencias centrales de cuarto orden) las genera desde la
   grilla. Las derivadas analíticas son recomendadas para $V_v$ con
   `erf` porque las series Taylor cerca de $r=0$ están implementadas
   con las cancelaciones exactas (ver docstrings de `meson_potential*.py`).

4. **Cache sobre $E_T$**. Sólo $A_L,A_U$ y los bloques que dependen
   de ellos se reconstruyen en cada llamada `MesonHamiltonian.matrix(E_T)`.
   El término diagonal $V_U$, las funciones de onda $u_n$, $u_n^{\prime}$,
   los pesos de Simpson y los autovalores HO se calculan una sola vez.

5. **Tamaño de la base**. `n_states` controla cuántos autovalores se
   conservan; internamente $N = n_\text{states}+20$ es el tamaño de la
   base usada en los productos. Para los estados altos ($n=3,4$ en
   $S$ o $P$) hay que usar $n_\text{states} \ge 20$ para que la cota
   variacional MacDonald esté convergida (con 15 estados, los autovalores
   de $\psi(4040)$ en adelante están sobre-estimados por decenas de MeV).

6. **Escala $b$ variacional por canal**. La función
   `salpeter_solver.find_variational_b_continuous` busca el $b^*$ óptimo
   por canal $(L,S,J)$ mediante `scipy.optimize.minimize_scalar`. El flag
   `--polish-variational` corre un LM extra al final del fit usando este
   $b^*$ por canal, recalibrando los parámetros para que sean consistentes
   con la base variacional.

---

## 8. Referencias internas

- Fórmula maestra: docstring de `H_full_matrix.py` (líneas 1–22).
- Ensamblado de bloques radiales: `build_radial_blocks` (línea 68).
- Pieza $K^\dagger K$: `K_dagK_matrix` (línea 246).
- Pieza $K_{LL}^\dagger(\cdot)K_{LL}$: `KLL_dagger_KLL_matrix` (línea 261).
- Pieza $W_s$: `W_s_matrix` (línea 327).
- Primitivas radiales y combinatores cerrados:
  `ho_primitives.py` (`acomm_p2_f_closed`, `p2_f_p2_closed`,
  `acomm_p4_f_closed`, `pdotr_f_rdotp_radial`, `K_K_tensor_radial`).
- Funciones de onda HO y derivadas analíticas: `ho_primitives.ho_radial_u`
  y `ho_radial_u_prime`.
- Solver de $E_T$ auto-consistente:
  `salpeter_solver.MesonHamiltonianSolver`.
- Búsqueda variacional de $b^*$: `salpeter_solver.find_variational_b_continuous`.
- Derivación detallada de la reducción espín-tensor:
  `V_T_matrix_elements.md`.
- Derivación manual de $W_s$: `work/W_s_calculation.tex`.
