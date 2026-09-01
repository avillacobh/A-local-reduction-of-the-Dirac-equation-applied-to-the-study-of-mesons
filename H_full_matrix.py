"""
Full effective Hamiltonian matrix in the truncated 3D-isotropic-HO basis with
definite L, total spin S in {0, 1}, and definite J in {|L-S|,...,L+S}.

    H = 2m + V_U + 2m * (K^dag K)
            + 2 * K_LL^dag (m + E_T/2 - V_U/2) K_LL,

where  A_X(r) = 1 / (m + E_T/2 - V_X(r)/2),  X in {L, U}, and

    (K^dag K) = {p^2, A_L^2}  +  nabla^2(A_L^2)
                + (l . sigma_T / r) (A_L^2)'

The long K_LL^dag(...) K_LL formula is implemented line-by-line below.

NO matrix products of operators (P^2 @ F, etc.) appear anywhere.  Every
{p^2,f}, {p^4,f}, p^2 f p^2 is computed via closed forms in `ho_primitives`
using the HO algebra; (p.r) f (r.p) is a single radial integral via
analytic u'.  Derivatives V_X', V_X'' may be passed analytically (recommended)
or computed by 4th-order finite differences (`deriv5`) from V_X.

Pure numpy (no scipy).
"""

import math
import numpy as np

from ho_primitives import (
    build_grid, simpson_weights, func_matrix,
    ho_E,
    pdotr_f_rdotp_radial,
    acomm_p2_f_closed, p2_f_p2_closed, acomm_p4_f_closed,
    spin_factors, K_K_tensor_radial,
    func_matrix_quad, HAS_SCIPY,
    ho_radial_u,
)


# ---------------------------------------------------------------------------
#  4th-order central finite-difference derivative on a uniform grid
# ---------------------------------------------------------------------------
def deriv5(y, dx):
    """5-point central difference (4th order) with lower-order endpoints."""
    n = len(y)
    d = np.empty_like(y)
    d[2:-2] = (-y[4:] + 8 * y[3:-1] - 8 * y[1:-3] + y[:-4]) / (12 * dx)
    d[0]  = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dx)
    d[1]  = (y[2] - y[0]) / (2 * dx)
    d[-2] = (y[-1] - y[-3]) / (2 * dx)
    d[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dx)
    return d


def _laplacian_radial(f, fp, r):
    """nabla^2 f for radial f(r): f'' + 2 f'/r."""
    return _ddx(fp, r) + 2.0 * fp / r if False else None     # never used directly


# ---------------------------------------------------------------------------
#  Build all radial multiplicative-function grids that enter the long formula
# ---------------------------------------------------------------------------
def build_radial_blocks(r, dr, V_L, V_U, m, E_T,
                        VLp=None, VLpp=None, VUp=None, VUpp=None,
                        V_v=None, VVp=None, VVpp=None):
    """
    Pre-compute every multiplicative function f(r) needed by the long
    K_LL^dag(...) K_LL formula and the simpler K^dag K piece.

    Returns a dict with keys (string -> ndarray on the grid).  Closed-form
    derivatives of A_L, A_U are obtained from V_L, V_U, V_L', V_L'', V_U', V_U''
    (analytic if supplied; 4th-order finite differences otherwise).
    """
    DL = m + 0.5 * E_T - 0.5 * V_L     # A_X = 1/D, D = m + E_T/2 - V_X/2
    DU = m + 0.5 * E_T - 0.5 * V_U
    if np.any(np.abs(DL) < 1e-12) or np.any(np.abs(DU) < 1e-12):
        raise ValueError("m + E_T/2 - V_X(r)/2 crosses zero on the grid: "
                         "A_X is singular.")
    AL = 1.0 / DL
    AU = 1.0 / DU

    VLp  = VLp  if VLp  is not None else deriv5(V_L, dr)
    VLpp = VLpp if VLpp is not None else deriv5(VLp,  dr)
    VUp  = VUp  if VUp  is not None else deriv5(V_U, dr)
    VUpp = VUpp if VUpp is not None else deriv5(VUp,  dr)

    # A_X' = -D'/D^2 = (V_X'/2) A_X^2
    AL_p = 0.5 * AL * AL * VLp
    AU_p = 0.5 * AU * AU * VUp

    # Functions that appear in the formula
    AL2          = AL * AL                           # A_L^2
    X            = AL2 * AU                          # A_L^2 A_U
    AU_pAL2      = AL2 * AU_p                        # A_L^2 A_U'
    ALp_sq_AU    = AL_p * AL_p * AU                  # (A_L')^2 A_U
    AL_ALp_AU    = AL * AL_p * AU                    # A_L A_L' A_U

    # Derivatives we need for {p^4, X} (closed form needs r * X')
    Xp = 2 * AL_ALp_AU + AU_pAL2                     # X'
    nab2_X = deriv5(Xp, dr) + 2.0 * Xp / r

    # nabla^4 X  (multiplicative function)
    nab2X_p = deriv5(nab2_X, dr)
    nab4_X  = deriv5(nab2X_p, dr) + 2.0 * nab2X_p / r

    # Auxiliary derivatives
    Y6a_p   = deriv5(AL_ALp_AU, dr)                  # (A_L A_L' A_U)'
    AL_p_AU_p = deriv5(AL_p * AU, dr)                # (A_L' A_U)'
    Y6b     = AL * AL_p_AU_p                          # A_L (A_L' A_U)'

    Xp_over_r        = Xp / r
    AU_pAL2_over_r   = AU_pAL2 / r
    AL_ALp_AU_over_r = AL_ALp_AU / r

    # Pieces of the (l.sigma_T) bracket
    half_one_over_r_dr_nab2X = 0.5 / r * nab2X_p
    nab2_Xp_or          = deriv5(Xp_over_r, dr)
    nab2_Xp_or          = deriv5(nab2_Xp_or, dr) + 2.0 * deriv5(Xp_over_r, dr) / r
    aux                 = AU_pAL2_over_r
    aux_p               = deriv5(aux, dr)
    nab2_AU_pAL2_or     = deriv5(aux_p, dr) + 2.0 * aux_p / r
    aux2                = AL_ALp_AU_over_r
    aux2_p              = deriv5(aux2, dr)
    nab2_AL_ALp_AU_or   = deriv5(aux2_p, dr) + 2.0 * aux2_p / r
    three_over_r_d_AL2_AUp_or = 3.0 / r * deriv5(AU_pAL2_over_r, dr)
    minus_one_over_r_d_AL_ALpAU_p = -1.0 / r * deriv5(Y6b, dr)
    inner               = AL * deriv5(AL_p * AU / r, dr)
    minus_d_inner       = -deriv5(inner, dr)

    # Y_8 for line 6 (the {(l.sig1),(l.sig2)} factor)
    Y8 = (1.0 / r) * (deriv5(AU_pAL2_over_r, dr) + ALp_sq_AU / (4 * r))

    # (A_L')^2 A_U / r^2 for line 5 (sandwich p.r ... r.p)
    Y5 = ALp_sq_AU / (r * r)

    # nabla^2 of Y6a_p, Y6b (line 3)
    Y6a_pp_grid     = deriv5(Y6a_p, dr)
    nab2_Y6a_p      = deriv5(Y6a_pp_grid, dr) + 2.0 * Y6a_pp_grid / r
    Y6b_p_grid      = deriv5(Y6b, dr)
    nab2_Y6b        = deriv5(Y6b_p_grid, dr) + 2.0 * Y6b_p_grid / r

    # The simpler K^dag K (first piece of H) needs nabla^2(A_L^2) and (A_L^2)'/r
    AL2_p           = 2 * AL * AL_p
    nab2_AL2        = deriv5(AL2_p, dr) + 2.0 * AL2_p / r
    AL2_pover_r     = AL2_p / r

    # Derivative of AU_pAL2_over_r is needed by the K-K tensor piece (line 8).
    # Pre-compute on the uniform grid so it can be spline-projected later.
    AU_pAL2_over_r_p = deriv5(AU_pAL2_over_r, dr)

    # ========================================================================
    #  W_s (spatial vector  W_s = alpha_1.alpha_2 V_v^s) building blocks.
    #  Activated only when V_v is supplied.  OGE consistency fixes the same
    #  radial function in the time and space components, V_v^s = s * V_v with
    #  s = ws_sign (default -1).  The caller passes V_v ALREADY scaled by s,
    #  so `V_v` below is V_v^s throughout this block.
    #  Decomposition of Eq. `spatial` in the appendix, projected onto |L,S,J>
    #  (see the docstring of W_s_matrix below, which is the authoritative form):
    #     hat W_s = -(1 - 2 kSS/3) [{p^2,F}   + e(r)]
    #             - (1 + 2 kSS/3) [{p^2,g_L} + nab2 g_L]
    #             - 2 T^{ab} p^a F p^b  +  2 T^{ab} p^a g_L p^b
    #             - 2 kLS (g_L'/r)
    #             + (T_LL / 3) h(r)
    #  with  f_U=V_v A_U,  F=V_v A_U A_L,  g_L=V_v A_L^2,
    #        e=A_L' f_U' + f_U nab2 A_L,
    #        h=A_L' f_U' + A_L (f_U'' - f_U'/r).
    # ========================================================================
    Ws_blocks = {}
    if V_v is not None:
        V_v = np.asarray(V_v, dtype=float)
        VVp_  = VVp  if VVp  is not None else deriv5(V_v, dr)
        VVpp_ = VVpp if VVpp is not None else deriv5(VVp_,  dr)

        # f_U = V_v * A_U   and analytic first derivative
        f_U   = V_v * AU
        f_U_p = VVp_ * AU + V_v * AU_p
        # second derivative of f_U (use deriv5 -- mixing analytic + numeric is fine)
        f_U_pp = deriv5(f_U_p, dr)

        # F = f_U * A_L  and g_L = V_v * A_L^2
        Ws_F   = f_U * AL
        Ws_F_p = deriv5(Ws_F, dr)
        Ws_gL  = V_v * AL * AL
        Ws_gL_p = deriv5(Ws_gL, dr)

        # nabla^2 g_L,   nabla^2 A_L
        Ws_nab2_gL = deriv5(Ws_gL_p, dr) + 2.0 * Ws_gL_p / r
        Ws_nab2_AL = deriv5(AL_p, dr) + 2.0 * AL_p / r

        # e(r), h(r)
        Ws_e = AL_p * f_U_p + f_U * Ws_nab2_AL
        Ws_h = AL_p * f_U_p + AL * (f_U_pp - f_U_p / r)

        # spin-orbit kernel: g_L'(r) / r
        Ws_gL_p_over_r = Ws_gL_p / r
        Ws_blocks = dict(
            Ws_F=Ws_F,           Ws_F_p=Ws_F_p,
            Ws_gL=Ws_gL,         Ws_gL_p=Ws_gL_p,
            Ws_gL_p_over_r=Ws_gL_p_over_r,
            Ws_nab2_gL=Ws_nab2_gL,
            Ws_e=Ws_e,           Ws_h=Ws_h,
            r2_Ws_F=r * r * Ws_F,
            r2_Ws_gL=r * r * Ws_gL,
        )

    out = dict(
        # core
        X=X, nab2_X=nab2_X, nab4_X=nab4_X,
        Y5=Y5, Y6a_p=Y6a_p, Y6b=Y6b,
        nab2_Y6a_p=nab2_Y6a_p, nab2_Y6b=nab2_Y6b,
        # bracket pieces of line 4-5
        Xp_over_r=Xp_over_r,
        AU_pAL2_over_r=AU_pAL2_over_r,
        AL_ALp_AU_over_r=AL_ALp_AU_over_r,
        half_one_over_r_dr_nab2X=half_one_over_r_dr_nab2X,
        nab2_Xp_or=nab2_Xp_or,
        nab2_AU_pAL2_or=nab2_AU_pAL2_or,
        nab2_AL_ALp_AU_or=nab2_AL_ALp_AU_or,
        three_over_r_d_AL2_AUp_or=three_over_r_d_AL2_AUp_or,
        minus_one_over_r_d_AL_ALpAU_p=minus_one_over_r_d_AL_ALpAU_p,
        minus_d_inner=minus_d_inner,
        Y8=Y8,
        # KdK (first piece of H)
        AL2=AL2, nab2_AL2=nab2_AL2, AL2_pover_r=AL2_pover_r,
        # derivative needed by K-K tensor (line 8)
        AU_pAL2_over_r_p=AU_pAL2_over_r_p,
        # for the closed-form combinators
        r2_AL2=r * r * AL2,
        r2_X=r * r * X,
        r4_X=r * r * r * r * X,
        rXp_X=r * Xp,
        r2_nab2_X=r * r * nab2_X,
        r2_Y6a_p=r * r * Y6a_p,
        r2_Y6b=r * r * Y6b,
        r2_Xp_over_r=r * r * Xp_over_r,
        r2_AU_pAL2_over_r=r * r * AU_pAL2_over_r,
        r2_AL_ALp_AU_over_r=r * r * AL_ALp_AU_over_r,
    )
    out.update(Ws_blocks)
    return out


# ---------------------------------------------------------------------------
#  K^dag K piece  (the simpler operator at the front of H)
# ---------------------------------------------------------------------------
def K_dagK_matrix(L, S, J, blocks_mat, E_arr, b, n_states):
    F   = blocks_mat["AL2_mat"]
    R2F = blocks_mat["r2_AL2_mat"]
    nab2F = blocks_mat["nab2_AL2_mat"]
    Fp_or = blocks_mat["AL2_pover_r_mat"]
    kappa_LS = spin_factors(L, S, J)["kappa_LS"]
    M = (acomm_p2_f_closed(F, R2F, E_arr, b)
         + nab2F
         + kappa_LS * Fp_or)
    return M[:n_states, :n_states]


# ---------------------------------------------------------------------------
#  K_LL^dag (m + E_T/2 - V_U/2) K_LL piece
# ---------------------------------------------------------------------------
def KLL_dagger_KLL_matrix(L, S, J, blocks, blocks_mat, E_arr, b,
                          u, du, r, weights, n_states):
    sf = spin_factors(L, S, J)
    kLS, kSS, kLL12 = sf["kappa_LS"], sf["kappa_SS"], sf["kappa_LL12"]

    F   = lambda k: blocks_mat[k + "_mat"]
    R2F = lambda k: blocks_mat["r2_" + k + "_mat"]
    AC2 = lambda k: acomm_p2_f_closed(F(k), R2F(k), E_arr, b)
    SW2 = lambda k: p2_f_p2_closed(F(k), R2F(k), blocks_mat["r4_" + k + "_mat"],
                                    E_arr, b)
    AC4 = lambda k: acomm_p4_f_closed(F(k), R2F(k), blocks_mat["r4_" + k + "_mat"],
                                       blocks_mat["rXp_" + k + "_mat"], E_arr, b)

    # ----- line 2 ---------------------------------------------------------
    line2 = (0.25 * AC4("X")
             + 0.25 * F("nab4_X")
             + 0.5  * SW2("X")
             + 0.5  * AC2("nab2_X")
             + 0.5  * pdotr_f_rdotp_radial(u, du, blocks["Y5"], r, weights))

    # ----- line 3 ---------------------------------------------------------
    line3 = -0.25 * (AC2("Y6a_p") + F("nab2_Y6a_p")
                     + AC2("Y6b")  + F("nab2_Y6b"))

    # ----- line 4-5  l.sigma_T bracket -----------------------------------
    bracket_LS = (
        AC2("Xp_over_r")
        + F("half_one_over_r_dr_nab2X")
        + 0.5 * F("nab2_Xp_or")
        + 0.5 * AC2("AU_pAL2_over_r")
        + 0.5 * F("nab2_AU_pAL2_or")
        + 0.5 * AC2("AL_ALp_AU_over_r")
        + 0.5 * F("nab2_AL_ALp_AU_or")
        + F("three_over_r_d_AL2_AUp_or")
        + F("minus_one_over_r_d_AL_ALpAU_p")
        + F("minus_d_inner")
    )
    line45 = 0.25 * kLS * bracket_LS

    # ----- line 6  {(l.sig1),(l.sig2)} multiplicative ---------------------
    line6 = 0.5 * kLL12 * F("Y8")

    # ----- line 7  sigma_1.sigma_2 [{p^2,Y} + nabla^2 Y] -----------------
    line7 = 0.5 * kSS * (AC2("AU_pAL2_over_r") + F("nab2_AU_pAL2_or"))

    # ----- line 8  -(1/2) [(sig.p) f (sig.p) + h.c.]  with f = A_L^2 A_U'/r
    # Method A canonical decomposition:
    #   (sig p) f (sig p) + h.c. = (1/3) sigma_1.sigma_2 [{p^2,f} + nabla^2 f]
    #                            + 2 T^{ab} p^a f p^b
    # No spin-orbit piece (sigma_1 sigma_2 cannot build a single-particle vector).
    f_grid  = blocks["AU_pAL2_over_r"]
    fp_grid = blocks["AU_pAL2_over_r_p"]    # pre-computed (uniform deriv5 +
                                            # spline projection if GL)
    line8_scalar = -(kSS / 6.0) * (AC2("AU_pAL2_over_r")
                                    + F("nab2_AU_pAL2_or"))
    line8_tensor = -0.5 * K_K_tensor_radial(L, J, S, f_grid, fp_grid,
                                             u, du, r, weights)
    line8 = line8_scalar + line8_tensor

    M = line2 + line3 + line45 + line6 + line7 + line8
    return M[:n_states, :n_states]


# ---------------------------------------------------------------------------
#  W_s = K^dag (alpha_1.alpha_2 V_v^s) K  contribution (spatial vector piece)
# ---------------------------------------------------------------------------
def W_s_matrix(L, S, J, blocks, blocks_mat, E_arr, b, u, du, r, weights, n_states):
    """
    W_s contribution to the reduced Hamiltonian, projected to |L,S,J>.

    Derivation (manual): starting from Eq. `spatial` of the appendix with
    V_v^s = V_v (OGE consistency), and using the per-particle identity
    sigma_i^a sigma_i^b = delta^{ab} + i epsilon_{abc} sigma_i^c on each of
    the (sigma_1.p)(sigma_1.sigma_2)(sigma_2.p) sandwiches.  Setting
        F_{VU}  = A_U * V_v       (== f_U)
        F_{VLL} = A_L^2 * V_v     (== g_L,  since [A_L, V_v] = 0)
    one finds, BEFORE projecting onto |L,S,J>,

        W_s = - (1 - sigma_1.sigma_2) { F_{VU}, p . A_L p }_+
              - (1/2)                  { F_{VU}, S_{A_L}    }_+
              - 2 (1 + sigma_1.sigma_2) p . F_{VLL} p
              - 2 (F_{VLL}'/r)  l . sigma_T
              +                  S_{F_{VLL}}

    with  S_f = (sigma_1.p) f (sigma_2.p) + (sigma_2.p) f (sigma_1.p)
    and   p . f p = (1/2) {p^2, f} + (1/2) nab^2 f.

    Projecting onto definite (L,S,J) using the canonical identity
        S_f -> (kSS/3)({p^2,f} + nab^2 f) + 2 T^{ab} p^a f p^b
    and  sigma_1.sigma_2 -> kSS,  l.sigma_T -> kLS,
    and using {F_{VU}, {p^2, A_L}} = 2{p^2, F} + 2 A_L' f_U'  (where F = f_U A_L),

        W_s | _{|L,S,J>} =
            - (1 - 2 kSS/3) [ {p^2, F}   + e(r) ]
            - (1 + 2 kSS/3) [ {p^2, g_L} + nab^2 g_L ]
            - 2 T^{ab} p^a F p^b
            + 2 T^{ab} p^a g_L p^b
            - 2 kLS (g_L'/r)
            + (T_LL / 3) h(r)

    with
        e(r) = A_L' f_U' + f_U nab^2 A_L
        h(r) = A_L' f_U' + A_L (f_U'' - f_U'/r)
        F    = f_U A_L,   g_L = V_v A_L^2,   f_U = V_v A_U.

    Note the *opposite* sign of the tensor F piece vs the g_L piece, and the
    new orbital-spin-orbit term -2 kLS (g_L'/r) absent from naive projection.

    Returns the n_states x n_states matrix in the HO basis.  If the W_s
    radial blocks are not present in `blocks` (V_v not supplied),
    returns zeros so this contribution is silently disabled.
    """
    if "Ws_F" not in blocks:
        return np.zeros((n_states, n_states))

    sf = spin_factors(L, S, J)
    kSS  = sf["kappa_SS"]
    kLS  = sf["kappa_LS"]
    T_LL = sf["T_LL"]

    F   = lambda k: blocks_mat[k + "_mat"]
    R2F = lambda k: blocks_mat["r2_" + k + "_mat"]
    AC2 = lambda k: acomm_p2_f_closed(F(k), R2F(k), E_arr, b)

    # Central pieces (no orbital tensor factor)
    cF  = -(1.0 - 2.0 * kSS / 3.0)         # coefficient on {p^2,F} + e
    cgL = -(1.0 + 2.0 * kSS / 3.0)         # coefficient on {p^2,g_L} + nab2 g_L
    central = (cF  * (AC2("Ws_F")  + F("Ws_e"))
             + cgL * (AC2("Ws_gL") + F("Ws_nab2_gL")))

    # Tensor pieces (each K_K_tensor_radial returns < 2 T^{ab} p^a f p^b >;
    # zero on singlet or L=0,S=1).  Note: opposite sign for F and g_L.
    tensor = (-1.0 * K_K_tensor_radial(L, J, S, blocks["Ws_F"],
                                        blocks["Ws_F_p"],
                                        u, du, r, weights)
              +1.0 * K_K_tensor_radial(L, J, S, blocks["Ws_gL"],
                                        blocks["Ws_gL_p"],
                                        u, du, r, weights))

    # Spin-orbit:  -2 kLS  (g_L'/r)
    spin_orbit = -2.0 * kLS * F("Ws_gL_p_over_r")

    # h(r) multiplicative correction:  +(T_LL / 3) h(r)
    # Origin: rank-2-tensor leftover of the anticommutator
    # {F_UV, T^{ab} p^a A_L p^b}_+ that the naive "arrow"
    # {F_UV, T^{ab} p^a A_L p^b}_+ -> 2 T^{ab} p^a F p^b ignores.
    # The exact identity is
    #   {F_UV, T^{ab} p^a A_L p^b}_+ = 2 T^{ab} p^a F p^b - (h/3) S_12(hat r),
    # with h(r) = A_L' f_U' + A_L (f_U'' - f_U'/r), derived from the commutator
    # [p^a, F_UV] = -i (F_UV'/r) x^a and using T^{ab} x^a x^b = (r^2/3) S_12.
    # After the overall -1 sign in -{F_UV, (1/2) S_{A_L}}_+, this enters
    # W_s|_{|L,S,J>} with sign + and coefficient T_LL/3 (NOT /6).
    # Verified with concrete test cases (A_L=1, F_UV=r and A_L=r^2, F_UV=r^2).
    h_piece = (T_LL / 3.0) * F("Ws_h")

    M = central + tensor + spin_orbit + h_piece
    return M[:n_states, :n_states]


# ---------------------------------------------------------------------------
#  MesonHamiltonian: caches every E_T-independent quantity
# ---------------------------------------------------------------------------
class MesonHamiltonian:
    """
    Cache the static parts of  H = 2m + V_U + 2m (K^dag K) + 2 K_LL^dag(...) K_LL
    so that scanning over E_T (e.g. for self-consistent root finding) only
    rebuilds the small E_T-dependent pieces (A_L, A_U and the multiplicative-
    function matrices that depend on them).

    QUADRATURE
    ----------
    Two integration schemes for the 1-D radial matrix elements
    int_0^inf u_i(r) u_j(r) f(r) dr, selected by `quadrature=`:

      * 'simpson' (default)
            composite Simpson 1/3 on a uniform truncated grid
            r in (r_max/N_grid, r_max).  Vectorised, fast.  Empirical
            accuracy with N_grid=4000 is ~1e-4 .. 1e-5; limited by the
            truncation tail at r_max.

      * 'scipy_quad'
            scipy.integrate.quad with infinite upper limit (np.inf), so the
            integration really IS from 0 to infinity with no truncation.
            Adaptive QUADPACK, accuracy ~1e-10 .. 1e-12 routinely.
            About 30-100x slower per H(E_T) call than Simpson (one quad call
            per matrix element per multiplicative function).  Recommended for
            accuracy spot-checks and final-result computations; use Simpson
            for parameter fits where each step needs to be fast.

    Quantities cached at construction (NOT dependent on E_T):
        r, dr, u, du, weights, E_arr, spin factors,
        V_L, V_L', V_L'', V_U, V_U', V_U'',  F_VU = <V_U>.

    Per-call cost (in `matrix(E_T)`): rebuild A-derived radial blocks and
    multiplicative-function matrices, then assemble.  Closed-form combinators
    produce matrices that are exactly symmetric in floating point, so no
    explicit symmetrisation is needed.
    """

    def __init__(self, L, S, J, V_L_func, V_U_func, m, b,
                 n_states=10, N=None, N_grid=4000, r_max=None,
                 V_L_prime=None, V_L_pp=None,
                 V_U_prime=None, V_U_pp=None,
                 V_v_func=None, V_v_prime=None, V_v_pp=None,
                 ws_style='full',
                 quadrature='simpson', quad_epsrel=1e-10):
        if N is None:
            N = n_states + 20
        if quadrature not in ('simpson', 'scipy_quad'):
            raise ValueError(f"quadrature must be 'simpson' or 'scipy_quad', "
                             f"got {quadrature!r}")
        if quadrature == 'scipy_quad' and not HAS_SCIPY:
            raise RuntimeError("quadrature='scipy_quad' requires scipy")

        self.L, self.S, self.J = L, S, J
        self.m, self.b = m, b
        self.n_states, self.N = n_states, N
        if ws_style != 'full':
            raise ValueError(f"ws_style must be 'full', got {ws_style!r}")
        self.ws_style = ws_style
        self.quadrature  = quadrature
        self.quad_epsrel = quad_epsrel

        # ----- E_T-independent: grid + wavefunctions ---------------------
        self.r, self.dr, self.u, self.du = build_grid(
            L, N, b, r_max=r_max, N_grid=N_grid)
        self.weights = simpson_weights(len(self.r), self.dr)

        # ----- E_T-independent: HO eigenvalues & spin factors ------------
        self.E_arr = ho_E(L, N, b)
        self.spin = spin_factors(L, S, J)

        # ----- E_T-independent: V_L, V_U and their derivatives -----------
        r = self.r
        self.V_L_func, self.V_U_func = V_L_func, V_U_func          # for scipy_quad
        self.V_L = np.asarray(V_L_func(r), float)
        self.V_U = np.asarray(V_U_func(r), float)
        self.V_L_prime, self.V_L_pp = V_L_prime, V_L_pp
        self.V_U_prime, self.V_U_pp = V_U_prime, V_U_pp
        self.VLp  = (np.asarray(V_L_prime(r), float)
                     if V_L_prime is not None else deriv5(self.V_L, self.dr))
        self.VLpp = (np.asarray(V_L_pp(r), float)
                     if V_L_pp is not None else deriv5(self.VLp, self.dr))
        self.VUp  = (np.asarray(V_U_prime(r), float)
                     if V_U_prime is not None else deriv5(self.V_U, self.dr))
        self.VUpp = (np.asarray(V_U_pp(r), float)
                     if V_U_pp is not None else deriv5(self.VUp, self.dr))

        # ----- E_T-independent: V_v (spatial vector for W_s) -----------------
        self.V_v_func = V_v_func
        if V_v_func is not None:
            self.V_v = np.asarray(V_v_func(r), float)
            self.VVp  = (np.asarray(V_v_prime(r), float)
                         if V_v_prime is not None else deriv5(self.V_v, self.dr))
            self.VVpp = (np.asarray(V_v_pp(r), float)
                         if V_v_pp is not None else deriv5(self.VVp, self.dr))
        else:
            self.V_v = None
            self.VVp = None
            self.VVpp = None

        # ----- u_n(r) as scalar-r callables (needed only by scipy_quad) ---
        if quadrature == 'scipy_quad':
            self._u_callables = [
                (lambda r_arg, n=n: float(ho_radial_u(n, L, np.array([r_arg]), b)[0]))
                for n in range(N)
            ]
        else:
            self._u_callables = None

        # ----- E_T-independent: V_U matrix in HO basis -------------------
        self.F_VU = self._fmat(self.V_U, V_U_func)[:n_states, :n_states]
        self.I_n  = np.eye(n_states)

    # -------- Internal: matrix element of u_i u_j f, two quadrature paths --
    def _fmat(self, f_grid, f_func):
        """Build  F[i,j] = int u_i u_j f dr  with the chosen quadrature.

        `f_grid` is f sampled on self.r (used by Simpson);
        `f_func` is a callable f(r) (used by scipy.quad over (0, inf))."""
        if self.quadrature == 'simpson':
            return func_matrix(self.u, f_grid, self.weights)
        else:                                  # scipy_quad
            if f_func is None:
                raise RuntimeError("scipy_quad needs a callable for f(r); "
                                   "a pre-sampled grid is not enough.")
            return func_matrix_quad(self._u_callables, f_func, self.N,
                                    epsabs=self.quad_epsrel * 1e-2,
                                    epsrel=self.quad_epsrel)

    # ---------------- E_T-dependent: build H(E_T) -----------------------
    def matrix(self, E_T):
        blocks = build_radial_blocks(
            self.r, self.dr, self.V_L, self.V_U, self.m, E_T,
            self.VLp, self.VLpp, self.VUp, self.VUpp,
            V_v=self.V_v, VVp=self.VVp, VVpp=self.VVpp)

        if self.quadrature == 'simpson':
            blocks_mat = {k + "_mat": func_matrix(self.u, blocks[k], self.weights)
                          for k in blocks}
        else:                                  # scipy_quad
            # Each multiplicative block is built on self.r; for adaptive
            # quad-from-0-to-inf we need a callable.  Wrap each block with
            # a cubic-spline interpolant so scipy.quad can sample at any r.
            from scipy.interpolate import CubicSpline as _CS
            blocks_mat = {}
            for k in blocks:
                interp_k = _CS(self.r, blocks[k], extrapolate=True)
                blocks_mat[k + "_mat"] = func_matrix_quad(
                    self._u_callables,
                    lambda r, fk=interp_k: float(fk(r)),
                    self.N, epsabs=self.quad_epsrel * 1e-2,
                    epsrel=self.quad_epsrel)

        H_KdK = K_dagK_matrix(self.L, self.S, self.J, blocks_mat,
                              self.E_arr, self.b, self.n_states)
        H_LL  = KLL_dagger_KLL_matrix(self.L, self.S, self.J,
                                       blocks, blocks_mat,
                                       self.E_arr, self.b,
                                       self.u, self.du, self.r, self.weights,
                                       self.n_states)
        if self.V_v is None:
            H_Ws = np.zeros((self.n_states, self.n_states))
        else:
            H_Ws = W_s_matrix(self.L, self.S, self.J,
                               blocks, blocks_mat,
                               self.E_arr, self.b,
                               self.u, self.du, self.r, self.weights,
                               self.n_states)
        # Closed forms produce exactly symmetric matrices -- no need to
        # average with the transpose.
        return (2 * self.m * self.I_n + self.F_VU + 2 * self.m * H_KdK
                + 2 * H_LL + H_Ws)

    # ---------------- Hermitian diagonalisation -------------------------
    def eigvals(self, E_T):
        """Sorted ascending eigenvalues (uses LAPACK DSYEVR via numpy.eigvalsh)."""
        return np.linalg.eigvalsh(self.matrix(E_T))

    def eigh(self, E_T):
        """(eigvals, eigvecs), sorted ascending."""
        return np.linalg.eigh(self.matrix(E_T))


# ---------------------------------------------------------------------------
#  One-shot wrapper for backward compatibility
# ---------------------------------------------------------------------------
def H_matrix(L, S, J, V_L_func, V_U_func, m, E_T, b,
             n_states=10, N=None, N_grid=4000, r_max=None,
             V_L_prime=None, V_L_pp=None, V_U_prime=None, V_U_pp=None,
             quadrature='simpson', quad_epsrel=1e-10):
    """Convenience wrapper: build a one-shot MesonHamiltonian and return H(E_T).

    `quadrature` is 'simpson' (fast, ~1e-4) or 'scipy_quad' (slower,
    ~1e-10; uses scipy.integrate.quad on (0, infinity))."""
    return MesonHamiltonian(L, S, J, V_L_func, V_U_func, m, b,
                            n_states=n_states, N=N, N_grid=N_grid, r_max=r_max,
                            V_L_prime=V_L_prime, V_L_pp=V_L_pp,
                            V_U_prime=V_U_prime, V_U_pp=V_U_pp,
                            quadrature=quadrature,
                            quad_epsrel=quad_epsrel).matrix(E_T)


# ---------------------------------------------------------------------------
#  Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Cornell potential as a quick smoke test
    kC, sC = 0.40, 0.18
    V    = lambda r: -kC / r + sC * r
    Vp   = lambda r:  kC / r**2 + sC
    Vpp  = lambda r: -2 * kC / r**3
    H = H_matrix(L=0, S=1, J=1, V_L_func=V, V_U_func=V,
                 V_L_prime=Vp, V_L_pp=Vpp, V_U_prime=Vp, V_U_pp=Vpp,
                 m=1.27, E_T=3.0, b=1.0, n_states=10)
    print("L=0  S=1  J=1  (^3S_1)  matrix elements (5x5):")
    print(np.array2string(H[:5, :5], precision=4, suppress_small=True))
    print("\nSymmetric (max |H - H^T|):", float(np.max(np.abs(H - H.T))))
