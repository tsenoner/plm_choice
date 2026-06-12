"""Evidence for the cross-pLM W1 decision: a pair-level two-sample null is severely
anti-conservative under dyadic dependence.

This is the committed, reproducible version of the simulation behind §3.3 of
docs/PLM_CHOICE_ANALYSIS_HANDOFF.md. It demonstrates why the cross-pLM arm reports
Wasserstein-1 (W1) as a DESCRIPTIVE distance + vertex-BCa CI only, with NO permutation
p-value and OUTSIDE the Holm correction.

The candidate null (rejected): a pair-level sign-flip two-sample null. For each upper-triangle
pair, swap (da_ij, db_ij) with probability 1/2, recompute W1(group1, group2), repeat -> a null
distribution; p = (1 + #{null >= obs}) / (n_perm + 1).

The problem: a distance matrix's m = n(n-1)/2 upper-triangle entries are derived from only n
proteins, so they are DYADICALLY DEPENDENT (pairs share vertices). The sign-flip permutes in
pair-space (m units) while the data's randomness lives in protein-space (n units). The null
distribution is far too narrow -> the observed statistic looks extreme almost always -> the
test rejects a true null far too often, and worsens as n grows.

A valid test should reject ~5% of the time under H0. Expected output (seeded):

    R0 i.i.d. pairs, NO dyadic structure  :  type-I @0.05 ~ 0.05   (CONTROL: the null is
                                             correctly calibrated when pairs are independent,
                                             proving the implementation is right and isolating
                                             dyadic dependence as the cause of R1-R4's failure)
    R1 shared-x eps=0.15 (near-identical) :  type-I @0.05 ~ 0.64
    R2 shared-x eps=0.60 (separated)      :  type-I @0.05 ~ 0.68
    R3 INDEP same-law n=40 (realistic)    :  type-I @0.05 ~ 0.72
    R4 INDEP same-law n=80 (realistic)    :  type-I @0.05 ~ 0.81   (worsens with n)
    PWR true alternative (H1)             :  reject ~ 1.00         (it does have power -
                                             but useless when type-I is 0.7)

The contrast R0 (~0.05) vs R1-R4 (0.64-0.81) is the whole argument: the null is sound on
independent pairs and fails ONLY under the dyadic dependence that real distance matrices always
have. This is not a coding artifact; it is intrinsic to permuting in pair-space.

A vertex-level null (which would inherit the dyadic structure) has no clean construction for
shared-vertex matrices, so W1 gets no p-value. Its dyadically-correct uncertainty is the
vertex-BCa CI already reported per cell.

Run: <repo>/.venv/bin/python scripts/cross_plm_w1_null_simulation.py
(needs only numpy + scipy)
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance as W1


def signflip_p(da, db, n_perm, rng):
    """Pair-level sign-flip two-sample p-value for W1(da, db) (the REJECTED null)."""
    m = da.size
    obs = W1(da, db)
    cnt = 0
    for _ in range(n_perm):
        f = rng.integers(0, 2, size=m).astype(bool)
        g1 = np.where(f, db, da)
        g2 = np.where(f, da, db)
        if W1(g1, g2) >= obs:
            cnt += 1
    return (1 + cnt) / (n_perm + 1), obs


def run(label, gen, reps, n_perm, seed):
    rng = np.random.default_rng(seed)
    ps, obss = [], []
    for _ in range(reps):
        da, db = gen(rng)
        p, obs = signflip_p(da, db, n_perm, rng)
        ps.append(p)
        obss.append(obs)
    ps = np.array(ps)
    obss = np.array(obss)
    print(f"{label}")
    print(f"   reject @0.05={(ps<=0.05).mean():.3f}  @0.10={(ps<=0.10).mean():.3f}  "
          f"@0.01={(ps<=0.01).mean():.3f}   median obs W1={np.median(obss):.4f}")


# R0 is the CONTROL: da, db are independent gaussian VECTORS (NOT pairwise distances), so the
# m "pairs" are genuinely independent -- no dyadic structure. H0 true (same marginal law). The
# null is correctly calibrated here (~0.05), which proves the failure in R1-R4 is dyadic
# dependence, not a bug. m is matched to R3's m=780 (n=40 -> 780 pairs).
def r0(rng, m=780):
    return rng.normal(size=m), rng.normal(size=m)


# All distance matrices below are over n vertices, so the m=n(n-1)/2 pairs are dyadically
# dependent within each matrix. R1-R4 are H0-TRUE (A and B have equal marginal laws).
def r1(rng, n=40, eps=0.15):  # shared geometry, near-identical pLMs (W1 ~ 0)
    x = rng.normal(size=(n, 3))
    return pdist(x + eps * rng.normal(size=(n, 3))), pdist(x + eps * rng.normal(size=(n, 3)))


def r2(rng, n=40, eps=0.6):  # shared geometry, more separated
    x = rng.normal(size=(n, 3))
    return pdist(x + eps * rng.normal(size=(n, 3))), pdist(x + eps * rng.normal(size=(n, 3)))


def r3(rng, n=40):  # INDEPENDENT same-law: two genuinely different pLMs, equal marginal law
    return pdist(rng.normal(size=(n, 3))), pdist(rng.normal(size=(n, 3)))


def r4(rng, n=80):  # INDEPENDENT same-law, larger n (anti-conservatism worsens)
    return pdist(rng.normal(size=(n, 3))), pdist(rng.normal(size=(n, 3)))


def pwr(rng, n=40):  # H1-TRUE: different marginal distance distributions (power check)
    return pdist(rng.normal(size=(n, 3))), pdist(rng.normal(size=(n, 8)) * 0.5)


if __name__ == "__main__":
    reps, n_perm = 400, 300
    print("=== W1 sign-flip null: type-I (R0-R4 are H0-TRUE) and power (PWR is H1-TRUE) ===")
    print("(a valid test rejects ~0.05 under H0; R0 control ~0.05, but R1-R4 dyadic -> 0.64-0.81)\n")
    run("R0 i.i.d. pairs, NO dyadic structure (CONTROL)", r0, reps, n_perm, 9)
    run("R1 shared-x eps=0.15 (near-identical)        ", r1, reps, n_perm, 0)
    run("R2 shared-x eps=0.60 (more separated)        ", r2, reps, n_perm, 1)
    run("R3 INDEP same-law n=40 (realistic null)      ", r3, reps, n_perm, 2)
    run("R4 INDEP same-law n=80 (realistic null, big n)", r4, reps, n_perm, 3)
    run("PWR true alternative n=40 (H1, want HIGH)     ", pwr, reps, n_perm, 4)
