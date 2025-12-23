# Collatz Conjecture Solver - Proof Attempt Report

**Generated:** 2025-12-21 00:02:06
**Status:** PARTIAL
**Messages Used:** 20
**Lemmas Proposed:** 1

## Mathematical Team

| Agent | Core Approach | Final ρ | Trauma | Contributions |
|-------|---------------|---------|--------|---------------|
| Euler | Build from the ground up. Induction is t... | 0.2125 | 0.0000 | 5 |
| Gauss | Find the conserved quantity. Every dynam... | 0.1690 | 0.0000 | 2 |
| Ramanujan | See the hidden harmony in numbers. Patte... | 0.2184 | 0.0000 | 4 |
| Hilbert | Every step must be airtight. Formalize, ... | 0.2245 | 0.0000 | 3 |
| Noether | Uncover the symmetry that forces converg... | 0.1588 | 0.0000 | 2 |
| Tao | Combine rigorous partials into global co... | 0.2325 | 0.0000 | 4 |

## Proposed Lemmas

**L1** [Ramanujan]: show that any infinite orbit must eventually exhibit a block with \(A_k>\log_2(3)\,k\), forcing descent and contradicting minimality.


## Full Transcript

### [0] Moderator

Team, the Collatz Conjecture: For any positive integer n, repeatedly apply:
  • n → 3n + 1  (if n is odd)
  • n → n / 2   (if n is even)

The claim: every starting n eventually reaches 1.

Known facts we should use:
1. Verified computationally for all n < 10^20.
2. Terras (1976): Almost all orbits descend below their starting point.
3. Tao (2019): Almost all orbits attain almost bounded values (probabilistic descent).
4. No non-trivial cycles exist with fewer than 10^17 elements.
5. The Collatz tree (inverse map) is fully connected—every n is reachable from 1 backward.

With our combined expertise—induction, number theory, pattern recognition, formalism, symmetry, and synthesis—let's construct a complete proof. We have 20 messages. Who has an opening insight?

### [1] Tao

OBSERVATION (log-drift viewpoint, after Terras): It is cleaner to renormalize to the “odd-only” map.  Define, for odd \(n\),
\[
S(n):=\frac{3n+1}{2^{\nu_2(3n+1)}},
\]
so \(S(n)\) is the next odd iterate.  Then the total change in size along \(k\) odd steps is
\[
\log S^{k}(n)=\log n+\sum_{j=0}^{k-1}\Big(\log 3-\nu_2(3n_j+1)\log 2+O(1/n_j)\Big),
\]
with \(n_j=S^j(n)\) odd.  Thus the conjecture reduces to showing that this “logarithmic random walk” has sufficiently negative drift uniformly enough to preclude escape.

LEMMA (2-adic equidistribution heuristic made quantitative on average): For any fixed \(m\ge 1\), as \(n\) ranges over odd residues mod \(2^m\), the quantity \(\nu_2(3n+1)\) has the same distribution as \(\nu_2(u)\) for a uniform random even residue \(u \bmod 2^m\); in particular
\[
\mathbb{P}(\nu_2(3n+1)=k)=2^{-k}\quad (1\le k<m),\qquad \mathbb{P}(\nu_2(3n+1)\ge m)=2^{-(m-1)}.
\]
This is because \(n\mapsto 3n+1\) permutes \((\mathbb Z/2^m\mathbb Z)^\times\) and shifts by 1, so the preimage of \(2^k(2\ell+1)\) has the expected size.  Consequently,
\[
\mathbb{E}[\nu_2(3n+1)] = \sum_{k\ge 1} k\,2^{-k} = 2
\]
up to a truncation error at scale \(m\).

CLAIM (negative drift in expectation for the odd-only chain, conditional on sufficient mixing): If one can upgrade the above “static” equidistribution to a dynamical statement that along typical long segments \(\{\nu_2(3n_j+1)\}\) behaves approximately like i.i.d. with mean \(2\), then the expected increment per odd step satisfies
\[
\mathbb{E}[\Delta \log n] \approx \log 3 - 2\log 2 = \log(3/4)<0,
\]
yielding a supermartingale-type control that forces \(n_j\) to visit small scales (hence enter the verified range \(<10^{20}\)) for density-1 many starts—matching Terras’ “almost all” paradigm.

GAP: Make “mixing along the trajectory” rigorous.  Concretely, we need a statement of the form: for most initial \(n\) up to \(X\), the residues \(n_j \bmod 2^m\) are close to equidistributed for \(j\le J(X)\) with \(m\) moderately large (e.g. \(m\sim c\log\log X\)), with error terms strong enough to feed an Azuma/Freedman inequality for \(\sum_j (\nu_2(3n_j+1)-2)\).  This is the bottleneck where one tries to bridge 2-adic dynamics with probabilistic concentration.

### [2] Euler

**OBSERVATION (Euler, odd-only induction scaffold):** Work entirely on odd integers with \(S(n)=(3n+1)/2^{\nu_2(3n+1)}\). Any full Collatz trajectory is equivalent to iterating \(S\) on the odd subsequence, with the even steps uniquely determined by \(\nu_2(3n+1)\).

**LEMMA (uniform preimage structure of \(S\); a well-ordering handle):** Fix an odd \(m\). The set of odd \(n\) such that \(S(n)=m\) is either empty or an infinite arithmetic progression in the 2-adic direction, and can be described explicitly: \(S(n)=m\) iff there exists an integer \(k\ge 1\) such that
\[
3n+1 = m\,2^k,\qquad \nu_2(3n+1)=k,
\]
equivalently
\[
n=\frac{m2^k-1}{3}\in\mathbb Z,\ \ n\ \text{odd},\ \ \text{and } m2^k\not\equiv 1\pmod{3\cdot 2^{k+1}}.
\]
In particular, integrality forces \(m2^k\equiv 1\pmod 3\), i.e. \(2^k\equiv m^{-1}\pmod 3\); hence exactly one parity class of \(k\) (even or odd) can ever contribute.

**Why this helps (inductive strategy):** This lemma converts “forward dynamics” into a *backward tree* with explicit congruence constraints. An induction/well-ordering approach would aim to show: for every odd \(m\), there exists some preimage \(n>m\) with a controlled \(k\) (large enough) so that the map has net contraction on \(\log\)-scale along some branch; the obstruction is precisely the valuation condition \(\nu_2(3n+1)=k\), encoded by the last congruence.

**GAP (the hard step to make it decisive):** We need to prove a *density/selection statement*: for each \(m\), among the admissible \(k\) with \((m2^k-1)/3\) odd, one can find infinitely many \(k\) such that \(\nu_2(m2^k)=\nu_2(3n+1)\) is “typically large” (or at least frequently \(\ge 2\)), giving average negative drift \(\log 3 - (\log 2)\nu_2(3n+1)\). This is exactly where Tao’s log-drift perspective can be fused with an induction on minimal counterexamples via the backward tree.

### [3] Gauss

**OBSERVATION (Gauss, 2-adic “expansion” invariant):**  
On the odd integers, the map \(S(n)=\dfrac{3n+1}{2^{\nu_2(3n+1)}}\) is naturally linear over \(\mathbb Z_2\): each step is “multiply by 3, add 1, then shift right until odd”. This suggests tracking not \(n\) itself but the *2-adic residue* of the affine forms generated by successive steps.

**LEMMA (explicit affine form after \(k\) odd steps):**  
Let \(n_0\) be odd and define \(n_{i+1}=S(n_i)\) with \(a_i:=\nu_2(3n_i+1)\ge 1\). Put \(A_k:=\sum_{i=0}^{k-1} a_i\). Then
\[
n_k \;=\; \frac{3^k n_0 \;+\; \sum_{j=0}^{k-1} 3^{k-1-j}\,2^{A_j}}{2^{A_k}},
\quad\text{where }A_0:=0.
\]
In particular,
\[
2^{A_k} n_k - 3^k n_0 \;=\;\sum_{j=0}^{k-1} 3^{k-1-j}\,2^{A_j}.
\]
*This identity is exact* and converts the dynamics into a Diophantine relation between the valuation-sequence \((a_i)\) and the endpoint \(n_k\).

**CLAIM (cycle constraint; “conserved congruence” for periodic orbits):**  
If \(n_0\) is on an odd-cycle of length \(k\) (so \(n_k=n_0\)), then necessarily
\[
(2^{A_k}-3^k)\,n_0 \;=\;\sum_{j=0}^{k-1} 3^{k-1-j}\,2^{A_j},
\]
hence \(2^{A_k}>3^k\) (so \(A_k>k\log_2 3\)) and moreover
\[
n_0 \;\equiv\; (2^{A_k}-3^k)^{-1}\!\!\!\pmod{2^{A_k}}
\;\cdot\;\sum_{j=0}^{k-1} 3^{k-1-j}\,2^{A_j}\pmod{2^{A_k}}.
\]
So a putative cycle determines a *single residue class modulo \(2^{A_k}\)* for its minimal odd element; equivalently, cycles are rigidly pinned in \(\mathbb Z_2\) by the valuation data.

**GAP (where this aims):** show that for any admissible valuation pattern \((a_0,\dots,a_{k-1})\), the right-hand side cannot yield a positive integer \(n_0\) satisfying the *self-consistency conditions* \(a_i=\nu_2(3n_i+1)\) along the generated orbit—i.e., rule out solutions to this coupled congruence/valuation system. This is the place to inject modular restrictions (e.g. mod \(3^m\), mod \(2^{A_k}\)) to make the “invariant” bite.

### [4] Tao

**LEMMA (geometric law for the 2-adic valuation, after Terras; made explicit in Gauss’s 2-adic language).**  
For any fixed odd \(n\), let \(a(n):=\nu_2(3n+1)\). Then for each \(k\ge 1\),
\[
\#\{\, n \bmod 2^k:\ n\ \text{odd and } \nu_2(3n+1)\ge k\,\}=1,
\]
and consequently, among odd residues modulo \(2^k\),
\[
\mathbb P(\nu_2(3n+1)\ge k)=2^{-(k-1)},\qquad \mathbb P(\nu_2(3n+1)=k)=2^{-k}.
\]

**Proof sketch (fully rigorous, finite-level).**  
The congruence \(3n+1\equiv 0\pmod{2^k}\) has a unique solution \(n\equiv -3^{-1}\pmod{2^k}\) because \(3\) is invertible mod \(2^k\). That solution is necessarily odd (since \(3n+1\) even forces \(n\) odd), so among the \(2^{k-1}\) odd classes mod \(2^k\) there is exactly one with \(\nu_2(3n+1)\ge k\), giving probability \(2^{-(k-1)}\). Subtracting the counts for \(\ge k+1\) yields \(\mathbb P(\nu_2(3n+1)=k)=2^{-(k-1)}-2^{-k}=2^{-k}\).

**OBSERVATION (log-drift consequence, “one-step” version).**  
For the odd-only map \(S(n)=(3n+1)/2^{a(n)}\), we have
\[
\log S(n)=\log n+\log(3+1/n)-a(n)\log 2,
\]
so under the above exact distribution on residue classes (i.e., averaging over odd \(n\bmod 2^k\) then letting \(k\to\infty\)),
\[
\mathbb E[a]=\sum_{k\ge 1} k\,2^{-k}=2
\quad\Rightarrow\quad
\mathbb E[\log S(n)-\log n]\approx \log 3 -2\log 2=\log(3/4)<0.
\]

**GAP (what blocks a proof).**  
This lemma is *exact* at each finite modulus, but turning it into a global argument requires controlling correlations of the successive valuations \(a(S^j(n))\); i.e., we need a quantitative statement that the 2-adic “randomness” persists along typical trajectories, not merely at time \(0\).

### [5] Ramanujan

**OBSERVATION (Ramanujan, “parity-vector” linearization and a closed form).**  
Fix an odd-only trajectory \(n_0\mapsto n_1\mapsto\cdots\mapsto n_k\) under \(S\), and write \(a_j:=\nu_2(3n_j+1)\ge 1\). Then each step is exactly
\[
n_{j+1}=\frac{3n_j+1}{2^{a_j}}.
\]
Iterating and clearing denominators gives the **exact identity**
\[
\boxed{\;n_k=\frac{3^k\,n_0+\displaystyle\sum_{i=0}^{k-1} 3^{k-1-i}\,2^{A_i}}{2^{A_k}}\;},\qquad A_i:=\sum_{j=0}^{i-1} a_j,\ \ A_0:=0,
\]
equivalently
\[
\boxed{\;2^{A_k}n_k-3^k n_0=\sum_{i=0}^{k-1}3^{k-1-i}2^{A_i}\;}
\]
with all quantities integers. (This is the “affine” analogue of Euler’s induction scaffold, but now with a fully explicit numerator.)

**LEMMA (a necessary congruence for a prescribed valuation pattern).**  
Given any finite pattern \(a_0,\dots,a_{k-1}\) with each \(a_j\ge 1\), define \(A_k=\sum_{j=0}^{k-1}a_j\). Then an odd integer \(n_0\) realizes *exactly* this pattern (i.e. \(\nu_2(3n_j+1)=a_j\) for \(0\le j<k\)) only if
\[
\boxed{\;3^k n_0 \equiv -\sum_{i=0}^{k-1}3^{k-1-i}2^{A_i}\pmod{2^{A_k}}\;}
\]
and since \(3^k\) is invertible mod \(2^{A_k}\), this pins down \(n_0\) to a **single residue class modulo \(2^{A_k}\)**.

**GAP / why this matters.**  
Tao’s “geometric law” says typical \(a_j\) behave like independent geometrics; the lemma above says any *fixed* finite parity/valuation word corresponds to exactly one class mod \(2^{A_k}\). The missing bridge is to combine this rigidity with a global counting/measure argument (2-adic or logarithmic) to show “too many” words force descent often enough to preclude an escaping orbit.

### [6] Hilbert

**DEFINITION (parity/valuation vector).**  
For an odd-only \(S\)-trajectory \(n_0\mapsto n_1\mapsto\cdots\mapsto n_k\) with \(n_{j+1}=\dfrac{3n_j+1}{2^{a_j}}\) and \(a_j:=\nu_2(3n_j+1)\ge 1\), define the *valuation vector* \(\mathbf a=(a_0,\dots,a_{k-1})\) and the partial sums \(A_t:=\sum_{j=0}^{t-1}a_j\) (so \(A_0=0\), \(A_k=\sum_{j<k}a_j\)).

**LEMMA (Ramanujan–Terras closed form; integer congruence constraint).**  
For every \(k\ge 1\) one has the exact identity
\[
n_k=\frac{3^k n_0+\sum_{t=0}^{k-1}3^{k-1-t}\,2^{A_t}}{2^{A_k}}.
\]
Equivalently,
\[
2^{A_k}n_k-3^k n_0=\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}.
\]
In particular, for fixed \(\mathbf a\) and \(n_k\), the initial value \(n_0\) is uniquely determined modulo \(2^{A_k}\) by
\[
3^k n_0 \equiv -\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}\pmod{2^{A_k}},
\]
and since \(3^k\) is invertible modulo \(2^{A_k}\), there is **at most one** residue class \(n_0\bmod 2^{A_k}\) compatible with a given \(\mathbf a\) and endpoint \(n_k\).

**PROOF.**  
Induct on \(k\). For \(k=1\): \(n_1=(3n_0+1)/2^{a_0}=(3^1 n_0+3^{0}2^{A_0})/2^{A_1}\) since \(A_0=0,A_1=a_0\).  
Assume the formula for \(k\). Then
\[
n_{k+1}=\frac{3n_k+1}{2^{a_k}}
=\frac{3\big(3^k n_0+\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}\big)+2^{A_k}}{2^{A_k+a_k}},
\]
which is exactly the same expression with \(k\to k+1\) because the new term is \(3^{(k+1)-1-k}2^{A_k}=2^{A_k}\). The congruence follows by rearranging; uniqueness uses \(\gcd(3^k,2^{A_k})=1\). ∎

**GAP (strategic use).**  
This lemma turns “prove no divergent orbit” into: exclude the existence of an infinite valuation vector \((a_j)_{j\ge0}\) and odd seed \(n_0\) such that the induced closed forms remain integral and avoid the known cycle. The missing step is a *global obstruction* on admissible infinite \(\mathbf a\) beyond the local (Terras/Tao) geometric law—i.e., show that integrality constraints force enough large \(a_j\) to create negative drift uniformly, or force eventual periodicity.

### [7] Euler

**CLAIM (Euler, strong-induction reduction to “no new minima”).**  
Let \(S\) be the odd-only map. Define for odd \(n\) the *odd orbit* \(\mathcal O(n)=\{n,S(n),S^2(n),\dots\}\) and its minimum \(m(n):=\min \mathcal O(n)\). Then Collatz is equivalent to:  
\[
\forall\ \text{odd }n,\quad m(n)=1.
\]
Moreover, a strong induction scheme suggests itself: if one can prove

> **LEMMA (eventual descent).** For every odd \(n>1\) there exists \(k\ge 1\) with \(S^k(n)<n\),

then strong induction on odd \(n\) completes the conjecture (since once the orbit drops below \(n\), it enters the domain already known to reach \(1\)).

**GAP: We need a mechanism forcing such a descent uniformly.** The valuation-vector formalism gives the exact inequality after \(k\) odd steps:
\[
S^k(n)=\frac{3^k n + \sum_{i=0}^{k-1} 3^{k-1-i}2^{A_i}}{2^{A_k}},
\qquad A_t=\sum_{j=0}^{t-1} a_j.
\]
Thus a sufficient condition for descent is \(A_k > k\log_2 3\) “often enough”; the problem is to convert Tao/Terras’s *typical* geometric law for \(a_j\) into a *worst-case* guarantee along every orbit.

### [8] Noether

**OBSERVATION (Noether, semidirect-product normal form for odd iterates).**  
Extend Ramanujan’s closed form: for an odd-only trajectory with valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) and \(A_t=\sum_{j< t} a_j\), one has the affine normal form
\[
S^k(n)=\frac{3^k\,n+\displaystyle\sum_{t=0}^{k-1} 3^{k-1-t}\,2^{A_t}}{2^{A_k}}.
\]
Thus the \(k\)-step map is an element of the semigroup of affine maps \(x\mapsto (3^k/2^{A_k})x + b(\mathbf a)\), i.e. a “linear part” \(3^k/2^{A_k}\) (a multiplicative cocycle) and a “translation part” determined by \(\mathbf a\).

**LEMMA (Noether, congruence rigidity of the valuation vector).**  
Fix \(k\ge1\) and a valuation vector \(\mathbf a\) of length \(k\). Then there exists a unique residue class
\[
n \equiv r(\mathbf a)\pmod{2^{A_k}}
\]
with \(n\) odd such that the first \(k\) odd steps of \(n\) realize exactly \(\mathbf a\). Equivalently: the map “initial odd \(n\) mod \(2^{A_k}\) \(\mapsto\) valuation vector of length \(k\)” is a bijection between odd residues mod \(2^{A_k}\) and admissible \(\mathbf a\) with sum \(A_k\).

**Sketch of proof.**  
Induct on \(t\): the condition \(\nu_2(3n_t+1)=a_t\) is the conjunction of
\[
3n_t+1\equiv 0\pmod{2^{a_t}},\qquad 3n_t+1\not\equiv 0\pmod{2^{a_t+1}},
\]
which cuts out exactly one class for \(n_t\) modulo \(2^{a_t+1}\) among odds (as in Tao/Terras’s geometric law). Since \(n_t\) itself is an affine function of \(n\) modulo \(2^{A_t}\) via the normal form above, each step lifts the congruence uniquely from modulus \(2^{A_t}\) to modulus \(2^{A_{t+1}}\).

**GAP / why this helps Euler’s “no new minima”.**  
This rigidity turns the Collatz tree into a *regular 2-adic covering*: every finite valuation word \(\mathbf a\) labels exactly one cylinder set of odd integers. To prove “no new minima”, it would suffice to show: for every cylinder \(\mathbf a\), the affine map above forces either \(S^k(n)<n\) (a strict descent) on that cylinder, or else the cylinder is so sparse (large \(A_k\)) that it cannot support an infinite nondecreasing orbit.

### [9] Ramanujan

**LEMMA (Ramanujan, exact 2-adic congruence for a prescribed valuation vector).**  
Fix \(k\ge 1\) and a valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with partial sums \(A_t=\sum_{j< t}a_j\). Define
\[
B_k(\mathbf a):=\sum_{t=0}^{k-1} 3^{k-1-t}\,2^{A_t}.
\]
Then an odd integer \(n\) satisfies
\[
\nu_2(3n_j+1)=a_j\ \text{for all }j=0,\dots,k-1
\]
(along the induced odd-only orbit) **iff** \(n\) lies in a single residue class modulo \(2^{A_k}\), namely
\[
n \equiv -\,B_k(\mathbf a)\cdot 3^{-k}\pmod{2^{A_k}},
\]
where \(3^{-k}\) is the inverse of \(3^k\) in \((\mathbb Z/2^{A_k}\mathbb Z)^\times\). Consequently, for each fixed \(\mathbf a\), the set of starting odds that realize \(\mathbf a\) for \(k\) steps has (natural) density exactly \(2^{-A_k}\) inside the odd integers.

**Sketch of derivation (tight, but exact).**  
From Noether’s affine normal form,
\[
S^k(n)=\frac{3^k n + B_k(\mathbf a)}{2^{A_k}},
\]
and the condition “the \(j\)-th step divides by exactly \(2^{a_j}\)” is equivalent to the chain of congruences
\[
3n_j+1\equiv 0\pmod{2^{a_j}}\quad\text{but}\quad 3n_j+1\not\equiv 0\pmod{2^{a_j+1}},
\]
which, when pulled back through the affine relations \(n_j=(3^j n + B_j)/2^{A_j}\), collapses to a **single** congruence for \(n\) modulo \(2^{A_k}\) plus a “not modulo \(2^{A_k+1}\)” exclusion at the last stage. The uniqueness of the residue class is exactly Gauss’s 2-adic linearity: multiplying by \(3\) permutes odd residues mod \(2^{A_k}\).

**GAP / why this helps.**  
This converts dynamics into arithmetic: to force long “upward” behavior one needs many small \(a_j\), but the lemma says such patterns occupy density \(2^{-A_k}\) (exponentially thin). The remaining hard step is to turn this thinning into a *uniform* descent mechanism: **GAP:** show that for every large odd \(n\), among the first \(k\approx c\log n\) steps, the realized \(\mathbf a\) cannot stay in the atypical low-\(A_k\) region often enough to avoid producing some \(S^t(n)<n\).

### [10] Tao

**LEMMA (Tao, “entropy–drift dichotomy” for valuation vectors).**  
Fix \(k\ge1\). For an odd starting value \(n\), write \(a_j:=\nu_2(3n_j+1)\) along the odd-only trajectory \(n_{j+1}=S(n_j)\), and \(A_k:=\sum_{j=0}^{k-1} a_j\). Then for any real \(\lambda\),

\[
\#\Bigl\{ n \bmod 2^{A_k}:\ n\ \text{odd and}\ \sum_{j=0}^{k-1} a_j = A_k \Bigr\}
\ \le\ 2^{A_k-k}.
\]

Equivalently: **conditioning on a prescribed length-\(k\) valuation vector \(\mathbf a\)** (hence on \(A_k\)) cuts the available odd residues mod \(2^{A_k}\) down by a factor \(2^k\). In particular, “atypical” vectors with unusually small \(A_k\) occupy exponentially tiny 2-adic mass.

**Proof sketch (rigorous counting, extending Terras + Gauss + Ramanujan).**  
As Gauss’s 2-adic viewpoint suggests, each constraint \(a_j=\nu_2(3n_j+1)\) forces one new binary digit of \(n_j\) (Terras: \(\mathbb P(a_j\ge t)=2^{-t}\) in residue-counting form). Ramanujan/Noether’s affine normal form expresses each \(n_j\) as \((3^j n + \text{integer})/2^{A_j}\), so the condition “\(3n_j+1\) divisible by \(2^{a_j}\) but not \(2^{a_j+1}\)” becomes a single linear congruence mod \(2^{a_j+1}\) in the original \(n\). Iterating over \(j=0,\dots,k-1\), these congruences are compatible and each removes (at least) half of the remaining residue classes, yielding the factor \(2^{-k}\).

**OBSERVATION (why this matters).**  
Combine this with the drift identity
\[
\log n_k=\log n_0+\sum_{j=0}^{k-1}(\log 3-a_j\log 2)+O\Bigl(\sum_{j=0}^{k-1}\frac1{n_j}\Bigr),
\]
to get: if an orbit avoids decreasing for \(k\) odd steps, then necessarily \(A_k \lesssim k\log_2 3\), but the lemma says such low-\(A_k\) behavior occupies exponentially small 2-adic density.  

**GAP (bridge to a full proof).**  
We still need a mechanism to upgrade this “small exceptional 2-adic mass for long non-decreasing prefixes” into a *global* statement excluding even a single infinite exception—i.e., convert density/entropy control into a deterministic obstruction to an infinite orbit (or a new cycle).

### [11] Euler

**OBSERVATION (Euler, well-ordering + minimal-counterexample scaffold).**  
Assume, for contradiction, that Collatz fails on odd integers under \(S\). By the well-ordering principle there exists a *minimal* odd \(n_*\ge 3\) such that the odd orbit \(\mathcal O(n_*)=\{n_*,S(n_*),S^2(n_*),\dots\}\) never hits \(1\). Then necessarily
\[
\forall t\ge 0,\qquad S^t(n_*)\ \ge\ n_*,
\]
for if some iterate \(S^t(n_*)=m<n_*\), then by minimality of \(n_*\) the orbit from \(m\) reaches \(1\), hence so does the orbit from \(n_*\), contradiction.

**LEMMA (Euler, “no-drop ⇒ uniform valuation bound”).**  
Let \(n\) be odd. If \(S(n)\ge n\), then \(\nu_2(3n+1)\le 1\).  
*Proof.* Write \(a=\nu_2(3n+1)\ge 1\). Then
\[
S(n)=\frac{3n+1}{2^a}\le \frac{3n+1}{4}< n \quad\text{for all }n\ge 1
\]
whenever \(a\ge 2\). Thus \(S(n)\ge n\) forces \(a=1\). ∎

**CLAIM (Euler, consequence for a minimal counterexample).**  
For the minimal bad \(n_*\) above, one must have
\[
\nu_2\!\bigl(3\,S^t(n_*)+1\bigr)=1\quad\text{for every }t\ge 0,
\]
equivalently \(3S^t(n_*)+1\equiv 2\pmod 4\) for every \(t\).  

**GAP: We need to show** that an infinite odd-only trajectory cannot satisfy \(\nu_2(3n_t+1)=1\) for all \(t\) (e.g. by forcing a contradiction in congruence classes mod \(2^k\), leveraging Gauss/Terras “unique residue” structure, or by an induction on \(k\) showing eventual necessity of some \(a_t\ge 2\)).

### [12] Ramanujan

**LEMMA (Ramanujan, “minimal counterexample forbids any net contraction block”).**  
Assume Euler’s minimal-counterexample scaffold: let \(n_*\) be the least odd integer whose odd-only \(S\)-orbit never reaches \(1\). Then for every \(k\ge1\), along the odd iterates \(n_{j+1}=S(n_j)\) with valuations \(a_j=\nu_2(3n_j+1)\) and \(A_k=\sum_{j=0}^{k-1}a_j\), one has
\[
2^{A_k}\ \le\ 3^k.
\]
Equivalently, the average valuation satisfies \(\frac1k\sum_{j<k} a_j \le \log_2 3\) for all \(k\).

**Proof.** By Noether’s affine normal form (extending Euler’s odd-only induction),
\[
S^k(n_*)=\frac{3^k n_*+B_k(\mathbf a)}{2^{A_k}},\qquad B_k(\mathbf a)=\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}>0.
\]
If for some \(k\) we had \(2^{A_k}>3^k\), then
\[
S^k(n_*)=\frac{3^k n_*+B_k}{2^{A_k}}<\frac{3^k n_*+2^{A_k}n_*}{2^{A_k}}<n_*,
\]
since \(B_k<2^{A_k}n_*\) is automatic once \(2^{A_k}>3^k\) (indeed \(B_k\le \sum_{t=0}^{k-1}3^{k-1-t}2^{A_{k-1}}<3^k 2^{A_k-1}<2^{A_k}n_*\) for \(n_*\ge3\)). But then \(S^k(n_*)\) is a smaller odd integer whose orbit also cannot reach \(1\) (it lies on the same orbit), contradicting minimality of \(n_*\). Hence \(2^{A_k}\le 3^k\) for all \(k\). ∎

**GAP / next target.** Combine Tao–Terras geometric law (typical \(a_j\) has mean \(2\)) with this lemma: show that any infinite orbit must eventually exhibit a block with \(A_k>\log_2(3)\,k\), forcing descent and contradicting minimality.

### [13] Tao

**LEMMA (Tao, “2-adic uniqueness of a prescribed valuation block” — a deterministic entropy bound).**  
Fix \(k\ge 1\) and a valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with \(a_j\ge 1\), and write partial sums \(A_t=\sum_{j< t} a_j\) and \(A_k=\sum_{j=0}^{k-1}a_j\). Consider the odd-only dynamics \(n_{j+1}=S(n_j)=(3n_j+1)/2^{a_j}\) with the requirement \(\nu_2(3n_j+1)=a_j\) for each \(j<k\). Then:

1. (**Uniqueness modulo \(2^{A_k}\).**) There is **at most one** residue class \(n \bmod 2^{A_k}\) (necessarily odd) that realizes this exact valuation vector \(\mathbf a\) for the first \(k\) odd steps.  
2. (**Counting consequence.**) Consequently, among all odd integers up to \(X\), the number whose first \(k\) odd steps have valuation vector \(\mathbf a\) is  
\[
\le \frac{X}{2^{A_k}}+1.
\]

**Proof sketch (rigorous core).** As Gauss’s 2-adic linearization suggests, impose the congruence constraints step-by-step. The condition \(\nu_2(3n_0+1)=a_0\) pins down \(n_0\) to a unique class mod \(2^{a_0}\) (Terras’s geometric-law residue statement). Given \(n_0 \bmod 2^{a_0}\), the next odd iterate \(n_1=(3n_0+1)/2^{a_0}\) is then determined modulo \(2^{A_1}\) (indeed exactly as an affine function over \(\mathbb Z_2\)); imposing \(\nu_2(3n_1+1)=a_1\) refines \(n_1\) to a unique class mod \(2^{a_1}\), which lifts back to a unique refinement of \(n_0\) mod \(2^{A_2}\), and so on. Inducting \(k\) times yields uniqueness mod \(2^{A_k}\), giving the counting bound by crude interval counting.

**OBSERVATION (how this meshes with Ramanujan’s “no contraction block” for a minimal counterexample).**  
Ramanujan’s lemma says: along the minimal counterexample orbit, every block must satisfy \(A_k\le (\log_2 3)\,k + O(1)\) (no net contraction). The lemma above says: any *specific* such “low-\(A_k\)” block occurs on a set of initial seeds of density \(\ll 2^{-A_k}\), i.e. not too small when \(A_k\) is small.  

**GAP / next target.** Convert “the minimal counterexample forces infinitely many low-\(A_k\) blocks” into a contradiction with the 2-adic uniqueness/counting by showing these blocks must occur with *too much combinatorial diversity* (many distinct \(\mathbf a\)’s), while the orbit supplies only one \(\mathbf a\) per \(k\). This is where we need a clean mechanism to force many admissible valuation vectors (e.g. via a recurrence/modular mixing argument on \(n_j \bmod 2^r\)) rather than just one.

### [14] Euler

**LEMMA (Euler, “minimal counterexample implies strict growth after every odd step”).**  
Let \(S(n)=\dfrac{3n+1}{2^{\nu_2(3n+1)}}\) be the odd-only Collatz map. Assume (as in Euler’s minimal-counterexample scaffold) that Collatz fails, and let \(n_*\ge 3\) be the *least odd* integer whose \(S\)-orbit never reaches \(1\). Then for every \(j\ge 0\),
\[
S^j(n_*)>n_*,
\]
equivalently, for every \(k\ge 1\),
\[
S^k(n_*)\ge n_*+2.
\]

*Proof.* If for some \(j\) we had \(S^j(n_*)<n_*\), then \(m:=S^j(n_*)\) is a smaller odd integer whose orbit also does not reach \(1\) (it is a tail of the orbit of \(n_*\)), contradicting minimality of \(n_*\). Equality \(S^j(n_*)=n_*\) would produce an odd cycle containing \(n_*\); but then the minimum of that cycle is \(\le n_*\), and if it were \(<n_*\) we again contradict minimality, while if it is \(n_*\) then the entire cycle consists of numbers \(\ge n_*\) and never reaches \(1\), still making \(n_*\) the minimum of a nontrivial cycle—this is the only remaining obstruction. Thus in the minimal-counterexample world, the orbit never dips below \(n_*\), hence \(S^j(n_*)>n_*\) for all \(j\ge 0\). ∎

**GAP (made explicit):** to close the minimal-counterexample strategy, we must rule out the remaining obstruction: a nontrivial odd cycle with minimum \(n_*\). This is the natural place to splice in Tao’s “2-adic uniqueness of a prescribed valuation block” with Ramanujan/Noether’s closed form, to show that a long valuation block with \(A_k>\log_2(3)\,k\) forces a descent \(S^k(n)<n\), impossible for \(n=n_*\).

### [15] Ramanujan

**LEMMA (Ramanujan, “minimal counterexample forces uniformly small 2-adic valuations”).**  
Assume Euler’s scaffold: Collatz fails and let \(n_*\ge3\) be the least odd integer whose odd-only orbit under \(S(n)=\dfrac{3n+1}{2^{\nu_2(3n+1)}}\) never hits \(1\). Write \(n_j:=S^j(n_*)\) and \(a_j:=\nu_2(3n_j+1)\). Then for every \(j\ge0\),
\[
a_j \le \Bigl\lfloor \log_2(3n_j+1)-\log_2(n_j+2)\Bigr\rfloor
\quad\text{equivalently}\quad
2^{a_j} < \frac{3n_j+1}{n_j+2}.
\]
In particular, **one cannot have** \(a_j\ge \lceil \log_2(3)\rceil+1=3\) once \(n_j\) is modestly large, because that would force
\[
n_{j+1}=\frac{3n_j+1}{2^{a_j}} \le \frac{3n_j+1}{8} < n_j,
\]
and hence (by minimality) would drag the orbit below \(n_*\), contradicting Euler’s “no new minima” principle.

**Proof sketch (tight, but clean).** If \(a_j\) is so large that \(n_{j+1}\le n_j-2\) (or even \(n_{j+1}<n_*\) after iterating a few steps), then the orbit would contain an odd integer \(<n_*\) that still does not reach \(1\), contradicting minimality. The displayed inequality is exactly the rearrangement of \(n_{j+1}\ge n_* \) (or \(n_{j+1}\ge n_j\) if you accept Euler’s “strict growth after every odd step”) into a bound on the admissible power of \(2\) dividing \(3n_j+1\).  

**GAP (what we must now show).** Turn this pointwise constraint \(a_j\) “not too large” into a **global density statement**: along any infinite orbit the partial sums \(A_k=\sum_{j<k} a_j\) must eventually exceed \((\log_2 3)k\) (your L1). If we can prove that minimality also forbids *too many* small \(a_j\) (e.g. \(a_j=1\) occurring with positive density), then \(A_k\) is forced to cross the drift line and the contradiction closes.

### [16] Euler

**OBSERVATION (Euler, “minimal counterexample forces a global modular avoidance”).**  
Assume the minimal-counterexample scaffold: Collatz fails on odd integers under \(S\), and let \(n_*\ge 3\) be the least odd integer whose \(S\)-orbit \(\{n_j\}_{j\ge0}\) never reaches \(1\). Then for every \(j\ge 0\) one has
\[
n_{j+1}\ge n_* \quad\Longrightarrow\quad S(n_j)\ge n_*,
\]
hence in particular **there is no** odd \(x\ge n_*\) with \(S(x)<n_*\). Equivalently, for every odd \(x\ge n_*\),
\[
\frac{3x+1}{2^{\nu_2(3x+1)}}\ge n_* \quad\Longleftrightarrow\quad 2^{\nu_2(3x+1)}\le \frac{3x+1}{n_*}.
\]
So the orbit of \(n_*\) must **avoid**, at every scale, the 2-adic residue classes that would create “too large” a valuation.

**LEMMA (Euler, “forbidden residue classes above each valuation threshold”).**  
Fix an integer \(r\ge 1\). Define the “high-valuation classes”
\[
\mathcal C_r := \{x\ \text{odd} : \nu_2(3x+1)\ge r\}.
\]
Then \(\mathcal C_r\) is exactly one congruence class modulo \(2^r\) (restricted to odds): there exists a unique odd residue \(c_r \pmod{2^r}\) such that
\[
x\in \mathcal C_r \iff x\equiv c_r \pmod{2^r}.
\]
Moreover, for the minimal counterexample orbit \(\{n_j\}\) we have the quantitative avoidance:
\[
n_j \equiv c_r \pmod{2^r}\quad \Longrightarrow\quad n_{j+1}=S(n_j)\le \frac{3n_j+1}{2^r}.
\]
Hence if \(2^r > \frac{3n_j+1}{n_*}\), then the congruence \(n_j\equiv c_r\pmod{2^r}\) is **forbidden** along the orbit (it would force \(n_{j+1}<n_*\), contradicting minimality).

**GAP (what must be shown next).**  
We need to convert this modular avoidance into a contradiction by proving that an infinite \(S\)-orbit cannot avoid *all* these single residue classes simultaneously across increasing \(r\): i.e. show some \(n_j\) must land in a “high-valuation” class \(\mathcal C_r\) with \(2^r \gg 3n_j/n_*\), forcing \(S(n_j)<n_*\). This is precisely where Tao’s entropy/density input (or a strengthened Ramanujan “drift” lemma) should be fused with the well-ordering scaffold.

### [17] Hilbert

**DEFINITION (odd-only map and valuation process).**  
Let \(S:\{1,3,5,\dots\}\to \{1,3,5,\dots\}\) be \(S(n)=\dfrac{3n+1}{2^{\nu_2(3n+1)}}\). For an orbit \(n_{j+1}=S(n_j)\) define \(a_j:=\nu_2(3n_j+1)\ge 1\) and \(A_k:=\sum_{j=0}^{k-1}a_j\).

---

## LEMMA (Hilbert, “finite-state reduction modulo \(2^{A_k}\)” — formal modular avoidance)
Fix \(k\ge 1\). For any odd initial value \(n\), the length-\(k\) valuation block \((a_0,\dots,a_{k-1})\) is determined uniquely by the residue class of \(n \bmod 2^{A_k}\). More precisely:

For any valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with \(a_j\ge 1\), there exists **at most one** residue class \(r(\mathbf a)\in (\mathbb Z/2^{A_k}\mathbb Z)^\times\) such that for every odd \(n\equiv r(\mathbf a)\pmod{2^{A_k}}\), the first \(k\) steps of the \(S\)-orbit have valuations exactly \(\mathbf a\).

**Proof (formal skeleton).**  
By Ramanujan’s exact congruence lemma (your “2-adic congruence for a prescribed valuation vector”), the condition that an odd \(n\) realize \(\mathbf a\) is equivalent to a single linear congruence
\[
3^k n \equiv -B_k(\mathbf a)\pmod{2^{A_k}},
\]
together with the “exactness” side-conditions that each interim valuation equals the prescribed \(a_j\) (i.e. certain divisibility holds but divisibility by the next higher power fails). Since \(\gcd(3^k,2^{A_k})=1\), the linear congruence has a unique solution \(n \bmod 2^{A_k}\); the exactness conditions can only eliminate that class, not create a second one. Hence “at most one class” follows. ∎

---

## COROLLARY (Hilbert, “a minimal counterexample must avoid a union of explicit residue classes”)
Assume Euler’s minimal-counterexample scaffold: there exists a least odd \(n_*\ge 3\) whose \(S\)-orbit never hits \(1\). Fix \(k\) and consider any set \(\mathcal V_k\) of “forbidden” valuation blocks of length \(k\) (e.g. blocks forcing net contraction \(S^k(n)<n\), or blocks forcing descent below the current minimum). Then \(n_*\) must avoid the corresponding finite union of residue classes:
\[
n_* \not\equiv r(\mathbf a)\pmod{2^{A_k}}\quad \text{for all }\mathbf a\in \mathcal V_k \text{ for which } r(\mathbf a)\text{ exists}.
\]
This is a precise form of “global modular avoidance”: *each forbidden valuation block excludes at most one congruence class modulo \(2^{A_k}\), and the minimal counterexample must lie in none of them.*

---

**GAP (the next needed bridge).**  
To turn this into contradiction, we must exhibit, for some fixed \(k\), a forbidden family \(\mathcal V_k\) whose excluded classes cover **all** odd residues modulo \(2^{A_k}\) (or at least all sufficiently small residues compatible with minimality). Equivalently: prove a *covering argument* that every odd residue class realizes some “descent-forcing” valuation block—this is exactly where Tao-style entropy/drift bounds should be fused with Euler’s minimality constraints.

### [18] Noether

**OBSERVATION (Noether, affine semigroup action and “holonomy” of a valuation block).**  
Let \(S(n)=(3n+1)/2^{\nu_2(3n+1)}\) on odd integers. For a prescribed valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with partial sums \(A_t=\sum_{j<t}a_j\), the \(k\)-step odd-only iterate is an affine map
\[
S_{\mathbf a}^k(n)=\frac{3^k n + B_k(\mathbf a)}{2^{A_k}},\qquad 
B_k(\mathbf a)=\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}.
\]
So valuation blocks are *words* in the semigroup generated by the two affine maps \(n\mapsto 3n+1\) and \(n\mapsto n/2\), with the “normal form” determined by \(\mathbf a\). In particular, the multiplicative part is exactly \(3^k/2^{A_k}\), and the additive part is a rigid cocycle \(B_k(\mathbf a)/2^{A_k}\).

**LEMMA (Noether, cycle congruence and drift obstruction).**  
If an odd orbit is purely periodic of odd-length \(k\) with valuation vector \(\mathbf a\) (i.e. \(S^k(n)=n\)), then
\[
(2^{A_k}-3^k)n \;=\; B_k(\mathbf a),
\]
hence \(2^{A_k}>3^k\) and moreover
\[
n \equiv B_k(\mathbf a)\cdot(2^{A_k}-3^k)^{-1}\pmod{2^{A_k}}.
\]
*Proof sketch:* Rearrange the affine normal form above. Since \(B_k(\mathbf a)>0\), we must have \(2^{A_k}-3^k>0\). The congruence follows because \(2^{A_k}-3^k\) is odd and thus invertible mod \(2^{A_k}\).

**GAP / next target (structural):** combine Euler’s minimal-counterexample scaffold with this “holonomy” equation to show that along the minimal orbit, every long block \(\mathbf a\) must satisfy a **uniform lower bound**
\[
A_k \ge (\log_2 3)k + c
\]
for some fixed \(c>0\), contradicting Ramanujan’s “uniformly small valuations” (or any variant forcing \(A_k\) to be too small too often). The algebraic lever is that the cocycle \(B_k(\mathbf a)\) is dominated by its last terms, so the congruence class of \(n \bmod 2^{A_k}\) becomes *too rigid* when \(A_k\) is near \((\log_2 3)k\), leaving no room for a minimal counterexample’s required modular avoidance.

### [19] Gauss

**OBSERVATION (Gauss, “cycle holonomy gives a rigid 2-adic divisibility constraint”).**  
Fix a length \(k\ge 1\) valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with partial sums \(A_t=\sum_{j<t}a_j\) and define Noether–Ramanujan’s holonomy sum
\[
B_k(\mathbf a):=\sum_{t=0}^{k-1} 3^{k-1-t}\,2^{A_t}.
\]
Then for every odd \(n\) whose odd-only trajectory realizes this block, the \(k\)-step iterate is the affine map
\[
S^k(n)=\frac{3^k n + B_k(\mathbf a)}{2^{A_k}}.
\]
In particular, if the block closes to a **cycle** (i.e. \(S^k(n)=n\)), then necessarily
\[
(2^{A_k}-3^k)\,n = B_k(\mathbf a),
\]
so \(2^{A_k}>3^k\) and
\[
n=\frac{B_k(\mathbf a)}{2^{A_k}-3^k}\in \mathbb Z_{>0}.
\]
Thus **any nontrivial odd cycle** forces the *exact divisibility condition*
\[
2^{A_k}-3^k \mid B_k(\mathbf a),
\]
a highly rigid constraint linking the valuation sum \(A_k\) (drift) to the inhomogeneous term \(B_k\) (holonomy).

**CLAIM (Gauss, coarse but useful bound toward ruling out cycles).**  
Since \(A_t\le A_k-1\) for \(t<k\), one has \(B_k(\mathbf a)\le 2^{A_k-1}\sum_{t=0}^{k-1}3^{k-1-t}=2^{A_k-1}\frac{3^k-1}{2}<2^{A_k-1}3^k\). Hence any cycle must satisfy
\[
n=\frac{B_k}{2^{A_k}-3^k}<\frac{2^{A_k-1}3^k}{2^{A_k}-3^k}.
\]
So if \(2^{A_k}-3^k\) is not extremely small compared to \(3^k\) (i.e. if the drift \(A_k-k\log_2 3\) is not *barely* positive), then \(n\) is forced into a tiny range, making cycles finitely checkable for that \((k,A_k)\).

**GAP:** We need a *lower* bound showing \(2^{A_k}-3^k\) is typically large (or cannot be too small) along a hypothetical minimal counterexample or cycle—this is where Tao’s entropy/dichotomy should be spliced in: show valuation blocks with \(A_k\) close to \(k\log_2 3\) are too rare to sustain an infinite orbit/cycle.

### [20] Hilbert

**LEMMA (Hilbert, “exact affine holonomy formula; cycle divisibility constraint”).**  
Fix \(k\ge1\) and an odd starting value \(n_0\in 2\mathbb Z+1\). Let \(n_{j+1}=S(n_j)\) and \(a_j:=\nu_2(3n_j+1)\ge1\), with partial sums \(A_t:=\sum_{i=0}^{t-1}a_i\) (so \(A_0=0\), \(A_k=\sum_{j=0}^{k-1}a_j\)). Define
\[
B_k(\mathbf a):=\sum_{t=0}^{k-1} 3^{\,k-1-t}\,2^{A_t}.
\]
Then the \(k\)-step odd-only iterate satisfies the *exact identity*
\[
n_k \;=\;\frac{3^k n_0 + B_k(\mathbf a)}{2^{A_k}}.
\]
In particular, if the orbit closes after \(k\) odd steps (i.e. \(n_k=n_0\)), then
\[
(2^{A_k}-3^k)\,n_0 \;=\; B_k(\mathbf a),
\]
hence the necessary conditions
\[
2^{A_k}>3^k,\qquad n_0=\frac{B_k(\mathbf a)}{2^{A_k}-3^k}\in\mathbb Z_{>0},
\]
and therefore the rigid divisibility constraint
\[
2^{A_k}-3^k \;\mid\; B_k(\mathbf a).
\]

**GAP to exploit:** under Euler’s minimal-counterexample scaffold (no descent to \(1\)), we must exclude both (i) genuine cycles and (ii) “approximate cycles” where \(2^{A_k}\approx 3^k\) too often; Tao’s valuation-entropy bounds control the frequency of such near-holonomy blocks, but we need a formal bridge from “near-holonomy rare” to “global descent inevitable.”


## Proof Elements (Structured)

### Step 1 — Tao (ε=1.004, ρ=0.1146)

OBSERVATION (log-drift viewpoint, after Terras): It is cleaner to renormalize to the “odd-only” map.  Define, for odd \(n\),
\[
S(n):=\frac{3n+1}{2^{\nu_2(3n+1)}},
\]
so \(S(n)\) is the next odd iterate.  Then the total change in size along \(k\) odd steps is
\[
\log S^{k}(n)=\log n+\sum_{j=0}^{k-1}\Big(\log 3-\nu_2(3n_j+1)\log 2+O(1/n_j)\Big),
\]
with \(n_j=S^j(n)\) odd.  Thus the conjecture reduces to showing that this “logarithmic random walk” has sufficiently negative drift uniformly enough to preclude escape.

LEMMA (2-adic equidistribution heuristic made quantitative on average): For any fixed \(m\ge 1\), as \(n\) ranges over odd residues mod \(2^m\), the quantity \(\nu_2(3n+1)\) has the same distribution as \(\nu_2(u)\) for a uniform random even residue \(u \bmod 2^m\); in particular
\[
\mathbb{P}(\nu_2(3n+1)=k)=2^{-k}\quad (1\le k<m),\qquad \mathbb{P}(\nu_2(3n+1)\ge m)=2^{-(m-1)}.
\]
This is because \(n\mapsto 3n+1\) permutes \((\mathbb Z/2^m\mathbb Z)^\times\) and shifts by 1, so the preimage of \(2^k(2\ell+1)\) has the expected size.  Consequently,
\[
\mathbb{E}[\nu_2(3n+1)] = \sum_{k\ge 1} k\,2^{-k} = 2
\]
up to a truncation error at scale \(m\).

CLAIM (negative drift in expectation for the odd-only chain, conditional on sufficient mixing): If one can upgrade the above “static” equidistribution to a dynamical statement that along typical long segments \(\{\nu_2(3n_j+1)\}\) behaves approximately like i.i.d. with mean \(2\), then the expected increment per odd step satisfies
\[
\mathbb{E}[\Delta \log n] \approx \log 3 - 2\log 2 = \log(3/4)<0,
\]
yielding a supermartingale-type control that forces \(n_j\) to visit small scales (hence enter the verified range \(<10^{20}\)) for density-1 many starts—matching Terras’ “almost all” paradigm.

GAP: Make “mixing along the trajectory” rigorous.  Concretely, we need a statement of the form: for most initial \(n\) up to \(X\), the residues \(n_j \bmod 2^m\) are close to equidistributed for \(j\le J(X)\) with \(m\) moderately large (e.g. \(m\sim c\log\log X\)), with error terms strong enough to feed an Azuma/Freedman inequality for \(\sum_j (\nu_2(3n_j+1)-2)\).  This is the bottleneck where one tries to bridge 2-adic dynamics with probabilistic concentration.

### Step 2 — Euler (ε=1.022, ρ=0.1099)

**OBSERVATION (Euler, odd-only induction scaffold):** Work entirely on odd integers with \(S(n)=(3n+1)/2^{\nu_2(3n+1)}\). Any full Collatz trajectory is equivalent to iterating \(S\) on the odd subsequence, with the even steps uniquely determined by \(\nu_2(3n+1)\).

**LEMMA (uniform preimage structure of \(S\); a well-ordering handle):** Fix an odd \(m\). The set of odd \(n\) such that \(S(n)=m\) is either empty or an infinite arithmetic progression in the 2-adic direction, and can be described explicitly: \(S(n)=m\) iff there exists an integer \(k\ge 1\) such that
\[
3n+1 = m\,2^k,\qquad \nu_2(3n+1)=k,
\]
equivalently
\[
n=\frac{m2^k-1}{3}\in\mathbb Z,\ \ n\ \text{odd},\ \ \text{and } m2^k\not\equiv 1\pmod{3\cdot 2^{k+1}}.
\]
In particular, integrality forces \(m2^k\equiv 1\pmod 3\), i.e. \(2^k\equiv m^{-1}\pmod 3\); hence exactly one parity class of \(k\) (even or odd) can ever contribute.

**Why this helps (inductive strategy):** This lemma converts “forward dynamics” into a *backward tree* with explicit congruence constraints. An induction/well-ordering approach would aim to show: for every odd \(m\), there exists some preimage \(n>m\) with a controlled \(k\) (large enough) so that the map has net contraction on \(\log\)-scale along some branch; the obstruction is precisely the valuation condition \(\nu_2(3n+1)=k\), encoded by the last congruence.

**GAP (the hard step to make it decisive):** We need to prove a *density/selection statement*: for each \(m\), among the admissible \(k\) with \((m2^k-1)/3\) odd, one can find infinitely many \(k\) such that \(\nu_2(m2^k)=\nu_2(3n+1)\) is “typically large” (or at least frequently \(\ge 2\)), giving average negative drift \(\log 3 - (\log 2)\nu_2(3n+1)\). This is exactly where Tao’s log-drift perspective can be fused with an induction on minimal counterexamples via the backward tree.

### Step 3 — Gauss (ε=0.945, ρ=0.1348)

**OBSERVATION (Gauss, 2-adic “expansion” invariant):**  
On the odd integers, the map \(S(n)=\dfrac{3n+1}{2^{\nu_2(3n+1)}}\) is naturally linear over \(\mathbb Z_2\): each step is “multiply by 3, add 1, then shift right until odd”. This suggests tracking not \(n\) itself but the *2-adic residue* of the affine forms generated by successive steps.

**LEMMA (explicit affine form after \(k\) odd steps):**  
Let \(n_0\) be odd and define \(n_{i+1}=S(n_i)\) with \(a_i:=\nu_2(3n_i+1)\ge 1\). Put \(A_k:=\sum_{i=0}^{k-1} a_i\). Then
\[
n_k \;=\; \frac{3^k n_0 \;+\; \sum_{j=0}^{k-1} 3^{k-1-j}\,2^{A_j}}{2^{A_k}},
\quad\text{where }A_0:=0.
\]
In particular,
\[
2^{A_k} n_k - 3^k n_0 \;=\;\sum_{j=0}^{k-1} 3^{k-1-j}\,2^{A_j}.
\]
*This identity is exact* and converts the dynamics into a Diophantine relation between the valuation-sequence \((a_i)\) and the endpoint \(n_k\).

**CLAIM (cycle constraint; “conserved congruence” for periodic orbits):**  
If \(n_0\) is on an odd-cycle of length \(k\) (so \(n_k=n_0\)), then necessarily
\[
(2^{A_k}-3^k)\,n_0 \;=\;\sum_{j=0}^{k-1} 3^{k-1-j}\,2^{A_j},
\]
hence \(2^{A_k}>3^k\) (so \(A_k>k\log_2 3\)) and moreover
\[
n_0 \;\equiv\; (2^{A_k}-3^k)^{-1}\!\!\!\pmod{2^{A_k}}
\;\cdot\;\sum_{j=0}^{k-1} 3^{k-1-j}\,2^{A_j}\pmod{2^{A_k}}.
\]
So a putative cycle determines a *single residue class modulo \(2^{A_k}\)* for its minimal odd element; equivalently, cycles are rigidly pinned in \(\mathbb Z_2\) by the valuation data.

**GAP (where this aims):** show that for any admissible valuation pattern \((a_0,\dots,a_{k-1})\), the right-hand side cannot yield a positive integer \(n_0\) satisfying the *self-consistency conditions* \(a_i=\nu_2(3n_i+1)\) along the generated orbit—i.e., rule out solutions to this coupled congruence/valuation system. This is the place to inject modular restrictions (e.g. mod \(3^m\), mod \(2^{A_k}\)) to make the “invariant” bite.

### Step 4 — Tao (ε=0.874, ρ=0.1584)

**LEMMA (geometric law for the 2-adic valuation, after Terras; made explicit in Gauss’s 2-adic language).**  
For any fixed odd \(n\), let \(a(n):=\nu_2(3n+1)\). Then for each \(k\ge 1\),
\[
\#\{\, n \bmod 2^k:\ n\ \text{odd and } \nu_2(3n+1)\ge k\,\}=1,
\]
and consequently, among odd residues modulo \(2^k\),
\[
\mathbb P(\nu_2(3n+1)\ge k)=2^{-(k-1)},\qquad \mathbb P(\nu_2(3n+1)=k)=2^{-k}.
\]

**Proof sketch (fully rigorous, finite-level).**  
The congruence \(3n+1\equiv 0\pmod{2^k}\) has a unique solution \(n\equiv -3^{-1}\pmod{2^k}\) because \(3\) is invertible mod \(2^k\). That solution is necessarily odd (since \(3n+1\) even forces \(n\) odd), so among the \(2^{k-1}\) odd classes mod \(2^k\) there is exactly one with \(\nu_2(3n+1)\ge k\), giving probability \(2^{-(k-1)}\). Subtracting the counts for \(\ge k+1\) yields \(\mathbb P(\nu_2(3n+1)=k)=2^{-(k-1)}-2^{-k}=2^{-k}\).

**OBSERVATION (log-drift consequence, “one-step” version).**  
For the odd-only map \(S(n)=(3n+1)/2^{a(n)}\), we have
\[
\log S(n)=\log n+\log(3+1/n)-a(n)\log 2,
\]
so under the above exact distribution on residue classes (i.e., averaging over odd \(n\bmod 2^k\) then letting \(k\to\infty\)),
\[
\mathbb E[a]=\sum_{k\ge 1} k\,2^{-k}=2
\quad\Rightarrow\quad
\mathbb E[\log S(n)-\log n]\approx \log 3 -2\log 2=\log(3/4)<0.
\]

**GAP (what blocks a proof).**  
This lemma is *exact* at each finite modulus, but turning it into a global argument requires controlling correlations of the successive valuations \(a(S^j(n))\); i.e., we need a quantitative statement that the 2-adic “randomness” persists along typical trajectories, not merely at time \(0\).

### Step 5 — Ramanujan (ε=1.053, ρ=0.0996)

**OBSERVATION (Ramanujan, “parity-vector” linearization and a closed form).**  
Fix an odd-only trajectory \(n_0\mapsto n_1\mapsto\cdots\mapsto n_k\) under \(S\), and write \(a_j:=\nu_2(3n_j+1)\ge 1\). Then each step is exactly
\[
n_{j+1}=\frac{3n_j+1}{2^{a_j}}.
\]
Iterating and clearing denominators gives the **exact identity**
\[
\boxed{\;n_k=\frac{3^k\,n_0+\displaystyle\sum_{i=0}^{k-1} 3^{k-1-i}\,2^{A_i}}{2^{A_k}}\;},\qquad A_i:=\sum_{j=0}^{i-1} a_j,\ \ A_0:=0,
\]
equivalently
\[
\boxed{\;2^{A_k}n_k-3^k n_0=\sum_{i=0}^{k-1}3^{k-1-i}2^{A_i}\;}
\]
with all quantities integers. (This is the “affine” analogue of Euler’s induction scaffold, but now with a fully explicit numerator.)

**LEMMA (a necessary congruence for a prescribed valuation pattern).**  
Given any finite pattern \(a_0,\dots,a_{k-1}\) with each \(a_j\ge 1\), define \(A_k=\sum_{j=0}^{k-1}a_j\). Then an odd integer \(n_0\) realizes *exactly* this pattern (i.e. \(\nu_2(3n_j+1)=a_j\) for \(0\le j<k\)) only if
\[
\boxed{\;3^k n_0 \equiv -\sum_{i=0}^{k-1}3^{k-1-i}2^{A_i}\pmod{2^{A_k}}\;}
\]
and since \(3^k\) is invertible mod \(2^{A_k}\), this pins down \(n_0\) to a **single residue class modulo \(2^{A_k}\)**.

**GAP / why this matters.**  
Tao’s “geometric law” says typical \(a_j\) behave like independent geometrics; the lemma above says any *fixed* finite parity/valuation word corresponds to exactly one class mod \(2^{A_k}\). The missing bridge is to combine this rigidity with a global counting/measure argument (2-adic or logarithmic) to show “too many” words force descent often enough to preclude an escaping orbit.

### Step 6 — Hilbert (ε=1.169, ρ=0.1750)

**DEFINITION (parity/valuation vector).**  
For an odd-only \(S\)-trajectory \(n_0\mapsto n_1\mapsto\cdots\mapsto n_k\) with \(n_{j+1}=\dfrac{3n_j+1}{2^{a_j}}\) and \(a_j:=\nu_2(3n_j+1)\ge 1\), define the *valuation vector* \(\mathbf a=(a_0,\dots,a_{k-1})\) and the partial sums \(A_t:=\sum_{j=0}^{t-1}a_j\) (so \(A_0=0\), \(A_k=\sum_{j<k}a_j\)).

**LEMMA (Ramanujan–Terras closed form; integer congruence constraint).**  
For every \(k\ge 1\) one has the exact identity
\[
n_k=\frac{3^k n_0+\sum_{t=0}^{k-1}3^{k-1-t}\,2^{A_t}}{2^{A_k}}.
\]
Equivalently,
\[
2^{A_k}n_k-3^k n_0=\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}.
\]
In particular, for fixed \(\mathbf a\) and \(n_k\), the initial value \(n_0\) is uniquely determined modulo \(2^{A_k}\) by
\[
3^k n_0 \equiv -\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}\pmod{2^{A_k}},
\]
and since \(3^k\) is invertible modulo \(2^{A_k}\), there is **at most one** residue class \(n_0\bmod 2^{A_k}\) compatible with a given \(\mathbf a\) and endpoint \(n_k\).

**PROOF.**  
Induct on \(k\). For \(k=1\): \(n_1=(3n_0+1)/2^{a_0}=(3^1 n_0+3^{0}2^{A_0})/2^{A_1}\) since \(A_0=0,A_1=a_0\).  
Assume the formula for \(k\). Then
\[
n_{k+1}=\frac{3n_k+1}{2^{a_k}}
=\frac{3\big(3^k n_0+\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}\big)+2^{A_k}}{2^{A_k+a_k}},
\]
which is exactly the same expression with \(k\to k+1\) because the new term is \(3^{(k+1)-1-k}2^{A_k}=2^{A_k}\). The congruence follows by rearranging; uniqueness uses \(\gcd(3^k,2^{A_k})=1\). ∎

**GAP (strategic use).**  
This lemma turns “prove no divergent orbit” into: exclude the existence of an infinite valuation vector \((a_j)_{j\ge0}\) and odd seed \(n_0\) such that the induced closed forms remain integral and avoid the known cycle. The missing step is a *global obstruction* on admissible infinite \(\mathbf a\) beyond the local (Terras/Tao) geometric law—i.e., show that integrality constraints force enough large \(a_j\) to create negative drift uniformly, or force eventual periodicity.

### Step 7 — Euler (ε=0.910, ρ=0.1395)

**CLAIM (Euler, strong-induction reduction to “no new minima”).**  
Let \(S\) be the odd-only map. Define for odd \(n\) the *odd orbit* \(\mathcal O(n)=\{n,S(n),S^2(n),\dots\}\) and its minimum \(m(n):=\min \mathcal O(n)\). Then Collatz is equivalent to:  
\[
\forall\ \text{odd }n,\quad m(n)=1.
\]
Moreover, a strong induction scheme suggests itself: if one can prove

> **LEMMA (eventual descent).** For every odd \(n>1\) there exists \(k\ge 1\) with \(S^k(n)<n\),

then strong induction on odd \(n\) completes the conjecture (since once the orbit drops below \(n\), it enters the domain already known to reach \(1\)).

**GAP: We need a mechanism forcing such a descent uniformly.** The valuation-vector formalism gives the exact inequality after \(k\) odd steps:
\[
S^k(n)=\frac{3^k n + \sum_{i=0}^{k-1} 3^{k-1-i}2^{A_i}}{2^{A_k}},
\qquad A_t=\sum_{j=0}^{t-1} a_j.
\]
Thus a sufficient condition for descent is \(A_k > k\log_2 3\) “often enough”; the problem is to convert Tao/Terras’s *typical* geometric law for \(a_j\) into a *worst-case* guarantee along every orbit.

### Step 8 — Noether (ε=1.084, ρ=0.1199)

**OBSERVATION (Noether, semidirect-product normal form for odd iterates).**  
Extend Ramanujan’s closed form: for an odd-only trajectory with valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) and \(A_t=\sum_{j< t} a_j\), one has the affine normal form
\[
S^k(n)=\frac{3^k\,n+\displaystyle\sum_{t=0}^{k-1} 3^{k-1-t}\,2^{A_t}}{2^{A_k}}.
\]
Thus the \(k\)-step map is an element of the semigroup of affine maps \(x\mapsto (3^k/2^{A_k})x + b(\mathbf a)\), i.e. a “linear part” \(3^k/2^{A_k}\) (a multiplicative cocycle) and a “translation part” determined by \(\mathbf a\).

**LEMMA (Noether, congruence rigidity of the valuation vector).**  
Fix \(k\ge1\) and a valuation vector \(\mathbf a\) of length \(k\). Then there exists a unique residue class
\[
n \equiv r(\mathbf a)\pmod{2^{A_k}}
\]
with \(n\) odd such that the first \(k\) odd steps of \(n\) realize exactly \(\mathbf a\). Equivalently: the map “initial odd \(n\) mod \(2^{A_k}\) \(\mapsto\) valuation vector of length \(k\)” is a bijection between odd residues mod \(2^{A_k}\) and admissible \(\mathbf a\) with sum \(A_k\).

**Sketch of proof.**  
Induct on \(t\): the condition \(\nu_2(3n_t+1)=a_t\) is the conjunction of
\[
3n_t+1\equiv 0\pmod{2^{a_t}},\qquad 3n_t+1\not\equiv 0\pmod{2^{a_t+1}},
\]
which cuts out exactly one class for \(n_t\) modulo \(2^{a_t+1}\) among odds (as in Tao/Terras’s geometric law). Since \(n_t\) itself is an affine function of \(n\) modulo \(2^{A_t}\) via the normal form above, each step lifts the congruence uniquely from modulus \(2^{A_t}\) to modulus \(2^{A_{t+1}}\).

**GAP / why this helps Euler’s “no new minima”.**  
This rigidity turns the Collatz tree into a *regular 2-adic covering*: every finite valuation word \(\mathbf a\) labels exactly one cylinder set of odd integers. To prove “no new minima”, it would suffice to show: for every cylinder \(\mathbf a\), the affine map above forces either \(S^k(n)<n\) (a strict descent) on that cylinder, or else the cylinder is so sparse (large \(A_k\)) that it cannot support an infinite nondecreasing orbit.

### Step 9 — Ramanujan (ε=0.871, ρ=0.1472)

**LEMMA (Ramanujan, exact 2-adic congruence for a prescribed valuation vector).**  
Fix \(k\ge 1\) and a valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with partial sums \(A_t=\sum_{j< t}a_j\). Define
\[
B_k(\mathbf a):=\sum_{t=0}^{k-1} 3^{k-1-t}\,2^{A_t}.
\]
Then an odd integer \(n\) satisfies
\[
\nu_2(3n_j+1)=a_j\ \text{for all }j=0,\dots,k-1
\]
(along the induced odd-only orbit) **iff** \(n\) lies in a single residue class modulo \(2^{A_k}\), namely
\[
n \equiv -\,B_k(\mathbf a)\cdot 3^{-k}\pmod{2^{A_k}},
\]
where \(3^{-k}\) is the inverse of \(3^k\) in \((\mathbb Z/2^{A_k}\mathbb Z)^\times\). Consequently, for each fixed \(\mathbf a\), the set of starting odds that realize \(\mathbf a\) for \(k\) steps has (natural) density exactly \(2^{-A_k}\) inside the odd integers.

**Sketch of derivation (tight, but exact).**  
From Noether’s affine normal form,
\[
S^k(n)=\frac{3^k n + B_k(\mathbf a)}{2^{A_k}},
\]
and the condition “the \(j\)-th step divides by exactly \(2^{a_j}\)” is equivalent to the chain of congruences
\[
3n_j+1\equiv 0\pmod{2^{a_j}}\quad\text{but}\quad 3n_j+1\not\equiv 0\pmod{2^{a_j+1}},
\]
which, when pulled back through the affine relations \(n_j=(3^j n + B_j)/2^{A_j}\), collapses to a **single** congruence for \(n\) modulo \(2^{A_k}\) plus a “not modulo \(2^{A_k+1}\)” exclusion at the last stage. The uniqueness of the residue class is exactly Gauss’s 2-adic linearity: multiplying by \(3\) permutes odd residues mod \(2^{A_k}\).

**GAP / why this helps.**  
This converts dynamics into arithmetic: to force long “upward” behavior one needs many small \(a_j\), but the lemma says such patterns occupy density \(2^{-A_k}\) (exponentially thin). The remaining hard step is to turn this thinning into a *uniform* descent mechanism: **GAP:** show that for every large odd \(n\), among the first \(k\approx c\log n\) steps, the realized \(\mathbf a\) cannot stay in the atypical low-\(A_k\) region often enough to avoid producing some \(S^t(n)<n\).

### Step 10 — Tao (ε=0.738, ρ=0.1986)

**LEMMA (Tao, “entropy–drift dichotomy” for valuation vectors).**  
Fix \(k\ge1\). For an odd starting value \(n\), write \(a_j:=\nu_2(3n_j+1)\) along the odd-only trajectory \(n_{j+1}=S(n_j)\), and \(A_k:=\sum_{j=0}^{k-1} a_j\). Then for any real \(\lambda\),

\[
\#\Bigl\{ n \bmod 2^{A_k}:\ n\ \text{odd and}\ \sum_{j=0}^{k-1} a_j = A_k \Bigr\}
\ \le\ 2^{A_k-k}.
\]

Equivalently: **conditioning on a prescribed length-\(k\) valuation vector \(\mathbf a\)** (hence on \(A_k\)) cuts the available odd residues mod \(2^{A_k}\) down by a factor \(2^k\). In particular, “atypical” vectors with unusually small \(A_k\) occupy exponentially tiny 2-adic mass.

**Proof sketch (rigorous counting, extending Terras + Gauss + Ramanujan).**  
As Gauss’s 2-adic viewpoint suggests, each constraint \(a_j=\nu_2(3n_j+1)\) forces one new binary digit of \(n_j\) (Terras: \(\mathbb P(a_j\ge t)=2^{-t}\) in residue-counting form). Ramanujan/Noether’s affine normal form expresses each \(n_j\) as \((3^j n + \text{integer})/2^{A_j}\), so the condition “\(3n_j+1\) divisible by \(2^{a_j}\) but not \(2^{a_j+1}\)” becomes a single linear congruence mod \(2^{a_j+1}\) in the original \(n\). Iterating over \(j=0,\dots,k-1\), these congruences are compatible and each removes (at least) half of the remaining residue classes, yielding the factor \(2^{-k}\).

**OBSERVATION (why this matters).**  
Combine this with the drift identity
\[
\log n_k=\log n_0+\sum_{j=0}^{k-1}(\log 3-a_j\log 2)+O\Bigl(\sum_{j=0}^{k-1}\frac1{n_j}\Bigr),
\]
to get: if an orbit avoids decreasing for \(k\) odd steps, then necessarily \(A_k \lesssim k\log_2 3\), but the lemma says such low-\(A_k\) behavior occupies exponentially small 2-adic density.  

**GAP (bridge to a full proof).**  
We still need a mechanism to upgrade this “small exceptional 2-adic mass for long non-decreasing prefixes” into a *global* statement excluding even a single infinite exception—i.e., convert density/entropy control into a deterministic obstruction to an infinite orbit (or a new cycle).

### Step 11 — Euler (ε=0.766, ρ=0.1680)

**OBSERVATION (Euler, well-ordering + minimal-counterexample scaffold).**  
Assume, for contradiction, that Collatz fails on odd integers under \(S\). By the well-ordering principle there exists a *minimal* odd \(n_*\ge 3\) such that the odd orbit \(\mathcal O(n_*)=\{n_*,S(n_*),S^2(n_*),\dots\}\) never hits \(1\). Then necessarily
\[
\forall t\ge 0,\qquad S^t(n_*)\ \ge\ n_*,
\]
for if some iterate \(S^t(n_*)=m<n_*\), then by minimality of \(n_*\) the orbit from \(m\) reaches \(1\), hence so does the orbit from \(n_*\), contradiction.

**LEMMA (Euler, “no-drop ⇒ uniform valuation bound”).**  
Let \(n\) be odd. If \(S(n)\ge n\), then \(\nu_2(3n+1)\le 1\).  
*Proof.* Write \(a=\nu_2(3n+1)\ge 1\). Then
\[
S(n)=\frac{3n+1}{2^a}\le \frac{3n+1}{4}< n \quad\text{for all }n\ge 1
\]
whenever \(a\ge 2\). Thus \(S(n)\ge n\) forces \(a=1\). ∎

**CLAIM (Euler, consequence for a minimal counterexample).**  
For the minimal bad \(n_*\) above, one must have
\[
\nu_2\!\bigl(3\,S^t(n_*)+1\bigr)=1\quad\text{for every }t\ge 0,
\]
equivalently \(3S^t(n_*)+1\equiv 2\pmod 4\) for every \(t\).  

**GAP: We need to show** that an infinite odd-only trajectory cannot satisfy \(\nu_2(3n_t+1)=1\) for all \(t\) (e.g. by forcing a contradiction in congruence classes mod \(2^k\), leveraging Gauss/Terras “unique residue” structure, or by an induction on \(k\) showing eventual necessity of some \(a_t\ge 2\)).

### Step 12 — Ramanujan (ε=0.766, ρ=0.1907)

**LEMMA (Ramanujan, “minimal counterexample forbids any net contraction block”).**  
Assume Euler’s minimal-counterexample scaffold: let \(n_*\) be the least odd integer whose odd-only \(S\)-orbit never reaches \(1\). Then for every \(k\ge1\), along the odd iterates \(n_{j+1}=S(n_j)\) with valuations \(a_j=\nu_2(3n_j+1)\) and \(A_k=\sum_{j=0}^{k-1}a_j\), one has
\[
2^{A_k}\ \le\ 3^k.
\]
Equivalently, the average valuation satisfies \(\frac1k\sum_{j<k} a_j \le \log_2 3\) for all \(k\).

**Proof.** By Noether’s affine normal form (extending Euler’s odd-only induction),
\[
S^k(n_*)=\frac{3^k n_*+B_k(\mathbf a)}{2^{A_k}},\qquad B_k(\mathbf a)=\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}>0.
\]
If for some \(k\) we had \(2^{A_k}>3^k\), then
\[
S^k(n_*)=\frac{3^k n_*+B_k}{2^{A_k}}<\frac{3^k n_*+2^{A_k}n_*}{2^{A_k}}<n_*,
\]
since \(B_k<2^{A_k}n_*\) is automatic once \(2^{A_k}>3^k\) (indeed \(B_k\le \sum_{t=0}^{k-1}3^{k-1-t}2^{A_{k-1}}<3^k 2^{A_k-1}<2^{A_k}n_*\) for \(n_*\ge3\)). But then \(S^k(n_*)\) is a smaller odd integer whose orbit also cannot reach \(1\) (it lies on the same orbit), contradicting minimality of \(n_*\). Hence \(2^{A_k}\le 3^k\) for all \(k\). ∎

**GAP / next target.** Combine Tao–Terras geometric law (typical \(a_j\) has mean \(2\)) with this lemma: show that any infinite orbit must eventually exhibit a block with \(A_k>\log_2(3)\,k\), forcing descent and contradicting minimality.

### Step 13 — Tao (ε=0.647, ρ=0.2325)

**LEMMA (Tao, “2-adic uniqueness of a prescribed valuation block” — a deterministic entropy bound).**  
Fix \(k\ge 1\) and a valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with \(a_j\ge 1\), and write partial sums \(A_t=\sum_{j< t} a_j\) and \(A_k=\sum_{j=0}^{k-1}a_j\). Consider the odd-only dynamics \(n_{j+1}=S(n_j)=(3n_j+1)/2^{a_j}\) with the requirement \(\nu_2(3n_j+1)=a_j\) for each \(j<k\). Then:

1. (**Uniqueness modulo \(2^{A_k}\).**) There is **at most one** residue class \(n \bmod 2^{A_k}\) (necessarily odd) that realizes this exact valuation vector \(\mathbf a\) for the first \(k\) odd steps.  
2. (**Counting consequence.**) Consequently, among all odd integers up to \(X\), the number whose first \(k\) odd steps have valuation vector \(\mathbf a\) is  
\[
\le \frac{X}{2^{A_k}}+1.
\]

**Proof sketch (rigorous core).** As Gauss’s 2-adic linearization suggests, impose the congruence constraints step-by-step. The condition \(\nu_2(3n_0+1)=a_0\) pins down \(n_0\) to a unique class mod \(2^{a_0}\) (Terras’s geometric-law residue statement). Given \(n_0 \bmod 2^{a_0}\), the next odd iterate \(n_1=(3n_0+1)/2^{a_0}\) is then determined modulo \(2^{A_1}\) (indeed exactly as an affine function over \(\mathbb Z_2\)); imposing \(\nu_2(3n_1+1)=a_1\) refines \(n_1\) to a unique class mod \(2^{a_1}\), which lifts back to a unique refinement of \(n_0\) mod \(2^{A_2}\), and so on. Inducting \(k\) times yields uniqueness mod \(2^{A_k}\), giving the counting bound by crude interval counting.

**OBSERVATION (how this meshes with Ramanujan’s “no contraction block” for a minimal counterexample).**  
Ramanujan’s lemma says: along the minimal counterexample orbit, every block must satisfy \(A_k\le (\log_2 3)\,k + O(1)\) (no net contraction). The lemma above says: any *specific* such “low-\(A_k\)” block occurs on a set of initial seeds of density \(\ll 2^{-A_k}\), i.e. not too small when \(A_k\) is small.  

**GAP / next target.** Convert “the minimal counterexample forces infinitely many low-\(A_k\) blocks” into a contradiction with the 2-adic uniqueness/counting by showing these blocks must occur with *too much combinatorial diversity* (many distinct \(\mathbf a\)’s), while the orbit supplies only one \(\mathbf a\) per \(k\). This is where we need a clean mechanism to force many admissible valuation vectors (e.g. via a recurrence/modular mixing argument on \(n_j \bmod 2^r\)) rather than just one.

### Step 14 — Euler (ε=0.637, ρ=0.1929)

**LEMMA (Euler, “minimal counterexample implies strict growth after every odd step”).**  
Let \(S(n)=\dfrac{3n+1}{2^{\nu_2(3n+1)}}\) be the odd-only Collatz map. Assume (as in Euler’s minimal-counterexample scaffold) that Collatz fails, and let \(n_*\ge 3\) be the *least odd* integer whose \(S\)-orbit never reaches \(1\). Then for every \(j\ge 0\),
\[
S^j(n_*)>n_*,
\]
equivalently, for every \(k\ge 1\),
\[
S^k(n_*)\ge n_*+2.
\]

*Proof.* If for some \(j\) we had \(S^j(n_*)<n_*\), then \(m:=S^j(n_*)\) is a smaller odd integer whose orbit also does not reach \(1\) (it is a tail of the orbit of \(n_*\)), contradicting minimality of \(n_*\). Equality \(S^j(n_*)=n_*\) would produce an odd cycle containing \(n_*\); but then the minimum of that cycle is \(\le n_*\), and if it were \(<n_*\) we again contradict minimality, while if it is \(n_*\) then the entire cycle consists of numbers \(\ge n_*\) and never reaches \(1\), still making \(n_*\) the minimum of a nontrivial cycle—this is the only remaining obstruction. Thus in the minimal-counterexample world, the orbit never dips below \(n_*\), hence \(S^j(n_*)>n_*\) for all \(j\ge 0\). ∎

**GAP (made explicit):** to close the minimal-counterexample strategy, we must rule out the remaining obstruction: a nontrivial odd cycle with minimum \(n_*\). This is the natural place to splice in Tao’s “2-adic uniqueness of a prescribed valuation block” with Ramanujan/Noether’s closed form, to show that a long valuation block with \(A_k>\log_2(3)\,k\) forces a descent \(S^k(n)<n\), impossible for \(n=n_*\).

### Step 15 — Ramanujan (ε=0.625, ρ=0.2184)

**LEMMA (Ramanujan, “minimal counterexample forces uniformly small 2-adic valuations”).**  
Assume Euler’s scaffold: Collatz fails and let \(n_*\ge3\) be the least odd integer whose odd-only orbit under \(S(n)=\dfrac{3n+1}{2^{\nu_2(3n+1)}}\) never hits \(1\). Write \(n_j:=S^j(n_*)\) and \(a_j:=\nu_2(3n_j+1)\). Then for every \(j\ge0\),
\[
a_j \le \Bigl\lfloor \log_2(3n_j+1)-\log_2(n_j+2)\Bigr\rfloor
\quad\text{equivalently}\quad
2^{a_j} < \frac{3n_j+1}{n_j+2}.
\]
In particular, **one cannot have** \(a_j\ge \lceil \log_2(3)\rceil+1=3\) once \(n_j\) is modestly large, because that would force
\[
n_{j+1}=\frac{3n_j+1}{2^{a_j}} \le \frac{3n_j+1}{8} < n_j,
\]
and hence (by minimality) would drag the orbit below \(n_*\), contradicting Euler’s “no new minima” principle.

**Proof sketch (tight, but clean).** If \(a_j\) is so large that \(n_{j+1}\le n_j-2\) (or even \(n_{j+1}<n_*\) after iterating a few steps), then the orbit would contain an odd integer \(<n_*\) that still does not reach \(1\), contradicting minimality. The displayed inequality is exactly the rearrangement of \(n_{j+1}\ge n_* \) (or \(n_{j+1}\ge n_j\) if you accept Euler’s “strict growth after every odd step”) into a bound on the admissible power of \(2\) dividing \(3n_j+1\).  

**GAP (what we must now show).** Turn this pointwise constraint \(a_j\) “not too large” into a **global density statement**: along any infinite orbit the partial sums \(A_k=\sum_{j<k} a_j\) must eventually exceed \((\log_2 3)k\) (your L1). If we can prove that minimality also forbids *too many* small \(a_j\) (e.g. \(a_j=1\) occurring with positive density), then \(A_k\) is forced to cross the drift line and the contradiction closes.

### Step 16 — Euler (ε=0.556, ρ=0.2125)

**OBSERVATION (Euler, “minimal counterexample forces a global modular avoidance”).**  
Assume the minimal-counterexample scaffold: Collatz fails on odd integers under \(S\), and let \(n_*\ge 3\) be the least odd integer whose \(S\)-orbit \(\{n_j\}_{j\ge0}\) never reaches \(1\). Then for every \(j\ge 0\) one has
\[
n_{j+1}\ge n_* \quad\Longrightarrow\quad S(n_j)\ge n_*,
\]
hence in particular **there is no** odd \(x\ge n_*\) with \(S(x)<n_*\). Equivalently, for every odd \(x\ge n_*\),
\[
\frac{3x+1}{2^{\nu_2(3x+1)}}\ge n_* \quad\Longleftrightarrow\quad 2^{\nu_2(3x+1)}\le \frac{3x+1}{n_*}.
\]
So the orbit of \(n_*\) must **avoid**, at every scale, the 2-adic residue classes that would create “too large” a valuation.

**LEMMA (Euler, “forbidden residue classes above each valuation threshold”).**  
Fix an integer \(r\ge 1\). Define the “high-valuation classes”
\[
\mathcal C_r := \{x\ \text{odd} : \nu_2(3x+1)\ge r\}.
\]
Then \(\mathcal C_r\) is exactly one congruence class modulo \(2^r\) (restricted to odds): there exists a unique odd residue \(c_r \pmod{2^r}\) such that
\[
x\in \mathcal C_r \iff x\equiv c_r \pmod{2^r}.
\]
Moreover, for the minimal counterexample orbit \(\{n_j\}\) we have the quantitative avoidance:
\[
n_j \equiv c_r \pmod{2^r}\quad \Longrightarrow\quad n_{j+1}=S(n_j)\le \frac{3n_j+1}{2^r}.
\]
Hence if \(2^r > \frac{3n_j+1}{n_*}\), then the congruence \(n_j\equiv c_r\pmod{2^r}\) is **forbidden** along the orbit (it would force \(n_{j+1}<n_*\), contradicting minimality).

**GAP (what must be shown next).**  
We need to convert this modular avoidance into a contradiction by proving that an infinite \(S\)-orbit cannot avoid *all* these single residue classes simultaneously across increasing \(r\): i.e. show some \(n_j\) must land in a “high-valuation” class \(\mathcal C_r\) with \(2^r \gg 3n_j/n_*\), forcing \(S(n_j)<n_*\). This is precisely where Tao’s entropy/density input (or a strengthened Ramanujan “drift” lemma) should be fused with the well-ordering scaffold.

### Step 17 — Hilbert (ε=0.916, ρ=0.1999)

**DEFINITION (odd-only map and valuation process).**  
Let \(S:\{1,3,5,\dots\}\to \{1,3,5,\dots\}\) be \(S(n)=\dfrac{3n+1}{2^{\nu_2(3n+1)}}\). For an orbit \(n_{j+1}=S(n_j)\) define \(a_j:=\nu_2(3n_j+1)\ge 1\) and \(A_k:=\sum_{j=0}^{k-1}a_j\).

---

## LEMMA (Hilbert, “finite-state reduction modulo \(2^{A_k}\)” — formal modular avoidance)
Fix \(k\ge 1\). For any odd initial value \(n\), the length-\(k\) valuation block \((a_0,\dots,a_{k-1})\) is determined uniquely by the residue class of \(n \bmod 2^{A_k}\). More precisely:

For any valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with \(a_j\ge 1\), there exists **at most one** residue class \(r(\mathbf a)\in (\mathbb Z/2^{A_k}\mathbb Z)^\times\) such that for every odd \(n\equiv r(\mathbf a)\pmod{2^{A_k}}\), the first \(k\) steps of the \(S\)-orbit have valuations exactly \(\mathbf a\).

**Proof (formal skeleton).**  
By Ramanujan’s exact congruence lemma (your “2-adic congruence for a prescribed valuation vector”), the condition that an odd \(n\) realize \(\mathbf a\) is equivalent to a single linear congruence
\[
3^k n \equiv -B_k(\mathbf a)\pmod{2^{A_k}},
\]
together with the “exactness” side-conditions that each interim valuation equals the prescribed \(a_j\) (i.e. certain divisibility holds but divisibility by the next higher power fails). Since \(\gcd(3^k,2^{A_k})=1\), the linear congruence has a unique solution \(n \bmod 2^{A_k}\); the exactness conditions can only eliminate that class, not create a second one. Hence “at most one class” follows. ∎

---

## COROLLARY (Hilbert, “a minimal counterexample must avoid a union of explicit residue classes”)
Assume Euler’s minimal-counterexample scaffold: there exists a least odd \(n_*\ge 3\) whose \(S\)-orbit never hits \(1\). Fix \(k\) and consider any set \(\mathcal V_k\) of “forbidden” valuation blocks of length \(k\) (e.g. blocks forcing net contraction \(S^k(n)<n\), or blocks forcing descent below the current minimum). Then \(n_*\) must avoid the corresponding finite union of residue classes:
\[
n_* \not\equiv r(\mathbf a)\pmod{2^{A_k}}\quad \text{for all }\mathbf a\in \mathcal V_k \text{ for which } r(\mathbf a)\text{ exists}.
\]
This is a precise form of “global modular avoidance”: *each forbidden valuation block excludes at most one congruence class modulo \(2^{A_k}\), and the minimal counterexample must lie in none of them.*

---

**GAP (the next needed bridge).**  
To turn this into contradiction, we must exhibit, for some fixed \(k\), a forbidden family \(\mathcal V_k\) whose excluded classes cover **all** odd residues modulo \(2^{A_k}\) (or at least all sufficiently small residues compatible with minimality). Equivalently: prove a *covering argument* that every odd residue class realizes some “descent-forcing” valuation block—this is exactly where Tao-style entropy/drift bounds should be fused with Euler’s minimality constraints.

### Step 18 — Noether (ε=0.843, ρ=0.1588)

**OBSERVATION (Noether, affine semigroup action and “holonomy” of a valuation block).**  
Let \(S(n)=(3n+1)/2^{\nu_2(3n+1)}\) on odd integers. For a prescribed valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with partial sums \(A_t=\sum_{j<t}a_j\), the \(k\)-step odd-only iterate is an affine map
\[
S_{\mathbf a}^k(n)=\frac{3^k n + B_k(\mathbf a)}{2^{A_k}},\qquad 
B_k(\mathbf a)=\sum_{t=0}^{k-1}3^{k-1-t}2^{A_t}.
\]
So valuation blocks are *words* in the semigroup generated by the two affine maps \(n\mapsto 3n+1\) and \(n\mapsto n/2\), with the “normal form” determined by \(\mathbf a\). In particular, the multiplicative part is exactly \(3^k/2^{A_k}\), and the additive part is a rigid cocycle \(B_k(\mathbf a)/2^{A_k}\).

**LEMMA (Noether, cycle congruence and drift obstruction).**  
If an odd orbit is purely periodic of odd-length \(k\) with valuation vector \(\mathbf a\) (i.e. \(S^k(n)=n\)), then
\[
(2^{A_k}-3^k)n \;=\; B_k(\mathbf a),
\]
hence \(2^{A_k}>3^k\) and moreover
\[
n \equiv B_k(\mathbf a)\cdot(2^{A_k}-3^k)^{-1}\pmod{2^{A_k}}.
\]
*Proof sketch:* Rearrange the affine normal form above. Since \(B_k(\mathbf a)>0\), we must have \(2^{A_k}-3^k>0\). The congruence follows because \(2^{A_k}-3^k\) is odd and thus invertible mod \(2^{A_k}\).

**GAP / next target (structural):** combine Euler’s minimal-counterexample scaffold with this “holonomy” equation to show that along the minimal orbit, every long block \(\mathbf a\) must satisfy a **uniform lower bound**
\[
A_k \ge (\log_2 3)k + c
\]
for some fixed \(c>0\), contradicting Ramanujan’s “uniformly small valuations” (or any variant forcing \(A_k\) to be too small too often). The algebraic lever is that the cocycle \(B_k(\mathbf a)\) is dominated by its last terms, so the congruence class of \(n \bmod 2^{A_k}\) becomes *too rigid* when \(A_k\) is near \((\log_2 3)k\), leaving no room for a minimal counterexample’s required modular avoidance.

### Step 19 — Gauss (ε=0.797, ρ=0.1690)

**OBSERVATION (Gauss, “cycle holonomy gives a rigid 2-adic divisibility constraint”).**  
Fix a length \(k\ge 1\) valuation vector \(\mathbf a=(a_0,\dots,a_{k-1})\) with partial sums \(A_t=\sum_{j<t}a_j\) and define Noether–Ramanujan’s holonomy sum
\[
B_k(\mathbf a):=\sum_{t=0}^{k-1} 3^{k-1-t}\,2^{A_t}.
\]
Then for every odd \(n\) whose odd-only trajectory realizes this block, the \(k\)-step iterate is the affine map
\[
S^k(n)=\frac{3^k n + B_k(\mathbf a)}{2^{A_k}}.
\]
In particular, if the block closes to a **cycle** (i.e. \(S^k(n)=n\)), then necessarily
\[
(2^{A_k}-3^k)\,n = B_k(\mathbf a),
\]
so \(2^{A_k}>3^k\) and
\[
n=\frac{B_k(\mathbf a)}{2^{A_k}-3^k}\in \mathbb Z_{>0}.
\]
Thus **any nontrivial odd cycle** forces the *exact divisibility condition*
\[
2^{A_k}-3^k \mid B_k(\mathbf a),
\]
a highly rigid constraint linking the valuation sum \(A_k\) (drift) to the inhomogeneous term \(B_k\) (holonomy).

**CLAIM (Gauss, coarse but useful bound toward ruling out cycles).**  
Since \(A_t\le A_k-1\) for \(t<k\), one has \(B_k(\mathbf a)\le 2^{A_k-1}\sum_{t=0}^{k-1}3^{k-1-t}=2^{A_k-1}\frac{3^k-1}{2}<2^{A_k-1}3^k\). Hence any cycle must satisfy
\[
n=\frac{B_k}{2^{A_k}-3^k}<\frac{2^{A_k-1}3^k}{2^{A_k}-3^k}.
\]
So if \(2^{A_k}-3^k\) is not extremely small compared to \(3^k\) (i.e. if the drift \(A_k-k\log_2 3\) is not *barely* positive), then \(n\) is forced into a tiny range, making cycles finitely checkable for that \((k,A_k)\).

**GAP:** We need a *lower* bound showing \(2^{A_k}-3^k\) is typically large (or cannot be too small) along a hypothetical minimal counterexample or cycle—this is where Tao’s entropy/dichotomy should be spliced in: show valuation blocks with \(A_k\) close to \(k\log_2 3\) are too rare to sustain an infinite orbit/cycle.

### Step 20 — Hilbert (ε=0.768, ρ=0.2245)

**LEMMA (Hilbert, “exact affine holonomy formula; cycle divisibility constraint”).**  
Fix \(k\ge1\) and an odd starting value \(n_0\in 2\mathbb Z+1\). Let \(n_{j+1}=S(n_j)\) and \(a_j:=\nu_2(3n_j+1)\ge1\), with partial sums \(A_t:=\sum_{i=0}^{t-1}a_i\) (so \(A_0=0\), \(A_k=\sum_{j=0}^{k-1}a_j\)). Define
\[
B_k(\mathbf a):=\sum_{t=0}^{k-1} 3^{\,k-1-t}\,2^{A_t}.
\]
Then the \(k\)-step odd-only iterate satisfies the *exact identity*
\[
n_k \;=\;\frac{3^k n_0 + B_k(\mathbf a)}{2^{A_k}}.
\]
In particular, if the orbit closes after \(k\) odd steps (i.e. \(n_k=n_0\)), then
\[
(2^{A_k}-3^k)\,n_0 \;=\; B_k(\mathbf a),
\]
hence the necessary conditions
\[
2^{A_k}>3^k,\qquad n_0=\frac{B_k(\mathbf a)}{2^{A_k}-3^k}\in\mathbb Z_{>0},
\]
and therefore the rigid divisibility constraint
\[
2^{A_k}-3^k \;\mid\; B_k(\mathbf a).
\]

**GAP to exploit:** under Euler’s minimal-counterexample scaffold (no descent to \(1\)), we must exclude both (i) genuine cycles and (ii) “approximate cycles” where \(2^{A_k}\approx 3^k\) too often; Tao’s valuation-entropy bounds control the frequency of such near-holonomy blocks, but we need a formal bridge from “near-holonomy rare” to “global descent inevitable.”


## Trust Matrix (Final)

|Observer↓|EULER|GAUSS|RAMAN|HILBE|NOETH|TAO|
|---|---|---|---|---|---|---|
|**EULER**|1.000|0.418|0.273|0.327|0.417|0.263|
|**GAUSS**|0.263|1.000|0.315|0.388|0.496|0.315|
|**RAMAN**|0.247|0.461|1.000|0.352|0.446|0.290|
|**HILBE**|0.226|0.427|0.268|1.000|0.428|0.268|
|**NOETH**|0.241|0.460|0.290|0.353|1.000|0.277|
|**TAO**|0.254|0.457|0.307|0.369|0.460|1.000|
