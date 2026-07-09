**Experiment Plan for Multi-stages Rank Order Tournament**

This project extends TEL--PPO from the one-stage Lazear--Rosen rank-order tournament \[1\] to dynamic multi-stage rank-order tournaments with terminal rewards. TEL--PPO is used as a computational solver that produces candidate state-dependent effort functions, ${\widehat{e}}_{t}(d)$, where *t* is the stage and *d* is the current cumulative score gap.

The objective is not to treat PPO training stability as evidence of equilibrium. Instead, equilibrium credibility is established through analytical recovery when closed-form benchmarks are available and through independent best-response verification when closed-form benchmarks are unavailable.

The key methodological pipeline is: one-stage validation $\rightarrow$ two-stage theoretical recovery $\rightarrow$ two-stage verifier calibration $\rightarrow$ three-stage verified approximate equilibrium $\rightarrow$ multi-stage benchmark.

The two-stage tournament plays a dual role. First, because it admits a closed-form symmetric Markov equilibrium benchmark, it tests whether TEL--PPO can recover a dynamic equilibrium effort function. Second, it provides a calibration environment for the independent best-response verifier. For three-stage and longer tournaments, where closed-form benchmarks are generally unavailable, the same calibrated verifier is used to assess whether learned effort functions have low exploitability.

Thus, TEL--PPO is used as a candidate equilibrium solver, while equilibrium quality is certified independently through best-response verification.

1.  **Model: Dynamic Lazear-Rosen rank-order Tournament with terminal rewards**

We study a dynamic Lazear--Rosen rank-order tournament in which all players remain active until the final stage. Early-stage performance affects future incentives through the evolving cumulative score gap, but early-stage wins do not determine the final winner. Rewards are paid only after the final cumulative performance is realized.

We begin with the two-player cases, N = 2, and consider finite-horizon tournament with stage T $\in \{ 2,3,4,\ldots\}$. We assume public interim score feedback: at the beginning of each stage $t$, both players observe the current cumulative score gap.

1.  **Stage performance**

For player $i$ in stage $t$, realized performance is,

$y_{i,t} = e_{i,t} + \varepsilon_{i,t}$, $e_{i,t} \in \lbrack 0,\overset{ˉ}{e}\rbrack$,

where $e_{i,t}$ is costly effort and $\varepsilon_{i,t}$ is random performance noise. To remain consistent with the Lazear-Rosen benchmark, the baseline assumes $\varepsilon_{i,t}\ \sim\ U\lbrack - q,q\rbrack$ and i.i.d. across players and stages. The parameter $q$ controls the magnitude of performance noise.

2.  **Cumulative score**

Cumulative performance after stage $t$ is：

$$S_{i,t} = \sum_{\tau = 1}^{t}y_{i,\tau}$$

with initial score: $S_{i,0} = 0$

The final winner is determined by the comparison of cumulative scores. Player $i$ wins if

$$S_{i,T} > S_{j,T}$$

Equivalently, player $i$ wins if the final score gap is positive.

3.  **Score gap state and State Transition**

Define player $i$\'s cumulative score gap over player $j$ at the beginning of stage $t$ as:

$d_{t} = S_{i,t - 1} - S_{j,t - 1}$.

Since both players start from zero cumulative score,

$$d_{1} = S_{i,0} - S_{j,0} = 0.
$$The score gap evolves according to:

$$d_{t + 1} = S_{i,t} - S_{j,t}.$$

Using:

$$S_{i,t} = S_{i,t - 1} + y_{i,t},$$

we obtain:

$$d_{t + 1} = d_{t} + y_{i,t} - y_{j,t}.$$

Substituting the performance equation gives:

$$d_{t + 1} = d_{t} + e_{i,t} - e_{j,t} + \varepsilon_{i,t} - \varepsilon_{j,t}$$

Let:

$$\xi_{t} = \varepsilon_{i,t} - \varepsilon_{j,t}$$

Then the state transition becomes:

$$d_{t + 1} = d_{t} + e_{i,t} - e_{j,t} + \xi_{t}$$

If $\varepsilon_{i,t} \sim U\lbrack - q,q\rbrack$, then $\xi_{t}$ follows a triangular distribution on $\lbrack - 2q,2q\rbrack$. This distribution is used to evaluate expectations over shock differences in the theoretical two-stage benchmark and in the independent numerical best-response verifier.

4.  **Terminal reward and Payoff**

Prizes are awarded only after the final stage. Player $i$\'s terminal reward is determined by the final score gap:

$$d_{T + 1} = S_{i,T} - S_{j,T}.$$

Define player $i$\'s terminal reward function as:

$$R_{i}(d_{T + 1}) = \left\{ \begin{matrix}
W_{H}, & d_{T + 1} > 0 \\
W_{L}, & d_{T + 1} < 0 \\
\frac{W_{H} + W_{L}}{2}, & d_{T + 1} = 0
\end{matrix} \right.\ $$

Thus, player $i$'s expected payoff is：

$$U_{i} = E\left\lbrack R_{i}(d_{T + 1}) - \sum_{t = 1}^{T}c(e_{i,t}) \right\rbrack$$

with quadratic effort cost

$$c(e_{i,t}) = {ke_{i.t}}^{2}$$

This formulation emphasizes that early-stage performance affects payoffs only through its effect on the evolving score gap, not through intermediate rewards or elimination.

2.  **Equilibrium object and Theoretical Benchmarks**

**2.1 State-Dependent Effort Function**

In the dynamic tournament, the equilibrium object is not a single effort level, but a state-dependent effort function. For each stage $t$, the equilibrium effort function is

$$e_{t}^{*}(d):D \rightarrow \lbrack 0,\overset{ˉ}{e}\rbrack,
$$where $D$ denotes the state space of possible cumulative score gaps, and $d \in D$ is the current cumulative score gap at the beginning of the stage.

The equilibrium effort-functions profile is

$$e^{*} = \left\{ e_{1}^{*},e_{2}^{*},\ldots,e_{T}^{*} \right\}.$$

For a symmetric two-player tournament, if player $i$\'s score gap at the beginning of stage $t$ is $d_{t},$

then player $i$\'s equilibrium effort is

$$e_{i,t}^{*} = e_{t}^{*}(d_{t}).$$

From player $j$\'s perspective, the score gap is $- d_{t}$. Therefore, player $j$\'s equilibrium effort is

$$e_{j,t}^{*} = e_{t}^{*}( - d_{t}).$$

Thus, symmetry means that both players use the same effort function, not that they necessarily choose the same effort level in every realized state.

**2.2 TEL-PPO Policy and Learned Effort Function**

TEL--PPO learns a stochastic policy,

$$\pi_{\theta}(e \mid t,d),$$

but the economic object recovered and reported is the policy-implied [mean/mode]{.mark} effort,

$${\widehat{e}}_{t}(d) = E_{e \sim \pi_{\theta}( \cdot \mid t,d)}\lbrack e\rbrack.$$

The recovery target is

$${\widehat{e}}_{t}(d) \approx e_{t}^{*}(d).
$$The verifier evaluates the deterministic policy-implied [mean/mode]{.mark} effort function ${\widehat{e}}_{t}(d)$, rather than sampled stochastic actions. This is consistent with the economic target, which is the deterministic equilibrium effort function.

**2.3 Markov State sufficiency**

In the two-player dynamic Lazear--Rosen tournament, suppose that performance shocks are independent across players and stages, and that the terminal reward depends only on the final score difference. Then the pair ($t,\ d_{t})$ is a sufficient Markov state.

The full history of past performances is not payoff-relevant once $t$ and $d_{t}$ are known. The current stage determines the number of remaining stages, and the current score gap summarizes all past performance differences that affect the final winning probability. Therefore, the equilibrium effort can be represented as $e_{t}^{*}(d_{t})$, and TEL-PPO should condition on ($t,\ d_{t})$, not the entire performance history.

**2.4 Symmetric Markov equilibrium condition**

Let $e = \{ e_{1},e_{2},\ldots,e_{T}\}$ denote a symmetric Markov effort-function profile, where $e_{t}(d)$ is the effort chosen at stage $t$ when the score gap is $d$. Given this effort-function profile, player $i$\'s continuation value at state ($t,\ d_{t})$ is:

$$V_{t}^{e}(d) = - c(\pi_{t}(d)) + E\left\lbrack V_{t + 1}^{e}\left( d + \pi_{t}(d) - \pi_{t}( - d) + \xi_{t} \right) \right\rbrack$$

with terminal value：

$$V_{T + 1}^{e}(d) = R_{i}(d).$$

A symmetric Markov perfect equilibrium \[2\] effort function $e^{*}\ $satisfies, for every stage $t$ and score gap $d$,

$$e_{t}^{*}(d) \in \arg\underset{e \in \lbrack 0,\overset{ˉ}{e}\rbrack}{\max}\left\{ - c(e) + E\left\lbrack V_{t + 1}^{e^{*}}\left( d + e - e_{t}^{*}( - d) + \xi_{t} \right) \right\rbrack \right\}.$$

Here, "perfect" means sequential optimality: the effort function must be a best response at every payoff-relevant state $(t,d)$, not only along the equilibrium path starting from $d_{1} = 0.$

This Bellman condition provides the theoretical basis for both the closed-form two-stage benchmark and the best-response verification procedure used in three-stage and longer tournaments.

**2.5 Closed-Form Two-Stage Benchmark**

For $T=2$, the dynamic Lazear--Rosen tournament admits a closed-form symmetric Markov benchmark under the interior, unconstrained, locally concave region described below. The two-stage equilibrium consists of the first-stage effort at the initial score gap and the second-stage score-gap-dependent effort function:

$$
e_{CF}^{*}=\{e_{1,CF}^{*}(0),e_{2,CF}^{*}(d)\}.
$$

We write

$$
e_{1,CF}^{*}(0)=g_1(W_H,W_L,k,q),
$$

and

$$
e_{2,CF}^{*}(d)=g_2(d;W_H,W_L,k,q).
$$

Let the prize spread be

$$
\Delta W=W_H-W_L>0,
$$

and let the stage-shock difference be

$$
\xi=\varepsilon_i-\varepsilon_j.
$$

Because $\varepsilon_i,\varepsilon_j\sim U[-q,q]$, the density of $\xi$ is triangular:

$$
f_\xi(x)=
\begin{cases}
\dfrac{2q-|x|}{4q^2}, & |x|\le 2q,\\[6pt]
0, & |x|>2q.
\end{cases}
$$

Throughout the closed-form benchmark below, use the quadratic cost convention

$$
c(e)=\frac{k}{2}e^2,\qquad c'(e)=ke,\qquad c''(e)=k.
$$

If the implementation instead uses $c(e)=ke^2$, replace $k$ by $2k$ in the denominator of all effort formulas.

**Final-stage derivation of $g_2(d)$.** In stage 2, player $i$ enters with score gap $d$. Given opponent effort $e_j$, player $i$ solves

$$
\max_{e_i\in[0,\bar e]}
\left\{
W_L+\Delta W\Pr(d+e_i-e_j+\xi>0)-\frac{k}{2}e_i^2
\right\}.
$$

Using symmetry of $\xi$,

$$
\Pr(d+e_i-e_j+\xi>0)=F_\xi(d+e_i-e_j),
$$

so the first-order condition for an interior optimum is

$$
ke_i=\Delta W f_\xi(d+e_i-e_j).
$$

In a symmetric Markov equilibrium, player $i$ at state $d$ uses $e_2^*(d)$ and player $j$ at the same history views the state as $-d$, so player $j$ uses $e_2^*(-d)$. The closed-form symmetric benchmark is even in $d$, so

$$
e_2^*(d)=e_2^*(-d).
$$

Therefore the argument of the density becomes $d$, and the final-stage benchmark is

$$
g_2(d;W_H,W_L,k,q)=\frac{\Delta W}{k}f_\xi(d).
$$

Equivalently,

$$
g_2(d;W_H,W_L,k,q)=
\begin{cases}
\dfrac{\Delta W}{k}\cdot\dfrac{2q-|d|}{4q^2}, & |d|\le 2q,\\[8pt]
0, & |d|>2q.
\end{cases}
$$

With the bounded effort constraint, the implementable benchmark is

$$
e_{2,CF}^{*}(d)=\Pi_{[0,\bar e]}\left[g_2(d;W_H,W_L,k,q)\right].
$$

Thus, $g_2(d)$ is highest when the contest is close $(d=0)$ and decreases linearly as $|d|$ increases, reaching zero outside the support $[-2q,2q]$.

**Second-order condition for $g_2(d)$.** For an interior final-stage optimum, the second derivative of player $i$'s objective with respect to $e_i$ is

$$
\frac{\partial^2 U_{i,2}}{\partial e_i^2}
=
\Delta W f_\xi'(d+e_i-e_j)-k.
$$

For the triangular density,

$$
f_\xi'(x)=
\begin{cases}
\dfrac{1}{4q^2}, & -2q<x<0,\\[6pt]
-\dfrac{1}{4q^2}, & 0<x<2q.
\end{cases}
$$

The potentially problematic region is $x<0$, where the density is increasing. A sufficient local concavity condition is therefore

$$
k>\frac{\Delta W}{4q^2}.
$$

Equivalently,

$$
q>q_{SOC}\equiv \sqrt{\frac{\Delta W}{4k}}.
$$

This is the main $q_{crit}$ condition for the unconstrained closed-form final-stage benchmark.

**First-stage derivation of $g_1$.** At the start of stage 1, the score gap is $d_1=0$. Suppose the opponent uses first-stage effort $g_1$. If player $i$ chooses $e_i$, the stage-2 score gap becomes

$$
d_2=e_i-g_1+\xi_1.
$$

Player $i$'s stage-1 problem is

$$
\max_{e_i\in[0,\bar e]}
\left\{
-\frac{k}{2}e_i^2+E[V_2^*(e_i-g_1+\xi_1)]
\right\}.
$$

The interior first-order condition is

$$
ke_i=E\left[V_2^{*\prime}(e_i-g_1+\xi_1)\right].
$$

At a symmetric equilibrium, $e_i=g_1$, so

$$
kg_1=E\left[V_2^{*\prime}(\xi_1)\right].
$$

Using the final-stage benchmark above, the stage-2 value can be written as

$$
V_2^*(d)=W_L+\Delta W F_\xi(d)-\frac{k}{2}g_2(d)^2.
$$

Therefore,

$$
V_2^{*\prime}(d)=\Delta W f_\xi(d)-kg_2(d)g_2'(d).
$$

Because $g_2(d)=\frac{\Delta W}{k}f_\xi(d)$ and the triangular density is symmetric, the cost-slope terms cancel in expectation across positive and negative score gaps. Hence

$$
E[V_2^{*\prime}(\xi_1)]
=
\Delta W\int_{-2q}^{2q}f_\xi(x)^2dx.
$$

For the triangular density,

$$
\int_{-2q}^{2q}f_\xi(x)^2dx=\frac{1}{3q}.
$$

Thus the closed-form first-stage benchmark is

$$
g_1(W_H,W_L,k,q)=\frac{\Delta W}{3kq}.
$$

With the bounded effort constraint,

$$
e_{1,CF}^{*}(0)=\Pi_{[0,\bar e]}\left[g_1(W_H,W_L,k,q)\right].
$$

For the parameter example $\Delta W=2$, $k=1/3500$, and $q=50$,

$$
g_1=\frac{2}{3(1/3500)(50)}\approx 46.7.
$$

**Second-order condition for $g_1$.** The second derivative of the stage-1 objective at the symmetric candidate is

$$
\frac{\partial^2 U_{i,1}}{\partial e_i^2}
=
-k+E[V_2^{*\prime\prime}(\xi_1)].
$$

Under the triangular-density benchmark,

$$
E[V_2^{*\prime\prime}(\xi_1)]
=-\frac{(\Delta W)^2}{16kq^4},
$$

so

$$
\frac{\partial^2 U_{i,1}}{\partial e_i^2}
=
-k-\frac{(\Delta W)^2}{16kq^4}<0.
$$

Therefore the first-stage benchmark satisfies the SOC whenever the final-stage benchmark is in the valid interior region.

**Participation constraint.** Let $\bar U$ denote the outside option. In the symmetric two-stage benchmark, the ex ante expected prize is

$$
E[\text{prize}]=\frac{W_H+W_L}{2}.
$$

The expected stage-1 effort cost is

$$
c(g_1)=\frac{k}{2}\left(\frac{\Delta W}{3kq}\right)^2
=\frac{(\Delta W)^2}{18kq^2}.
$$

The expected stage-2 effort cost is

$$
E[c(g_2(d_2))]
=\frac{k}{2}E[g_2(\xi_1)^2]
=\frac{(\Delta W)^2}{16kq^2}.
$$

Therefore the ex ante participation constraint is

$$
\frac{W_H+W_L}{2}
-
\frac{17(\Delta W)^2}{144kq^2}
\ge \bar U.
$$

Equivalently, if $\frac{W_H+W_L}{2}>\bar U$, the participation constraint imposes

$$
q\ge q_{PC}
\equiv
\Delta W\sqrt{
\frac{17}{144k\left(\frac{W_H+W_L}{2}-\bar U\right)}
}.
$$

If there is no outside option in the experiment, set $\bar U=0$ and report that the participation constraint is automatically satisfied only when the above inequality holds.

**Effective closed-form validity region and $q_{crit}$.** The unprojected closed-form benchmark is valid only in the region where the interior solution is locally concave, the effort upper bound does not bind, and the participation constraint is satisfied.

The SOC condition is

$$
q>q_{SOC}=\sqrt{\frac{\Delta W}{4k}}.
$$

The final-stage effort bound requires

$$
g_2(0)=\frac{\Delta W}{2kq}\le \bar e,
$$

or

$$
q\ge q_{B,2}=\frac{\Delta W}{2k\bar e}.
$$

The first-stage effort bound requires

$$
g_1=\frac{\Delta W}{3kq}\le \bar e,
$$

or

$$
q\ge q_{B,1}=\frac{\Delta W}{3k\bar e}.
$$

Combining the constraints gives the effective threshold

$$
q_{crit}
=
\max\left\{
q_{SOC},
q_{B,2},
q_{B,1},
q_{PC}
\right\}.
$$

The closed-form comparison should therefore be restricted to

$$
q\ge q_{crit}
\quad\text{and}\quad
|d|\le 2q.
$$

For $|d|>2q$, the final-stage marginal winning-probability density is zero, so the unprojected benchmark gives

$$
g_2(d)=0.
$$

If $q<q_{crit}$, the simple closed-form benchmark may fail because the final-stage objective may be locally non-concave, effort bounds may bind, or participation may fail. Those parameter cases should be excluded from the main closed-form recovery test or reported separately as boundary/corner cases.

In the empirical comparison, TEL--PPO is evaluated against the closed-form benchmark only over the parameter and state regions where the corresponding analytical expression is valid.

The two-stage TEL-PPO recovery targets are

$$
\widehat e_1(0)\approx e_{1,CF}^{*}(0),
$$

and

$$
\widehat e_2(d)\approx e_{2,CF}^{*}(d).
$$

**2.6 Exploitability-Based Multi-stage approximate MPE certificate**

For $T \geq 3$, closed-form equilibrium is generally unavailable for the model class considered here. Let $\widehat{e} = \{{\widehat{e}}_{1},\ {\widehat{e}}_{2},\ \ldots,\ {\widehat{e}}_{T}\}\ $denote the effort functions recovered from TEL-PPO. Holding the opponent's learned effort function fixed, define player $i$'s best response \[4\] \[5\] as

$${BR}_{i}\left( {\widehat{e}}_{- i\ } \right)\  \in arg\max_{e_{i}}U_{i}(e_{i},{\widehat{e}}_{- i})$$

Exploitability is defined as

$$Exp\left( \widehat{e} \right) = \ U_{i}({BR}_{i}\left( {\widehat{e}}_{- i}),{\widehat{e}}_{- i}, \right) - \ U_{i}\left( {\widehat{e}}_{i},{\widehat{e}}_{- i}, \right)$$

The learned effort function is certified as an $\varepsilon$-approximate Markov perfect equilibrium if

$${EXP}^{UCB}\left( \widehat{e} \right) \leq \ \varepsilon$$

where

$${EXP}^{UCB}\left( \widehat{e} \right) = EXP\left( \widehat{e} \right) + 1.96\ SE(EXP)$$

Here, $EXP\left( \widehat{e} \right)$ is the estimated exploitability, $SE(EXP)$ is the standard error of the exploitability estimate(calculated by paired simulation-?depend on the result), and 1.96 corresponds to an approximate 95% confidence interval. ${EXP}^{UCB}\left( \widehat{e} \right)$ denotes the upper confidence bound(UCB) of the estimated exploitability.

Because exploitability is estimated using simulation and numerical approximation, certification is based on the upper confidence bound rather than the point estimate alone. Thus, a learned effort function in accepted as an $\varepsilon$-approximate Markov perfect equilibrium only if ${EXP}^{UCB}\left( \widehat{e} \right)$ is below the tolerance level $\varepsilon$.

Therefore, for three-stage and longer tournaments, TEL-PPO is used to compute candidate effort functions, while equilibrium quality is assessed independently through best-response verification.

3.  **TEL-PPO solver**

TEL--PPO is used to generate candidate state-dependent effort functions. It is not itself the equilibrium proof. Equilibrium credibility comes from analytical comparison in the two-stage case and independent best-response verification in three-stage and longer tournaments.

**3.1 State input and Normalization**

For the two-player model, the state input is:

$$s_{t} = (t,{\widetilde{d}}_{t},\ W_{H},W_{L},k),
$$where ${\widetilde{d}}_{t}$ is the normalized score gap.

A practical normalization is

$$s_{t} = \left\lbrack \frac{t}{T},\frac{d_{t}}{q\sqrt{t}},W_{H},W_{L},k,q \right\rbrack$$

The score-gap normalization stabilizes training because the scale of $d_{t}$ increases with noise and accumulated stages.

**3.2 Beta Policy for Bounded Continuous Effort**

TEL-PPO keeps the Beta policy \[3\] for bounded continuous effort as the one-stage paper. The actor outputs positive parameters $\alpha_{\theta}(t,d) > 0$ and $\beta_{\theta}(s_{t}) > 0$ . The normalized action is sampled as

$a_{t} \sim Beta(\alpha_{\theta}(t,d),\beta_{\theta}(t,d))$,

Where

$$a_{t} \in \lbrack 0,1\rbrack.$$

The normalized action is mapped to effort by:

$$e_{t} = e_{\min} + a_{t}(e_{\max} - e_{\min})$$

The policy-implied [mean/(mode]{.mark}) effort is

$${\widehat{e}}_{t}(d) = \overset{ˉ}{e}\frac{\alpha_{\theta}(t,d)}{\alpha_{\theta}(t,d) + \beta_{\theta}(t,d)}.
$$

This deterministic [mean/(mode) effort]{.mark} is the object compared against closed-form benchmarks and passed to the independent verifier.

**3.3 Shared symmetric policy**

Because players are identical, the main model uses a shared symmetric policy:

$$\pi_{\theta}(e \mid t,d).$$

Player $i$ observes $d_{t}$ and player $j$ observes $- d_{t}$, therefore,

$e_{i,t} \sim \pi_{\theta}( \cdot \mid t,d_{t})$ and$
$$$e_{j,t} \sim \pi_{\theta}( \cdot \mid t, - d_{t})$$

This respects symmetry while allowing leaders and followers to exert different effort levels when the score gap is nonzero.

**3.4 Reward structure, Return, and GAE**

Intermediate-stage rewards contain only effort costs:

$$r_{i,t} = - c\left( e_{i,t} \right),\ t < T$$

At the final stage, the terminal prize is added:

$$r_{i,T} = R_{i}(d_{T + 1}) - c(e_{i,T})$$

The total economic return is:

$$G_{i} = \sum_{t = 1}^{T}r_{i,t}
$$In PPO training, the return from stage $t$ is computed as

$$G_{i,t}^{\gamma} = \sum_{\tau = t}^{T}\gamma^{\tau - t}r_{i,\tau}.$$

We set

$$\gamma = 1
$$so that the RL return coincides with the finite-horizon economic payoff rather than an artificially discounted objective.

For advantage estimation, we use generalized advantage estimation. The temporal-difference residual is

$$\delta_{i,t} = r_{i,t} + \gamma V_{\phi}\left( s_{i,t + 1} \right) - V_{\phi}\left( s_{i,t} \right).$$

The GAE advantage is

$${\widehat{A}}_{i,t}^{GAE(\gamma,\lambda)} = \sum_{\mathcal{l} = 0}^{T - t}{(\gamma\lambda})^{\mathcal{l}}\delta_{i,t + \mathcal{l}}.$$

The main specification uses

$$\gamma = 1,\lambda = 1,$$

with $\lambda = 0.95$ as a robustness check.

5.  **Policy Extraction**

After training, TEL--PPO produces a stochastic policy

$$\pi_{\theta}(e \mid t,d).$$

For economic interpretation and verification, we extract the policy-implied mean(?mode) effort:

$${\widehat{e}}_{t}(d) = E_{\pi_{\theta}}\lbrack e \mid t,d\rbrack.$$

The resulting deterministic learned effort-function profile is

$$\widehat{e} = \{{\widehat{e}}_{1},{\widehat{e}}_{2},\ldots,{\widehat{e}}_{T}\}.$$

This is the object used in closed-form comparison, exploitability estimation, and state-wise deviation analysis.

6.  **Curriculum Learning**

A curriculum can be used as a training stabilization device:

$$T = 1 \rightarrow T = 2 \rightarrow T = 3.$$

Specifically, one-stage learning can initialize final-stage intuition. The two-stage model can then be trained and validated against the closed-form benchmark. After that, the trained two-stage actor and critic can be used to initialize the three-stage model.

Curriculum learning is not part of the equilibrium definition and must not enter the verifier. It is only a training procedure. The final learned effort function must still be evaluated independently through best-response verification.

An ablation comparing training with and without curriculum can be included in the baseline three-stage case.

7.  **Training Protocol**

All experiments should use a pre-specified training protocol. The protocol should report:

1)  number of random seeds;

2)  total training episodes or environment steps;

3)  PPO learning rate;

4)  clipping parameter;

5)  entropy coefficient;

6)  actor and critic network architecture;

7)  batch size;

8)  number of PPO epochs per update;

9)  evaluation frequency;

10) stopping criterion;

11) checkpoint selection rule.

The final policy should be selected according to a pre-specified rule, not by visual fit to the theoretical benchmark. For example, the selected checkpoint may be the final checkpoint after a fixed training budget or the checkpoint with the lowest validation exploitability under a pre-specified validation procedure.

4.  **Independent Verification Framework**

The independent verifier evaluates whether the learned effort function is approximately a best response to itself under symmetry. TEL--PPO generates candidate policies; the verifier assesses equilibrium quality.

**4.1 Dynamic-Programming(DP) Best-Response Verifier**

After training, fix the learned effort function ${\widehat{e}}_{t(d)}$. The opponent\'s effort at state *d* is ${\widehat{e}}_{t( - d)}$. The dynamic-programing best-response value function is

$$V_{t}^{BR}(d) = \underset{e \in \lbrack 0,\overset{ˉ}{e}\rbrack}{\max}\left\{ - c(e) + E\left\lbrack V_{t + 1}^{BR}\left( d + e - {\widehat{e}}_{t}( - d) + \xi_{t} \right) \right\rbrack \right\}$$

The terminal condition is:

$$V_{T + 1}^{BR}(d) = R_{i}(d)$$

The payoff under the learned effort function is

$$V_{t}^{\widehat{e}}(d) = - c({\widehat{e}}_{t}(d)) + E\left\lbrack V_{t + 1}^{\widehat{e}}\left( d + {\widehat{e}}_{t}(d) - {\widehat{e}}_{t}( - d) + \xi_{t} \right) \right\rbrack,$$

with

$$V_{T + 1}^{\widehat{e}}(d) = R_{i}(d).$$

Then exploitability is:

$$EXP(\widehat{e}) = V_{1}^{BR}(0) - V_{1}^{\widehat{e}}(0)$$

A learned effort function is certified as an $\varepsilon$-approximate Markov perfect equilibrium only if

$${EXP}^{UCB}\left( \widehat{e} \right) = EXP\left( \widehat{e} \right) + 1.96\ SE(EXP) \leq \ \varepsilon$$

2.  **Verifier Calibration in the Closed-Form Two-Stage Case**

The DP verifier solves the best-response problem by backward induction over a discretized state-action space.

Let

$$\mathcal{D = \{}d_{1},\ldots,d_{M}\}$$

denote the score-gap grid and

$$\mathcal{E = \{}e_{1},\ldots,e_{K}\}$$

denote the effort grid.

The score-gap grid is defined over

$$\lbrack - D_{\max},D_{\max}\rbrack,$$

where $D_{\max}$ should be large enough to contain the relevant reachable score-gap region. A conservative choice is

$$D_{\max} = T(\overset{ˉ}{e} + 2q) + m,$$

where $m$ is an additional margin.

Because the next score gap

$$d_{t + 1} = d + e - {\widehat{e}}_{t}( - d) + \xi_{t}$$

does not necessarily lie exactly on the grid, value interpolation is used to evaluate

$$V_{t + 1}(d_{t + 1}).$$

[Expectations over]{.mark} $\xi_{t}$ [can be computed using deterministic quadrature based on the known triangular distribution of]{.mark} $\xi_{t}$[, or by Monte Carlo integration with confidence intervals. Deterministic quadrature is preferred when feasible because it reduces simulation noise.]{.mark}

**4.3 State-Wise One-Step Deviation Gap**

For any learned effort function $\widehat{e}$, define the state-action value associated with a one-step deviation as

$$Q_{t}^{\widehat{e}}(d,e) = - c(e) + E\left\lbrack V_{t + 1}^{\widehat{e}}\left( d + e - {\widehat{e}}_{t}( - d) + \xi_{t} \right) \right\rbrack
$$Here, $Q_{t}^{\widehat{e}}(d,e)$ is the payoff obtained by player $i$ when the current state is $(t,d)$, the player chooses effort $e$ in the current stage, the opponent follows the learned effort function ${\widehat{e}}_{t}( - d)$, and both players follow the learned effort functions thereafter. The term $- c(e)$ is the current effort cost, while the continuation value captures the expected payoff from the next state onward.

The one-step deviation gap is defined as:

$\Delta_{t}(d) = \underset{e}{\max}Q_{t}^{\widehat{e}}(d,e) - Q_{t}^{\widehat{e}}(d,{\widehat{e}}_{t}(d))$.

This gap measures the maximum payoff gain from deviating from the TEL--PPO learned effort ${\widehat{e}}_{t}(d)$ in a single state $(t,d)$, while keeping future behavior fixed at the learned effort function $\widehat{e}$. If

$$\Delta_{t}(d) \approx 0,$$

then the learned effort ${\widehat{e}}_{t}(d)$ is close to a best response in that state. A large value of $\Delta_{t}(d)$ indicates that a player could profitably deviate from the learned effort at that specific state.

We will report two versions of the deviation gap. The first is the worst-case deviation gap:

$$\underset{t,d}{\max}\Delta_{t}(d)
$$This is a conservative off-path metric because it searches over all stages and score gaps, including states that may occur with very low probability under the learned effort functions.

The second is the on-path deviation gap:

$$E_{d_{t} \sim \widehat{e}}\lbrack\Delta_{t}(d_{t})\rbrack$$

This measures average deviation incentives in states that are actually reached with positive probability when both players follow the learned effort functions. Therefore, the worst-case measure evaluates robustness over the full state space, while the on-path measure evaluates equilibrium quality in empirically relevant states.

Together, these state-wise deviation gaps complement the aggregate exploitability measure. Exploitability evaluates the payoff gain from a full dynamic best response starting from the initial state, while $\Delta_{t}(d)$ provides a local diagnostic of whether the learned effort function satisfies the Bellman optimality condition at each state. If both exploitability and state-wise deviation gaps are small, the learned effort function provides stronger evidence of approximate Markov perfect equilibrium behavior. []{.mark}

**4.4 Numerical Error Controls**

The independent verifier is numerical and is therefore subject to approximation error. We track four sources of verification error:

$$\epsilon_{total} = \epsilon_{BR} + \epsilon_{MC/int} + \epsilon_{grid} + \epsilon_{interp}.$$

Here, $\epsilon_{BR}$ is the optimization error of the best-response solver; $\epsilon_{MC/int}$ is Monte Carlo or numerical integration error; $\epsilon_{grid}$ is state/action discretization error; $\epsilon_{interp}$ is value interpolation error.

We do not claim a closed-form upper bound for each component. Instead, we can control these errors through:

1)  grid refinement over both score-gap and effort grids;

2)  confidence intervals for payoff and exploitability estimates;

3)  deterministic integration checks when feasible;

4)  calibration against the two-stage closed-form equilibrium.

Certification is based on

$$EXP^{UCB}(\widehat{e}),$$

not on the point estimate alone.

Thus, low exploitability must be robust to numerical approximation, not only to the best-response calculation itself.

**4.5 Two-Stage Verifier Calibration**

Because the two-stage case has a closed-form equilibrium, we will use it to calibrate the verifier before applying the verifier to three-stage tournaments.

First, run the verifier on the [closed-form two-stage equilibrium]{.mark}:

${e^{*}}_{CF} = \ {\{ e^{*}}_{1,CF}(0),\ {e^{*}}_{2,CF}(d)\}\ $.

The verifier should report exploitability close to zero:

$EXP\left( {e^{*}}_{CF} \right)\ approx\ 0$.

Second, run the verifier on deliberately misspecified two-stage policies, such as constant low effort, constant high effort, one-stage effort repeated in both stages, and a no-gap second-stage effort function.

The verifier should assign much higher exploitability to these policies:

$$EXP(\text{bad~policy}) \gg EXP(e_{CF}^{*}).
$$This confirms that the verifier has discriminatory power before it is used in the three-stage setting where no closed-form benchmark exists.

**5. Experiment plan**

1.  **Experimental Protocol**

All experiments use pre-specified parameter grids, random seeds, training budgets, and acceptance thresholds.

The two-stage thresholds should be chosen before observing the main results and calibrated using the one-stage benchmark and the two-stage verifier error floor. This avoids choosing criteria after observing outcomes.

For each experiment, report:

1)  parameter values;

2)  number of training seeds;

3)  mean and standard deviation across seeds;

4)  learned effort-function plots;

5)  exploitability estimates;

6)  confidence intervals;

7)  certification status;

8)  robustness checks.

    1.  **One-stage foundation**

Our companion one-stage tournament study validates TEL-PPO in the analytically tractable one-stage Lazear-Rosen tournament. It establishes bounded continuous effort learning, self-play stability, exploitability-based verification, and analytical benchmark recovery.

The present study uses that result as the foundation and focuses on dynamic tournaments with evolving score gaps and terminal rewards.

2.  **Two-Stage Closed-Form Recovery and Verifier Calibration**

**Objective.** The two-stage experiment has two objectives. First, it tests whether TEL--PPO recovers the closed-form dynamic equilibrium effort functions:

$$e_{1,CF}^{*}(0),e_{2,CF}^{*}(d).$$

Second, it calibrates the independent best-response verifier that will later be used for three-stage tournaments.

**Environment.** $T = 2$, $N = 2$, initial gap $d_{1} = 0$, state $s_{t} = (t,d_{t})$. The tournament has public interim score feedback and terminal rewards only, with accumulated effort costs.

Use only parameter regions where the closed-form two-stage benchmark is valid. Boundary or corner cases should be either excluded from the main validation grid or reported separately.

**Parameter Grid(try the following set of parameters first)**

  ------------------------------------------------------------------------------------------------------------------------
  **Parameter**                               **Suggested values**     **[Purpose]{.mark}**
  ------------------------------------------- ------------------------ ---------------------------------------------------
  Prize spread $\Delta W = \ W_{H} - W_{L}$   $$\Delta W = \ 6 - 2$$   [Tests incentive strength]{.mark}

  Cost k                                      1/3500                   [Tests effort-cost sensitivity]{.mark}

  Noise q                                     50                       [Tests stochastic performance uncertainty]{.mark}

  Theoretical Equilibrium Effort              46.7                     

  Effort upper bound $\overline{e}$           \[0,100\]                [Avoids unintended boundary equilibria]{.mark}
  ------------------------------------------------------------------------------------------------------------------------

[Start by varying]{.mark} $q$ [while holding]{.mark} $\Delta W$[,]{.mark} $k$[, and]{.mark} $\overset{ˉ}{e}$ [fixed. After the pipeline works, expand to the full grid.]{.mark}

**Experiment 1 TEL--PPO vs. Closed-Form Equilibrium**

**Closed-Form Equilibrium**

The player's expected payoff is

$$Eu_{i}\left( \sum_{}^{}e_{it},\sum_{}^{}e_{- it} \right) = \ w^{L} + \ p\left( \sum_{}^{}e_{it},\sum_{}^{}e_{- it} \right)\left\lbrack w^{H} - \ w^{L} \right\rbrack - \ ke_{i1}^{2} - E(\ ke_{i2}^{2}\mathbf{)\ \ }$$

The symmetric equilibrium effort in stage 1: $e^{*} = \ \frac{w^{H} - \ w^{L}}{6qk}$; The symmetric expected equilibrium effort in stage 2: $e^{*} = \ \frac{w^{H} - \ w^{L}}{6qk}$

**TEL--PPO**

Train TEL--PPO for $T = 2$ and extract the learned effort functions

$${\widehat{e}}_{1}(0)\ \text{and }{\widehat{e}}_{2}(d).$$

Compare them with the closed-form two-stage equilibrium benchmark:

$$e_{1,CF}^{*}(0)\text{ and}{\ e}_{2,CF}^{*}(d).$$

Report the following metrics.

First-stage absolute error:

$$AE_{1} = \mid {\widehat{e}}_{1}(0) - e_{1,CF}^{*}(0) \mid .$$

First-stage relative error:

$$RE_{1} = \frac{\mid {\widehat{e}}_{1}(0) - e_{1,CF}^{*}(0) \mid}{1 + e_{1,CF}^{*}(0)}.$$

Second-stage mean absolute error:

$$MAE_{2} = \frac{1}{\mid \mathcal{D} \mid}\sum_{d \in \mathcal{D}}^{}{\mid {\widehat{e}}_{2}(d) - e_{2,CF}^{*}(d) \mid}.$$

Second-stage root mean squared error:

$$RMSE_{2} = \sqrt{\frac{1}{\mid \mathcal{D} \mid}\sum_{d \in \mathcal{D}}^{}\left( {\widehat{e}}_{2}(d) - e_{2,CF}^{*}(d) \right)^{2}}.$$

Second-stage relative policy error:

$$RPE_{2} = \frac{RMSE_{2}}{1 + \frac{1}{\mid \mathcal{D} \mid}\sum_{d \in \mathcal{D}}^{}e_{2,CF}^{*}(d)}.$$

Payoff loss:

$${PL}_{2} = U(e_{CF}^{*}) - U(\widehat{e}).$$

Here,

$$e_{CF}^{*} = \left\{ e_{1,CF}^{*}(0),e_{2,CF}^{*}(d) \right\}$$

denotes the closed-form two-stage equilibrium effort-function profile, and

$$\widehat{e} = \left\{ {\widehat{e}}_{1}(0),{\widehat{e}}_{2}(d) \right\}$$

denotes the TEL--PPO learned effort-function profile.

## **Experiment 2: TEL--PPO Output Verified by Independent Best Response** {#experiment-2-telppo-output-verified-by-independent-best-response .unnumbered}

Using the same independent best-response verifier that will later be used for three-stage tournaments, compute the exploitability of the TEL--PPO learned effort function:

$$EXP(\widehat{e}) = U(BR_{i}({\widehat{e}}_{- i}),{\widehat{e}}_{- i}) - U({\widehat{e}}_{i},{\widehat{e}}_{- i}).$$

Here,

$$BR_{i}({\widehat{e}}_{- i})$$

is player $i$\'s best response when the opponent uses the learned effort function ${\widehat{e}}_{- i}$. The first term,

$$U(BR_{i}({\widehat{e}}_{- i}),{\widehat{e}}_{- i}),$$

is player $i$\'s payoff from optimally deviating against the fixed TEL--PPO opponent. The second term,

$$U({\widehat{e}}_{i},{\widehat{e}}_{- i}),$$

is player $i$\'s payoff when both players follow the TEL--PPO learned effort functions.

Certification requires

$$EXP^{UCB}(\widehat{e}) < \epsilon,$$

This demonstrates that the TEL--PPO output is strategically stable, not merely numerically close to the closed-form solution.

## **Experiment 3: Closed-Form Equilibrium Verified by the Same Verifier** {#experiment-3-closed-form-equilibrium-verified-by-the-same-verifier .unnumbered}

Evaluate the exact closed-form equilibrium using the same independent best-response verifier:

$$EXP(e_{CF}^{*}) = U(BR_{i}(e_{CF, - i}^{*}),e_{CF, - i}^{*}) - U(e_{CF,i}^{*},e_{CF, - i}^{*}).$$

The expected result is

$$EXP(e_{CF}^{*}) \approx 0.$$

This establishes the numerical error floor of the verifier. If the verifier assigns near-zero exploitability to the known closed-form equilibrium, then the verifier is able to recognize a true equilibrium in a setting where the theoretical benchmark is known.

## **Experiment 4: Two-Stage Falsification** {#experiment-4-two-stage-falsification .unnumbered}

Evaluate deliberately non-equilibrium policies, including:

a)  constant low effort;

b)  constant high effort;

c)  random effort;

d)  one-stage effort repeated in both stages;

e)  no-gap second-stage effort.

The desired pattern is

$$EXP(\text{bad~policy}) \gg EXP({\widehat{e}}^{TEL\text{-}PPO}) \approx EXP(e_{CF}^{*}).$$

This falsification test verifies that the independent best-response procedure can distinguish equilibrium-like effort functions from poor effort functions. It also shows that low exploitability is not an automatic outcome of the verifier.

  ---------------------------------------------------------------------------------------------------------------
  **Policy**                             **Expected exploitability**   **Interpretation**
  -------------------------------------- ----------------------------- ------------------------------------------
  Closed-form equilibrium $e_{CF}^{*}$   Near zero                     Verifier recognizes the true equilibrium

  TEL-PPO learned effort $\widehat{e}$   Low and below $\varepsilon$   Learned output is certified

  Bad policies                           High                          Verifier detects profitable deviations
  ---------------------------------------------------------------------------------------------------------------

## **[Acceptance Criteria for Two-Stage Validation]{.mark}** {#acceptance-criteria-for-two-stage-validation .unnumbered}

Use pre-specified thresholds for the two-stage validation. For example:

$${RE_{1} < 5\% - 10\%,
}{RPE_{2} < 5\% - 10\%,}$$

and

$$\frac{EXP^{UCB}(\widehat{e})}{W_{H} - W_{L}} < 1\% - 3\%.$$

Here, $RE_{1}$ measures the relative error in the first-stage effort, $RPE_{2}$ measures the relative policy error for the second-stage effort function, and

$$\frac{EXP^{UCB}(\widehat{e})}{W_{H} - W_{L}}$$

normalizes exploitability by the prize spread.

The exact thresholds should be chosen before running the main experiments. They should be calibrated using the one-stage results and the two-stage verifier error floor.

## **5.3 Three-Stage Verified Equilibrium Computation** {#three-stage-verified-equilibrium-computation .unnumbered}

This is the main contribution of the multi-stage experiment. For $T = 3,$ there is no closed-form benchmark. Therefore, TEL--PPO is used to compute candidate state-dependent effort functions:

$${\widehat{e}}_{1}(d),{\widehat{e}}_{2}(d),{\widehat{e}}_{3}(d).$$

These effort functions are evaluated using the same independent best-response verifier that was calibrated in the two-stage case.

## **Main Outputs** {#main-outputs .unnumbered}

### Learned effort functions

Plot the learned effort functions for each stage: ${\widehat{e}}_{1}(d),{\widehat{e}}_{2}(d),{\widehat{e}}_{3}(d).$

The x-axis is the score gap $d$, and the y-axis is effort.

### Best response vs. learned effort

For each stage, plot the best-response effort function against the TEL--PPO learned effort function:

$$BR_{t}(d)\text{ vs. }{\widehat{e}}_{t}(d).$$

If these two functions are close, the learned effort function is close to a state-wise best response.

### One-step deviation gaps

Plot the one-step deviation gap for each stage, $\Delta_{t}(d)$, and also report both the worst-case and on-path deviation gaps: $\underset{t,d}{\max}\Delta_{t}(d)$ and $E_{d_{t} \sim \widehat{e}}\left\lbrack \Delta_{t}(d_{t}) \right\rbrack.$

The worst-case measure checks all states, including off-path states. The on-path measure focuses on states that are reached with high probability under the learned effort functions.

### Exploitability certificate

Report the estimated exploitability $EXP(\widehat{e}),$ the normalized exploitability $\frac{EXP(\widehat{e})}{W_{H} - W_{L}},$ the upper confidence bound $EXP^{UCB}(\widehat{e}),$ and the certification status.

A learned three-stage effort function is certified as an $\epsilon$-approximate Markov perfect equilibrium if $EXP^{UCB}(\widehat{e}) \leq \epsilon.$

## **[Expected Economic Patterns]{.mark}** {#expected-economic-patterns .unnumbered}

## The learned effort functions are expected to display several economically interpretable patterns: {#the-learned-effort-functions-are-expected-to-display-several-economically-interpretable-patterns .unnumbered}

a)  effort is high when the contest is close;

b)  effort is lower for players who are far ahead;

c)  effort may also fall for players who are far behind, reflecting discouragement;

d)  moderately behind players may exert catch-up effort;

e)  later-stage effort functions may be more sensitive to the score gap because fewer future opportunities remain.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **[T]{.mark}**   **[Prize spread]{.mark}**   **[Noise]{.mark}**   **[EXP]{.mark}**   **[EXP /]{.mark}** $\mathbf{\mathrm{\Delta}W}$   **[UCB]{.mark}**   **[Certified?]{.mark}**
  ---------------- --------------------------- -------------------- ------------------ ------------------------------------------------ ------------------ -------------------------
  [3]{.mark}       [medium]{.mark}             [low]{.mark}         [\...]{.mark}      [\...]{.mark}                                    [\...]{.mark}      [yes/no]{.mark}

  [3]{.mark}       [medium]{.mark}             [medium]{.mark}      [\...]{.mark}      [\...]{.mark}                                    [\...]{.mark}      [yes/no]{.mark}

  [3]{.mark}       [medium]{.mark}             [high]{.mark}        [\...]{.mark}      [\...]{.mark}                                    [\...]{.mark}      [yes/no]{.mark}
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## **5.4 Robustness, Ablations and Falsification** {#robustness-ablations-and-falsification .unnumbered}

To ensure that the equilibrium certification is not driven by numerical artifacts, training instability, or a weak verifier, we will consider conducting the following robustness and falsification checks.

### **Grid refinement** {#grid-refinement .unnumbered}

Run the independent best-response verifier with multiple state and action grids for effort and score gap, such as $51,101,201$ grid points. Compare $EXP_{51},EXP_{101},EXP_{201}$, and report whether exploitability remains stable. If exploitability is stable, it proves that the result is not numerical artifact.

### **Monte Carlo confidence intervals** {#monte-carlo-confidence-intervals .unnumbered}

Report exploitability with confidence intervals:

$$EXP(\widehat{e}) \pm 1.96\text{ }SE(EXP).$$

Certification should be based on the upper confidence bound:

$$EXP^{UCB}\left( \widehat{e} \right) = EXP\left( \widehat{e} \right) + 1.96\text{ }SE(EXP).
$$

### **Independent adversarial RL best response** {#independent-adversarial-rl-best-response .unnumbered}

As a secondary verification check, train a separate PPO best-response agent against the frozen TEL--PPO opponent:

$$BR_{i}^{RL}({\widehat{e}}_{- i}).$$

Dynamic programming remains the primary verifier, while adversarial RL provides an additional robustness check. If both DP best response \[6\] and adversarial RL best response \[4\] fail to find profitable deviations, the result becomes much stronger.

### **Falsification tests** {#falsification-tests .unnumbered}

Evaluate deliberately misspecified policies, such as constant effort policies, random effort policies, no-gap policies, or policies that mechanically repeat one-stage effort across all stages.

The desired pattern is

$$EXP(\text{bad~policy}) \gg EXP({\widehat{e}}^{TEL\text{-}PPO})$$

and, in the two-stage case,

$$EXP(\text{bad~policy}) \gg EXP(e_{CF}^{*}).$$

This verifies that the best-response procedure can distinguish equilibrium-like effort functions from poor effort functions.

### **[Curriculum ablation]{.mark}** {#curriculum-ablation .unnumbered}

Compare training with curriculum and without curriculum in the baseline three-stage case:

$$T = 3.$$

The curriculum version uses the training sequence

$$T = 1 \rightarrow T = 2 \rightarrow T = 3,$$

while the no-curriculum version trains the three-stage model directly from random initialization.

This ablation tests whether the certified result is robust to the training procedure and is not merely a warm-start artifact.

### **Seed robustness** {#seed-robustness .unnumbered}

Run multiple random seeds and report:

1)  Policy-function variability;

2)  Exploitability variability;

3)  Certification rates.

A robust result should display similar effort-function shapes, low exploitability, and high certification rates across seeds.

## **5.5 Multi-Stage Extension** {#multi-stage-extension .unnumbered}

After the three-stage experiment, extend the analysis to $T = 4\ \text{and }T = 5.$ These experiments will be presented as benchmark extensions, not as the main validation layer.

For each horizon, report the learned effort functions: ${\widehat{e}}_{t}(d),t = 1,\ldots,T,$ as well as total effort, expected effort cost, exploitability, and certification status.

## Multi-Stage Summary Table {#multi-stage-summary-table .unnumbered}

  -------------------------------------------------------------------------------------------
  **T**   **Total effort**   **Expected effort cost**   **Exploitability**   **Certified?**
  ------- ------------------ -------------------------- -------------------- ----------------
  2       \...               \...                       \...                 yes

  3       \...               \...                       \...                 yes/no

  4       \...               \...                       \...                 yes/no

  5       \...               \...                       \...                 yes/no
  -------------------------------------------------------------------------------------------

## **[Main Questions]{.mark}** {#main-questions .unnumbered}

The multi-stage extension is used to study how effort incentives change as the tournament horizon increases. The main questions are:

a)  Does effort increase as the tournament approaches the final stage?

b)  Do leader and follower effort functions diverge?

c)  Is score-gap-dependent effort hump-shaped?

d)  Does total expected effort increase with the number of stages?

e)  Does public interim feedback amplify or reduce total effort?

## **Verification for Larger Horizons** {#verification-for-larger-horizons .unnumbered}

For larger horizons, exact verification may become computationally more costly because the state space grows and the dynamic-programming best-response calculation becomes more demanding.

If verification becomes difficult for $T = 4$ or $T = 5$, report both DP-based approximate exploitability and adversarial RL best-response checks: $EXP^{DP}(\widehat{e})$ and $EXP^{RL}(\widehat{e}).$

Dynamic programming remains the primary verifier, while adversarial RL best response is used as a secondary robustness check.

[When reporting larger-horizon results, be explicit about numerical limitations, including grid resolution, interpolation error, Monte Carlo or integration error, and best-response solver accuracy. This makes clear that the]{.mark} $T = 4$ [and]{.mark} $T = 5$ [experiments are computational benchmark extensions rather than the core validation layer.]{.mark}

## **6. Deliverables and Figures** {#deliverables-and-figures .unnumbered}

The final experimental section will include the following figures and tables.

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Figure/Table**   **Content**                                                                                                        **Purpose**
  ------------------ ------------------------------------------------------------------------------------------------------------------ ----------------------------------
  Figure 1           Two-stage closed-form vs. TEL--PPO effort: $e_{1}(0)$ and $e_{2}(d)$                                               Analytical recovery

  Figure 2           Two-stage verifier calibration: exploitability of closed-form, TEL-PPO, and bad policies                           Verifier credibility

  Figure 3           Three-stage learned effort functions: ${\widehat{e}}_{1}(d)$, ${\widehat{e}}_{2}(d)$, and ${\widehat{e}}_{3}(d)$   Main economic result

  Figure 4           Three-stage best response vs learned effort                                                                        Equilibrium verification

  Figure 5           State-wise deviation gaps: $\Delta_{t}(d)$                                                                         Local deviation incentives

  Table 1            Two-stage recovery metrics and exploitability                                                                      Dynamic analytical validation

  Table 2            Three-stage exploitability certificate                                                                             Approximate MPE certification

  Table 3            Grid refinement, seed robustness, and falsification tests                                                          Robustness

  Table 4            T = 2,3,4,5 benchmark comparison                                                                                   Scalability and benchmark output
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Figures 1 and 2 establish credibility in the two-stage setting where the theoretical benchmark is known. Figures 3--5 present the main three-stage results and verification diagnostics. Tables 1--3 document recovery accuracy, exploitability, and robustness. Table 4 summarizes how the method scales to longer tournament horizons.

**7. Summary**

TEL--PPO is used as a candidate equilibrium solver rather than as evidence of equilibrium by itself. In the two-stage tournament, the closed-form benchmark provides an analytical recovery test and calibrates the independent best-response verifier. In three-stage and longer tournaments, where closed-form benchmarks are unavailable, the same calibrated verifier evaluates whether the learned effort functions have low exploitability.

The resulting object is therefore not merely an RL simulation outcome, but a numerically verified computational equilibrium benchmark for dynamic Lazear--Rosen rank-order tournaments.

More specifically, the project contributes a methodology that combines:

$$\text{closed-form~recovery} + \text{verifier~calibration} + \text{best-response~exploitability~certification} + \text{multi-stage~computational~benchmarking}.
$$

This structure allows TEL--PPO to be evaluated not only by training stability or visual policy convergence, but by economically meaningful equilibrium diagnostics.

.

Reference:

1.  Lazear, Edward P., and Sherwin Rosen. \"Rank-order tournaments as optimum labor contracts.\" *Journal of political Economy* 89, no. 5 (1981): 841-864

2.  <https://colab.research.google.com/github/QuantEcon/lecture-python.notebooks/blob/main/markov_perf.ipynb?utm_source=chatgpt.com>

3.  Petrazzini, Irving GB, and Eric A. Antonelo. \"Proximal policy optimization with continuous bounded action space via the beta distribution.\" In *2021 IEEE symposium series on computational intelligence (SSCI)*, pp. 1-8. IEEE, 2021

4.  Martin, Carlos, and Tuomas Sandholm. \"ApproxED: Approximate exploitability descent via learned best responses.\" *arXiv preprint arXiv:2301.08830* (2023).

5.  <https://deepwiki.com/google-deepmind/open_spiel/6.4-best-response-and-exploitability?utm_source=chatgpt.com>

6.  Saure, Denis, and Gabriel Y. Weintraub. \"An Approximate Dynamic Programming Approach to Solving Dynamic Oligopoly Models Vivek Farias MIT Sloan School.\"
