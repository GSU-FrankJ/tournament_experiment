import numpy as np
# Model as implied by the reports: dW=4; e1*(0)=dW/(6kq) with e1*(50)=46.67 -> k=1/3500.
# Per-player shocks U[-q,q] -> xi = difference, triangular on [-2q,2q]; f(0)=1/(2q) (matches e2*(0)=70 at q=50).
dW, k = 4.0, 1/3500.0
def F(x, q):
    a = 2*q; x = np.clip(x, -a, a)
    return np.where(x < 0, (x + a)**2/(2*a*a), 1 - (a - x)**2/(2*a*a))
def f(x, q):
    a = 2*q; return np.clip((a - np.abs(x))/(a*a), 0, None)
for q in (50, 60):
    r = dW/(k*q*q)
    e1s = dW/(6*k*q); e2s0 = dW/(2*k)*f(0.0, q)
    E = np.linspace(0, 100, 200001)          # dense, independent action grid
    # terminal one-step gain vs zero-effort opponent, as a function of d
    def W(d):  # best terminal value at gap d vs zero opponent (stage-2 cost included)
        return np.max(dW*F(d + E, q) - k*E**2)
    ds = np.linspace(-2*q-100, 2*q+100, 1201)
    G2 = np.array([W(d) - dW*F(d, q) for d in ds])
    # root dynamic BR vs zero-effort opponent: max_e1 -k e1^2 + E_xi1[ W(e1 + xi1) ]
    xi = np.linspace(-2*q, 2*q, 4001); wxi = f(xi, q); wxi /= wxi.sum()
    dgrid = np.linspace(-2*q, 2*q + 100, 3001)
    Wg = np.array([W(d) for d in dgrid])
    E1 = np.linspace(0, 100, 2001)
    vals = [-k*e1**2 + np.sum(wxi*np.interp(e1 + xi, dgrid, Wg)) for e1 in E1]
    G1 = max(vals) - dW*0.5
    print(f"q={q}: r={r:.4f}  e1*={e1s:.3f}  e2*(0)={e2s0:.3f}")
    print(f"   closed form r/(16+2r) = {r/(16+2*r):.4f}   numeric G2(0)/dW = {G2[np.argmin(abs(ds))]/dW:.4f}")
    print(f"   max_d G2(d)/dW = {G2.max()/dW:.4f} at d={ds[G2.argmax()]:.1f}")
    print(f"   G1(0)/dW (root, dynamic) = {G1/dW:.4f} at e1={E1[int(np.argmax(vals))]:.2f}")
    print(f"   => Gmax_full(e=0)/dW = {max(G1, G2.max())/dW:.4f}")
    # stage-1 best-response slope at the analytic equilibrium (derived: E[V2'']/(E[V2'']-2k), E[V2'']=dW^2/(32 k q^4))
    EV2pp = dW**2/(32*k*q**4)
    print(f"   stage-1 BR slope at e1* = {EV2pp/(EV2pp-2*k):.3f};  own-curvature share E[V2'']/2k = {EV2pp/(2*k):.3f}")
    # peak-error <-> residual factor and noise-bias coefficient
    print(f"   Delta2(0)/dW per eps^2 (symmetric under-effort) = {r/(16+2*r):.4f};  5% peak -> {0.0025*r/(16+2*r):.2e} dW")
