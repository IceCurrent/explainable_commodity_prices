# E0: Cross-window factor alignment & stability audit

- Windows: **2618**, factors K = **5**
- Reference window: 0
- **Median adjacent |corr| (anchor 1): 0.999**
- Mean / min adjacent |corr|: 0.998 / 0.850
- Median decoder |cosine| (anchor 2): 0.929
- Anchor agreement rate: 1.000
- Permutation-event rate: 0.000
- Sign-flip rate per factor: f1=0.000, f2=0.000, f3=0.000, f4=0.000, f5=0.000

## Gate verdict: **STABLE**

Median matched |corr| >= 0.8: factors have a stable identity. Proceed with cross-window pooled mapping at full strength.