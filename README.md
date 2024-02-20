## Photon assisted tunneling:

$$I_{p}(V_0,\alpha) = \sum_{n=-\infty}^{+\infty} J_n^2(\alpha) \cdot I_{dc}(V_0 + n\hbar \omega / e)$$

## Components of $Y_{mm\prime} = G_{mm\prime} + i B_{mm\prime}$

```math
G_{mm\prime} = \frac{e}{2 \hbar \omega_{m\prime}} \cdot
\;\sum_{n,n\prime=-\infty}^{\infty} J_n(\alpha) J_{n\prime}(\alpha) \delta_{m-m\prime, n\prime-n}
\left\{ \left[ I_{dc}(V_0+n\prime \hbar \omega /e + \hbar \omega_{m\prime}/e) -
I_{dc}(V_0 + n\prime \hbar \omega/e) \right] +
\left[ I_{dc}(V_0 + n\hbar \omega/e) -
I_{dc}(V_0 + n \hbar \omega/e - \hbar \omega_{m\prime}/e) \right]  \right\}
```

```math
B_{mm\prime} = \frac{e}{2 \hbar \omega_{m\prime}} \cdot
\sum_{n,n\prime=-\infty}^{\infty} J_n(\alpha) J_{n\prime}(\alpha) \delta_{m-m\prime, n\prime-n}
\left\{ \left[ I_{kk}(V_0+n\prime \hbar \omega /e + \hbar \omega_{m\prime}/e) -
I_{kk}(V_0 + n\prime \hbar \omega/e) \right] -
\left[ I_{kk}(V_0 + n\hbar \omega/e) -
I_{kk}(V_0 + n \hbar \omega/e - \hbar \omega_{m\prime}/e) \right]  \right\}
```


$\omega_m = m \cdot \omega + \omega_0$

$\omega$ - LO rate, $\omega_0$ - IF rate, $\omega_m$ - Signal rate


__Augmented__ 
```math
Y'_{mm} =
\begin{bmatrix}
    Y_{11} + Y_{S} & Y_{10} & Y_{1-1}  \\
    Y_{01} & Y_{00} + Y_L & Y_{0-1}   \\
    Y_{-11} & Y_{-10} & Y_{-1-1}+Y_I  \\
\end{bmatrix}
```

$Y_{IF} = 1/Z'_{00}$



__All there formulas from Tucker & Feldman theory__

For I-V curve analysis used RespFnFromIVData from QMix https://github.com/garrettj403/QMix


## Releases are published automatically when a tag is pushed to GitHub.

```bash
sh deploy.sh -t {tag}
```

```
.\deploy.bat {tag}
```