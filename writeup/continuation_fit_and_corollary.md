§7 — Fitting n, and robustness

In the pair setup, the functional form above predicts that the breaking perturbation at each $$\varphi$$ satisfies

$$\left(\frac{\theta(\varphi)\cos\varphi}{\theta_1}\right)^{n} + \left(\frac{\theta(\varphi)\sin\varphi}{\theta_2}\right)^{n} = 1,$$

a superellipse of exponent $$n$$ in normalized coordinates. We fit this single parameter by minimizing the squared residual of this equation over the measured $$(\theta(\varphi)\cos\varphi,\ \theta(\varphi)\sin\varphi)$$ points. The fitted $$n$$ is consistently close to 2 across conditions: different contrastive pairs; perturbations along combinations of random (orthogonalized) directions in higher dimensions, not just pairs; different models; different perturbation and measurement layers; and all three response metrics. Median fitted $$n$$ sits between about 2.0 and 2.2, with per-pair fit residuals of order 1%. We therefore take $$n = 2$$ as our empirical starting point, in which case the response function reduces to

$$R(v) = F\!\left(\sum_i w_i\, (d_i^\top v)^2\right) = F\!\left(v^\top A v\right), \qquad A = \sum_i w_i\, d_i d_i^\top,$$

a convex function of a positive semidefinite quadratic form.


§8 — Corollary

This is in tension with the universal reading of LRH under superposition. $$A \succeq 0$$, so its top eigenvector $$u$$ satisfies $$u^\top A u \geq d_i^\top A d_i$$ for every feature $$i$$ — the model is at least as sensitive along $$u$$ as along any feature it is built from. Under superposition $$M > D$$ the features span the activation space, and cannot all lie in the top eigenspace of $$A$$; generically $$u$$ is not itself a feature direction. The model is therefore strictly more sensitive along at least one non-feature direction than along every feature, contradicting the reading of LRH in which features are the uniquely most-sensitive directions in activation space. Empirically observed $$n = 2$$ thus rules out strict LRH and superposition as a joint description: one of $$\{$$the strong form of LRH, superposition as assumed, the quadratic-response finding$$\}$$ has to be softened.
