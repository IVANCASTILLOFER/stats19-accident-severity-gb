"""Recalibración de probabilidades por desplazamiento de priors (factor multiplicativo por clase).

Corrige un nivel medio erróneo (deriva) sin cambiar el orden relativo de los casos dentro de cada clase.
Las decisiones (umbrales por clase y modo alerta) se siguen tomando con las probabilidades originales.
"""
import numpy as np
from sklearn.metrics import average_precision_score, log_loss


def fit_factors(p: np.ndarray, y: np.ndarray, iters: int = 100) -> np.ndarray:
    """Factores f tales que la media de normalizar(p * f) por clase iguala la frecuencia observada."""
    obs = np.bincount(y, minlength=3) / len(y)
    f = np.ones(3)
    for _ in range(iters):
        q = p * f
        q /= q.sum(1, keepdims=True)
        f *= obs / q.mean(0)
    return f / f[0]


def fit_slope_bias(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Recalibración con pendiente y sesgo: softmax(a·log p + [0, b1, b2]); se ajusta minimizando la log-pérdida.

    A diferencia de los factores por clase, la pendiente `a` corrige también la *forma* del error
    (p. ej. subestimar los casos de bajo riesgo y sobrestimar los de alto riesgo)."""
    from scipy.optimize import minimize
    from scipy.special import softmax
    lp = np.log(np.clip(p, 1e-9, 1))
    oh = np.eye(3)[y]

    def nll(t):
        q = softmax(t[0] * lp + np.array([0.0, t[1], t[2]]), axis=1)
        return -np.mean(np.sum(oh * np.log(np.clip(q, 1e-12, 1)), axis=1))

    r = minimize(nll, x0=[1.0, 0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400})
    return r.x


def apply_slope_bias(p: np.ndarray, t) -> np.ndarray:
    from scipy.special import softmax
    t = np.asarray(t)
    return softmax(t[0] * np.log(np.clip(p, 1e-9, 1)) + np.array([0.0, t[1], t[2]]), axis=1)


def apply_factors(p: np.ndarray, f) -> np.ndarray:
    q = p * np.asarray(f)
    return q / q.sum(1, keepdims=True)


def ece_ksi(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Error de calibración esperado de P(grave o mortal), con deciles de la probabilidad predicha."""
    s, t = 1 - p[:, 0], (y > 0).astype(float)
    edges = np.quantile(s, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, bins - 1)
    return float(sum(abs(t[idx == b].mean() - s[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    onehot = np.eye(3)[y]
    return {
        "pred_ksi": float((1 - p[:, 0]).mean()), "obs_ksi": float((y > 0).mean()),
        "pred_fatal": float(p[:, 2].mean()), "obs_fatal": float((y == 2).mean()),
        "brier": float(((p - onehot) ** 2).sum(1).mean()),
        "logloss": float(log_loss(y, p, labels=[0, 1, 2])),
        "ece_ksi": ece_ksi(y, p),
        "ap_ksi": float(average_precision_score(y > 0, 1 - p[:, 0])),
    }
