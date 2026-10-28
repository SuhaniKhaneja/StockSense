import numpy as np


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / y.sum())


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def bias(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float((p - y).sum() / y.sum())


def coverage(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def bootstrap_wape_reduction(preds, model, baseline, n=1000, seed=0):
    """Relative WAPE reduction of `model` vs `baseline`, with a 95% CI from resampling products."""
    g = preds.assign(err=(preds["y"] - preds["p"]).abs())
    a = g[g["model"] == model].groupby("sku")["err"].sum()
    b = g[g["model"] == baseline].groupby("sku")["err"].sum()
    skus = a.index.intersection(b.index)
    ea, eb = a.loc[skus].to_numpy(), b.loc[skus].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(skus), len(skus))
        draws.append(1 - ea[i].sum() / eb[i].sum())
    return {"relative_wape_reduction": float(1 - ea.sum() / eb.sum()),
            "ci95": [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))],
            "n_skus": int(len(skus))}
