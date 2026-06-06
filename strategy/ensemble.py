"""strategy/ensemble.py — XGB + LGBM + LogReg soft-voting ensemble."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("clawdbot.ensemble")

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class EnsemblePredictor:
    """Soft-voting ensemble of XGBClassifier + LGBMClassifier + LogisticRegression.

    Weights default to equal thirds. Caller may override via `weights` param.
    """

    def __init__(self, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        self._w = np.array(weights, dtype=np.float64)
        self._w /= self._w.sum()
        self._xgb: Any = None
        self._lgbm: Any = None
        self._lr: Any = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsemblePredictor":
        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression

        spw = float(max(1.0, (y == 0).sum() / max((y == 1).sum(), 1)))

        self._xgb = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=spw,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
        )
        self._xgb.fit(X, y)

        try:
            import lightgbm as lgb
            self._lgbm = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=spw,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                verbose=-1,
            )
            self._lgbm.fit(X, y)
        except ImportError:
            logger.info("lightgbm not installed — ensemble uses XGB+LR only (2-model average)")
            self._lgbm = None
            # rebalance weights to XGB + LR
            w = np.array([self._w[0] + self._w[1] / 2, 0.0, self._w[2] + self._w[1] / 2])
            self._w = w / w.sum()

        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self._lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1,
            )),
        ])
        self._lr.fit(X, y)

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EnsemblePredictor not fitted")

        proba_xgb = self._xgb.predict_proba(X)[:, 1]
        proba_lr = self._lr.predict_proba(X)[:, 1]

        if self._lgbm is not None:
            proba_lgbm = self._lgbm.predict_proba(X)[:, 1]
            combined = (
                self._w[0] * proba_xgb
                + self._w[1] * proba_lgbm
                + self._w[2] * proba_lr
            )
        else:
            combined = self._w[0] * proba_xgb + self._w[2] * proba_lr

        out = np.column_stack([1.0 - combined, combined])
        return out

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int32)

    # ------------------------------------------------------------------
    def save(self, out_dir: Path, symbol: str, suffix: str = "_ensemble_v2") -> Path:
        """Persist sub-models. XGB → JSON, LGBM → txt, LR → JSON."""
        import pickle

        out_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace("/", "_")
        base = out_dir / f"{safe}{suffix}"

        xgb_path = base.parent / f"{base.stem}_xgb.json"
        self._xgb.save_model(str(xgb_path))

        lgbm_path = None
        if self._lgbm is not None:
            lgbm_path = base.parent / f"{base.stem}_lgbm.txt"
            self._lgbm.booster_.save_model(str(lgbm_path))

        lr_pkl = base.parent / f"{base.stem}_lr.pkl"
        with open(lr_pkl, "wb") as f:
            pickle.dump(self._lr, f)  # Pipeline (scaler + LR) or bare LR

        meta = {
            "type": "EnsemblePredictor",
            "weights": self._w.tolist(),
            "xgb_path": xgb_path.name,
            "lgbm_path": lgbm_path.name if lgbm_path else None,
            "lr_pkl": lr_pkl.name,
        }
        meta_path = base.parent / f"{base.stem}.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Ensemble saved → %s", meta_path)
        return meta_path

    @classmethod
    def load(cls, meta_path: Path) -> "EnsemblePredictor":
        import pickle

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        d = meta_path.parent

        inst = cls(weights=tuple(meta["weights"]))  # type: ignore[arg-type]

        from xgboost import XGBClassifier
        inst._xgb = XGBClassifier()
        inst._xgb.load_model(str(d / meta["xgb_path"]))

        if meta.get("lgbm_path"):
            try:
                import lightgbm as lgb
                inst._lgbm = lgb.Booster(model_file=str(d / meta["lgbm_path"]))
            except ImportError:
                inst._lgbm = None

        with open(d / meta["lr_pkl"], "rb") as f:
            inst._lr = pickle.load(f)

        inst._fitted = True
        return inst
