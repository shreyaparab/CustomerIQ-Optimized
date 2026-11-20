"""Optional probability calibration using held-out data."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV


def calibrate_pipeline(fitted_pipeline, X_val, y_val, method: str = "sigmoid"):
    """Calibrate an already-fitted pipeline on validation data."""
    from sklearn.frozen import FrozenEstimator
    cal = CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method=method)
    cal.fit(X_val, y_val)
    return cal
