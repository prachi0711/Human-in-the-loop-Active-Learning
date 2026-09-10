# Use this for batch runner: results/rq2_3_analysis_combine.py
"""
Manages personalisation for the stress inference system.
Modified to allow constant blend weights and synchronous background threads for simulation runs.
"""

import os, threading, datetime
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Thresholds
MIN_LABELS_TO_TRAIN = 2
PERSONAL_RF_ESTIMATORS = 50
RANDOM_STATE = 42
LABELS_FOR_MAX_WEIGHT = 20  

LABEL_MAP = {"calm": 0, "stress": 1, "stressed": 1}


class ActiveLearningManager:
    """
    Parameters
    ----------
    user_id         : str
    feature_columns : list[str]  - must match base model feature order
    data_dir        : str - directory for the CSV label file
    on_retrain      : callable(n_labels, blend_w) | None
                      Called after each retrain.
    fixed_weight    : float - The exact constant mix ratio to assign to the personal model.
    is_simulation   : bool - If True, forces retraining to block execution synchronously.
    """

    def __init__(self, user_id: str, feature_columns: list, data_dir: str = ".", on_retrain=None,
        fixed_weight: float = 0.2,    # Overridden dynamically by sweep loops
        is_simulation: bool = False,  # Set to True during optimization simulation runs
    ):
        self.user_id = user_id
        self.feature_columns = feature_columns 
        self.data_dir = data_dir
        self.on_retrain = on_retrain   
        self.fixed_weight = fixed_weight    # Store constant target weight
        self.is_simulation = is_simulation   # Store loop tracking flag

        safe_id = "".join(c if c.isalnum() else "_" for c in user_id)
        self.csv_path = os.path.join(data_dir, f"labels_{safe_id}.csv")

        self.personal_model: RandomForestClassifier | None = None
        self.personal_scaler: StandardScaler | None = None

        self.labelled_df = self._load_existing_labels()

        self._lock = threading.Lock()
        self._retrain_thread: threading.Thread | None = None

        # retrain immediately if labels from a previous session are already available
        if len(self.labelled_df) >= MIN_LABELS_TO_TRAIN:
            self._trigger_retrain()

    
    # Public API
    def add_label(self, features: list, label_str: str, timestamp: datetime.datetime | None = None,) -> bool:
        """
        Save a labelled window and trigger retraining.
        Return True if accepted, False if label_str is unrecognised.
        """
        label_int = LABEL_MAP.get(label_str.lower().strip())
        if label_int is None:
            return False

        ts = timestamp or datetime.datetime.now()
        row = {col: val for col, val in zip(self.feature_columns, features)}
        row["label"] = label_int
        row["label_str"] = label_str
        row["timestamp"] = ts.isoformat()

        with self._lock:
            if self.labelled_df.empty:
                self.labelled_df = pd.DataFrame([row])
            else:
                self.labelled_df = pd.concat(
                    [self.labelled_df, pd.DataFrame([row])], ignore_index=True
                )
            self._save_csv()

        self._trigger_retrain()
        return True

    # associating the particular features with the label provided the user
    def blend_with_features(self, base_probs: np.ndarray, features: list,) -> tuple[np.ndarray, float]:
        """
        Blend base model probabilities with personal model probabilities.
        Return (blended_probs, blend_weight).
        """
        w = self._blend_weight()

        with self._lock:
            model = self.personal_model
            scaler = self.personal_scaler

        if model is None or w == 0.0:
            return base_probs.copy(), 0.0

        try:
            x_scaled = scaler.transform(np.array(features).reshape(1, -1))
            personal_probs = model.predict_proba(x_scaled)[0]

            # Align class order (if personal model has not seen both classes)
            blended = np.zeros(2)
            for idx, cls in enumerate(model.classes_):
                blended[int(cls)] = personal_probs[idx]

            result = (1.0 - w) * base_probs + w * blended
            result /= result.sum()   # re-normalise
            return result, w

        except Exception:
            return base_probs.copy(), 0.0


    @property
    def n_labels(self) -> int:
        with self._lock:
            return len(self.labelled_df)

    @property
    def label_counts(self) -> dict:
        with self._lock:
            if self.labelled_df.empty:
                return {"calm": 0, "stress": 0}
            vc = self.labelled_df["label"].value_counts()
            return {"calm": int(vc.get(0, 0)), "stress": int(vc.get(1, 0))}

    @property
    def blend_weight(self) -> float:
        return self._blend_weight()

    @property
    def personal_model_ready(self) -> bool:
        with self._lock:
            return self.personal_model is not None

    # count the total number of labels per state (calm or stress)
    def status_summary(self) -> str:
        counts = self.label_counts
        return (
            f"AL labels: calm={counts['calm']} stress={counts['stress']} "
            f"| blend_w={self._blend_weight():.2f} "
            f"| personal={'ready' if self.personal_model_ready else 'not ready'}"
        )

    # blending the base model with the personal model as the labels are provided
    def _blend_weight(self) -> float:
        n = self.n_labels
        if n < MIN_LABELS_TO_TRAIN:
            return 0.0
             
        return min(self.fixed_weight, n/LABELS_FOR_MAX_WEIGHT)
    
    def _trigger_retrain(self):
        if self.is_simulation:
            self._retrain()
            return

        if self._retrain_thread and self._retrain_thread.is_alive():
            return
        self._retrain_thread = threading.Thread(target=self._retrain, daemon=True)
        self._retrain_thread.start()

    # retraining the model with new labels
    def _retrain(self):
        with self._lock:
            df = self.labelled_df.copy()

        if len(df) < MIN_LABELS_TO_TRAIN:
            return

        X = df[self.feature_columns].values.astype(float)
        y = df["label"].values.astype(int)

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=PERSONAL_RF_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        rf.fit(X_scaled, y)

        with self._lock:
            self.personal_model  = rf
            self.personal_scaler = scaler

        n = len(df)
        w = self._blend_weight()

        if self.on_retrain:
            try:
                self.on_retrain(n, w)
            except Exception:
                pass

    # load the existing label when the model is used again
    def _load_existing_labels(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                required = self.feature_columns + ["label", "label_str", "timestamp"]
                if all(c in df.columns for c in required):
                    return df
            except Exception:
                pass
        cols = self.feature_columns + ["label", "label_str", "timestamp"]
        return pd.DataFrame(columns=cols)

    def _save_csv(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.labelled_df.to_csv(self.csv_path, index=False)
