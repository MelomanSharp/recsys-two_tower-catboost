import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict
from src.data.features import FeatureEngineer


@dataclass
class DriftReport:
    feature_name: str
    psi: float
    threshold: float
    drifted: bool


class DriftDetector:
    """Computes and aggregates drift metrics across feature families."""

    THRESHOLDS = {
        "price": 0.15,  
        "item_trendiness": 0.20, 
        "department_name": 0.25, # Категорийный дрейф
    }

    def __init__(self):
        self.engineer = FeatureEngineer()

    def compare_windows(
        self, ref: pd.DataFrame, tar: pd.DataFrame, features: list
    ) -> Dict[str, DriftReport]:
        reports = {}
        for feat in features:
            if feat not in ref.columns or feat not in tar.columns:
                continue
            ref_vals = ref[feat].dropna().values
            tar_vals = tar[feat].dropna().values
            if len(ref_vals) < 30 or len(tar_vals) < 30:
                continue

            psi = self.engineer.calculate_psi(ref_vals, tar_vals, num_bins=10)
            threshold = self.THRESHOLDS.get(feat, 0.25)
            reports[feat] = DriftReport(
                feature_name=feat,
                psi=float(psi),
                threshold=threshold,
                drifted=psi > threshold,
            )
        return reports

    def should_retrain(self, reports: Dict[str, DriftReport]) -> bool:
        """
        Decision policy: 
        Если цена пробила 0.15 (сезонность) ИЛИ тренды/ассортимент улетели за 0.20.
        """
        if not reports:
            return False
        
        # catch seasonal price drift (threshold 0.15)
        price_drift = reports.get("price")
        if price_drift and price_drift.drifted:
            return True
            
        # track the drift of real drivers (trends, product range)
        trend_drift = reports.get("item_trendiness")
        if trend_drift and trend_drift.drifted:
            return True

        return False