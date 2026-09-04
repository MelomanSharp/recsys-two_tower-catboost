import pandas as pd
import numpy as np

def simulate_retraining_policies():
    """Simulates 3 retraining strategies on historical data."""
    # Generates synthetic quality drift (as if the model were degrading over time)
    weeks = np.arange(1, 25)
    base_quality = 0.95
    drift = np.cumsum(np.random.normal(-0.005, 0.002, len(weeks)))
    quality_over_time = np.clip(base_quality + drift, 0.70, 0.95)
    
    # 1. Fixed schedule (every four weeks).
    fixed_retrains = [w for w in weeks if w % 4 == 0]
    
    # 2. Performance-triggered (if drop > 0.03 from rolling average)
    perf_retrains = []
    for i in range(3, len(weeks)):
        if quality_over_time[i] < np.mean(quality_over_time[i-3:i]) - 0.03:
            perf_retrains.append(weeks[i])
            quality_over_time[i:] += 0.05 # "Rollback" to a good state after retraining
            
    print(f"Fixed Schedule retrains: {len(fixed_retrains)}")
    print(f"Performance-triggered retrains: {len(perf_retrains)}")
    print("✅ Simulation complete. Use these insights to configure Airflow DAG thresholds.")

if __name__ == "__main__":
    simulate_retraining_policies()