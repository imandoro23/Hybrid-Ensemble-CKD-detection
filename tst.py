import pandas as pd
import numpy as np

# Analytical threshold
p_uci = 0.625; p_nhanes = 0.184; t_uci = 0.50
t_star = (p_nhanes/p_uci * t_uci) / (p_nhanes/p_uci * t_uci + (1-p_nhanes)/(1-p_uci) * (1-t_uci))
print(f"Analytically derived threshold: {t_star:.4f}")

# Then recompute metrics using t_star instead of 0.984
# y_pred_new = (y_proba_nh >= t_star).astype(int)
# ... recompute sensitivity, specificity, PPV, NPV, accuracy