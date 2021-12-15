import pandas as pd
from utility_plot import visualise_data
from pathlib import Path

# ============================================================================
if __name__ == "__main__":

    path_result = Path(__file__).parent / "result_syn.csv"
    df = pd.read_csv(path_result)
    vd = visualise_data(df)
    vd.line_plot("ecart-type", "MSE", Path(__file__).parent / "curve_syn.png")
