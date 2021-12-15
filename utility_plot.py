import seaborn as sns
import pandas as pd
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt


class visualise_data:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def scatter_plot(
        self,
        col_name1: str,
        col_name2: str,
        hue_name: Optional[str],
        path_save_fig: Path,
    ) -> None:
        sns.scatterplot(data=self.df, x=col_name1, y=col_name2, hue=hue_name)
        plt.savefig(path_save_fig)

    def line_plot(
        self,
        col_name1: str,
        col_name2: str,
        path_save_fig: Path,
    ) -> None:
        sns.lineplot(data=self.df, x=col_name1, y=col_name2)
        plt.savefig(path_save_fig)
