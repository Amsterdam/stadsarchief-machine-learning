import pandas as pd


def split_dataframe(certain: int, uncertain: int) -> pd.DataFrame:
    total = certain + uncertain
    certain_percentage = certain / total * 100
    uncertain_percentage = uncertain / total * 100

    counts_df = pd.DataFrame([
        [certain, uncertain, total],
        [certain_percentage, uncertain_percentage, 100.0]
    ],
        columns=['certain', 'uncertain', 'total'],
        index=['absolute', 'relative']
    )
    return counts_df
