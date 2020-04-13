import utils
import pandas as pd
import numpy as np
from evaluate import rmsse

input_dir = "data/train/"

df = pd.read_csv(f'{input_dir}sales_train_validation.csv').pipe(utils.reduce_mem_usage)
price_df = pd.read_csv(f'{input_dir}sell_prices.csv').pipe(utils.reduce_mem_usage)
calendar_df = pd.read_csv(f'{input_dir}calendar.csv').pipe(utils.reduce_mem_usage)
submission = pd.read_csv('data/submission/sample_submission.csv')

def walmart_baseline(df=df,price_df=price_df,calendar_df=calendar_df):
    calendar_df["d"] = calendar_df["d"].apply(lambda x: int(x.split("_")[1]))
    price_df["id"] = price_df["item_id"] + "_" + price_df["store_id"] + "_validation"

    for day in range(1858, 1886):
        wk_id = list(calendar_df[calendar_df["d"] == day]["wm_yr_wk"])[0]
        wk_price_df = price_df[price_df["wm_yr_wk"] == wk_id]
        df = df.merge(wk_price_df[["sell_price", "id"]], on=["id"], how='inner')
        df["unit_sales_" + str(day)] = df["sell_price"] * df["d_" + str(day)]
        df.drop(columns=["sell_price"], inplace=True)

    df["dollar_sales"] = df[[c for c in df.columns if c.find("unit_sales") == 0]].sum(axis=1)
    df.drop(columns=[c for c in df.columns if c.find("unit_sales") == 0], inplace=True)
    df["weight"] = df["dollar_sales"] / df["dollar_sales"].sum()
    df.drop(columns=["dollar_sales"], inplace=True)


    for d in range(1, 29):
        df["F_7_" + str(1885 + d)] = df["d_" + str(1885 + d - 28)]
    agg_df = pd.DataFrame(df[[c for c in df.columns if c.find("d_") == 0 or c.find("F_") == 0]].sum()).transpose()
    agg_df["level"] = 1
    agg_df["weight"] = 1 / 12
    column_order = agg_df.columns
    level_groupings = {2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"],
                       6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"], 8: ["store_id", "cat_id"],
                       9: ["store_id", "dept_id"],
                       10: ["item_id"], 11: ["item_id", "state_id"]}

    for level in level_groupings:
        temp_df = df.groupby(by=level_groupings[level]).sum().reset_index(drop=True)
        temp_df["level"] = level
        temp_df["weight"] /= 12
        agg_df = agg_df.append(temp_df[column_order])

    del temp_df

    df["weight"] /= 12

    train_series_cols = [c for c in df.columns if c.find("d_") == 0][:-28]
    ground_truth_cols = [c for c in df.columns if c.find("d_") == 0][-28:]
    forecast_cols = [c for c in df.columns if c.find("F_") == 0]

    df["rmsse"] = rmsse(np.array(df[ground_truth_cols]),
                        np.array(df[forecast_cols]), np.array(df[train_series_cols]))
    agg_df["rmsse"] = rmsse(np.array(agg_df[ground_truth_cols]),
                            np.array(agg_df[forecast_cols]), np.array(agg_df[train_series_cols]))

    df["wrmsse"] = df["weight"] * df["rmsse"]
    agg_df["wrmsse"] = agg_df["weight"] * agg_df["rmsse"]

    print(df["wrmsse"].sum() + agg_df["wrmsse"].sum())
    utils.submission_file(df)

if __name__ == "__main__":
    walmart_baseline()