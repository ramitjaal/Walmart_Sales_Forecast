import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def reduce_mem_usage(df, verbose=False):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

def submission_file(df):
    submit = df[["id"]]
    for i in range(1,29):
        submit["F" + str(i)] = df["F_7_" + str(1885 + i)]
    submit1 = submit.copy()
    submit1["id"] = submit1["id"].apply(lambda x: x.replace('validation','evaluation'))
    submit = submit.append(submit1).reset_index(drop=True)
    submit.to_csv("submission.csv",index=False)