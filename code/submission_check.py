import pandas as pd

def check_value(path) :
    csv = pd.read_csv(path)

    print(csv["ans"].value_counts())

csv_path = "/opt/ml/input/data/eval/submission_test2.csv"
check_value(csv_path)