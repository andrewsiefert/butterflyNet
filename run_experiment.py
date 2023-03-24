import pandas as pd
from sklearn.metrics import roc_auc_score

exec(open('neural_net_functions.py').read())


# import data
df = pd.read_csv("H1ForNeural.csv")
df.shape
df.head()

df.host = df.host - 1


# run experiments with different numbers and sizes of hidden layers
hidden = [[4],[8],[16],[4,4],[8,8],[16,16],[4,4,4],[4,8,16],[16,8,4]]

output = []
for h in hidden:
    vals = run_experiment_loo(df, hidden=h)
    output.append(vals)


pred = pd.concat([pd.Series(i.tolist()) for i in output], axis=1)
pred.columns = ["hidden_4", "hidden_8", "hidden_16", "hidden_4-4", "hidden_8-8", "hidden_16-16",
                "hidden_4-4-4", "hidden_4-8-16", "hidden_16-8-4"]
pred.to_csv("neural_net_predictions.csv")

auc = [roc_auc_score(df.host, x) for x in output]
loo_auc = pd.DataFrame({"hidden": [x.replace("hidden_", "") for x in pred.columns],
                        "loo_auc": auc})
loo_auc.to_csv("loo_auc.csv")