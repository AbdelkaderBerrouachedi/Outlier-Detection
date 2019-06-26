import arff
import pandas as pd
import numpy as np

data = pd.read_csv('ALOI_new.csv').iloc[:, 0:-1]
input_data = data.iloc[:, 1:]
target_data = data.iloc[:, 0]
target_values = []
for t in target_data:
    if t == 'yes':
        target_values.append(1)
    else:
        target_values.append(0)
complete_data = pd.DataFrame(data=np.concatenate(([input_data.values,
                                                   np.array(target_values).reshape(-1, 1)]), axis=1))
complete_data.to_csv('aloi.csv')
