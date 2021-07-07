from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
def standardScaler(data, denomrmlization = False):
    scaler = StandardScaler().fit(data)
    if denomrmlization:
        data = scaler.transform(data)
    else:
        data = scaler.inverse_transform(data)
    return data

df = pd.read_csv('data/data_Poland_to_2021_05.csv')
data_df = df.iloc[:,[-2,-1]]
data_desc = data_df.describe()
data = data_df.values

data_nor = (data - data_desc.loc['mean'].values)/ (data_desc.loc['max'].values - data_desc.loc['min'].values)

df_ns = pd.DataFrame(columns= data_df.columns, data = data_nor)

df_ns.to_csv('data/data_Poland_to_2021_05_ns.csv', index= False)
# %%
data = data.reshape(len(data),1)
scaler = StandardScaler()

data_st = scaler.fit_transform(data)

data_org = scaler.inverse_transform(data_st)
# %%
data_st = data_st.reshape(len(data))
df_st = pd.DataFrame({df.columns[-1]:data_st})

# %%
# data_org_1 = data_nor * (data.max() - data.min()) + data.mean()

# data_st = data_st.reshape(len(data))

