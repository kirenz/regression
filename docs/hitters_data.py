# Hitters data preparation



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import
df = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/master/Hitters.csv")

# drop missing cases
df = df.dropna()

# Create dummies
dummies = pd.get_dummies(df[['League', 'Division','NewLeague']])

# Create our label y:
y = df[['Salary']]
X_numerical = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Make a list of all numerical features
list_numerical = X_numerical.columns

# Create all features
X = pd.concat([X_numerical, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
feature_names = X.columns

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Data standardization
scaler = StandardScaler().fit(X_train[list_numerical]) 

X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])

# Make pandas dataframes
df_train = y_train.join(X_train)
df_test = y_test.join(X_test)