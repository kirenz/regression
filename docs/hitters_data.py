# # Hitters data preparation

# ## Import
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/kirenz/datasets/master/Hitters.csv")
# drop missing cases
df = df.dropna()
# ## Create label and features
# Since we will use algorithms from scikit learn, we need to encode our categorical features as one-hot numeric features (dummy variables):
dummies = pd.get_dummies(df[['League', 'Division','NewLeague']])
# Next, we create our label y:
y = df['Salary']
X_numerical = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
# Make a list of all numerical features (we need them later):
list_numerical = X_numerical.columns
# Create all features
X = pd.concat([X_numerical, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
# ### Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


df_train = pd.DataFrame(X_train)
df_train['Salary'] = pd.DataFrame(y_train)

