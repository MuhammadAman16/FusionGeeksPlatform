import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv(r'_YOUR_LOCATION_\house_data.csv')

# Select columns
columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

# Separate features and target
X = df[['bedrooms', 'bathrooms', 'floors', 'yr_built']]
y = df['price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Define preprocessing steps
numeric_features = ['bedrooms', 'bathrooms', 'yr_built']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['floors']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LinearRegression())])

# Fit model
clf.fit(X_train, y_train)

# Evaluate model
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")

# Save model
pickle.dump(clf, open('model.pkl', 'wb'))
