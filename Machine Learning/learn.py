import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2. Split into Train, Validation, and Test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# 3. Identify categorical columns
cat_col = X_train.select_dtypes(include=["object", "category"]).columns

# 4. One-Hot Encode (pd.get_dummies) for X_train
train_dummies = pd.get_dummies(X_train[cat_col], drop_first=True)
X_train = pd.concat([X_train.drop(columns=cat_col), train_dummies], axis=1)

# 5. One-Hot Encode for X_val and align columns with X_train
val_dummies = pd.get_dummies(X_val[cat_col], drop_first=True)
X_val = pd.concat([X_val.drop(columns=cat_col), val_dummies], axis=1)
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

# 6. One-Hot Encode for X_test and align columns with X_train
test_dummies = pd.get_dummies(X_test[cat_col], drop_first=True)
X_test = pd.concat([X_test.drop(columns=cat_col), test_dummies], axis=1)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)

X_encoded = encoder.fit_transform(X)


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[
        ["School","College","Masters"],
        ["low","meduim","high"]
        ]
)
catgorical = ["Education","Stasfication"]

encoded = encoder.fit_transform(df[catgorical])

encoded_df = pd.DataFrame(
    encoded,
    columns =catgorical,
    index=df.index

)

from category_encoders import BinaryEncoder

encoder = BinaryEncoder(
    cols=["City"]
)

X_encoded = encoder.fit_transform(X)
