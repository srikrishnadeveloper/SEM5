
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
