## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  DEVELOPED BY : V.SAI SRUTHI
  
  REG NO : 21222300061
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/4222f3a0-37e1-4915-9e31-10f10b4cb53c)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
oe=OrdinalEncoder(categories=[pm])
oe.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/36e1b9c2-2ade-46e9-9eaa-3ff5d5b3b901)

```
 df['bo2']=oe.fit_transform(df[["ord_2"]])
 print(df)
```
![image](https://github.com/user-attachments/assets/2640628f-e2b8-41a3-bb60-c5cc364c0175)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/a8e8d945-23b4-4607-bf4d-94ecbd337fd3)


```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2 = pd.concat([df2, enc], axis=1)
df2
```
![image](https://github.com/user-attachments/assets/2719189b-f557-4bb9-ad49-2076b156ae80)

```
 pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/e00f8952-d36f-4708-98b6-472cb6e43451)

```
pip install category_encoders
```
![image](https://github.com/user-attachments/assets/67a51a4f-114c-473a-b514-71cef214fc43)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/545efd9c-30f5-4b02-9b57-d8ce822a2930)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/da208581-8e7b-4855-9ba7-9a106604e5d3)

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/75522c18-aefc-4644-92f4-146cf4cb0fb4)

```
 from category_encoders import TargetEncoder
 te=TargetEncoder()
 CC=df.copy()
 new=te.fit_transform(X=CC["City"],y=CC["Target"])
 CC=pd.concat([CC,new],axis=1)
 CC
```
![image](https://github.com/user-attachments/assets/5ea6ea66-9e00-4978-9621-dfac61ada222)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/c672ae1c-a4aa-418c-8abb-6e3f4fa04ac9)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/37e74176-9d00-4a99-87c7-ffd849cf4122)

```
 np.log(df["Highly Positive Skew"])

```
![image](https://github.com/user-attachments/assets/ba73f51e-8eaa-43ae-9120-783feb1360f8)

```
 np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c6c36099-d6b5-4ab2-9994-db83b01bb74e)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/61098a66-2dcf-4c22-b9af-586ccb17fa2f)

```
 np.square(df["Highly Positive Skew"])
```

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/c34a9466-8d0e-430c-b919-57a42410ed9c)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/7938e0f0-9f50-471e-aff0-4ac661a086cb)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/0f708371-de26-4595-b672-5b83a89dff2b)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/eae393e0-adf9-4acf-b865-19057ccf2815)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/6fc5015e-61d2-4fc5-b014-f72680cf916f)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/fe79dacc-15dd-451b-9f55-c74e1e64084a)

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/0baae28e-3414-4a03-b718-1eeabb286e2e)

```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```
![image](https://github.com/user-attachments/assets/9f361ffa-428b-41a3-a251-f643f270c35d)


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
