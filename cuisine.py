# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df=pd.read_csv('Cuisine_rating.csv')


print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# missing value treatment
print(df.isnull().sum()) 


# removing duplicates
dups = df.duplicated()
print('no. of duprows=%d'%(dups.sum()))
df[dups]
df.drop_duplicates(inplace=True)


# outlier treatment
def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range=Q1-(1.5*IQR)
    upper_range=Q3+(1.5*IQR)
    return lower_range,upper_range

#  standardization

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_col = df.select_dtypes(include=['float64','int64']).columns
df[num_col] = scaler.fit_transform(df[num_col])
print(df[num_col[:5]].head())

# onehot encoding
df = pd.get_dummies(df, drop_first=True)
print(df.head())

#univariate analysis 
for col in num_col:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True , bins = 30)
    plt.title(f'Distribution of {col}')
    plt.show()
    
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col)
    plt.title(f'count plot of {col}')
    plt.xticks(rotation=45)
    plt.show()
    
# bivariate analysis
sns.heatmap(df.corr(), annot = True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[num_col])
plt.show()

for col in cat_cols:
    for num in num_col:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df, x=col, y=num)
        plt.title(f'{num} vs {col}')
        plt.xticks(rotation=45)
        plt.show()