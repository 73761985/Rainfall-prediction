# -*- coding: utf-8 -*-
"""weather.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Tf-2cuTptS_a0x_Q_olqsQ-v0Oelp5Kg
"""

# Commented out IPython magic to ensure Python compatibility.
import zipfile as zf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# %matplotlib inline
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/archive (1).zip',compression='zip')
df.head()

df.shape

df.info()

df.describe()

df.isnull().sum()

num_features = [feature for feature in df.columns if df[feature].dtype=='float']

print("there are {} numerical features-\n ".format(len(num_features)))
print("Columns with numerical features are-\n",num_features)

cat_features = [feature for feature in df.columns if df[feature].dtype=='O']

print("there are {} categorical features-\n ".format(len(cat_features)))
print("Columns with categorical features are-\n",cat_features)

sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']=(10,6)

plt.figure(figsize=(10, 6))
sns.displot(
    data=df.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=1.20
)
plt.show()

df['RainTomorrow'].unique()

df[df['RainTomorrow']=="No"].shape

df[df['RainTomorrow']=="Yes"].shape

df['RainTomorrow'].isnull().sum()

df['RainToday'].isnull().sum()

missing = df.isnull().sum().sort_values(ascending=False)
missing_per = (missing/len(df))*100
pd.DataFrame({"Missing_Records":missing,"percentage of missing data":missing_per.values})

# sns.pairplot(df)

for feature in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], bins=25, kde=True)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(f"{feature} Distribution with KDE")
    plt.show()

data=df.copy()

columns_to_exclude = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
filtered_data = data.drop(columns=columns_to_exclude)

fig, axes = plt.subplots(nrows=1, ncols=len(filtered_data.columns), figsize=(14, 5))
colors = sns.color_palette('Set2', n_colors=len(filtered_data.columns))

for i, column in enumerate(filtered_data.columns):
    sns.boxplot(y=filtered_data[column], ax=axes[i], color=colors[i % len(colors)])
    axes[i].set_xlabel(None)

plt.tight_layout()
plt.show()

data1=df.copy()

data2= data1.dropna(subset=['RainToday','RainTomorrow'])

data2.shape

data2[data2['RainTomorrow']=="No"].shape

data2[data2['RainTomorrow']=="Yes"].shape

def calculate_percentage_of_yes_no(data, column_name='RainTomorrow'):
    value_counts = data[column_name].value_counts(normalize=True) * 100
    yes_percentage = value_counts.get('Yes', 0)
    no_percentage = value_counts.get('No', 0)
    print(f"Percentage of 'Yes': {yes_percentage:.2f}%")
    print(f"Percentage of 'No': {no_percentage:.2f}%")
    return {'Yes': yes_percentage, 'No': no_percentage}
percentages = calculate_percentage_of_yes_no(data2, 'RainTomorrow')
print(percentages)

px.histogram(data2,x="Location",title="Location vs Rainy Days",color="RainToday")

px.histogram(data2,x="Temp3pm",title="Temp. at 3pm vs RainTomorrow",color="RainTomorrow")

#mean temperature at 3pm depending on RainTomorrow
print("mean temp when it rains tomorrow")
print(data2[["RainTomorrow","Temp3pm"]].groupby("RainTomorrow").mean())

px.histogram(data2,x="RainTomorrow",color="RainToday",title="RainTomorrow vs RainToday")

px.scatter(data2.sample(2000),title="Min.Temp vs Max.Temp",x="MinTemp",y="MaxTemp",color="RainToday")

px.scatter(data2.sample(2000),title="Temp (3 pm) vs Humidity (3 pm) " , x="Temp3pm",y='Humidity3pm',color='RainTomorrow')

data2.drop(['Cloud9am', 'Cloud3pm','Evaporation','Sunshine'], axis=1)

data1.dropna(subset=['RainToday','RainTomorrow'], inplace = True)

data1.drop(['Cloud9am', 'Cloud3pm','Evaporation','Sunshine'], axis=1, inplace=True)

plt.figure(figsize=(10,6))
sns.displot( data=data1.isna().melt(value_name="missing"), y="variable", hue="missing", multiple="fill", aspect=1.25 )

data1['Date'] = pd.to_datetime(data1['Date'])
data1['Year'] = data1['Date'].dt.year
data1['Month'] = data1['Date'].dt.month
data1.drop('Date',inplace=True, axis=1)
data1

sns.set_theme(style='darkgrid')
sns.countplot(x='Year', hue='RainTomorrow', data=data1)

sns.countplot(x='Month', hue='RainTomorrow', data=data1)

#Listing columns with same datatype in vector
num_features = [column_name for column_name in data1.columns if data1[column_name].dtype=='float64']
cat_features = [column_name for column_name in data1.columns if data1[column_name].dtype=='object' or data1[column_name].dtype=='int64']
print("Columns with numerical datatypes-\n",(num_features))
print("columns with categorical datatypes-\n",(cat_features))

#List of numbers of different categorical values in each categorical columns
print(data1[cat_features].nunique())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data1[num_features]=scaler.fit_transform(data1[num_features])
data1[num_features].head()

#Creating boxplot for every numerical columns in data
plt.figure(figsize=(17, 20))
for i in range(len(num_features)):
    plt.subplot(4, 4, i+1)
    fig = data1.boxplot(column=num_features[i])
    fig.set_title("")
    fig.set_ylabel(num_features[i])
plt.show()

print("count of `RainTomorrow` before removing outliers")
print(Counter(data1["RainTomorrow"]))

#Removing outliers from numerical columns of the dataset
Q1 = data1[num_features].quantile(0.25)
Q3 = data1[num_features].quantile(0.75)
IQR = Q3-Q1
pd.concat([Q1,Q3,IQR],axis=1,keys=['Q1','Q2','IQR'])

for i, feature in enumerate(num_features):
    data1[feature] = np.where(data1[feature] > Q3[i] + 1.5 * IQR[i], Q3[i] + 1.5 * IQR[i], data1[feature])
    data1[feature] = np.where(data1[feature] < Q1[i] - 1.5 * IQR[i], Q1[i] - 1.5 * IQR[i], data1[feature])

data1[num_features].describe()

#Boxplot after Replacing the outliers with the max and min value from the numerical columns from the dataset
plt.figure(figsize=(17, 20))
for i in range(len(num_features)):
    plt.subplot(4, 4, i+1)
    fig = data1.boxplot(column=num_features[i])
    fig.set_title("")
    fig.set_ylabel(num_features[i])
plt.show()

print("Count of RainTomorrow values after removing the outliers")
print(Counter(data1['RainTomorrow']))

#Import the KNNImputer Class
imputer=KNNImputer(n_neighbors=5)
imputer=imputer.fit(data1[num_features])
data1[num_features]=imputer.transform(data1[num_features])
data1

plt.figure(figsize=(10,6))
sns.displot( data=data1.isna().melt(value_name="missing"), y="variable", hue="missing", multiple="fill", aspect=1.25 )

sns.heatmap(data1.isnull(),yticklabels=False,cmap='winter')

print(data1['WindGustDir'].mode())
print(data1['WindDir9am'].mode())
print(data1['WindDir3pm'].mode())

data1['WindGustDir'].fillna('W',inplace=True)
data1['WindDir9am'].fillna('N',inplace=True)
data1['WindDir3pm'].fillna('SE',inplace=True)

sns.heatmap(data1.isnull(),yticklabels=False,cmap='autumn')

def convert_categorical_features(feature):
    le = LabelEncoder()
    feature_encoded = le.fit_transform(feature)
    return(feature_encoded)

for feature in cat_features:
    data1[feature]=convert_categorical_features(data1[feature])
data1[cat_features]

data1.describe()

#Creating a copy of preprocessed dataset
df1=data1.copy()

#creating a temporary dataframe for numerical features
temp_df = df1[num_features]
temp_df

#correlation matrix
plt.figure(figsize=(16,25))
plt.title('Correlation heatmap of Rain in Australia dataset')
ax = sns.heatmap(temp_df.corr(),annot=True,vmax=0.9,center=0, square=True,linewidths=0.5,cbar_kws={'shrink':.5},cmap=sns.diverging_palette(20,200,n=200))
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_yticklabels(ax.get_yticklabels(),rotation=30)
plt.show()

for column1 in temp_df.columns:
    for column2 in range(temp_df.columns.get_loc(column1) + 1, len(temp_df.columns)):
        if temp_df[column1].corr(temp_df.iloc[:, column2]) >= 0.9:
            print(f"{column1} and {temp_df.columns[column2]} are highly correlated")

morning_temp = (df1['MinTemp'] + df1['Temp9am']) / 2
noon_temp = (df1['MaxTemp'] + df1['Temp3pm']) / 2
new_pressure = (df1['Pressure9am'] + df1['Pressure3pm']) / 2

temp_df['Morning_temp'] = morning_temp
temp_df['Noon_temp'] = noon_temp
temp_df['New_pressure'] = new_pressure

temp_df.drop(["MinTemp", "Temp9am", "MaxTemp", "Temp3pm", "Pressure9am", "Pressure3pm"], inplace=True, axis=1)

temp_df.head()

df1['Morning_temp'] = morning_temp
df1['Noon_temp'] = noon_temp
df1['New_pressure'] = new_pressure
df1.drop(["MinTemp","Temp9am","MaxTemp","Temp3pm","Pressure9am","Pressure3pm"], inplace=True, axis=1)
df1.head()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16,16))
plt.title('Correlation heatmap of Rain in Australia dataset')
ax = sns.heatmap(temp_df.corr(), annot=True, vmax=0.9, center=0, square=True, linewidths=0.5, cbar_kws={'shrink':0.5}, cmap=sns.diverging_palette(20,200,n=200))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.show()

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(Z3):
    vif_data = pd.DataFrame()
    vif_data["feature"] = Z3.columns
    vif_data["VIF"] = [variance_inflation_factor(np.array(Z3), i) for i in range(len(Z3.columns))]
    return vif_data

vif_df = vif(temp_df)
vif_df.sort_values(by=['VIF'],ascending=False)

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in range(X.iloc[:, variables].shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True
    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

calculate_vif_(temp_df,5)

df1.drop(['Noon_temp'], axis=1, inplace=True)
temp_df.drop(['Noon_temp'], axis=1, inplace=True)
temp_df

data1

num_features = [column_name for column_name in df1.columns if df1[column_name].dtype=='float64']
data1['Morning_temp']=morning_temp
data1['New_pressure']=new_pressure
data1.drop(['MinTemp','Temp9am','MaxTemp','Temp3pm','Pressure9am','Pressure3pm'],inplace=True , axis=1)
data1[num_features].info()

#best variable selected using Stepwise Feature Selection
best_var = ['Location','Rainfall','WindGustSpeed','WindDir9am','WindSpeed9am','WindSpeed3pm','Humidity9am', 'Humidity3pm','RainToday','Year','Month','Morning_temp','New_pressure']

#Splitting dataset into X(explanatory) and Y(target variable)
X = data1[best_var]
y=data1['RainTomorrow']

# Split X and y into trainning testing sets
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

X_train.shape , X_test.shape

from imblearn.over_sampling import SMOTE

oversample= SMOTE()
X_train_smote , y_train_smote = oversample.fit_resample(X_train,y_train)

from xgboost import XGBClassifier

xgb_model=XGBClassifier()

xgb_model.fit(X_train_smote,y_train_smote)

y_pred_xgb=xgb_model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred_xgb))

from sklearn.ensemble import RandomForestClassifier

RF_model=RandomForestClassifier()

RF_model.fit(X_train_smote,y_train_smote)

y_pred_RF=RF_model.predict(X_test)

print(accuracy_score(y_test,y_pred_RF))

# Make predictions on both training and test sets
y_pred_train = RF_model.predict(X_train_smote)
y_pred_test = RF_model.predict(X_test)

# Calculate accuracy for both training and test sets
train_accuracy = accuracy_score(y_train_smote, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

plt.bar(['Training', 'Test'], [train_accuracy, test_accuracy], color=['blue', 'red'])
plt.title('Model Performance Comparison')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

from sklearn.metrics import confusion_matrix

conf_matrix_train = confusion_matrix(y_train_smote, y_pred_train)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)

# Plot confusion matrices
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_train, annot=True, cmap='Blues', fmt='d', xticklabels=RF_model.classes_, yticklabels=RF_model.classes_)
plt.title('Training Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_test, annot=True, cmap='Blues', fmt='d', xticklabels=RF_model.classes_, yticklabels=RF_model.classes_)
plt.title('Testing Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.tight_layout()
plt.show()

from sklearn.ensemble import GradientBoostingClassifier

GB_model=GradientBoostingClassifier()

GB_model.fit(X_train_smote,y_train_smote)

y_pred_GB=GB_model.predict(X_test)

print(accuracy_score(y_pred_RF,y_pred_GB))

X_reset_index = X_train_smote.reset_index(drop=True)
y_reset_index = y_train_smote.reset_index(drop=True)

df_concatenated = pd.concat([X_reset_index, y_reset_index], axis=1)

# random_10_rows = df_concatenated.sample(n=50)
# random_10_rows

correct_mapping = all(df_concatenated.index == y_reset_index.index)

if correct_mapping:
    print("Every row of X is correctly mapped with the corresponding y.")
else:
    print("There is a mismatch between X and y.")

df_concatenated.shape

sampled_dataframes = []
num_samples = int(0.8 * len(df_concatenated))
num_dataframes = 5

for i in range(num_dataframes):
    sampled_df = df_concatenated.sample(n=num_samples, replace=True, random_state=i)
    sampled_dataframes.append(sampled_df)

from sklearn.tree import DecisionTreeClassifier

base_classifier = DecisionTreeClassifier()
classifiers = []

for i in range(5):
    X = sampled_dataframes[i].drop(columns=['RainTomorrow'])
    y = sampled_dataframes[i]['RainTomorrow']

    classifier = DecisionTreeClassifier()
    classifier.fit(X, y)

    classifiers.append(classifier)

def aggregate_predictions(classifiers, test_data):
    all_predictions = []

    for classifier in classifiers:
        predictions = classifier.predict(test_data)
        all_predictions.append(predictions)

    ensemble_predictions = []

    for i in range(len(test_data)):
        sample_predictions = [pred[i] for pred in all_predictions]
        majority_vote = max(set(sample_predictions), key=sample_predictions.count)
        ensemble_predictions.append(majority_vote)

    return ensemble_predictions

# predictions = aggregate_predictions(classifiers, X_test)

# pd.DataFrame(predictions).shape

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

predictions = aggregate_predictions(classifiers, X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(y_test, predictions)
print("Classification Report:")
print(class_report)

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

def evaluate_ensemble(classifiers, sampled_dataframes, X_test, y_test):
    train_accuracies = []
    test_accuracies = []
    test_conf_matrices = []

    for i, clf in enumerate(classifiers):
        X_train_sampled = sampled_dataframes[i].drop(columns=['RainTomorrow'])
        y_train_sampled = sampled_dataframes[i]['RainTomorrow']

        train_pred = clf.predict(X_train_sampled)
        test_pred = clf.predict(X_test)

        train_accuracy = accuracy_score(y_train_sampled, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        test_conf_matrix = confusion_matrix(y_test, test_pred)
        test_conf_matrices.append(test_conf_matrix)

    return train_accuracies, test_accuracies, test_conf_matrices

# Evaluate the ensemble model
train_accuracies, test_accuracies, test_conf_matrices = evaluate_ensemble(classifiers, sampled_dataframes, X_test, y_test)

# Plotting the comparison graph for accuracies
plt.plot(range(1, num_dataframes + 1), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(range(1, num_dataframes + 1), test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Ensemble Model')
plt.ylabel('Accuracy')
plt.title('Ensemble Model Performance')
plt.xticks(range(1, num_dataframes + 1))
plt.legend()
plt.show()

# Plotting confusion matrices for testing dataset
plt.figure(figsize=(16, 6))

for i in range(num_dataframes):
    plt.subplot(1, num_dataframes, i + 1)
    sns.heatmap(test_conf_matrices[i], annot=True, cmap='Blues', fmt='d', xticklabels=RF_model.classes_, yticklabels=RF_model.classes_)
    plt.title(f'Testing Confusion Matrix {i+1}')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Calculate accuracy for RF_model
RF_train_accuracy = accuracy_score(y_train, RF_model.predict(X_train))
RF_test_accuracy = accuracy_score(y_test, RF_model.predict(X_test))

# Calculate accuracy for the custom ensemble model
ensemble_train_accuracy = np.mean(train_accuracies)
ensemble_test_accuracy = np.mean(test_accuracies)

# Plotting the comparison graph for accuracies
labels = ['Random Forest', 'Custom Ensemble']
train_accuracies = [RF_train_accuracy, ensemble_train_accuracy]
test_accuracies = [RF_test_accuracy, ensemble_test_accuracy]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, train_accuracies, width, label='Training Accuracy')
bars2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

from sklearn.metrics import accuracy_score

# Calculate accuracy for RF_model
RF_train_accuracy = accuracy_score(y_train, RF_model.predict(X_train))
RF_test_accuracy = accuracy_score(y_test, RF_model.predict(X_test))

# Calculate accuracy for the custom ensemble model
ensemble_train_accuracy = np.mean(train_accuracies)
ensemble_test_accuracy = np.mean(test_accuracies)

# Print out the accuracy scores
print("Random Forest Model:")
print("Training Accuracy:", RF_train_accuracy)
print("Testing Accuracy:", RF_test_accuracy)

print("\nCustom Ensemble Model:")
print("Training Accuracy:", ensemble_train_accuracy)
print("Testing Accuracy:", ensemble_test_accuracy)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_accuracy = 0
best_num_dataframes = 0
best_classifiers = []

# Define the range of values for num_dataframes to search over
num_dataframes_values = [3, 4, 5, 6, 7,9,10,12,14,15,16,17,18,19,20]

for num_dataframes in num_dataframes_values:
    sampled_dataframes = []
    num_samples = int(0.8 * len(df_concatenated))

    for i in range(num_dataframes):
        sampled_df = df_concatenated.sample(n=num_samples, replace=True, random_state=i)
        sampled_dataframes.append(sampled_df)

    classifiers = []

    for i in range(num_dataframes):
        X = sampled_dataframes[i].drop(columns=['RainTomorrow'])
        y = sampled_dataframes[i]['RainTomorrow']

        classifier = DecisionTreeClassifier()
        classifier.fit(X, y)

        classifiers.append(classifier)

    predictions = aggregate_predictions(classifiers, X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_dataframes = num_dataframes
        best_classifiers = classifiers

print("Best number of dataframes:", best_num_dataframes)
print("Best accuracy:", best_accuracy)

# Now you can use the best classifiers and the best number of dataframes for further analysis

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_accuracy = 0
best_num_dataframes = 0
best_classifiers = []

# Define the range of values for num_dataframes to search over
num_dataframes_values = [21,22,23,24]

for num_dataframes in num_dataframes_values:
    sampled_dataframes = []
    num_samples = int(0.8 * len(df_concatenated))

    for i in range(num_dataframes):
        sampled_df = df_concatenated.sample(n=num_samples, replace=True, random_state=i)
        sampled_dataframes.append(sampled_df)

    classifiers = []

    for i in range(num_dataframes):
        X = sampled_dataframes[i].drop(columns=['RainTomorrow'])
        y = sampled_dataframes[i]['RainTomorrow']

        classifier = DecisionTreeClassifier()
        classifier.fit(X, y)

        classifiers.append(classifier)

    predictions = aggregate_predictions(classifiers, X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_dataframes = num_dataframes
        best_classifiers = classifiers

print("Best number of dataframes:", best_num_dataframes)
print("Best accuracy:", best_accuracy)

# Now you can use the best classifiers and the best number of dataframes for further analysis

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_accuracy = 0
best_num_dataframes = 0
best_classifiers = []

# Define the range of values for num_dataframes to search over
num_dataframes_values = [100]

for num_dataframes in num_dataframes_values:
    sampled_dataframes = []
    num_samples = int(0.8 * len(df_concatenated))

    for i in range(num_dataframes):
        sampled_df = df_concatenated.sample(n=num_samples, replace=True, random_state=i)
        sampled_dataframes.append(sampled_df)

    classifiers = []

    for i in range(num_dataframes):
        X = sampled_dataframes[i].drop(columns=['RainTomorrow'])
        y = sampled_dataframes[i]['RainTomorrow']

        classifier = DecisionTreeClassifier()
        classifier.fit(X, y)

        classifiers.append(classifier)

    predictions = aggregate_predictions(classifiers, X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_dataframes = num_dataframes
        best_classifiers = classifiers

print("Best number of dataframes:", best_num_dataframes)
print("Best accuracy:", best_accuracy)

# Now you can use the best classifiers and the best number of dataframes for further analysis

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_accuracy = 0
best_num_dataframes = 0
best_classifiers = []

# Define the range of values for num_dataframes to search over
num_dataframes_values = [110]

for num_dataframes in num_dataframes_values:
    sampled_dataframes = []
    num_samples = int(0.8 * len(df_concatenated))

    for i in range(num_dataframes):
        sampled_df = df_concatenated.sample(n=num_samples, replace=True, random_state=i)
        sampled_dataframes.append(sampled_df)

    classifiers = []

    for i in range(num_dataframes):
        X = sampled_dataframes[i].drop(columns=['RainTomorrow'])
        y = sampled_dataframes[i]['RainTomorrow']

        classifier = DecisionTreeClassifier()
        classifier.fit(X, y)

        classifiers.append(classifier)

    predictions = aggregate_predictions(classifiers, X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_dataframes = num_dataframes
        best_classifiers = classifiers

print("Best number of dataframes:", best_num_dataframes)
print("Best accuracy:", best_accuracy)

# Now you can use the best classifiers and the best number of dataframes for further analysis

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_accuracy = 0
best_num_dataframes = 0
best_classifiers = []

# Define the range of values for num_dataframes to search over
num_dataframes_values = [115]

for num_dataframes in num_dataframes_values:
    sampled_dataframes = []
    num_samples = int(0.8 * len(df_concatenated))

    for i in range(num_dataframes):
        sampled_df = df_concatenated.sample(n=num_samples, replace=True, random_state=i)
        sampled_dataframes.append(sampled_df)

    classifiers = []

    for i in range(num_dataframes):
        X = sampled_dataframes[i].drop(columns=['RainTomorrow'])
        y = sampled_dataframes[i]['RainTomorrow']

        classifier = DecisionTreeClassifier()
        classifier.fit(X, y)

        classifiers.append(classifier)

    predictions = aggregate_predictions(classifiers, X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_dataframes = num_dataframes
        best_classifiers = classifiers

print("Best number of dataframes:", best_num_dataframes)
print("Best accuracy:", best_accuracy)

# Now you can use the best classifiers and the best number of dataframes for further analysis

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sns.set()

print(X_train_smote.shape)
print(y_train_smote.shape)

x = X_train_smote.iloc[:, 0]
y = X_train_smote.iloc[:, 1]

plt.figure(figsize=(7, 5))
plt.scatter(x[y_train_smote.values==0], y[y_train_smote.values==0], c='orange', edgecolors='w', s=100, label='class 0')
plt.scatter(x[y_train_smote.values==1], y[y_train_smote.values==1], c='crimson', edgecolors='w', s=100, label='class 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(fontsize=14)
plt.show()

df2=df_concatenated.copy()

print(Counter(df2['RainTomorrow']))

num_samples = int(0.95 * len(df2))

subset = df2.sample(n=num_samples, replace=True, random_state=42)

print(Counter(subset['RainTomorrow']))

num_1 = (subset['RainTomorrow'] == 1).sum()
num_0 = (subset['RainTomorrow'] == 0).sum()

subset['pre1(log-odds)'] = np.log(num_1/num_0)
subset

subset['pre1(probability)'] = 1/(1+np.exp(-np.log(num_1/num_0)))
subset

subset['res1'] = subset['RainTomorrow'] - subset['pre1(probability)']
subset

var=['Location','Rainfall','WindGustSpeed','WindDir9am','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','RainToday','Year','Month','Morning_temp','New_pressure']

x_=subset[var]
y_=subset['res1']

from sklearn.tree import DecisionTreeRegressor

model2=DecisionTreeRegressor()
model2.fit(x_,y_)

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# plot_tree(model2, feature_names=['Location','Rainfall','WindGustSpeed','WindDir9am','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','RainToday','Year','Month','Morning_temp','New_pressure'],filled=True, node_ids=True)
# plt.show()

subset['leaf_entry1'] = model2.apply(x_)
subset

def return_logs(leaf):
  temp_df = subset[subset['leaf_entry1'] == leaf]
  num = temp_df['res1'].sum()

  den = sum(temp_df['pre1(probability)'] * (1 - temp_df['pre1(probability)']))
  return round(num/den,2)

subset['pre2(log-odds)'] = subset['pre1(log-odds)'] + subset['leaf_entry1'].apply(return_logs)

subset['pre2(probability)'] = 1/(1+np.exp(-subset['pre2(log-odds)']))

subset['res2'] = subset['RainTomorrow'] - subset['pre2(probability)']

reg2 = DecisionTreeRegressor()

reg2.fit(x_,subset['res2'])

subset['leaf_entry2'] = reg2.apply(x_)

def return_logs(leaf):
  num = subset[subset['leaf_entry2'] == leaf]['res2'].sum()
  den = sum(subset[subset['leaf_entry2'] == leaf]['pre2(probability)'] * (1 - subset[subset['leaf_entry2'] == leaf]['pre2(probability)']))
  return round(num/den,2)

subset['pre3(log-odds)'] = subset['pre1(log-odds)'] + subset['pre2(log-odds)'] + subset['leaf_entry2'].apply(return_logs)

subset['pre3(probability)'] = 1/(1+np.exp(-subset['pre3(log-odds)']))
subset

subset['res_final'] = subset['RainTomorrow'] - subset['pre3(probability)']

subset[['res1','res2','res_final']]

from sklearn.metrics import roc_curve, roc_auc_score

predicted_probabilities = subset['pre3(probability)']

fpr, tpr, thresholds = roc_curve(subset['RainTomorrow'], predicted_probabilities)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

auc = roc_auc_score(subset['RainTomorrow'], predicted_probabilities)
print('AUC:', auc)

optimal_threshold_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_index]
print('Optimal Threshold:', optimal_threshold)

predicted_labels = (predicted_probabilities >= optimal_threshold).astype(int)

# Let's calculate accuracy

accuracy = (predicted_labels == subset['RainTomorrow']).sum() / len(subset['RainTomorrow'])

print('Accuracy:', accuracy)

num_1_t = (y_test == 1).sum()
num_0_t = (y_test == 0).sum()

X_test['pre1(log-odds)'] = np.log(num_1_t / num_0_t)
X_test['pre1(probability)'] = 1 / (1 + np.exp(-np.log(num_1_t / num_0_t)))

# Iteration 2
X_test['leaf_entry1'] = model2.apply(X_test[var])
X_test['pre2(log-odds)'] = X_test['pre1(log-odds)'] + X_test['leaf_entry1'].apply(return_logs)
X_test['pre2(probability)'] = 1 / (1 + np.exp(-X_test['pre2(log-odds)']))

# Iteration 3
X_test['leaf_entry2'] = reg2.apply(X_test[var])
X_test['pre3(log-odds)'] = X_test['pre1(log-odds)'] + X_test['pre2(log-odds)'] + X_test['leaf_entry2'].apply(return_logs)
X_test['pre3(probability)'] = 1 / (1 + np.exp(-X_test['pre3(log-odds)']))

predicted_probabilities_test = X_test['pre3(probability)']
optimal_threshold =  0.49920153260687306
predicted_labels_test = (predicted_probabilities_test >= optimal_threshold).astype(int)

# y_test

# predicted_labels_test

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predicted_labels_test)
print("Accuracy:", accuracy)

# X_test

# num_1_t = (y_test == 1).sum()
# num_1_t

# num_0_t = (y_test == 0).sum()
# num_0_t

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

df3=df_concatenated.copy()

xx=df3[var]
yy=df3['RainTomorrow']

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# Define your classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    # "SVM": SVC(probability=True)
}

classifier_names = {
    "Logistic Regression": "LR",
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "Gradient Boosting": "GB",
    # "SVM": "SVM"
}

# Function for k-fold indices
def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds

# Data
k = 5
predictions_df = pd.DataFrame()

# Loop through classifiers
for clf_name, clf in classifiers.items():
    print(f"Processing {clf_name}...")
    fold_predictions = []
    fold_indices = []

    for train_indices, test_indices in kfold_indices(xx, k):
        X_train_fold, X_test_fold = xx.iloc[train_indices], xx.iloc[test_indices]
        y_train_fold, y_test_fold = yy.iloc[train_indices], yy.iloc[test_indices]

        clf.fit(X_train_fold, y_train_fold)
        fold_pred = clf.predict(X_test_fold)
        fold_predictions.append(fold_pred)
        fold_indices.extend(X_test_fold.index)

    predictions_df[classifier_names[clf_name] + "_pred"] = np.concatenate(fold_predictions)
    predictions_df.index = fold_indices

print(predictions_df)

print(predictions_df.shape)
print(yy.shape)

indices_not_common = xx.index.difference(predictions_df.index)

print(indices_not_common)

y_train_aligned = yy.drop(indices_not_common)

df_combined = pd.concat([predictions_df, y_train_aligned], axis=1)

print(df_combined)

xt=df_combined.drop(["RainTomorrow"],axis=1)
yt=df_combined["RainTomorrow"]

from sklearn.tree import DecisionTreeClassifier

modelm=DecisionTreeClassifier()

modelm.fit(xt,yt)

X_test

X_test[var]

base_models = {
    "LR_pred": LogisticRegression(),
    "DT_pred": DecisionTreeClassifier(),
    "RF_pred": RandomForestClassifier(),
    "GB_pred": GradientBoostingClassifier(),
    # "SVM_pred": SVC(probability=True)
}

base_model_predictions = {}
for model_name, model in base_models.items():
    model.fit(xx, yy)
    predictions = model.predict(X_test[var])
    base_model_predictions[model_name] = predictions

base_model_predictions_df = pd.DataFrame(base_model_predictions)

base_model_predictions_df

base_model_predictions_df = base_model_predictions_df.rename(columns={
    'LR_Pred': 'LR_pred',
    'DT_Pred': 'DT_pred',
    'RF_Pred': 'RF_pred',
    'GB_Pred': 'GB_pred',
    # 'SVM_Pred': 'SVM_pred'
})

base_model_predictions_df

metamodel_predictions = modelm.predict(base_model_predictions_df)

metamodel_predictions

metamodel_predictions.shape

from sklearn.metrics import accuracy_score

metamodel_accuracy = accuracy_score(y_test, metamodel_predictions)
print("Metamodel Accuracy:", metamodel_accuracy)

class_report = classification_report(y_test, metamodel_predictions)
print("Classification Report:")
print(class_report)





from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# Define your classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    # "SVM": SVC(probability=True)  # Uncomment if needed
}

classifier_abbreviations = {
    "Logistic Regression": "LR",
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "Gradient Boosting": "GB",
    # "SVM": "SVM"  # Uncomment if needed
}

# Function for k-fold indices
def generate_kfold_indices(data, num_folds):
    fold_size = len(data) // num_folds
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    folds = []
    for i in range(num_folds):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds

# Data
num_folds = 10
predictions_df = pd.DataFrame()

# Loop through classifiers
for clf_name, clf in classifiers.items():
    print(f"Processing {clf_name}...")
    fold_predictions = []
    all_test_indices = []

    for train_indices, test_indices in generate_kfold_indices(xx, num_folds):
        X_train, X_test = xx.iloc[train_indices], xx.iloc[test_indices]
        y_train, y_test = yy.iloc[train_indices], yy.iloc[test_indices]

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        fold_predictions.append(predictions)
        all_test_indices.extend(test_indices)

    predictions_df[classifier_abbreviations[clf_name] + "_pred"] = np.concatenate(fold_predictions)
    predictions_df.index = all_test_indices

print(predictions_df)
print(predictions_df.shape)
print(yy.shape)

# Find indices not common between xx and predictions_df
uncommon_indices = xx.index.difference(predictions_df.index)

print(uncommon_indices)

aligned_y_train = yy.drop(uncommon_indices)

combined_df = pd.concat([predictions_df, aligned_y_train], axis=1)

print(combined_df)







base_models_new = {
    "LR_pred": LogisticRegression(),
    "DT_pred": DecisionTreeClassifier(),
    "RF_pred": RandomForestClassifier(),
    "GB_pred": GradientBoostingClassifier(),
    # "SVM_pred": SVC(probability=True)
}

base_model_predictions_new = {}
for model_name, model in base_models_new.items():
    model.fit(xx, yy)
    predictions_new = model.predict(X_test[var])
    base_model_predictions_new[model_name] = predictions_new

base_model_predictions_df_new = pd.DataFrame(base_model_predictions_new)

base_model_predictions_df_new = base_model_predictions_df_new.rename(columns={
    'LR_Pred': 'LR_pred',
    'DT_Pred': 'DT_pred',
    'RF_Pred': 'RF_pred',
    'GB_Pred': 'GB_pred',
    # 'SVM_Pred': 'SVM_pred'
})

base_model_predictions_df_new

xt_new=df_combined.drop(["RainTomorrow"],axis=1)
yt_new=df_combined["RainTomorrow"]

xt_new

yt_new

metamodel_new=DecisionTreeClassifier()

metamodel_new.fit(xt_new,yt_new)

metamodel_predictions_new = metamodel_new.predict(base_model_predictions_df_new)

metamodel_predictions_new

metamodel_accuracy_new = accuracy_score(y_test, metamodel_predictions_new)
print("Metamodel Accuracy:", metamodel_accuracy_new)

class_report_new = classification_report(y_test, metamodel_predictions_new)
print("Classification Report:")
print(class_report_new)
