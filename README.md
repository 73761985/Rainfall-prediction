# Rainfall-prediction
# Introduction & Problem Statement
Rainfall prediction plays a critical role in various sectors such as agriculture, water resource management, urban planning, and disaster preparedness. Accurate forecasting of rainfall patterns can aid in decision-making processes, mitigate risks, and improve overall resilience to weather-related events. This project focuses on developing a robust and reliable rainfall prediction model using machine learning techniques.
Objective of this project is to build a machine learning model using Ensemble Technique to Predict the rainfall.
# Proposed Methodology
# Choosing dataset and preprocessing: 
Name of Dataset: Used a real dataset named “weatherAUS”.
Size of Dataset: It contains 145460 records with 23 Features about 10 years of daily weather observations from many locations across Australia. RainTomorrow is the target variable to predict. It means did it rain the next day, Yes or No? This column is Yes if the rain for that day was 1mm or more. Dataset is in csv format.
Source of Dataset: This dataset is available and can be accessed through
below link--
https://www.kaggle.com/datasets/gauravduttakiit/weather-in-aus
# Exploratory Data Analysis (EDA):
• Removal of missing values: - Deleted records with missing values at the Rain Today and Rain Tomorrow columns as it was only 3.25%. Also, approximately 40% of Evaporation, Sunshine, Cloud9am,Cloud3pm, columns have missing values . filling these columns could result in outliers. Therefore, they has also been removed from the dataset.
• Imputing missing values:- remaining categorical features with missing values will be imputed with most repeated values.
• Imputed the missing values in numerical columns by their mean values. Since the mean is sensitive to Outliers so first standardize the numerical columns above and input the missing values using the mean.
• Feature Construction: -Separated the Date Variable into two variables Year and month and then dropped the date variable.
• Scaling the Feature:- Applied Standard Scalar technique to scale the features
• Outlier Handling:- Applied IQR method to remove Outliers
• Correlation and Multicollinearity:-Utilized correlation matrix to see correlation between features Since  MinTemp and Temp9amn highly correlated with each other  so created new feature morning_temp by combining these . same  process applied to create noon_temp from MaxTemp and  Temp3pm also created new_pressure from Pressure9am and  Pressure3pm and dropped MinTemp, Temp9am, MaxTemp, Temp3pm,Pressure9am, Pressure3pm.  Utilized VIF score to see multicollinearity and dropped  Noon_Temp 
• Best Features :-At last finalized best input features with combination of below givenfeatures Location,Rainfall,WindGustSpeed,WindDir9am,WindSpeed9am,
 WindSpeed3pm,Humidity9am,Humidity3pm,RainToday,Year,Month,Morning_temp, New_pressure 
• Output Column:-Output Column is  RainTomorrow 
• Imbalance Handling : 
After EDA shape of dataset is (140787,14).Since dataset is  imbalanced as number of “Yes” in output column is 22.16% and a  number of “No” in output column is 77.84%. So, applied 
SMOTE(synthetic minority oversampling) to handle imbalance in dataset. 
# Implementing Bagging Model:
Dataset Partitioning
The dataset is partitioned into multiple subsets using random sampling with replacement. For each subset, a decision tree classifier is trained independently. These classifiers collectively form the ensemble of base classifiers in the bagging model.
Prediction Aggregation
To make predictions, the outputs of individual classifiers are aggregated using a majority voting scheme. This process results in an ensemble prediction that represents the consensus of the base classifiers.
Performance Evaluation
The performance of the bagging ensemble model is evaluated using standard metrics such as accuracy score. 

# Implementing Boosting Model : 
Since we know that  Boosting is a powerful ensemble learning technique designed to 
enhance the performance of weak learners by combining them into a strong learner. Unlike bagging, where multiple models are trained independently, boosting sequentially trains a series of weak learners, with each subsequent learner focusing on correcting the errors made by its predecessors. The final prediction is then determined through a weighted combination of the individual weak learners. 
• The fundamental idea behind boosting is to iteratively improve the model's performance by giving more weight to the observations that were misclassified in previous iterations. This iterative process effectively adjusts the model's focus towards the difficult-to-classify instances, thereby improving its overall accuracy and robustness. 
So for this followed some steps:
Base Model Creation: 
• A Decision Tree Regressor (model2) was instantiated and trained on features (var) and the residual (res1) obtained from the initial prediction. 
• Each instance of the Decision Tree Regressor was trained to predict the residuals of the previous iteration. 
Iterative Model Training: 
• Sequentially, two more iterations of model training were conducted. 
• In the second iteration, another Decision Tree Regressor (reg2) was trained on features (var) and the residual (res2) obtained from the second prediction. 
• Similarly, in the third iteration, predictions were refined using the residuals from the previous two iterations. 
Prediction Refinement: 
• At each iteration, predictions were refined by combining the previous predictions with the new prediction obtained from the current model. 
• Log-odds and probability values were computed at each step to refine the predictions. 
Evaluation on Training Set: 
• The accuracy of the final predictions was evaluated using the area under the ROC curve (AUC) and the optimal threshold. 
• The ROC curve was plotted to visualize the trade-off between true positive rate and false positive rate. 
# Implementing Stacking Model
Base Model Training:
Two base models, Support Vector Classifier (SVC) and Logistic Regression (LR), are trained independently on the provided training data. Each model learns patterns and relationships within the data using its respective algorithm.

# K-Fold Cross Validation for Meta Model Creation:
K-Fold Cross Validation is implemented on the training data. For each fold, the training data is split into K subsets (folds). Iterating over each fold, both SVC and LR models are trained on K-1 folds and then predict on the remaining fold. Predictions from both models are collected for each fold.

# Creating Meta Model Dataset:
The predictions obtained from the previous step are used to construct a new dataset. Each instance in this dataset corresponds to a data point from the original training set. Features for each instance consist of the predictions made by the base models (SVC and LR) in the previous step. The target variable for each instance remains the same as the original training data.

# Meta Model Training:
A meta-model, also known as a meta-classifier or a second-level model, is trained on the dataset created in the previous step. This meta-model learns to combine the predictions from the base models to make the final prediction. Any suitable algorithm, such as logistic regression, random forest, or gradient boosting, can be used for training the meta-model.


# Prediction:
Once the meta-model is trained, it can be utilized to make predictions on new, unseen data. For each new data point, predictions are made using the base models (SVC and LR). These predictions serve as input to the meta-model to obtain the final prediction.
