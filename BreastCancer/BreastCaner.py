
#### BREAST CANCER CLASSIFICATION PROBLEM ####


#About Dataset
#Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
#n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

#This database is also available through the UW CS ftp server:
#ftp ftp.cs.wisc.edu
#cd math-prog/cpo-dataset/machine-learn/WDBC/

#Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

#Attribute Information:

#1) ID number
#2) Diagnosis (M = malignant, B = benign)
#3-32)

#Ten real-valued features are computed for each cell nucleus:

#a) radius (mean of distances from center to points on the perimeter)
#b) texture (standard deviation of gray-scale values)
#c) perimeter
#d) area
#e) smoothness (local variation in radius lengths)
#f) compactness (perimeter^2 / area - 1.0)
#g) concavity (severity of concave portions of the contour)
#h) concave points (number of concave portions of the contour)
#i) symmetry
#j) fractal dimension ("coastline approximation" - 1)

#The mean, standard error and "worst" or largest (mean of the three
#largest values) of these features were computed for each image,
#resulting in 30 features. For instance, field 3 is Mean Radius, field
#13 is Radius SE, field 23 is Worst Radius.

#All feature values are recoded with four significant digits.

#Missing attribute values: none

#Class distribution: 357 benign, 212 malignant




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis,LocalOutlierFactor
from sklearn.decomposition import PCA

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)

#warning library
import warnings
warnings.filterwarnings("ignore")





#### DATASET UPLOAD ####



df = pd.read_csv("Python/data.csv")
df.drop(["Unnamed: 32","id"],inplace=True,axis=1)

# Unwanted variables removed.

df = df.rename(columns={"diagnosis":"target"})

import re
sns.countplot(df["target"])
print()
df.target.value_counts()

df["target"] = [1 if i.strip() =="M" else 0 for i in df.target]


def basic_eda(dataFrame,head=5):
    print("################# SHAPE ##############")
    print(dataFrame.shape)
    print("########### Types ##################")
    print(dataFrame.dtypes)
    print("################## Head ###############")
    print(dataFrame.head(head))
    print("################## Tail ###############")
    print(dataFrame.tail(head))
    print("################## NA #################")
    print(dataFrame.isnull().sum())
    print("#################### Quantiles ###############")
    print(dataFrame.describe([0,0.05,0.25,0.5,0.75,0.95,0.99,1]).T)
    print("#################### INFO ###############")
    print(dataFrame.info)
    print("#################### NA ###############")
    print(dataFrame.isnull().sum())

basic_eda(df)

describe = df.describe()



#### EXPLORATORY DATA ANALYSIS (EDA) ####



# Correlation

corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot=True, fmt = ".2f")
plt.title("Correlation Matrix Between Features")
plt.show()


threshold = 0.75
filter = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filter].tolist()
sns.clustermap(df[corr_features].corr(), annot=True, fmt=".2f")
plt.title("Corr Matrix Between Features w Corr Threshold 0.75")
plt.show()


# box plot

df_melted = pd.melt(df,
                    id_vars = "target",   # 2 farklı class olduğu için 2 farklı class şeklinde görselleştirilmek isteniyor
                    var_name = "features",
                    value_name = "value"
                    )

plt.figure()
sns.boxplot(x="features", y="value", hue="target", data=df_melted)
plt.xticks(rotation=90) # featurelerin isimleri 90 derece dik olması için
plt.show()

## standardization - normalization

# pair plot

sns.pairplot(df[corr_features], diag_kind="kde", markers="+", hue="target")
plt.show()


######  OUTLIER DETECTION AND REMOVAL -> LOCAL OUTLIER FACTOR ####


y = df.target
x = df.drop(["target"], axis=1)

columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
x_score =  clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["outlier_score"] = x_score

#threshold
threshold = -2
filter = outlier_score["outlier_score"] < threshold
outlier_index = outlier_score[filter].index.tolist()


plt.figure()
plt.scatter(x.iloc[outlier_index,0],
            x.iloc[outlier_index,1],
            color = "blue",
            s = 50,
            label = "Outliers")

plt.scatter(x.iloc[:,0],
            x.iloc[:,1],
            color="k", s=3 ,
            label="Data Points")

radius = (x_score.max() - x_score)/(x_score.max() - x_score.min())
outlier_score["radius"] = radius

plt.scatter(x.iloc[:,0] , x.iloc[:,1],
            edgecolors= "r" ,
            s = 1000*radius ,
            facecolors= "none",
            label = " Outlier Score")
plt.legend()
plt.show()


# drop outliers

x = x.drop(outlier_index)

y = y.drop(outlier_index).values


#### SEPARATION OF DATA SET AS TRAINING AND TEST DATA SET ####
#### TEST - TRAİN - SPLİT ####


x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.3, random_state=35)



#### STANDARDIZATION ####

x_scaled = StandardScaler().fit_transform(x_train)
x_train = pd.DataFrame(x_scaled,columns=columns)

#Since we will train the knn algorithm with x_train, it is necessary to standardize the x_test according to x_train
x_test = StandardScaler().fit_transform(x_test)


#boxplot()
x_train_df = pd.DataFrame(x_train,columns=columns)
x_train_df["target"] = y_train
df_melted = pd.melt(x_train_df,
                    id_vars = "target",   # 2 farklı class olduğu için 2 farklı class şeklinde görselleştirilmek isteniyor
                    var_name = "features",
                    value_name = "value"
                    )

plt.figure()
sns.boxplot(x="features", y="value", hue="target", data=df_melted)
plt.xticks(rotation=90)
plt.show()


####  KNN MODEL FIT ####

knn_model = KNeighborsClassifier(n_neighbors=2).fit(x_train,y_train)
y_prediction = knn_model.predict(x_test)
cm = confusion_matrix(y_test,y_prediction)
acc = accuracy_score(y_test,y_prediction)
score = knn_model.score(x_test,y_test)

print("SCORE:",score)
print("CM:",cm)
print("Basic KNN Acc:",acc)


cv_results1 = cross_validate(knn_model,x_train,y_train,cv=10,
                            scoring=["accuracy","f1","roc_auc"])

cv_results1["test_accuracy"].mean()
cv_results1["test_f1"].mean()
cv_results1["test_roc_auc"].mean()

predictonvalues_and_reelvalues = pd.Series(y_prediction) , pd.Series(y_test)
predictonvalues_and_reelvalues = pd.DataFrame(predictonvalues_and_reelvalues)

print(classification_report(y_test,y_prediction))


#### KNN FINDING THE BEST PARAMETERS ####


knn_model_2 = KNeighborsClassifier()

k_range = list(range(1,51))
weight_option = ["uniform","distance"]
knn_params = dict(n_neighbors=k_range,
                      weights = weight_option)


knn_gs_best = GridSearchCV(knn_model_2,
                           knn_params,
                           cv=10,
                           n_jobs=-1,
                           verbose=1,
                           scoring="accuracy").fit(x_train,y_train)

knn_gs_best.best_params_

knn_model_2 = KNeighborsClassifier(**knn_gs_best.best_params_).fit(x_train,y_train)

y_prediction_test = knn_model.predict(x_test)

print(classification_report(y_test,y_prediction_test))

cv_results2 = cross_validate(knn_model_2,x_train,y_train,cv=10,
                            scoring=["accuracy","f1","roc_auc"])

cv_results2["test_accuracy"].mean()
cv_results2["test_f1"].mean()
cv_results2["test_roc_auc"].mean()

random_user = x_train.sample(1)
knn_model_2.predict(random_user)


####  NEIGHBORHOOD COMPONENT ANALYSIS (NCA)  ####

nca = NeighborhoodComponentsAnalysis(n_components=2,random_state=35)
nca.fit(x_scaled,y)
x_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(x_reduced_nca, columns = ["p1","p2"])
nca_data["target"] = y

sns.scatterplot(x = "p1", y="p2", hue="target", data=nca_data)
plt.title("NCA: p1 vs p2")
plt.show()


x_train_nca, x_test_nca , y_train_nca , y_test_nca = train_test_split(x_reduced_nca,y,random_state=35)

knn_model_nca = KNeighborsClassifier().fit(x_train_nca,y_train_nca)

y_prediction_nca = knn_model_nca.predict(x_test_nca)
print(classification_report(y_test_nca,y_prediction_nca))

cv_results_nca = cross_validate(knn_model_nca,x_train_nca,y_train_nca,cv=10,scoring=["accuracy","f1","roc_auc"])

cv_results_nca["test_accuracy"].mean()
cv_results_nca["test_f1"].mean()
cv_results_nca["test_roc_auc"].mean()

#### BEST PARAMETER KNN NCA MODEL ####

knn_gridsearch_best_nca = GridSearchCV(knn_model_nca,
                           knn_params,
                           cv=10,
                           n_jobs=-1,
                           verbose=1,).fit(x_train_nca,y_train_nca  )

knn_gridsearch_best_nca.best_params_

knn_final_model = KNeighborsClassifier(**knn_gridsearch_best_nca.best_params_).fit(x_train_nca,y_train_nca)
y_prediction_nca_final_model = knn_final_model.predict(x_test_nca)

cv_results_nca_2 = cross_validate(knn_final_model,x_train_nca,y_train_nca,cv=10,scoring=["accuracy","f1","roc_auc"])
cv_results_nca_2["test_accuracy"].mean()
cv_results_nca_2["test_f1"].mean()
cv_results_nca_2["test_roc_auc"].mean()

knn_final_model.score(x_test_nca,y_test_nca)
knn_final_model.score(x_train_nca,y_train_nca)

print(classification_report(y_test_nca,y_prediction_nca_final_model))



df_x_train_nca = pd.DataFrame(x_train_nca)
random_user = df_x_train_nca.sample(1)
knn_final_model.predict(random_user)



