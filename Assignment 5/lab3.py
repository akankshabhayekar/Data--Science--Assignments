#!/usr/bin/env python
# coding: utf-8

# # Part A

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


train_file=pd.read_csv(r"C:\Users\akank\Downloads\SteelPlateFaults-train1")
test_file=pd.read_csv(r"C:\Users\akank\Downloads\SteelPlateFaults-test1")

del train_file[train_file.columns[0]]
del test_file[test_file.columns[0]]


# In[3]:


train_file_class0=train_file.groupby("Class").get_group(0)
train_file_class1=train_file.groupby("Class").get_group(1)


train_labels_0=train_file_class0["Class"]
train_labels_1=train_file_class1["Class"]

test_labels=test_file["Class"]


drop_features=["X_Minimum","Y_Minimum","TypeOfSteel_A300","TypeOfSteel_A400","Class"]

for drops in drop_features:
    train_file_class0=train_file_class0.drop(drops,axis=1)
    train_file_class1=train_file_class1.drop(drops,axis=1)
    test_file=test_file.drop(drops,axis=1)
    


# # GMM Models initialization

# In[4]:


gmm2=GaussianMixture(n_components=2,covariance_type="full")
gmm4=GaussianMixture(n_components=4,covariance_type="full")
gmm8=GaussianMixture(n_components=8,covariance_type="full")
gmm16=GaussianMixture(n_components=16,covariance_type="full")

gmms=[gmm2,gmm4,gmm8,gmm16]
accuracy_store0=[]
accuracy_store1=[]


# ## GMM Model for q = [2 ,4, 8, 16] with training for class 0

# In[5]:


for gmm in gmms:
    gmm.fit(train_file_class0,train_labels_0)
    log_prob=gmm.score_samples(train_file_class0)
    prediction=gmm.predict(test_file)
    accuracy=metrics.accuracy_score(test_labels,prediction)
    cfm=metrics.confusion_matrix(test_labels, prediction)
    cfm_disp=metrics.ConfusionMatrixDisplay(confusion_matrix = cfm)
    cfm_disp.plot()
    accuracy_store0.append(round(accuracy,2))
    cfm_disp.ax_.set_title("Accuracy: "+ str(round(accuracy,2)))
    plt.show()
    print(log_prob)


# ## GMM Model for q = [2 ,4, 8, 16] with training for class 1

# In[6]:


for gmm in gmms:
    gmm.fit(train_file_class1,train_labels_1)
    log_prob=gmm.score_samples(train_file_class1)
    prediction=gmm.predict(test_file)
    accuracy=metrics.accuracy_score(test_labels,prediction)
    cfm=metrics.confusion_matrix(test_labels, prediction)
    cfm_disp=metrics.ConfusionMatrixDisplay(confusion_matrix = cfm)
    cfm_disp.plot()
    accuracy_store1.append(round(accuracy,2))
    cfm_disp.ax_.set_title("Accuracy: "+ str(round(accuracy,2)))
    plt.show()
    print(log_prob)


# In[7]:


rel_order=["KNN without Normalization","KNN with Normalization","Bayes Classifier","Inbuilt Naive Bayes","Bayes-GMM (trained on Class 0)","Bayes-GMM (trained on Class 1)"]
rel_accuracy=[0.90,0.97,0.93,0.91,round(max(accuracy_store0),2),round(max(accuracy_store1),2)]
table={}
for i in range(len(rel_order)):
    table[rel_order[i]]=rel_accuracy[i]
data=pd.DataFrame(table.values(),table.keys())
data.columns=["Most Accurate Values"]
pd.DataFrame(data)


# # Part B

# In[284]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


# In[26]:


file=pd.read_csv(r"C:\Users\akank\Documents\ic160\abalone.csv")


# In[27]:


x=file[file.columns[:-1]]
pd.DataFrame(x)


# In[28]:


y=file[file.columns[-1]]
pd.DataFrame(y)


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True,random_state=42)


# In[30]:


x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
training_data= pd.concat([x_train,y_train],axis=1)
training_data.to_csv("abalone-train.csv")
pd.DataFrame(training_data)


# In[31]:


x_test.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
testing_data= pd.concat([x_test,y_test],axis=1)
testing_data.to_csv("abalone-test.csv")
pd.DataFrame(testing_data)


# Finding attribute with the highest pearson correlation coefficient 

# In[41]:


pcc_dict={}
max_pcc=-1000
max_pcc_col="null"
for col in x.columns:
    pcc,_=pearsonr(x[col],y)
    pcc_dict[col]=pcc
    if(max_pcc<pcc):
        max_pcc=pcc
        max_pcc_col=col

print("Attribute with the highest pearson correlation coefficient:",max_pcc_col)
print("Value of Pearson Correlation Coefficient for that attribute:",max_pcc)

df=pd.DataFrame(pcc_dict.values(),pcc_dict.keys())
df.columns=["PCC"]
pd.DataFrame(df)


# Linear Regression of attribute with highest PCC with Rings

# In[153]:


reg=LinearRegression()


# In[154]:


pcc_train=np.array(x_train[max_pcc_col])
pcc_train=pcc_train.reshape(1,-1)
pcc_train=pcc_train.transpose()

pcc_test=np.array(x_test[max_pcc_col])
pcc_test=pcc_test.reshape(1,-1)
pcc_test=pcc_test.transpose()

reg.fit(pcc_train,y_train)


# In[155]:


prediction_training=reg.predict(pcc_train)
prediction_testing=reg.predict(pcc_test)


# In[279]:


plt.scatter(pcc_train,y_train,color="red")
# plt.scatter(pcc_test,y_test,color="green")
m,c=np.polyfit(pcc_train[:,0],y_train, 1)
# m2,c2=np.polyfit(pcc_test[:,0],y_test, 1)
plt.plot(pcc_train[:,0], m*pcc_train[:,0] + c,color="black") 
# plt.plot(pcc_test[:,0], m*pcc_test[:,0] + c,color="blue")
plt.xlabel(max_pcc_col)
plt.ylabel("Rings")
plt.title("Best Fit Line")
plt.grid()
plt.show()


# In[157]:


rmse_training=metrics.mean_squared_error(y_train,prediction_training,squared=False)
rmse_testing=metrics.mean_squared_error(y_test,prediction_testing,squared=False)

print("RMSE in training:",rmse_training)
print("RMSE in testing:", rmse_testing)


# In[158]:


plt.scatter(y_test,prediction_testing)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Predicted V/S Actual")
plt.show()


# Multivariate Linear Regression

# In[162]:


reg.fit(x_train,y_train)


# In[169]:


prediction_training=reg.predict(x_train)
prediction_testing=reg.predict(x_test)


# In[174]:


rmse_training=metrics.mean_squared_error(y_train,prediction_training,squared=False)
rmse_testing=metrics.mean_squared_error(y_test,prediction_testing,squared=False)

print("RMSE in training:",rmse_training)
print("RMSE in testing:", rmse_testing)


# In[175]:


plt.scatter(y_test,prediction_testing)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Predicted V/S Actual")
plt.show()


# Single Variate Non-Linear Regression

# In[224]:


reg=LinearRegression()
poly_features2=PolynomialFeatures(2)
poly_features3=PolynomialFeatures(3)
poly_features4=PolynomialFeatures(4)
poly_features5=PolynomialFeatures(5)

poly_features=[poly_features2,poly_features3,poly_features4,poly_features5]
p=["Degree 2","Degree 3","Degree 4","Degree 5"]
poly_dict={}


# In[226]:


for features in range(len(poly_features)):
    x_poly_training=poly_features[features].fit_transform(pcc_train)
    x_poly_testing=poly_features[features].fit_transform(pcc_test)

    reg.fit(x_poly_training,y_train)
    
    prediction_training=reg.predict(x_poly_training)
    prediction_testing=reg.predict(x_poly_testing)
    
    rmse_training=metrics.mean_squared_error(y_train,prediction_training,squared=False)
    rmse_testing=metrics.mean_squared_error(y_test,prediction_testing,squared=False)
    
    poly_dict[p[features]]=[rmse_training,rmse_testing]


# In[231]:


df=pd.DataFrame(poly_dict.values(),poly_dict.keys())
df.columns=["RMSE-Train","RMSE-Test"]
pd.DataFrame(df)


# It can be seen Degree 4 has the least error on Test data set

# In[236]:


rmse_list=list(poly_dict.values())
rmse_train_list=[]
rmse_test_list=[]
for lists in rmse_list:
    rmse_train_list.append(lists[0])
    rmse_test_list.append(lists[1])


# In[287]:


plt.bar(p,rmse_train_list)
plt.plot(p,rmse_train_list,"-o",color="black")
plt.xlabel("Degrees")
plt.ylabel("RMSE Errors")
plt.title("Train Data")
plt.show()


# In[261]:


plt.bar(p,rmse_test_list)
plt.plot(p,rmse_train_list,"o-",color="black")
plt.plot(p,rmse_test_list,"*-",color="red")
plt.title("Test Data")
plt.show()


# In[330]:


def best_fit_curve(x,a,b,c,d,e):
    curve=a*x**4+b*x**3+c*x**2+d*x+e
    return curve


# In[337]:


x_poly_training=poly_features4.fit_transform(pcc_train)
x_poly_testing=poly_features4.fit_transform(pcc_test)
reg.fit(x_poly_training,y_train)
prediction_training=reg.predict(x_poly_training)
prediction_testing=reg.predict(x_poly_testing)
plt.scatter(pcc_train,y_train)

parameters,_=curve_fit(best_fit_curve,pcc_train[:,0],prediction_training)
a,b,c,d,e=parameters

component_x=np.linspace(min(pcc_train),max(pcc_train),200)
component_y=best_fit_curve(component_x,a,b,c,d,e)

plt.plot(component_x,component_y,color="black")

plt.xlabel(max_pcc_col)
plt.ylabel("Rings")
plt.title("Best Fit Curve")
plt.grid()


# In[338]:


plt.scatter(y_test,prediction_testing)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Predicted V/S Actual")
plt.show()


# Multi-variate non-linear regression

# In[340]:


for features in range(len(poly_features)):
    x_poly_training=poly_features[features].fit_transform(x_train)
    x_poly_testing=poly_features[features].fit_transform(x_test)

    reg.fit(x_poly_training,y_train)
    
    prediction_training=reg.predict(x_poly_training)
    prediction_testing=reg.predict(x_poly_testing)
    
    rmse_training=metrics.mean_squared_error(y_train,prediction_training,squared=False)
    rmse_testing=metrics.mean_squared_error(y_test,prediction_testing,squared=False)
    
    poly_dict[p[features]]=[rmse_training,rmse_testing]


# In[341]:


df=pd.DataFrame(poly_dict.values(),poly_dict.keys())
df.columns=["RMSE-Train","RMSE-Test"]
pd.DataFrame(df)


# Degree 2 is the best for multivariate polynomial regression

# In[345]:


rmse_list=list(poly_dict.values())
rmse_train_list=[]
rmse_test_list=[]
for lists in rmse_list:
    rmse_train_list.append(lists[0])
    rmse_test_list.append(lists[1])


# In[346]:


plt.bar(p,rmse_train_list)
plt.plot(p,rmse_train_list,"-o",color="black")
plt.xlabel("Degrees")
plt.ylabel("RMSE Errors")
plt.title("Train Data")
plt.show()


# In[347]:


plt.bar(p,rmse_test_list)
plt.plot(p,rmse_train_list,"o-",color="black")
plt.plot(p,rmse_test_list,"*-",color="red")
plt.title("Test Data")
plt.show()


# In[351]:


x_poly_training=poly_features2.fit_transform(x_train)
x_poly_testing=poly_features2.fit_transform(x_test)
reg.fit(x_poly_training,y_train)
prediction_testing=reg.predict(x_poly_testing)


# In[353]:


plt.scatter(y_test,prediction_testing)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Predicted V/S Actual")
plt.show()

