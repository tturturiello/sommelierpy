#!/usr/bin/env python
# coding: utf-8

# # Wine Quality

# ## Dataset Credits:
# 
# http://www3.dsi.uminho.pt/pcortez/wine/
# 
# *P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.*

# In[1]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sb


# In[3]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sb
from plotly.subplots import make_subplots

#from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import classification_report, confusion_matrix


# In[4]:


dfr = pd.read_csv('./data/winequality/winequality-red.csv', sep=';')
dfw = pd.read_csv('./data/winequality/winequality-white.csv', sep=';')


# In[5]:


dfw['type'] = 'white'
dfr['type'] = 'red'
df = pd.concat([dfw, dfr])
display(df.head())
display(df.tail())


# In[6]:


df.shape


# ### **Features :**
# - **wine type** - 1096 Red and 3451 White wine
# 
# - **fixed acidity** - Most acids involved with wine or fixed or nonvolatile
# 
# - **volatile acidity** - The amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
# 
# - **citric acid** - the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
# 
# - **residual sugar** - The amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet
# 
# - **chlorides** - The amount of salt in the wine
# 
# - **free sulfur dioxide** - The free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine
# 
# - **total sulfur dioxide** - Amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
# 
# - **density** - the density of wine is close to that of water depending on the percent alcohol and sugar content
# 
# - **pH** - Describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale
# 
# - **sulphates** - a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant
# 
# - **alcohol** - the percent alcohol content of the wine
# 
# ### **Outcome Variable:**
# - **quality** - score between 0 and 10
# 

# ## Sanity Check

# In[7]:


df.info()


# ## Exploring Data

# We combined two different but very similar data sets. One that covered red wines and one that covered white wines. Very similar because the columns were exactly the same.

# In[8]:


fig, ax = plt.subplots(figsize=(15, 8))
ax = sb.heatmap(df.corr(), annot=True)


# In[9]:


fig = make_subplots(rows=1, 
                    cols=2,
                    shared_yaxes=True,
                    horizontal_spacing=0.01,
                   )
fig.update_layout( height=1200, 
                  showlegend=False,
                  title='Quality distribution per type'
                 )
fig.add_trace(go.Box(x=dfw['type'], y=dfw['quality']), row=1, col=1)
fig.add_trace(go.Box(x=dfr['type'], y=dfr['quality']), row=1, col=2)


# There are no particular differences between the two rating distributions among different wine qualities. Although we combined two different data sets.

# In[10]:


px.histogram(data_frame=df, x='quality', height=600, width=1200, title='<b>Quality distribution</b>')


# # Linear Regression

# ## Purpose

# The intent of this work is to be able to predict the quality of a wine based on its its organoleptic characteristics.

# ## Making a guess

# Assuming that there can be a direct correlation between a single characteristic and wine quality, can we already identify which characteristic might help us predict the value of quality?
# 
# Correlations between quality and a given combination of values of multiple characteristics cannot be identified by these indices.
# But this will have to be understood by the model we build.
# 
# If we look at the heat map below, we can identify which features have a higher correlation index. But they may not necessarily be the ones relevant to the prediction.

# In[11]:


fig, ax = plt.subplots(figsize=(15, 8))
ax = sb.heatmap(df.corr(), annot=True)


# # Linear Regression

# ## Ridge

# In[12]:


dummies = pd.get_dummies(df['type'])
y = df['quality']
X_ = df.drop(['type', 'quality'], axis=1)
X = pd.concat([X_, dummies[['red']]], axis = 1)
X


# In[13]:


alphas = 10**np.linspace(2,-2,100)*0.5
alphas


# In[14]:


ridge = Ridge(normalize=True)

coefs = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)


# We varied the regularization parameter to find the best value of alpha

# In[15]:


ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')


# In[16]:


X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[17]:


ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True) # con Ridge fa..., con RidgeCV oltre a fare quello che fa con Ridge(), sa gia' che deve fare ridge regressor
ridgecv.fit(X_train, y_train)
ridgecv.alpha_


# The best alpha for the lasso is 1.12. What stands out is the enormous preponderance of one feature over the others.

# In[18]:


ridge2 = Ridge(alpha=ridgecv.alpha_, normalize=True)
ridge2.fit(X_train, y_train)
pred2 = ridge2.predict(X_test)
print(pd.Series(ridge2.coef_, index = X.columns)) # stampiamo i coefficienti
print('MSE: ', mean_squared_error(y_test, pred2)) # Calcoliamo MSE


# Our conjecture was wrong. Instead, the feature that seems to prevail is density, with overwhelming evidence.
# The MSE is about 0.53, a relatively low error. Let us now see what happens with Lasso.

# ## Lasso

# In[19]:


alphas = 10**np.linspace(0,-10,100)*0.5

lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')


# In[20]:


lasso = Lasso(max_iter = 10000, normalize = True)
lassocv = LassoCV(alphas = alphas, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
mean_squared_error(y_test, lasso.predict(X_test))
lassocv.alpha_


# In[21]:


print(pd.Series(lasso.coef_, index=X.columns))
print('MSE: ', mean_squared_error(y_test, pred2)) # Calcoliamo MSE


# Density is the most relevant datum for our Lasso model. The MSE is 0.53, which is actually a good value for this datum.

# # Classification

# Classification between white wine and red wine could be predicted.

# In[22]:


df.index.values


# In[23]:


from sklearn.model_selection import train_test_split
X = df.drop(['type'], axis=1)
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y)


# ## Decision Tree

# In[24]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score # NB: ci sono misure specifiche per classificatori e regressori

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predicted_train = model.predict(X_train) # per vedere se le risposte sono simili a quelle desiderate

predicted_test = model.predict(X_test)

print('Train accuracy')
print(accuracy_score(y_train, predicted_train)) # misuro accuratezza train set

print('Test score')
print(accuracy_score(y_test, predicted_test)) # misuro accuratezza test set


# In[25]:


conf_mat = confusion_matrix(y_train, predicted_train)
print(conf_mat)
print(classification_report(y_test, predicted_test))


# The data collected from the classifier report indicate that:
# - the percentage of correct predictions is 0.97 for red wine and 0.99 for white wine (accuracy)
# - the percentage of positive cases detected is 0.97 for red wine and 0.99 for white wine (recall)
# - the percentage of correct positive predictions is 0.97 for red wine and 0.99 for white wine (f1-score)

# In[26]:


# Data handling
import pandas as pd

# Exploratory Data Analysis & Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Model improvement and Evaluation 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix


# Plotting confusion matrix
matrix = pd.DataFrame(conf_mat, 
                      ('Fake', 'Real'), 
                      ('Fake', 'Real'))
print(matrix)

# Visualising confusion matrix
plt.figure(figsize = (16,14),facecolor='white')
heatmap = sns.heatmap(matrix, annot = True, annot_kws = {'size': 20}, fmt = 'd', cmap = 'YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = 18, weight='bold')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = 18, weight='bold')

plt.title('Decision Tree\n', fontsize = 18, color = 'darkblue')
plt.ylabel('True type', fontsize = 14)
plt.xlabel('Predicted type', fontsize = 14)
plt.show()


# The prediction made with the decision tree is particularly good. In fact, we notice that on the diagonal we find most of the results. The classification made with the type of wine is easy for the model to understand, perhaps because wines take on very different characteristics between red and white wine.

# ## K-Neighbors

# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[28]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

print(conf_mat)
print(classification_report(y_test, y_pred))


# The data collected from the classifier report indicate that:
# - the percentage of correct predictions is 0.99 for both red and white wine (accuracy)
# - the percentage of positive cases detected is 0.97 for red wine and 1.00 for white wine (recall)
# - the percentage of correct positive predictions is 0.98 for red wine and 0.99 for white wine (f1-score).
# 
# In general, we can say that the classifier seems to work well, to the point of being very close in terms of accuracy to the decision tree.

# In[29]:


# Data handling
import pandas as pd

# Exploratory Data Analysis & Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Model improvement and Evaluation 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix


# Plotting confusion matrix
matrix = pd.DataFrame(conf_mat, 
                      ('Fake', 'Real'), 
                      ('Fake', 'Real'))
print(matrix)

# Visualising confusion matrix
plt.figure(figsize = (16,14),facecolor='white')
heatmap = sns.heatmap(matrix, annot = True, annot_kws = {'size': 20}, fmt = 'd', cmap = 'YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = 18, weight='bold')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = 18, weight='bold')

plt.title('K-Neighbors\n', fontsize = 18, color = 'darkblue')
plt.ylabel('True type', fontsize = 14)
plt.xlabel('Predicted type', fontsize = 14)
plt.show()


# For K-Neighbors the classification seems to work well, as we find a concentration on diagonal TP-TN values compared to the other FP-FNs. Apparently it seems to perform less well than the previous classifier (Decision Tree), although in reality, comparing the ratios described above (precision, recall and f1-score), we find that they are more or less equivalent.

# ## Conclusion

# We considered two very similar data sets, one of white wines and one of red wines, combined them and created a "type" column that we then used for classification. 
# 
# Looking at the correlation indices, we noticed a high correlation between wine quality and alcohol quantity, only to find that it was instead density that was particularly useful in determining the quality of a wine.
# 
# Through linear regression, our model is able to predict wine quality with an accuracy of ~0.532 with the Ridge and Lasso regressors.
# Finally, we compared two classifiers (K-Neighbors and Decision Tree) to try to accurately determine the type of wine (white or red).
