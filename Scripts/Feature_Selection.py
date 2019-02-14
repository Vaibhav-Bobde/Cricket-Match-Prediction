import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

matches_data = pd.read_csv("G:\CSUF Study\ADBMS\Project\DataSet\IPL\processed_data.csv")
outcome_var = ['winner']
l=['team1', 'team2', 'toss_winner','city','toss_decision','p1','p2','SAB','team1_bat_score',
   'team2_bat_score','team1_ball_score','team2_ball_score']
predictor_var = l
pd_var=[]
for pp in l:
    pd_var.append(pp)

X=matches_data[predictor_var]
y=matches_data[outcome_var]
clf = RandomForestClassifier()
clf = clf.fit(X, y)
print(X.shape)

print(clf.feature_importances_)
ll=zip(clf.feature_importances_,predictor_var)
for importance,feature in ll:
    print(feature + ' =')
    print('%s' % '{0:.3%}'.format(importance))
#     print feature,importance
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape[1]

#removing features---------------------
ii=1
print(len(predictor_var))
print(X_new.shape[1])
for  importance,feature in ll:
    if ii<len(predictor_var)-X_new.shape[1]+3:
        print(feature)
        predictor_var.remove(feature)
    else:
        break
    ii+=1
print(len(predictor_var))

#Principal Component Analysis---------------------------
scaler = StandardScaler()
X1=scaler.fit_transform(X)

#using pca
from sklearn import decomposition
pca = decomposition.PCA() #all features
train_x_fit=pca.fit_transform(X1)
train_x_cov=pca.get_covariance()
exp_var=pca.explained_variance_ratio_ #Percentage of variance explained by each of the selected components.
# print exp_var.shape
print(exp_var)
zipped = zip(exp_var,pd_var)

def getKey(item):
    return item[0]
# zipped.sort()
zipped=sorted(zipped, key=getKey)
print(zipped)
#plotting explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(exp_var.shape[0]), exp_var, alpha=0.5, align='center',
        label='individual explained variance')
plt.ylabel('Explained variance ratios')
plt.xlabel('Principal components ids')
plt.legend(loc='best')
plt.savefig('pca.png')
plt.tight_layout()
print("features to removed = ",zipped[:3])


predictor_var.remove('team2_ball_score')
predictor_var.remove('team1_ball_score')
predictor_var.remove('team2_bat_score')

#Finding best parameters for Random Forest using GridSearchCV------------------
# Number of trees in random forest
n_estimators = [int(i) for i in np.linspace(start = 200, stop = 2000, num = 10)]

# Maximum number of levels in tree
max_depth = [int(i) for i in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Number of features which we have to consider at every split
max_features = ['auto', 'sqrt']

# Method for selecting samples for training each tree
bootstrap = [True, False]

# Minimum number of samples which are required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples which are required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# First create the base model to tune

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rfc = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(matches_data[predictor_var],matches_data['winner'])
print(rf_random.best_params_)
print(rf_random.best_score_)

print("----------------SVM Best Params-----------------------")
C = 10. ** np.arange(-3, 8)
gamma = 10. ** np.arange(-5, 4)
print(C)
param_grid = {'C': [0.001,0.01,0.1,1,10],'gamma':gamma,'kernel':['linear','rbf']}
svm_clf = RandomizedSearchCV(svm.SVC(), param_distributions = param_grid,n_iter = 100, cv = 3, scoring='accuracy')
svm_clf.fit(matches_data[predictor_var], matches_data['winner'])
print(svm_clf.best_params_)
print(svm_clf.best_score_)

print("------------Logistic Regression Best Params-------------")

param_grid_LR={ 'C': [0.001,0.01,0.1,1,10,100,1000],
            'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }
logistic_clf = GridSearchCV(LogisticRegression(n_jobs=-1), param_grid_LR,cv=3, scoring='accuracy')
logistic_clf.fit(matches_data[predictor_var], matches_data['winner'])
print(logistic_clf.best_params_)
print(logistic_clf.best_score_)

params_knn = {
    "n_neighbors": np.arange(1, 31, 2),
	"metric": ["euclidean", "cityblock"],
    'leaf_size':[1,2,3,5],
    'weights':['uniform', 'distance'],
    'algorithm':['auto', 'ball_tree','kd_tree','brute']
    }
knn_clf = RandomizedSearchCV(KNeighborsClassifier(n_jobs=-1), param_distributions = params_knn,n_iter = 200, cv = 3, scoring='accuracy')
knn_clf.fit(matches_data[predictor_var], matches_data['winner'])
print(knn_clf.best_params_)
print(knn_clf.best_score_)
print("OK")