import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm

path_ProccessedData="G:\CSUF Study\ADBMS\Project\DataSet\IPL\processed_data.csv"
path_2018_IplData="G:\CSUF Study\ADBMS\Project\DataSet\IPL\ipl-2018.csv"
matches_data = pd.read_csv(path_ProccessedData)
matches_data = matches_data.query('season>2013')
predictor_var=['team1', 'team2', 'toss_winner','city','toss_decision','p1','p2','SAB','team1_bat_score']
outcome_var = ['winner']

def classify_data(model, data, X, Y,num_folds):
    # model.fit(data[X],data[Y].values.ravel())
    # prediction=model.predict(data[X])
    kfold=KFold(n_splits=num_folds)
    error=[]
    success_rate_train=[]
    for train_index, test_index in kfold.split(data[X]):
        train_x = (data[X].iloc[train_index,:])
        train_y = (data[Y].iloc[train_index])
        model.fit(train_x,train_y.values.ravel())
        score = model.score(data[X].iloc[test_index,:], data[Y].iloc[test_index])
        print(score)
        error.append(score)
        success_rate_train.append(model.score(train_x, train_y))
    print('Cross validation Score : %s' % '{0:.3%}'.format(np.mean(error)))
    return model

print("-----------Logistic Regression---------")
log_model = LogisticRegression(n_jobs=-1,C= 0.01, solver= 'newton-cg',multi_class='auto')
log_model = classify_data(log_model,matches_data,predictor_var,outcome_var,5)

print("-----------Random Forest-----------")
rf_model = RandomForestClassifier(n_jobs=-1,bootstrap=True, min_samples_leaf=4, n_estimators=200, min_samples_split=5, max_features='auto', max_depth=80 ) # -- best param
#rf_model = RandomForestClassifier(n_jobs=-1,bootstrap=True, min_samples_leaf=5, n_estimators=2000, min_samples_split=2, max_features='auto', max_depth=100 )
rf_model = classify_data(rf_model,matches_data,predictor_var,outcome_var,5)

print("----------Support Vector Machines-----------")
svm_model=svm.SVC(kernel='rbf',C=10,gamma=0.1) #-- best param
#svm_model=svm.SVC(kernel='linear',C=10,gamma=0.00001)
classify_data(svm_model,matches_data,predictor_var,outcome_var,5)


def predict():
    ipl2018_data = pd.read_csv(path_2018_IplData)
    team_numbers = {'MI': 1, 'KKR': 2, 'RCB': 3, 'DC': 4, 'CSK': 5, 'RR': 6, 'DD': 7, 'GL': 8, 'KXIP': 9, 'SRH': 10,
                    'RPS': 11, 'KTK': 12, 'PW': 13}
    team_numbers_rev = {1: 'MI', 2: 'KKR', 3: 'RCB', 4: 'DC', 5: 'CSK', 6: 'RR', 7: 'DD', 8: 'GL', 9: 'KXIP', 10: 'SRH',
                        11: 'RPS', 12: 'KTK', 13: 'PW'}
    city_ids = {
        'Hyderabad': 1,
        'Pune': 2,
        'Rajkot': 3,
        'Indore': 4,
        'Bangalore': 5,
        'Bengaluru': 5,
        'Mumbai': 6,
        'Kolkata': 7,
        'Delhi': 8,
        'Chandigarh': 9,
        'Kanpur': 10,
        'Jaipur': 11,
        'Chennai': 12,
        'Cape Town': 13,
        'Port Elizabeth': 14,
        'Durban': 15,
        'Centurion': 16,
        'East London': 17,
        'Johannesburg': 18,
        'Kimberley': 19,
        'Bloemfontein': 20,
        'Ahmedabad': 21,
        'Cuttack': 22,
        'Nagpur': 23,
        'Dharamsala': 24,
        'Kochi': 25,
        'Visakhapatnam': 26,
        'Raipur': 27,
        'Ranchi': 28,
        'Abu Dhabi': 29,
        'Sharjah': 30,
        'Dubai': 31,
        'Mohali': 32
    }
    toss_decision_ids = {'field': 0, 'bat': 1}
    encode = {'team1': team_numbers,
              'team2': team_numbers,
              'city': city_ids,
              'toss_decision': toss_decision_ids,
              'toss_winner': team_numbers,
              'winner': team_numbers
              }
    ipl2018_data.replace(encode, inplace=True)
    predictor_var2 = ['team1', 'team2', 'city', 'toss_winner', 'toss_decision']

    print("Logistic Regression")
    log_model = LogisticRegression(n_jobs=-1, C=0.01, solver='newton-cg', multi_class='auto')
    log_model.fit(matches_data[predictor_var2], matches_data[outcome_var].values.ravel())
    pred_log = log_model.predict(ipl2018_data[predictor_var2])
    unique, counts = np.unique(pred_log, return_counts=True)
    print(dict(zip(unique, counts)))

    print("Random Forest")
    rf_model2 = RandomForestClassifier(n_jobs=-1,bootstrap=True, min_samples_leaf=4, n_estimators=200, min_samples_split=5, max_features='auto', max_depth=80 ) # -- best param
    rf_model2.fit(matches_data[predictor_var2], matches_data[outcome_var].values.ravel())
    pred_rf2 = rf_model2.predict(ipl2018_data[predictor_var2])
    unique, counts = np.unique(pred_rf2, return_counts=True)
    print(dict(zip(unique, counts)))

    print("SVM")
    svm_model=svm.SVC(kernel='rbf',C=10,gamma=0.1) #-- best paramsvm_model=svm.SVC(kernel='rbf',C=10,gamma=0.1) #-- best param
    svm_model.fit(matches_data[predictor_var2], matches_data[outcome_var].values.ravel())
    pred_svm = svm_model.predict(ipl2018_data[predictor_var2])
    unique, counts = np.unique(pred_svm, return_counts=True)
    print(dict(zip(unique, counts)))

    ipl2018_data['prediction'] = pred_log

    encode_rev = {
             'prediction': team_numbers_rev
    }

    ipl2018_data_predicted = pd.read_csv(path_2018_IplData)
    ipl2018_data_predicted['prediction'] = pred_rf2
    ipl2018_data_predicted.replace(encode_rev, inplace=True)
    ipl2018_data_predicted.to_csv("Predicted_data.csv")
    print('Done')

predict()