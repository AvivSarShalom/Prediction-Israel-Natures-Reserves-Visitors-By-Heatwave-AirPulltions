import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.metrics import r2_score

def find_correlated(tmp):
    correlated_features = set()
    correlation_matrix = tmp.corr()
    targets = ['Total' , 'Israelis_Count', 'Tourists_Count' ]

    for i in range(len(correlation_matrix .columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.4:
                colname = correlation_matrix.columns[i]
                if colname not in targets:
                    correlated_features.add(colname)
    return correlated_features
def run_poly(models,target,path):
    model_idx = 0
    f = open(path, 'a')
    for model in models:    
        for i in range(1,5):
            train, test = poli_regression(model,i,target)
            #print(train.columns)
            correct_rows = train.loc[train.predicted_entries_train==train.target].any(axis=1).count()
            accuracy = round(correct_rows/len(train),3)
            train_mse = metrics.mean_squared_error(train.target, train.predicted_entries_train)
            train_rmse = np.sqrt(metrics.mean_squared_error(train.target, train.predicted_entries_train))
            train_mae = metrics.mean_absolute_error(train.target, train.predicted_entries_train)
            test_mse = metrics.mean_squared_error(test.target, test.predicted_entries_test)
            test_rmse = np.sqrt(metrics.mean_squared_error(test.target, test.predicted_entries_test))
            test_mae = metrics.mean_absolute_error(test.target, test.predicted_entries_test)


            f.write("Model "+str(model_idx)+" Degrees "+str(i)+" Accuracy: "+str(accuracy)+"\n")
            f.write("------ TRAIN DATA ------\n")
            f.write("MSE : "+str(train_mse)+", RMSE: "+str(train_rmse)+", MAE : "+str(train_mae)+"\n")
            f.write("R2 TRAIN "+ str(r2_score(train.target, train.predicted_entries_train))+"\n")
            f.write("------ TEST DATA ------\n")
            f.write("MSE : "+str(test_mse)+", RMSE: "+str(test_rmse)+", MAE : "+str(test_mae)+"\n")
            f.write("R2 TEST "+ str(r2_score(test.target, test.predicted_entries_test))+"\n")
            f.write("--------------------------------\n")
        model_idx += 1    
    f.close()


def poli_regression(model,degrees,target):
    corr_df = model.corr()
    correlated = corr_df.target.loc[(abs(corr_df.target)>=0.15)]
    
    correlated = correlated.drop([target]).index.tolist()
    X = model[correlated]
    y = model.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7254)

    train_df = pd.merge(left=X_train, right=y_train, left_index=True, right_index=True)
    test_df = pd.merge(left=X_test, right=y_test, left_index=True, right_index=True)

    x_train_scaler = MinMaxScaler()
    x_test_scaler = MinMaxScaler()
    y_train_scaler = MinMaxScaler()
    y_test_scaler = MinMaxScaler()
    X_train_scaled = x_train_scaler.fit_transform(X_train)
    X_test_scaled = x_test_scaler.fit_transform(X_test)
    y_train_scaled = y_train_scaler.fit_transform(pd.DataFrame(y_train))
    y_test_scaled = y_test_scaler.fit_transform(pd.DataFrame(y_test))


    poly = PolynomialFeatures(degree=degrees)

    #fit the x variable to fit a 2rd degree polynomial value
    X_poly = poly.fit_transform(X_train_scaled)
    poly.fit(X_poly, y_train_scaled)

    pol_lin_reg = LinearRegression()
    pol_lin_reg.fit(X_poly, y_train_scaled)

    #predict the training data

    y_train_pred_scaled = pol_lin_reg.predict(poly.fit_transform(X_train_scaled))
    y_train_pred = y_train_scaler.inverse_transform(y_train_pred_scaled)

    #create a pandas series of the results
    y_train_pred = round(pd.Series(y_train_pred[:,0], index=y_train.index, name='predicted_entries_train'),ndigits=2)

    #Add the results to the DF
    train_df = pd.merge(left=train_df, right=y_train_pred , left_index=True, right_index=True)

    #train_df.head()


    y_test_pred_scaled = pol_lin_reg.predict(poly.fit_transform(X_test_scaled))
    y_test_pred = y_test_scaler.inverse_transform(y_test_pred_scaled)

    #create a pandas series of the results
    y_test_pred = round(pd.Series(y_test_pred[:,0], index=y_test.index, name='predicted_entries_test'),ndigits=2)

    #Add the results to the DF
    test_df = pd.merge(left=test_df, right=y_test_pred , left_index=True, right_index=True)
    return train_df,test_df

def add_to_res(descripton,model_type,model_number,Site_in_this_model,ACC_Training,MAE_Training,MSE_Training,RMSE_Training,R2_Training,ACC_Test,MAE_Test,MSE_Test,RMSE_Test,R2_Test):
    res=pd.read_excel("../res.xlsx")
    res.head()
    new_row={'descripton':descripton,'Model_type':model_type,'Model_number':model_number,
        'Site_in_this_model':Site_in_this_model,'ACC_Training':ACC_Training,'MAE_Training':MAE_Training,'MSE_Training':MSE_Training,
        'RMSE_Training':RMSE_Training,'R2_Training':R2_Training,'ACC_Test':ACC_Test,'MAE_Test':MAE_Test,'MSE_Test':MSE_Test,'RMSE_Test':RMSE_Test,'R2_Test':R2_Test}
    res=res.append(new_row,ignore_index=True)   
    res.to_excel("../res.xlsx",index=False)
