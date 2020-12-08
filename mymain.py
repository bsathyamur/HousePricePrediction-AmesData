# To handle datasets
import pandas as pd
import numpy as np
import xgboost

# For Feature scaling
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import Lasso,LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error 
from sklearn.ensemble import GradientBoostingRegressor

# Get dataset from the input CSV files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
testY = pd.read_csv("test_y.csv")
test["Sale_Price"] = testY["Sale_Price"]

class dataPreProcess:
    
    def __init__(self):
        pass

    def getCategoryFeatures(self,data):
        vars_categories = [var for var in data.columns if data[var].dtypes == 'O']
        return vars_categories
 
    def getNumericFeatures(self,data):
        vars_numeric = [var for var in data.columns if data[var].dtypes != 'O']
        return vars_numeric
    
    def printNAFeaturesMean(self,data):
        vars_with_na = [var for var in data.columns if data[var].isnull().sum() > 0]
        print(data[vars_with_na].isnull().mean())
        
    def getNACategoryFeatures(self,data):
        vars_with_na = [var for var in data.columns
                                if data[var].isnull().sum() > 0 and data[var].dtypes == 'O'
                        ]
        return vars_with_na
    
    def getNANumericFeatures(self,data):
        vars_with_na = [var for var in data.columns
                                if data[var].isnull().sum() > 0 and data[var].dtypes != 'O'
                        ]
        return vars_with_na        
    
    def replaceNACategory(self,data,vars_with_na):
        data[vars_with_na] = data[vars_with_na].fillna('Missing')
        return data
        
    def replaceNANumeric(self,data,vars_with_na):
        for var in vars_with_na:
            mode_val = data[var].mode()[0]
            # compute Mode and replace with Mode
            data[var] = data[var].fillna(mode_val)
        
        return data
    
    def handleTemporalVariable(self,data,vars_temporal,target_var):
        for var in vars_temporal:
            data[var] = data[target_var] - data[var]
            
        return data
    
    def transformLog(self,data,vars_numeric):
        for var in vars_numeric:
            data[var] = np.log(data[var])
            
        return data
    
    def getRareLabel(self,data,var,target_var):
        tmp = data.groupby(var)[target_var].count() / len(data)
        frequent_ls = tmp[tmp > 0.01].index
        return frequent_ls
    
    def handleRareLabel(self,frequent_ls,var_categories,data):
        data[var] = np.where(data[var_categories].isin(frequent_ls), data[var_categories], 'Rare')     
        return data
    
    def encodeCategories(self,data,var,target_var):
        ordered_labels = data.groupby([var])[target_var].mean().sort_values().index
        ordered_label = {k: i for i, k in enumerate(ordered_labels, 0)}   
        return ordered_label
    
    def replaceCategories(self,data,var,ordered_label):
        data[var] = data[var].map(ordered_label)
        return data
    
    def dropColumns(self,data,del_col_vars):
        data = data.drop(columns = del_col_vars)
        return data
        
    def writetoCSV(self,data,filename):
        data.to_csv(filename, index=False)
        return True    

class featureSelection:
    def __init__(self,train,Xtrain,Ytrain,target_var):
        self.train = train
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.target_var = target_var
    
    def selectFeaturesLasso(self):
        selFeatures = SelectFromModel(Lasso(alpha=0.005, random_state=0))
        selFeatures.fit(self.Xtrain,self.Ytrain)
        selected_feats = self.Xtrain.columns[(selFeatures.get_support())]
        features = selected_feats.to_list() 
        return features
    
    # Pearson correlation coefficient
    def selectFeatsPearsonCorr(self):  
        corr = self.train.corr()[self.target_var].sort_values(ascending=False)[1:]
        abs_corr = abs(corr)
        relevant_features = abs_corr[abs_corr>0.4]
        return relevant_features.index

#Create a class object for dataPreProcess
objdataPreprocess = dataPreProcess()

# Get category NA features from train data
na_category_vars = objdataPreprocess.getNACategoryFeatures(train)

# Process train data
train = objdataPreprocess.replaceNACategory(train,na_category_vars)

#Process test data
test = objdataPreprocess.replaceNACategory(test,na_category_vars)

# Get Numeric NA features from train data
na_numeric_vars = objdataPreprocess.getNANumericFeatures(train)

# Process train data
train = objdataPreprocess.replaceNANumeric(train,na_numeric_vars)

# Process test data
test = objdataPreprocess.replaceNANumeric(test,na_numeric_vars)


# Process temporal variables
vars_temporal = ['Year_Built','Year_Remod_Add','Garage_Yr_Blt']
target_var = 'Year_Sold'

#Process train data
train = objdataPreprocess.handleTemporalVariable(train,vars_temporal,target_var)

#Process test data
test = objdataPreprocess.handleTemporalVariable(test,vars_temporal,target_var)


# Transform Log
vars_numeric = ['Sale_Price']

#Process train data
train = objdataPreprocess.transformLog(train,vars_numeric)

#Process test data
test = objdataPreprocess.transformLog(test,vars_numeric)


# Handle Rare Labels for train data
target_var = 'Sale_Price'

var_categories = objdataPreprocess.getCategoryFeatures(train)

for var in var_categories:
    frequent_ls = objdataPreprocess.getRareLabel(train,var,target_var)
    train = objdataPreprocess.handleRareLabel(frequent_ls,var,train)
    test = objdataPreprocess.handleRareLabel(frequent_ls,var,test)  

#Drop some unncessary columns
#dropColsList = ['Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 
#'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude']

#Drop from train data
#train = objdataPreprocess.dropColumns(train,dropColsList)

#Drop from test data
#test = objdataPreprocess.dropColumns(test,dropColsList)


#Retrieve the categorical variables
var_categories = objdataPreprocess.getCategoryFeatures(train)

# Retrieve the numeric variables to be used for feature scaling
var_numeric = objdataPreprocess.getNumericFeatures(train)
var_numeric.remove("PID")
var_numeric.remove("Sale_Price")

for var in var_categories:
    dfDummies = pd.get_dummies(train[var], prefix = var)
    train = pd.concat([train,dfDummies],axis = 1)

for var in var_categories:
    dfDummies = pd.get_dummies(test[var], prefix = var)
    test = pd.concat([test,dfDummies],axis = 1)    

#Keep the dummy one hot encoded categorical columns and remove the original categorical columns    
train = objdataPreprocess.dropColumns(train,var_categories)
test = objdataPreprocess.dropColumns(test,var_categories)

scaler_vars = var_numeric

# create scaler
scaler = StandardScaler()

#  fit  the scaler to the train set
scaler.fit(train[scaler_vars]) 

# transform the train 
train[scaler_vars] = scaler.transform(train[scaler_vars])

#transform the test data
test[scaler_vars] = scaler.transform(test[scaler_vars])

#Check for no NA in the train
#objdataPreprocess.printNAFeaturesMean(train)

#Check for no NA in the test
#objdataPreprocess.printNAFeaturesMean(test)


#objdataPreprocess.writetoCSV(train,"train_preprocess.csv")
#objdataPreprocess.writetoCSV(test,"test_preprocess.csv")


Xtrain = train.drop(columns=['Sale_Price'])
Ytrain = pd.DataFrame(train["Sale_Price"])
Xtest = test.drop(columns=['Sale_Price'])
Ytest = pd.DataFrame(test[["PID","Sale_Price"]])


#objFeatureSelection = featureSelection(train,Xtrain,Ytrain,"Sale_Price")
#selFeatures = objFeatureSelection.selectFeaturesLasso()

# Selected Features via Lasso
selFeatures = ['Lot_Frontage', 'Lot_Area', 
               'Year_Built', 'Year_Remod_Add', 'Mas_Vnr_Area', 
               'BsmtFin_SF_1', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF', 
               'Low_Qual_Fin_SF', 'Gr_Liv_Area', 'Bsmt_Full_Bath', 
               'Full_Bath', 'Kitchen_AbvGr', 'TotRms_AbvGrd', 
               'Fireplaces', 'Garage_Cars', 'Garage_Area', 'Wood_Deck_SF', 
               'Enclosed_Porch', 'Screen_Porch', 'Longitude', 'Latitude', 
               'MS_Zoning_Residential_Low_Density', 'Lot_Shape_Regular', 
               'Condition_1_Norm', 'Overall_Qual_Below_Average', 
               'Overall_Cond_Good', 'Foundation_PConc', 'Bsmt_Qual_Excellent', 
               'Bsmt_Exposure_Gd', 'Bsmt_Exposure_No', 'BsmtFin_Type_1_GLQ', 
               'Heating_QC_Excellent', 'Heating_QC_Typical', 'Central_Air_N', 
               'Kitchen_Qual_Excellent', 'Kitchen_Qual_Typical', 'Functional_Typ', 
               'Fireplace_Qu_Good', 'Garage_Type_Attchd', 'Garage_Cond_Typical', 
               'Sale_Condition_Abnorml']

Xtrain = pd.DataFrame(Xtrain[selFeatures]) 
Xtest = pd.DataFrame(Xtest[selFeatures]) 

X_train = Xtrain
X_test = Xtest
Y_train = Ytrain
Y_test = Ytest[["Sale_Price"]]

lin_model = Lasso(alpha=0.0005,random_state = 0)
lin_model.fit(X_train,Y_train)
pred = lin_model.predict(X_test)

#rmse = np.sqrt(mean_squared_error(Y_test, pred))
#print("Test RMSE for LASSO " + str(rmse))

tmp = pd.DataFrame(pred)
mysub1 = pd.DataFrame(Ytest["PID"])
mysub1["Sale_Price"] = np.exp(tmp)
mysub1.to_csv('mysubmission1.txt',index=False)


best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.05,
                 max_depth=5,#3
                 min_child_weight=1.5,
                 n_estimators=500,#5                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.8,
                 seed=42)
best_xgb_model.fit(X_train,Y_train)
pred_xgboost = best_xgb_model.predict(X_test)

#rmse = np.sqrt(mean_squared_error(Y_test, pred_xgboost))
#print("Test RMSE for XGBoost " + str(rmse))

tmp = pd.DataFrame(pred_xgboost)
mysub1 = pd.DataFrame(Ytest["PID"])
mysub1["Sale_Price"] = np.exp(tmp)
mysub1.to_csv('mysubmission2.txt',index=False)