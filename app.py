# Importing required libraries
from operator import imod
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import sklearn
import datetime
from time import time
from time import time
import pickle
import shutil
import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import datetime
import random

# Creating a Flask instance and setting a static folder for file uploads
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

# Getting current date and year and loading a model parameter file
currentDateTime = datetime.datetime.now()
date = currentDateTime.date()
year = date.strftime("%Y")
thersold = pickle.load(open('static/model_para/default_thersold.pkl', 'rb'))


def categorical_featute_engg(df):
    # defining all the features
    one_hot_feature = ['D_63', 'D_64'] # Features to encode using one-hot encoding
    cat = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_66", "D_68"] # Categorical features
    offset = [2, 1, 2, 2, 3, 2, 3, 2, 2]  # Subtracting the minimal value in full train csv

    # Making a dataframe with only the categorical features
    df_cat = df[['customer_ID'] + cat + one_hot_feature]

    # Replacing empty string and -1 with X and Y respectively in one_hot_feature
    df_cat[one_hot_feature] = df_cat[one_hot_feature].replace(r'', 'X', regex=True)
    df_cat[one_hot_feature] = df_cat[one_hot_feature].replace(r'-1', 'Y', regex=True)

    # Grouping by customer_ID and counting the number of rows in each group
    df_count = df.groupby('customer_ID')['S_2'].agg(['count'])
    df_count.columns = ['count']

    # Deleting the original dataframe to save memory
    del df

    # Adding offset to the categorical columns and converting the values to integer
    for col, s in zip(cat, offset):
        df_cat[col] = np.array(df_cat[col].values) + s
        df_cat[col] = df_cat[col].fillna(1).astype('int8')

    # Resetting the index and dropping the original index column
    df_cat = df_cat.reset_index(drop=True)
    # df_cat = df_cat.drop(['index'],axis=1)

    # Loading the saved one-hot encoder and its corresponding feature names
    pipe_one_hot_encoder = pickle.load(open('static/model_para' + '/one_hot_encoder_D_63_D_64.pkl', 'rb'))
    one_hot_ = pickle.load(open('static/model_para' + '/one_hot_encoder_D_63_D_64.pkl_feature_names', 'rb'))

    # One-hot encoding the one_hot_feature columns and converting the values to integer
    one_hot_df = pd.DataFrame(pipe_one_hot_encoder.transform(df_cat[one_hot_feature]).toarray().astype('int8'),
                              columns=one_hot_)

    # Dropping the one_hot_feature columns and concatenating the one-hot encoded columns
    df_cat = df_cat.drop(['D_63', 'D_64'], axis=1)
    df_cat = pd.concat([df_cat, one_hot_df], axis=1)

    # Deleting the one_hot_df dataframe to save memory
    del one_hot_df

    # Grouping by customer_ID and getting the first, last, and nunique values of each column
    df_cat = df_cat.groupby('customer_ID')[list(df_cat.columns)[1:]].agg(['first', 'last', 'nunique'])
    df_cat.columns = ['_'.join(x) for x in df_cat.columns]

    # Adding customer_ID column back to the dataframe
    df_cat['customer_ID'] = df_cat.index

    # Reordering the columns and resetting the index
    df_cat = df_cat[['customer_ID'] + list(df_cat.columns)[:-1]]
    df_cat = df_cat.reset_index(drop=True)

    # Converting all columns except customer_ID to integer
    for col in df_cat.columns[1:]:
        df_cat[col] = df_cat[col].astype('int8')

    df_cat = df_cat.merge(df_count, on='customer_ID')
    del df_count

    return df_cat


def numerical_featute(df):
    # create a list of all columns except for 'customer_ID' and 'S_2'
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID', 'S_2']]

    # create separate lists of categorical and numerical features
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    num_features = [col for col in all_cols if col not in cat_features]

    # group the dataframe by customer_ID and calculate the mean, std, min, max, and last values for each numerical feature
    new_df = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])

    # flatten the multi-level column index and join the levels with underscores
    new_df.columns = ['_'.join(x) for x in new_df.columns]

    # convert all columns in the new dataframe to float16 data type
    for col in list(new_df.columns):
        new_df[col] = new_df[col].astype('float16')

    # create a list of the column names in the new dataframe
    column_name = list(new_df.columns)

    # return the new dataframe
    return new_df



def feature_standardization(num_df, cat_df):
    # loading the standardized and non-standardized categorical columns from saved files
    cat_colum_std = pickle.load(open('static/model_para' + '/cat_col_std.pkl', 'rb'))
    cat_colum_not_std = pickle.load(open('static/model_para' + '/cat_colum_not_std.pkl', 'rb'))

    # loading the saved numerical and categorical data preprocessing pipelines
    pipe_num = pickle.load(open('static/model_para' + '/numerical_data_preprocessing_pipeline.pkl', 'rb'))
    pipe_cat = pickle.load(open('static/model_para' + '/categorical_data_preprocessing_pipeline.pkl', 'rb'))

    # transforming the numerical data using the saved pipeline
    train_num = pipe_num.transform(num_df.iloc[:, 1:])

    # transforming the categorical data using the saved pipeline
    train_cat_std = pipe_cat.transform(cat_df[cat_colum_std])
    customer = cat_df['customer_ID']
    train_cat_non_std = np.array(cat_df[cat_colum_not_std[1:]])

    # concatenating the transformed numerical and categorical data
    final_data = np.concatenate([train_num, train_cat_std, train_cat_non_std], axis=1)

    # loading the saved XGBoost models and predicting the probability of default
    for i in range(5):
        # loading the model
        if i == 0:
            model = xgb.XGBClassifier()
            model.load_model('static/model_para/' + 'model/xgb' + f'/model_{i}.xgb')
            # predicting the probability of default
            predication_prob = model.predict_proba(final_data)
            predication = model.predict(final_data)
        else:
            model.load_model('static/model_para/' + 'model/xgb' + f'/model_{i}.xgb')
            predication_prob += model.predict_proba(final_data)
            predication += model.predict(final_data)
    
    # averaging the predicted probability of default
    predication_prob = predication_prob / 5

    # setting the threshold for predicting default
    thersold = 0.5
    predication = [1 if x >= (thersold * 100) else 0 for x in predication_prob[:, 1]]

    # generating random profile pictures for each customer
    random_list = [f'static/assests/profile_pictures/{random.randint(1, 10)}.png' for x in range(len(customer))]

    # creating a list of tuples with customer ID, predicted probability of default, and profile picture path for progress showing
    result = [x for x in zip(random_list[:15], customer[:15], predication_prob[:, 1][:15])]

    # creating a DataFrame with customer ID, probability of non-default, probability of default, and predicted default
    output_df = [x for x in zip(customer, predication_prob[:, 0], predication_prob[:, 1], predication)]
    output_df = pd.DataFrame(output_df, columns=['customer_ID', 'Probability 0', 'Probability 1', 'Prediction'])
    # saving the DataFrame as a CSV file
    output_df.to_csv('static/download_data/default_report.csv', index=False)
    return result


def dataframe_creator(file_path):
    # Read the CSV file at file_path, replace missing values with NaN
    df = pd.read_csv(file_path, na_values=[np.NaN])
    columns = list(df.columns)
    # Drop the 'target' column if it exists
    if 'target' in columns:
        df = df.drop(['target'], axis=1)

    # Apply categorical feature engineering to the data
    cat_df = categorical_featute_engg(df)

    # Apply numerical feature engineering to the data
    num_df = numerical_featute(df)
    num_df = num_df.reset_index()
    num_df = num_df.replace(np.inf, np.nan)

    # Apply feature standardization to the numerical and categorical data, then combine them
    result = feature_standardization(num_df, cat_df)

    return result


@app.route('/')
def home():
    # Render the home page
    return render_template('index.html', year=year)


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        currentDateTime = datetime.datetime.now()
        date = currentDateTime.date()
        year = date.strftime("%Y")

        # Save the uploaded file to the UPLOAD_FOLDER directory
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(app.config['UPLOAD_FOLDER'] + filename)

        # Call dataframe_creator function to process the uploaded file and get the predictions
        content = dataframe_creator(app.config['UPLOAD_FOLDER'] + filename)

        # Set variables to render the results on the home page
        show_result = True
        result_path = 'static/download_data/default_report.csv'

        return render_template('index.html',
                               content=content,
                               show_result=show_result,
                               result_path=result_path,
                               year=year,
                               thersold=thersold)

# running the app
if __name__ == '__main__':
    app.run(debug=True,port=5002)
