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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'
# drive_path = 'static/model_para'

currentDateTime = datetime.datetime.now()
date = currentDateTime.date()
year = date.strftime("%Y")
thersold = pickle.load(open('static/model_para/default_thersold.pkl', 'rb'))


def categorical_featute_engg(df):
    # defining all the faetures

    one_hot_feature = ['D_63', 'D_64']
    cat = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_66", "D_68"]
    offset = [2, 1, 2, 2, 3, 2, 3, 2, 2]  # 2 minus minimal value in full train csv

    # Making categoriacl data drame with only categorical features
    df_cat = df[['customer_ID'] + cat + one_hot_feature]

    df_cat[one_hot_feature] = df_cat[one_hot_feature].replace(r'', 'X', regex=True)
    df_cat[one_hot_feature] = df_cat[one_hot_feature].replace(r'-1', 'Y', regex=True)

    df_count = df.groupby('customer_ID')['S_2'].agg(['count'])
    df_count.columns = ['count']

    del df

    for col, s in zip(cat, offset):
        df_cat[col] = np.array(df_cat[col].values) + s
        df_cat[col] = df_cat[col].fillna(1).astype('int8')

    df_cat = df_cat.reset_index(drop=True)
    # df_cat = df_cat.drop(['index'],axis=1)

    pipe_one_hot_encoder = pickle.load(open('static/model_para' + '/one_hot_encoder_D_63_D_64.pkl', 'rb'))
    one_hot_ = pickle.load(open('static/model_para' + '/one_hot_encoder_D_63_D_64.pkl_feature_names', 'rb'))

    one_hot_df = pd.DataFrame(pipe_one_hot_encoder.transform(df_cat[one_hot_feature]).toarray().astype('int8'),
                              columns=one_hot_)

    df_cat = df_cat.drop(['D_63', 'D_64'], axis=1)
    df_cat = pd.concat([df_cat, one_hot_df], axis=1)

    del one_hot_df

    df_cat = df_cat.groupby('customer_ID')[list(df_cat.columns)[1:]].agg(['first', 'last', 'nunique'])
    df_cat.columns = ['_'.join(x) for x in df_cat.columns]

    df_cat['customer_ID'] = df_cat.index

    df_cat = df_cat[['customer_ID'] + list(df_cat.columns)[:-1]]
    df_cat = df_cat.reset_index(drop=True)

    for col in df_cat.columns[1:]:
        df_cat[col] = df_cat[col].astype('int8')

    df_cat = df_cat.merge(df_count, on='customer_ID')
    del df_count

    return df_cat


def numerical_featute(df):
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID', 'S_2']]
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    num_features = [col for col in all_cols if col not in cat_features]

    new_df = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    new_df.columns = ['_'.join(x) for x in new_df.columns]

    for col in list(new_df.columns):
        new_df[col] = new_df[col].astype('float16')

    column_name = list(new_df.columns)
    return new_df


def feature_standardization(num_df, cat_df):
    cat_colum_std = pickle.load(open('static/model_para' + '/cat_col_std.pkl', 'rb'))
    cat_colum_not_std = pickle.load(open('static/model_para' + '/cat_colum_not_std.pkl', 'rb'))

    pipe_num = pickle.load(open('static/model_para' + '/numerical_data_preprocessing_pipeline.pkl', 'rb'))
    pipe_cat = pickle.load(open('static/model_para' + '/categorical_data_preprocessing_pipeline.pkl', 'rb'))

    train_num = pipe_num.transform(num_df.iloc[:, 1:])

    train_cat_std = pipe_cat.transform(cat_df[cat_colum_std])
    customer = cat_df['customer_ID']
    train_cat_non_std = np.array(cat_df[cat_colum_not_std[1:]])

    final_data = np.concatenate([train_num, train_cat_std, train_cat_non_std], axis=1)

    for i in range(5):

        if i == 0:
            model = xgb.XGBClassifier()
            model.load_model('static/model_para/' + 'model/xgb' + f'/model_{i}.xgb')

            predication_prob = model.predict_proba(final_data)
            predication = model.predict(final_data)

        else:
            model.load_model('static/model_para/' + 'model/xgb' + f'/model_{i}.xgb')
            predication_prob += model.predict_proba(final_data)
            predication += model.predict(final_data)

    predication_prob = predication_prob / 5

    # predication = [1 if x >= 3 else 0 for x in predication]
    # model = xgb.XGBClassifier()
    # model.load_model('static/model_para/model_sklearn.txt')
    predication_prob = model.predict_proba(final_data)

    # probability of defualting or result == 1

    predication_prob_0 = [round(x[0] * 100, 4) for x in predication_prob]
    predication_prob_1 = [round(x[1] * 100, 4) for x in predication_prob]
    predication = [1 if x >= (thersold * 100) else 0 for x in predication_prob_1]
    # profile picture genrating

    random_list = [f'static/assests/profile_pictures/{random.randint(1, 10)}.png' for x in range(len(customer))]

    #   output list for progress showing
    result = [x for x in zip(random_list[:15], customer[:15], predication_prob_1[:15])]

    # making df for csv file
    output_df = [x for x in zip(customer, predication_prob_0, predication_prob_1, predication)]
    output_df = pd.DataFrame(output_df, columns=['customer_ID', 'Probability 0', 'Probability 1', 'Prediction'])
    output_df.to_csv('static/download_data/default_report.csv', index=False)
    return result


def dataframe_creator(file_path):
    df = pd.read_csv(file_path, na_values=[np.NaN])
    columns = list(df.columns)
    if 'target' in columns:
        df = df.drop(['target'], axis=1)

    # # categorical Feature Engg
    cat_df = categorical_featute_engg(df)

    # # Numerical Feature Engg
    num_df = numerical_featute(df)
    num_df = num_df.reset_index()
    num_df = num_df.replace(np.inf, np.nan)

    # return num_df
    result = feature_standardization(num_df, cat_df)

    return result


@app.route('/')
def home():
    return render_template('index.html', year=year)


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        currentDateTime = datetime.datetime.now()
        date = currentDateTime.date()
        year = date.strftime("%Y")

        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(app.config['UPLOAD_FOLDER'] + filename)

        # file = open(app.config['UPLOAD_FOLDER']+filename,'rb')
        content = dataframe_creator(app.config['UPLOAD_FOLDER'] + filename)
        show_result = True
        result_path = 'static/download_data/default_report.csv'

        return render_template('index.html',
                               content=content,
                               show_result=show_result,
                               result_path=result_path,
                               year=year,
                               thersold=thersold)


if __name__ == '__main__':
    app.run(debug=True)
