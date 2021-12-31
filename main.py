from flask import Flask, jsonify, request
from flask import flash
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import logging
logging.basicConfig(level=logging.DEBUG)


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # categorical features
    cat_cols = ['channelGrouping', 'device.browser', 'device.operatingSystem',
                'device.deviceCategory',  'geoNetwork.continent', 'geoNetwork.subContinent',
                'geoNetwork.country',  'geoNetwork.region',  'geoNetwork.metro',
                'geoNetwork.city', 'geoNetwork.networkDomain', 'trafficSource.campaign',
                'trafficSource.source',  'trafficSource.medium', 'trafficSource.keyword',
                'trafficSource.referralPath',
                'browser_category', 'browser_os','source_country',
                'channelGrouping_browser','channelGrouping_OS']

    #numeric features
    num_cols = ['visitId','visitNumber','visitStartTime',
                'totals.hits','totals.pageviews','totals.sessionQualityDim',
                'totals.timeOnSite','totals.transactions','totals.transactionRevenue',
                'totals.totalTransactionRevenue']


    #load saved models
    lbgclf = joblib.load('pkllbgclf')
    lbgreg = joblib.load('pkllbgreg')
    for col in cat_cols:
       model =  'le' + col
       pkl_fl = 'pkl' + col
       model = joblib.load(pkl_fl)

    #https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/
    file = request.files['file'] 
    test_data = pd.read_csv(file,converters={"fullVisitorId": str})
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
    if file.filename == '':
            flash('No selected file')
    
 
 
    #https://www.kaggle.com/robikscube/tutorial-time-series-forecasting-with-xgboost
    def transform_date_int(indata):
        """
        Create temporal features from date fields
        """
        indata['date'] = pd.to_datetime(indata['date'])
        indata['dayofweek'] = indata['date'].dt.dayofweek 
        indata['quarter'] = indata['date'].dt.quarter 
        indata['month'] = indata['date'].dt.month 
        indata['year'] = indata['date'].dt.year 
        indata['dayofyear'] = indata['date'].dt.dayofyear 
        indata['dayofmonth'] = indata['date'].dt.day 
        indata['weekofyear'] = indata['date'].dt.isocalendar().week.astype(float)
        #features for visitStartTime
        indata['vis_date'] = pd.to_datetime(indata['visitStartTime'], unit='s')
        indata['sess_date_hours'] = indata['vis_date'].dt.hour 
        #https://www.kaggle.com/ashishpatel26/permutation-importance-feature-imp-measure-gacrp/notebook
        indata['hits_per_day']   = indata.groupby('dayofyear')['totals.hits'].transform('nunique') 
        indata['hits_per_month'] = indata.groupby('month')['totals.hits'].transform('nunique') 
        indata['hits_per_dom'] = indata.groupby('dayofmonth')['totals.hits'].transform('nunique') 
        indata['hits_per_dow'] = indata.groupby('dayofweek')['totals.hits'].transform('nunique') 
        indata['pageviews_per_day'] = indata.groupby('dayofyear')['totals.pageviews'].transform('nunique') 
        indata['pageviews_per_month'] = indata.groupby('month')['totals.pageviews'].transform('nunique') 
        indata['pageviews_per_dom'] = indata.groupby('dayofmonth')['totals.pageviews'].transform('nunique') 
        indata['pageviews_per_dow'] = indata.groupby('dayofweek')['totals.pageviews'].transform('nunique') 
        indata['month_unique_user_count'] = indata.groupby('month')['fullVisitorId'].transform('nunique')
        indata['day_unique_user_count'] = indata.groupby('dayofyear')['fullVisitorId'].transform('nunique')
        indata['weekday_unique_user_count'] = indata.groupby('dayofweek')['fullVisitorId'].transform('nunique')
        indata['monthday_unique_user_count'] = indata.groupby('dayofmonth')['fullVisitorId'].transform('nunique') 
        indata['browser_category'] = indata['device.browser'] + '_' + indata['device.deviceCategory']
        indata['browser_os'] = indata['device.browser'] + '_' + indata['device.operatingSystem']
        indata['source_country'] = indata['trafficSource.source'] + '_' + indata['geoNetwork.country']
        indata['channelGrouping_browser'] = indata['device.browser'] + "_" + indata['channelGrouping']
        indata['channelGrouping_OS'] = indata['device.operatingSystem'] + "_" + indata['channelGrouping']
        indata['tran_per_day']   = indata.groupby('dayofyear')['totals.transactions'].transform('nunique') 
        indata['tran_per_month'] = indata.groupby('month')['totals.transactions'].transform('nunique') 
        indata['tran_per_dom'] = indata.groupby('dayofmonth')['totals.transactions'].transform('nunique') 
        indata['tran_per_dow'] = indata.groupby('dayofweek')['totals.transactions'].transform('nunique') 
        indata['timeOnSite_per_day']   = indata.groupby('dayofyear')['totals.timeOnSite'].transform('nunique') 
        indata['timeOnSite_per_month'] = indata.groupby('month')['totals.timeOnSite'].transform('nunique') 
        indata['timeOnSite_per_dom'] = indata.groupby('dayofmonth')['totals.timeOnSite'].transform('nunique') 
        indata['timeOnSite_per_dow'] = indata.groupby('dayofweek')['totals.timeOnSite'].transform('nunique') 
        # 
        #convert int to float to avoid truncations
        for col in indata.columns:
           if indata[col].dtype == 'int64':   
              indata[col] = indata[col].astype(float)
        return indata
    
    def data_transform(indata):
       '''
       transform user level features
       '''
       #https://stackoverflow.com/questions/19078325/naming-returned-columns-in-pandas-aggregate-function
       def diff(x):
          #get difference between max & min date  
          time_d =  x.max() -x.min() 
          return float(time_d.days)
       #https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values  
       def vcount(x):
          #get count of rows for visitStartTime
          return len(indata.loc[indata['visitStartTime'].isin(x)])
       #
       date_min= min(indata['date'])
       date_max= max(indata['date'])   
       # difference between preiod start and first visit
       def min_diff(x):
         time_d= x.min() - date_min
         return float(time_d.days)
       # difference between preiod end and last visit
       def max_diff(x):
         time_d=  date_max - x.max()
         return float(time_d.days)
       # 
       dfg=indata.groupby('fullVisitorId').agg({ 
       'channelGrouping'             : ['max'],              
       'device.browser'              : 'max',                   
       'device.operatingSystem'      : 'max',                     
       'device.isMobile'             : 'max',      
       'device.deviceCategory'       : 'max',                      
       'geoNetwork.continent'        : 'max',                      
       'geoNetwork.subContinent'     : 'max',                        
       'geoNetwork.country'          : 'max',                       
       'geoNetwork.region'           : 'max',                        
       'geoNetwork.metro'            : 'max',                          
       'geoNetwork.city'             : 'max',                         
       'geoNetwork.networkDomain'    : 'max',                          
       'trafficSource.campaign'      : 'max',                         
       'trafficSource.source'        : 'max',                          
       'trafficSource.medium'        : 'max',                         
       'trafficSource.keyword'       : 'max',                          
       'trafficSource.referralPath'  : 'max',                         
       'date'                      : [('_diff', diff),('_min',min_diff),('_max',max_diff)], 
       'visitStartTime'            : [('_count', vcount)],                    
       'totals.hits'               : [ 'max', 'min', 'sum','mean'],    
       'totals.pageviews'          : ['max', 'min', 'sum','mean'],
       'totals.timeOnSite'         : ['max', 'min', 'sum','mean'], 
       'totals.sessionQualityDim'  : 'max',
       'totals.transactions'       : 'sum',                
       'totals.transactionRevenue' : 'sum',               
       'totals.totalTransactionRevenue' :  'sum',
       'dayofweek'                 :  [ 'max', 'min'],                
       'quarter'                   :  [ 'max', 'min'],                                
       'month'                     :  [ 'max', 'min'],                                     
       'year'                      :  [ 'max', 'min'],                                 
       'dayofyear'                 :  [ 'max', 'min'],                                 
       'dayofmonth'                :  [ 'max', 'min'],                                   
       'weekofyear'                :  [ 'max', 'min'],      
       'sess_date_hours'           :  [ 'max', 'min', 'sum','mean'],
       'hits_per_day'              :  [ 'max', 'min', 'sum','mean'],   
       'hits_per_month'            :  [ 'max', 'min', 'sum','mean'],    
       'hits_per_dom'              :  [ 'max', 'min', 'sum','mean'],    
       'hits_per_dow'              :  [ 'max', 'min', 'sum','mean'],    
       'pageviews_per_day'         :  [ 'max', 'min', 'sum','mean'], 
       'pageviews_per_month'       :  [ 'max', 'min', 'sum','mean'], 
       'pageviews_per_dom'         :  [ 'max', 'min', 'sum','mean'], 
       'pageviews_per_dow'         :  [ 'max', 'min', 'sum','mean'], 
       'month_unique_user_count'   :  [ 'max', 'min', 'sum','mean'],    
       'day_unique_user_count'     :  [ 'max', 'min', 'sum','mean'],    
       'weekday_unique_user_count' :  [ 'max', 'min', 'sum','mean'],    
       'monthday_unique_user_count' :  [ 'max', 'min', 'sum','mean'],    
       'browser_category'          : 'max', 
       'browser_os'                : 'max',
       'source_country'            : 'max',   
       'channelGrouping_browser'   : 'max',   
       'channelGrouping_OS'        : 'max',   
       'tran_per_day'              :  [ 'max', 'min', 'sum','mean'],   
       'tran_per_month'            :  [ 'max', 'min', 'sum','mean'],  
       'tran_per_dom'              :  [ 'max', 'min', 'sum','mean'],  
       'tran_per_dow'              :  [ 'max', 'min', 'sum','mean'],  
       'timeOnSite_per_day'        :  [ 'max', 'min', 'sum','mean'],   
       'timeOnSite_per_month'      :  [ 'max', 'min', 'sum','mean'],  
       'timeOnSite_per_dom'        :  [ 'max', 'min', 'sum','mean'],  
       'timeOnSite_per_dow'        :  [ 'max', 'min', 'sum','mean']  
     
        
       })
       #rename column by appending the aggregate name
       dfg.columns = ["_".join(x) for x in dfg.columns.ravel()]
       return dfg

    def transform_cat(indata):
        '''
        encode categorical features
        '''
        for col in cat_cols: 
            labelencoder = LabelEncoder()
            indata[col] = labelencoder.fit_transform(indata[col].values.astype('str' ))

    def process_data(test_data):
        '''
        process input query point and compute predicted value
        '''
        #-----------------------------------
        #pre-processing data
        #handle boolean data
        test_data['device.isMobile'] = test_data['device.isMobile'].astype(bool)
        #handle numeric data for nulls
        for col in num_cols:
            test_data[col].fillna(0, inplace=True)           # replace nulls with zeros
            test_data[col] = test_data[col].astype('float')
        #feature transforms
        #
        test_data = transform_date_int(test_data)
        transform_cat(test_data)
        trans_data =  data_transform(test_data)
        trans_data['fullVisitorId'] = 0
        trans_data= trans_data.drop('totals.totalTransactionRevenue_sum',axis=1)
        #
        target_cols= [ 'fullVisitorId']
        #Predicting probility on test data
        classifier_pred = lbgclf.predict_proba(trans_data.drop(target_cols,axis=1))
        regressor_pred = lbgreg.predict(trans_data.drop(target_cols,axis=1))
        final_pred = (classifier_pred[:,1]*regressor_pred)
        return final_pred 

    predictions = process_data(test_data)
    print(predictions)
    
    return jsonify({'prediction': predictions.tolist()})
    

if __name__ == '__main__':
    app.config['SECRET_KEY']  = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_PERMANENT']= False
    app.run(host='0.0.0.0', port=8080,debug = True)
