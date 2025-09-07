import pandas as pd
import numpy as np
from pandas import read_csv
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

tt_fridge = 'Get the csv file path'
tt_door = 'Get the csv file path'
tt_gps = 'Get the csv file path'
tt_modbus = 'Get the csv file path'
tt_motion = 'Get the csv file path'
tt_thermostat = 'Get the csv file path'
tt_weather = 'Get the csv file path'

tt_network = 'Get the csv file path'


le = LabelEncoder()
MMsc = MinMaxScaler()

# combine all csv files
all_filenames_iot = [tt_fridge, tt_door, tt_gps,  tt_modbus, tt_motion, tt_thermostat, tt_weather]
df_iot = pd.concat([pd.read_csv(f) for f in all_filenames_iot])
df_iot


iot_first_le_cols = ['motion_status', 'thermostat_status']

for col in iot_first_le_cols:
    df_iot[col] = le.fit_transform(df_iot[col])
    
df_iot.to_csv('../tt_orig_iot_le1.csv', index=False)


net_first_le_cols = ['proto', 'service', 'conn_state', 'dns_qclass', 'dns_rcode']

for col in net_first_le_cols:
    df_net[col] = le.fit_transform(df_net[col])

df_net.to_csv('../tt_orig_net_le1.csv', index=False)

df_net['proto'].unique()  # ([1, 2, 0])

tt_orig_iot_le1 = '../tt_orig_iot_le1.csv'
tt_orig_net_le1 = '../tt_orig_net_le1.csv'

all_filenames_iot_net = [tt_orig_iot_le1, tt_orig_net_le1]
df_iot_net = pd.concat([pd.read_csv(f) for f in all_filenames_iot_net])
df_iot_net

# type & label cols, move to end
type_label_to_move = ['type', 'label']
all_cols = df_iot_net.columns.tolist()
new_order = [col for col in all_cols if col not in type_label_to_move] + type_label_to_move
df_iot_net = df_iot_net.reindex(columns=new_order)
type_label_to_move = df_iot_net.pop('type')
df_iot_net['type'] = type_label_to_move
df_iot_net
df_iot_net.to_csv('../tt_orig_iot_net_concat1.csv', index=False)

# 24 columns remove
tt_orig_iot_net_concat1 = '../tt_orig_iot_net_concat1.csv'
tt_orig_iot_net_concat1_data = read_csv(tt_orig_iot_net_concat1)
tt_orig_iot_net_concat1_data.dtypes
tt_orig_iot_net_concat1_data.shape  #(472162, 63)
# ['ts'] not found in axis
columns_to_remove_25 = [
    'date', 'time', 'src_port', 'dst_port', 'src_ip', 'dst_ip', 
    'http_uri', 'weird_name', 'weird_addl', 'weird_notice', 
    'dns_query', 'ssl_version', 'ssl_cipher', 
    'ssl_subject', 'ssl_issuer', 'http_user_agent', 
    'http_method', 'http_version', 'http_request_body_len', 
    'http_response_body_len', 'http_status_code', 'http_user_agent', 
    'http_orig_mime_types', 'http_resp_mime_types', 'http_trans_depth'
    ]
# len(columns_to_remove_25)  # 25
tt_orig_iot_net_concat1_data = tt_orig_iot_net_concat1_data.drop(columns_to_remove_25, axis=1)
tt_orig_iot_net_concat1_data.dtypes
tt_orig_iot_net_concat1_data.shape  #(472162, 39)
tt_orig_iot_net_concat1_data


cat_pipeline = make_pipeline(
    tt_orig_iot_net_concat1_data[cat_cols] = tt_orig_iot_net_concat1_data[cat_cols].astype(str), 
    tt_orig_iot_net_concat1_data[cat_cols] = tt_orig_iot_net_concat1_data[cat_cols].str.strip(),  # remove namespaces
    tt_orig_iot_net_concat1_data[cat_cols] = tt_orig_iot_net_concat1_data[cat_cols].apply(lambda x: 1 if x == 'open' else 0),  # open/close to 1/0
    tt_orig_iot_net_concat1_data[cat_cols] = tt_orig_iot_net_concat1_data[cat_cols].apply(lambda x: 1 if x == 'true' else 0),  # true/false to 1/0
    tt_orig_iot_net_concat1_data[cat_cols].replace('-', 'F', inplace=True), #inplace=True
    tt_orig_iot_net_concat1_data[cat_cols] = tt_orig_iot_net_concat1_data[cat_cols].apply(lambda x: 1 if x == 'T' else 0)
    SimpleImputer(missing_values =np.nan, strategy ='constant', fill_value =0),
    LabelEncoder(),
    
    )                    
cat_data_tf = cat_pipeline.fit_transform(tt_orig_iot_net_concat1_data[cat_cols]).toarray()

strip_cols = ['temp_condition' , 'door_state' , 'sphone_signal' , 'light_status']  # cat cols --> ([0, 1], dtype=int64), high/low, on/off, open/close, true/false

true_false_cols = ['dns_AA', 'dns_RA', 'dns_RD', 'dns_rejected', 'ssl_resumed', 'ssl_established']  # cat cols --> ([0, 1], dtype=int64), '-'/T/F

imputer_cat_cols = [
    'motion_status', 'thermostat_status', 'proto', 'service', 'conn_state', 
    'dns_qclass', 'dns_rcode' 
    ]

imputer_cont_cols = [
    'fridge_temperature', 'latitude', 'longitude', 'FC1_Read_Input_Register', 
    'FC2_Read_Discrete_Value', 'FC3_Read_Holding_Register', 'FC4_Read_Coil', 
    'current_temperature', 'temperature', 'pressure', 'humidity', 'duration', 
    'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 'missed_bytes', 'src_ip_bytes', 
    'dst_ip_bytes','dns_qtype'
    ]

########## strip_cols to 1/0 & convert to num (int64)
tt_orig_iot_net_concat1_data[strip_cols] = tt_orig_iot_net_concat1_data[strip_cols].fillna('missing')
tt_orig_iot_net_concat1_data[strip_cols] = tt_orig_iot_net_concat1_data[strip_cols].apply(lambda x: x.str.strip())

tt_orig_iot_net_concat1_data['temp_condition'].unique()  #['high', 'low', 'missing']
tt_orig_iot_net_concat1_data['thermostat_status'].unique()  #[nan,  1.,  0.]
tt_orig_iot_net_concat1_data['sphone_signal'].unique()  #['missing', '0', '1', 'true', 'false']
tt_orig_iot_net_concat1_data['light_status'].unique()  #['missing', 'off', 'on']
tt_orig_iot_net_concat1_data['door_state'].unique()  #['missing', 'closed', 'open']


tt_orig_iot_net_concat1_data['temp_condition'] = tt_orig_iot_net_concat1_data['temp_condition'].apply(lambda x: 1 if x == 'high' else 0)  #([1, 0], dtype=int64)
tt_orig_iot_net_concat1_data['light_status'] = tt_orig_iot_net_concat1_data['light_status'].apply(lambda x: 1 if x == 'on' else 0)  #([0, 1], dtype=int64)
tt_orig_iot_net_concat1_data['door_state'] = tt_orig_iot_net_concat1_data['door_state'].apply(lambda x: 1 if x == 'open' else 0)  #([0, 1], dtype=int64)
tt_orig_iot_net_concat1_data['sphone_signal'] = tt_orig_iot_net_concat1_data['sphone_signal'].apply(lambda x: 1 if x == 'true' else 0)  #([0, 1], dtype=int64)

########## true_false_cols to 1/0 & convert to num (int64)
for c in true_false_cols:
    tt_orig_iot_net_concat1_data[c].replace('-', 'F', inplace=True) #inplace=True
    tt_orig_iot_net_concat1_data[c] = tt_orig_iot_net_concat1_data[c].apply(lambda x: 1 if x == 'T' else 0)  # T/F to 1/0

########## imputer_cat_cols
tt_orig_iot_net_concat1_data['motion_status'].unique()  # ([ 2.,  0.,  1., nan]) --> ([2., 0., 1.])
imputer_cat_zero = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=0)
imputer_cat_zero.fit(tt_orig_iot_net_concat1_data[imputer_cat_cols])
tt_orig_iot_net_concat1_data[imputer_cat_cols]= imputer_cat_zero.fit_transform(tt_orig_iot_net_concat1_data[imputer_cat_cols])
tt_orig_iot_net_concat1_data
tt_orig_iot_net_concat1_data.dtypes


tt_kaggle_iot_net_le_concat1_data
tt_kaggle_iot_net_le_concat1_data.dtypes
tt_orig_iot_net_concat1_data['proto'].unique()  #([nan, 'tcp', 'udp', 'icmp'], dtype=object) --> ([0, 'tcp', 'udp', 'icmp'], dtype=object)
tt_orig_iot_net_concat1_data['motion_status'].unique()  #([nan,  0.,  1.]) --> ([0, 1.0], dtype=object)
tt_orig_iot_net_concat1_data['thermostat_status'].unique()  # --> ([0, 1.0], dtype=object)
tt_orig_iot_net_concat1_data['service'].unique()  # --> ([0, '-', 'smb;gssapi', 'dce_rpc', 'smb', 'dns', 'ssl', 'http', 'ftp', 'gssapi'], dtype=object)
tt_orig_iot_net_concat1_data['conn_state'].unique()  # --> ([0, 'OTH', 'REJ', 'S1', 'RSTR', 'SF', 'RSTO', 'SH', 'S3', 'S0', 'SHR', 'S2', 'RSTOS0', 'RSTRH'], dtype=object)
tt_orig_iot_net_concat1_data['dns_qclass'].unique()  # --> ([0, 1.0, 32769.0], dtype=object)
tt_orig_iot_net_concat1_data['dns_rcode'].unique()  # --> ([0, 3.0, 2.0, 5.0], dtype=object)

########## imputer_cont_cols
tt_orig_iot_net_concat1_data['fridge_temperature']  # --> 
imputer_cont_median = SimpleImputer(missing_values=np.nan, strategy='median')
imputer_cont_median.fit(tt_orig_iot_net_concat1_data[imputer_cont_cols])
tt_orig_iot_net_concat1_data[imputer_cont_cols]= imputer_cont_median.fit_transform(tt_orig_iot_net_concat1_data[imputer_cont_cols])
tt_orig_iot_net_concat1_data
tt_orig_iot_net_concat1_data.dtypes

tt_orig_iot_net_concat1_data.to_csv('../tt_orig_iot_net_concat1_le_nan1.csv', index=False)

#####
for column in combined_data.select_dtypes(include=['object']).columns:
    combined_data[column] = label_encoder.fit_transform(combined_data[column])
    
for column in combined_data.select_dtypes(include=['object']).columns:
    combined_data[column] = label_encoder.fit_transform(combined_data[column])
    
num_cols
len(num_cols)  #25
# ['fridge_temperature',
#  'latitude',
#  'longitude',
#  'FC1_Read_Input_Register',
#  'FC2_Read_Discrete_Value',
#  'FC3_Read_Holding_Register',
#  'FC4_Read_Coil',
#  'motion_status',
#  'current_temperature',
#  'thermostat_status',
#  'temperature',
#  'pressure',
#  'humidity',
#  'duration',
#  'src_bytes',
#  'dst_bytes',
#  'missed_bytes',
#  'src_pkts',
#  'src_ip_bytes',
#  'dst_pkts',
#  'dst_ip_bytes',
#  'dns_qclass',
#  'dns_qtype',
#  'dns_rcode',
#  'label']
########## Net dataset - drop more - 24 col -FT : OneHotEncoder() for 'proto' , 'service' , 'conn_state'

########## tt_orig_iot_net_concat MinMax
# tt_orig_iot_net_concat1_data['fridge_temperature']   #for example
tt_orig_iot_net_concat1_le_nan1 = '../tt_orig_iot_net_concat1_le_nan1.csv'
tt_orig_iot_net_concat1_le_nan1_data = read_csv(tt_orig_iot_net_concat1_le_nan1)
MinMax_scaler = MMsc.fit(tt_orig_iot_net_concat1_le_nan1_data[imputer_cont_cols])
tt_orig_iot_net_concat1_le_nan1_data[imputer_cont_cols] = MinMax_scaler.fit_transform(tt_orig_iot_net_concat1_le_nan1_data[imputer_cont_cols])
# tt_orig_iot_net_concat1_le_nan1_data['fridge_temperature']   #for example
tt_orig_iot_net_concat1_le_nan1_data.to_csv('../tt_orig_iot_net_concat1_le_nan1_MinMax1.csv', index=False)
# (# You are right, MinMaxScaler will scale your data from 0 to 1. 0 will be the min of your column and 1 the max.
# Apply function will not actually transform your features, it will just return a dataframe with the transformed columns. 
# So you need to affect your transformation to your features :
# features = features.apply(lambda x: MinMaxScaler().fit_transform(x))
# )









