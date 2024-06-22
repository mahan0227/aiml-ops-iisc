import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import *

'''
bike_pipeline = Pipeline([

    ('weekday_imputation', WeekdayImputer(feature='weekday')),
    ('weatherisimp', WeathersitImputer(feature = 'weathersit')),
    ('map_yr',Mapper('yr',{2011: 0, 2012: 1})),
    ('map_mnth',Mapper('mnth', {'January': 0, 'February': 1, 'December': 11, 'March': 2, 'November': 10, 'April': 3,'October': 9, 'May': 4, 'September': 8, 'June': 5, 'July': 6, 'August': 7} )),
    ('map_season',Mapper('season', {'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3})),
    ('map_weathersit',Mapper('weathersit',{'Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3})),
    ('map_holiday',Mapper('holiday', {'Yes': 1, 'No': 0})),
    ('map_workingday',Mapper('workingday', {'No': 0, 'Yes': 1})),
    ('map_hr',Mapper('hr', {'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8,'10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16,'12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23})),
    ('outlier_handler_temp', OutlierHandler('temp')),
    ('outlier_handler_atemp', OutlierHandler('atemp')),
    ('outlier_handler_hum', OutlierHandler('hum')),
    ('outlier_handler_windspeed', OutlierHandler('windspeed')),
    ('onehot_encoder', WeekdayOneHotEncoder(col = 'weekday')),
    ('unused_column_dropper',columnDropperTransformer(['dteday', 'casual', 'registered'])),
    # scale
    ('scaler', StandardScaler()),
    # Model fit
    ('model_rf', RandomForestRegressor(n_estimators=150, max_depth=10,random_state=42))
])

'''


bike_pipeline = Pipeline([
    ('weekday_imputation', WeekdayImputer(feature=config.model_config.weekday_var)),
    ('weatherisimp', WeathersitImputer(feature = config.model_config.weathersit_var)),
    ('map_yr',Mapper(config.model_config.yr_var, config.model_config.yr_mappings)),
    ('map_mnth',Mapper(config.model_config.mnth_var, config.model_config.mnth_mappings)),
    ('map_season',Mapper(config.model_config.season_var, config.model_config.season_mappings)),
    ('map_weathersit',Mapper(config.model_config.weathersit_var, config.model_config.weathersit_mappings)),
    ('map_holiday',Mapper(config.model_config.holiday_var, config.model_config.holiday_mappings)),
    ('map_workingday',Mapper(config.model_config.workingday_var, config.model_config.workingday_mappings)),
    ('map_hr',Mapper(config.model_config.hr_var, config.model_config.hr_mappings)),
    ('outlier_handler_temp', OutlierHandler(config.model_config.temp_var)),
    ('outlier_handler_atemp', OutlierHandler(config.model_config.atemp_var)),
    ('outlier_handler_hum', OutlierHandler(config.model_config.hum_var)),
    ('outlier_handler_windspeed', OutlierHandler(config.model_config.windspeed_var)),
    ('onehot_encoder', WeekdayOneHotEncoder(col = config.model_config.weekday_var)),
    ('unused_column_dropper',ColumnDropperTransformer(config.model_config.unused_fields)),
    # scale
    ('scaler', StandardScaler()),
    # Model fit
    ('model_rf', RandomForestRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,
                                     random_state=config.model_config.random_state))
])

