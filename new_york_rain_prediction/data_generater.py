import os
import pandas as pd
import numpy as np

class weatherDataGenerator:

    def __init__(self, dataset_dir, city):
        self.dataset_dir=dataset_dir
        self.city=city
        
    def extract_city_info(self, data_file, column_name, city):
        df=pd.read_csv(data_file)
        df=df[['datetime',city]]
        df['datetime']=pd.to_datetime(df['datetime'])
        df.rename(columns={city:column_name}, inplace=True)
        df.set_index('datetime',inplace=True)
        return df

    def create_city_dataset(self):
        data_files = os.listdir(self.dataset_dir)
        if 'city_attributes.csv' in data_files: data_files.remove('city_attributes.csv') # do not need it for now
        column_names = [x.replace('.csv','') for x in data_files]
    #     print(column_names)
        dfs=[]
        for i in range(len(data_files)):
            df = self.extract_city_info(os.path.join(self.dataset_dir,data_files[i]),column_names[i],self.city)
            dfs.append(df)
            
        aggregated_df = pd.concat(dfs, axis=1) # We are lucky here because dataset has no missing datetime index across files
        # aggregated_df = self.rm_extra_data(aggregated_df)
        return aggregated_df


    def simple_fill_na(self, df):
        # Fill in missing values
        df['weather_description'].fillna('None', axis=0, inplace=True)
        df.fillna(method='ffill', axis=0, inplace=True)
        df.fillna(method='bfill', axis=0, inplace=True)

    def consolidate_weather_condition(self, interested_features, df):
        def is_rainny(value):
            keywords = ['rain','thunderstorm', 'drizzle'] # will label all instances containing these keywords as "rain", otherwise "no rain"
            for x in keywords:
                if x in value:
                    return 1
            return 0
        df['rainny']=df['weather_description'].apply(is_rainny)
        df = df[interested_features]
        return df

    def rm_extra_data(self,df):
        # thow hours that do not fill 24-hour day
        # this is optional to call depend on how you want to generate your dataset
        df=df['2012-10-02':]
        return df



class datasetGenerator:
    def __init__(self, ts_df, selected_features, target_col):
        self.df=ts_df
        self.selected_features=selected_features
        self.target=target_col
        self.X, self.y = self.create_x_y()

    def create_x_y(self):
        X = self.df[self.selected_features].values
        y = self.df[self.target].values
        return X,y

    
    def generate_rolling_sequence_data(self, input_step, output_step):
        '''
            generate rolling sequence inputs with [# input timestep, # features]
        '''
        inputs=[]
        outputs=[]
        for i in range(input_step,len(self.X)-output_step+1): # make sure to add 1 since range does not include the last value
            sample=self.X[i-input_step:i]
            label=self.y[i:i+output_step]
            inputs.append(sample)
            outputs.append(label)
        return np.array(inputs), np.array(outputs)

    def generate_discrete_sequence_data(self, input_step, output_step, step_size):
        '''
            generate discrete sequence inputs with [# input timestep, # features]
        '''
        inputs=[]
        outputs=[]
        index_range=int((len(self.X)-(input_step+output_step)*step_size)/step_size)+1 # make sure to add 1 since range does not include the last value

        for i in range(index_range):
            sample=self.X[i*step_size:(i+input_step)*step_size]
            label=self.y[(i+input_step)*step_size:(i+input_step+output_step)*step_size]
            inputs.append(sample)
            outputs.append(label)
        return np.array(inputs), np.array(outputs)

    def generate_feature(self, df, target_features, func_ptr, rename=False):
        '''
            generate additional feature by grouping by date and appling func_ptr
        '''
        new_df=df[target_features].groupby('date').apply(func_ptr)

        if rename:
            col_names=self.rm_ignored(target_features)
            # new_col_names=[col+'_{}'.format(func_ptr) for col in col_names]
            titles={col:col+'_{}'.format(func_ptr) for col in col_names}

        new_df.rename(columns=titles, inplace=True)
        return new_df



    def scale_features(self, start, end, target):
        '''
        MIN-MAX scale for each column(feature)
        source: 2-D numpy array for all weather data (values for weather_df)
        target: 3-D numpy array with shape (# samples, # timesteps, # features)
        start, end: denotes where in the source pool of weather data do you want to extract mean and max for every feature. 
        Note we should do feature scaling for train, validation and test set separately as they may have different distributions.
        Returns:
            Normalized target dataset

        '''
    #     print(target.shape)
        feature_min = np.min(self.X[start:end], axis=0)
        feature_max = np.max(self.X[start:end], axis=0)
        denom_factor=np.subtract(feature_max,feature_min)
    #     print(feature_max)
    #     print(feature_min)
    #     print(denom_factor)
        results = np.array([np.divide(np.subtract(x,feature_min),denom_factor) for x in target])
    #     print(results.shape)
        return results
    
    def split_data(self, sequence_inputs, sequence_labels, train, val, test):
        '''
        This train, val, test split function does not do randomization given the fact that this problem is a time series
        '''

        assert train+val+test==len(sequence_inputs)
        Xtrain, Xval, Xtest = sequence_inputs[:train],sequence_inputs[train:train+val],sequence_inputs[train+val:]
        ytrain, yval, ytest = sequence_labels[:train],sequence_labels[train:train+val],sequence_labels[train+val:]
        # print(Xtrain.shape)
        # print(ytrain.shape)
        # print(Xval.shape)
        # print(yval.shape)
        # print(Xtest.shape)
        # print(ytest.shape)
        return Xtrain, Xval, Xtest, ytrain, yval, ytest

    @staticmethod
    def rm_ignored(self, target):
        ignored=['date','day','month','year','rainny','rain_count']
        return [col for col in target if col not in ignored]
        
