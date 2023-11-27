import os
import pandas as pd
import datetime

class DataPreparator:
    def __init__(self, path):
        self.__path = path
        self.__required_packages = {
            'pandas': '1.5.0',  # Min version of pandas
            'numpy': '1.23.0',   # Min version of numpy
            'matplotlib': '3.6.0'   # Min version of numpy
        }
        self.__hole_data_removers = {
            'linear', # linearly fill empty days
            'last_day', # fill with the value of the last working day
            'shift' # shift the data so that there are no spaces
        }
        self.__is_readed = False
        self.__is_parsed = False
        if not os.path.exists(self.__path):
            raise FileNotFoundError(f"File '{self.__path}' doesn't exist")
    
    @property
    def get_hole_data_removers(self):
        return self.__hole_data_removers
    
    def read(self):
        self.__data = pd.read_csv(self.__path)
        self.__is_readed = True

    def parse(self):
        try:
            data = self.__data
            data['Timestamp'] = pd.DataFrame(data.Date)
            data.Date = pd.to_datetime(data.Date, unit = 's')
            data = data.sort_values(by=["Date"])
            data.Close = data.Close.astype(float)
            data.High = data.High.astype(float)
            data.Low = data.Low.astype(float)
            data.Open = data.Open.astype(float)
            self.__is_parsed = True
        except Exception as ex:
            self.__is_parsed = False
        if(not self.__is_parsed):
            try:
                data = self.__data
                data['Timestamp'] = pd.DataFrame(data.Date)
                data.Date = pd.to_datetime(data.Date)
                data = data.sort_values(by = ["Date"])
                data.Close = data.Price.str.replace(',', '').astype(float) # Here price is equal to close
                data.High = data.High.str.replace(',', '').astype(float)
                data.Low = data.Low.str.replace(',', '').astype(float)
                data.Open = data.Open.str.replace(',', '').astype(float)
                data = data.drop("Vol.", axis = 1) # in this version we are not interested in Volume
            except Exception as ex:
                print("DataPreparator can't parse data")
                return
    
    def cut_data_by_date(self, begin, end):
        return
    

    def __date_freq__(self):
        data = self.__data
        timestemp = 0
        date_freq = 'None'
        if 'Date' in data: 
                timestemp = int(pd.to_datetime(data.Date[1]).timestamp()) - \
                    int(pd.to_datetime(data.Date[0]).timestamp())
        else:
                timestemp = int(pd.to_datetime(data.index[1]).timestamp()) - \
                    int(pd.to_datetime(data.index[0]).timestamp())
        if timestemp == 1:
            date_freq = 's'
        elif timestemp == 60:
            date_freq = 'm'
        elif timestemp == 3600:
            date_freq = 'h'
        elif timestemp == 86400:
            date_freq = 'D'
        elif timestemp == 604800:
            date_freq = 'W'
        else:
            raise Exception('Date freq detection problem. Please fix Date column or index to prevent this problem.')
        return date_freq, timestemp
    
    def remove_empty_days(self, remover = 'shift'):
        data = self.__data
        if not self.__is_readed or \
            not self.__is_parsed or \
                data is None:
            raise Exception("Data is not initialized")
        modified_data = pd.DataFrame(data)
        date_freq, ts = self.__date_freq__()
        if remover == 'linear':
            if 'Date' in modified_data: 
                all_dates = pd.date_range(start=modified_data.Date.min(), \
                                          end=modified_data.Date.max(), freq = date_freq)
            else:
                all_dates = pd.date_range(start=modified_data.index.min(), \
                                          end=modified_data.index.max(), freq = date_freq)
            modified_data['Date'] = all_dates
            modified_data = modified_data.sort_values(by = 'Date')
            min_date = modified_data['Date'].min()
            max_date = modified_data['Date'].max()
            date_range = pd.date_range(min_date, max_date, freq = date_freq)
            new_df = pd.DataFrame({'Date': date_range})
            merged_df = pd.merge(new_df, modified_data, on = 'Date', how = 'left')
            modified_data = merged_df.interpolate(method='linear', axis=0)
            new_order = ['Date', 'Price', 'Open', 'High', 'Low']
            modified_data = modified_data[new_order]
        elif remover == 'last_day':
            if 'Date' in modified_data: 
                all_dates = pd.date_range(start=modified_data.Date.min(), \
                                          end=modified_data.Date.max(), freq = date_freq)
            else:
                all_dates = pd.date_range(start=modified_data.index.min(), \
                                          end=modified_data.index.max(), freq = date_freq)
            new_df = pd.DataFrame(index = all_dates)
            new_df = new_df.merge(modified_data, how = 'left', left_index = True, right_index = True)
            new_df.fillna(method = 'ffill', inplace = True)
            new_df['Date'] =  new_df.index
            new_order = ['Date', 'Price', 'Open', 'High', 'Low']
            modified_data = new_df[new_order]
        elif remover == 'shift':
            modified_data['new_index'] = modified_data.Close
            modified_data['new_date'] = modified_data.Close
            if 'Date' in modified_data:
                modified_data.loc[0, 'new_index'] = modified_data.Date[0].timestamp()
            else:
                modified_data.loc[0, 'new_index'] = int(pd.to_datetime(modified_data.index[0]).timestamp())
            modified_data.loc[0, 'new_date'] = datetime.datetime.fromtimestamp(modified_data.loc[0, 'new_index']).strftime('%Y-%m-%d %H:%M:%S')
            for i in range(len(modified_data.new_index)):
                if i > 0:
                    modified_data.loc[i, 'new_index'] = modified_data.new_index[i - 1] + ts
                    modified_data.loc[i, 'new_date'] = datetime.datetime.fromtimestamp(modified_data.loc[i, 'new_index']).strftime('%Y-%m-%d %H:%M:%S')
            modified_data.set_index('new_date', inplace = True)
        else:
            raise Exception("Something went wrong")
        return modified_data