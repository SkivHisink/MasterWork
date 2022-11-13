import pandas as pd

class DataManipulator:
    def __init__(self, path):
        self.data = self.__getData(path)
        self.file = path

    def __getData(self, path):
        data = pd.read_csv(path)
        return data
    
    def transformData(self):
        self.data = self.data.replace(",", "", regex=True)
        self.data.Date = pd.to_datetime(self.data.Date)
        self.data = self.data.sort_values(by=["Date"])
        self.data.set_index('Date')
        self.data.Price = self.data.Price.astype(float)
        self.data.High = self.data.High.astype(float)
        self.data.Low = self.data.Low.astype(float)
        self.data.Open = self.data.Open.astype(float)
        self.data["Vol."] = self.data["Vol."].astype(float)

    def dateToIndex(self):
        self.data.set_index('Date', inplace=True)

    def plot(self):
        self.data.plot()