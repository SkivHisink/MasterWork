import pandas

class BaseModel:
    def __init__(self, data):
        """Base model constructor.

        Args:
            price : Data object with price.
            date : Data object with date in date format.
        
        """
        self._data = data
        self._result = None
        self._is_predicted = False
    
    def get_data(self) -> pandas.DataFrame:
        return self._data
    
    def get_result(self):
        if(self._is_predicted):
            return self._result
    
    def predict():
        raise NotImplementedError()
