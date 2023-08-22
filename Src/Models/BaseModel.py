import pandas

class BaseModel:
    def __init__(self, price, date):
        """Base model constructor.

        Args:
            price : Data object with price.
            date : Data object with date in date format.
        
        """
        self._price = price
        self._date = date
        self._result = None
        self._is_predicted = False
    
    def get_price(self) -> pandas.DataFrame:
        return self._price
    
    def get_date(self) -> pandas.DataFrame:
        return self._date
    
    def get_result(self):
        if(self._is_predicted):
            return self._result
    
    def predict():
        raise NotImplementedError()
