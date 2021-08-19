from datetime import timedelta

import numpy as np
import pandas as pd
from dateutil.parser import isoparse

if __name__ == '__main__':
    columns = ["bicycles", "cars", "motorcycles", "buses", "trucks"]
    dti = pd.date_range("2021-08-15", periods=60 * 24, freq="1min")
    df = pd.DataFrame(index=dti, columns=["bicycles", "cars", "motorcycles", "buses", "trucks"])
    df = df.fillna(0)

    csv = pd.read_csv('dict_file.csv').T
    csv = csv.transpose()
    csv.dropna()
    for item in columns:
        for occurrence in csv.get(item).dropna():
            datetime = isoparse(str(occurrence.split(':', 1)[1]))
            datetime_masked = np.datetime64(datetime - timedelta(minutes=datetime.minute % 1,
                                                                 seconds=datetime.second,
                                                                 microseconds=datetime.microsecond), 'ns')
            df.loc[datetime_masked][item] += 1
    df.to_csv('test.csv')
