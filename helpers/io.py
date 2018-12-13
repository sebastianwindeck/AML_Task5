from datetime import datetime
import pandas as pd
import os
import numpy as np


def outputter(array):
    y = pd.DataFrame(array, dtype=np.dtype('U25'))
    ids = list(range(0, y.shape[0]))
    ids = pd.DataFrame(ids)
    output = pd.concat([ids, y], axis=1)
    output.columns = ["Id", "y"]

    cwd = os.getcwd()
    date = datetime.date(datetime.now())
    time = datetime.time(datetime.now())
    s = "_"
    seq = (str(date), str(time.strftime('%H_%M_%S')), "solution.csv")
    file_name = s.join(seq)  # type: str
    path = cwd + "/output/" + file_name
    path = os.path.abspath(path)
    print(path)
    output.to_csv(path_or_buf=path, sep=',', na_rep='', float_format='U25',
                  header=True, index=False,
                  mode='w', encoding=None, compression=None,
                  quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=None,
                  date_format=None, doublequote=True, escapechar=None, decimal='.')
