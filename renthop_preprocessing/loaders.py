
import features

import numpy as np
from pandas.io.json import read_json
from pandas import DataFrame, get_dummies

from sklearn.utils import shuffle

from datetime import datetime
import re

FTRAIN = 'data/train.json'
FTEST = 'data/test.json'

