# Convolutional Neural Network #

import snowflake.connector
import os

PASSWORD = os.getenv('SNOWSQL_PWD')
WAREHOUSE = os.getenv('WAREHOUSE')
ACCOUNT = os.getenv('SNOWACCT')