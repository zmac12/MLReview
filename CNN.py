# Convolutional Neural Network #

import snowflake.connector
import os

PASSWORD = os.getenv('SNOWSQL_PWD')
WAREHOUSE = os.getenv('SNOWWAREHOUSE')
ACCOUNT = os.getenv('SNOWACCT')
USER = os.getenv('SNOWUSER')
