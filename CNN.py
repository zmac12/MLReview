# Convolutional Neural Network #

import snowflake.connector
import os

PASSWORD = os.getenv('SNOWSQL_PWD')
WAREHOUSE = os.getenv('SNOWWAREHOUSE')
ACCOUNT = os.getenv('SNOWACCT')
USER = os.getenv('SNOWUSER')

con = snowflake.connector.connect(
    user='zachmcq',
    password='seQ2nsatio2nal',
    account='cfa83386.us-east-1',
    warehouse=WAREHOUSE,
    schema='PUBLIC',
    database='CNNPOC'
)

cur = con.cursor()

sql = 'SELECT * FROM CNNPOC.PUBLIC.LABELLEDREVIEWSNEW'

cur.execute(sql)

df = cur.fetch_pandas_all()
print(df)