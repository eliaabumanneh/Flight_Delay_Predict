import boto3
import csv
from io import StringIO

#set up S3 connection without access keys
s3 = boto3.resource('s3', aws_access_key_id='', aws_secret_access_key='')

#set up file path
bucket_name = 'noaa-ghcn-pds'
key = 'csv/by_year/2019.csv'

#read CSV file from S3
s3_object = s3.Object(bucket_name, key)
csv_bytes = s3_object.get()['Body'].read().decode('utf-8')
csv_file = StringIO(csv_bytes)
reader = csv.reader(csv_file)

#iterate over rows in CSV file
for row in reader:
    print(row)
