from google.cloud import bigquery

# Initialize client
client = bigquery.Client()
dataset_id = 'healthcare_data'
table_id = 'diabetic_data'

# Create dataset
dataset = bigquery.Dataset(f"healthcare-analytics-461619.{dataset_id}")
client.create_dataset(dataset, exists_ok=True)

# Upload CSV
table_ref = client.dataset(dataset_id).table(table_id)
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    autodetect=True
)
with open('diabetic_data_preprocessed.csv', 'rb') as source_file:
    job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
job.result()
print("Data uploaded to BigQuery")