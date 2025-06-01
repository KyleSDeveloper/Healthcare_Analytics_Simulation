from google.cloud import bigquery

# Initialize client
client = bigquery.Client()

# Run query on large dataset
query = """
SELECT age, AVG(num_hospitalizations) as avg_hospitalizations
FROM `healthcare-analytics-461619.healthcare_data.diabetic_data_large`
GROUP BY age
ORDER BY age
"""
query_job = client.query(query)
results = query_job.to_dataframe()
print(results)