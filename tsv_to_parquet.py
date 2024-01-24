import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Audio

csv_file = '/home/dhaval/Downloads/cv-corpus-16.1-2023-12-06-id/cv-corpus-16.1-2023-12-06/id/train.tsv'
parquet_file = './my.parquet'

audio_paths = []
text_arr = []
speaker_ids = []

df = pd.read_csv(csv_file, sep='\t')
df = df.reset_index()
for index, row in df.iterrows():
    audio_paths.append(f"/home/dhaval/Downloads/cv-corpus-16.1-2023-12-06-id/cv-corpus-16.1-2023-12-06/id/clips/{row['path']}")
    text_arr.append(row["sentence"])
    speaker_ids.append(row["client_id"])

audio_dataset = Dataset.from_dict({
    "audio": audio_paths,
    "text": text_arr,
    "speaker_id": speaker_ids
}).cast_column("audio", Audio())

audio_dataset.push_to_hub("dhavalgala/mozilla-ind")
# print(audio_paths)

# for i, chunk in enumerate(csv_stream):
#     print("Chunk", i)
#     if i == 0:
#         # Guess the schema of the CSV file from the first chunk
#         parquet_schema = pa.Table.from_pandas(df=chunk).schema
#         print(parquet_schema)
#         # Open a Parquet file for writing
#         parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')
#     # Write CSV chunk to the parquet file
#     table = pa.Table.from_pandas(chunk, schema=parquet_schema)
#     print("table", table)
#     parquet_writer.write_table(table)

# parquet_writer.close()
