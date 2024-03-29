import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Audio

# base_path = '/home/dhaval/Downloads/cv-corpus-16.1-2023-12-06-mr/cv-corpus-16.1-2023-12-06'
base_path = '/home/dhaval/Downloads/openslr-marathi-dataset'
lang = 'mr'
# csv_file = f'{base_path}/{lang}/train.tsv'
csv_file = f'{base_path}/line_index.tsv'

audio_paths = []
text_arr = []
speaker_ids = []

df = pd.read_csv(csv_file, sep='\t')
df = df.reset_index()
for index, row in df.iterrows():
    # if index >= 128 and index < 256:
    # audio_paths.append(f"{base_path}/{lang}/clips/{row['path']}")
    audio_paths.append(f"{base_path}/{row['Path']}.wav")
    text_arr.append(row["Text"])
    # speaker_ids.append(row["client_id"])

audio_dataset = Dataset.from_dict({
    "audio": audio_paths,
    "text": text_arr,
    # "speaker_id": speaker_ids
}).cast_column("audio", Audio())

audio_dataset.push_to_hub("dhavalgala/openslr-mar")
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
