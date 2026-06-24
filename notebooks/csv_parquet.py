"""
CSV to Parquet Converter
"""
# DuckDB handles the conversion with its own internal memory management, making it well suited for very large data files. 
# PyArrow is used at the end to read and work with the output

import duckdb
import pyarrow.parquet as pq
from pathlib import Path


## Configuration
# Use `snappy` for fast reads in active workloads, 
# `zstd` when storage size matters, `gzip` for compatibility with older tools for compression


input_csv = "..\\csv_data\\company_data_info.csv"
OUTPUT_PARQUET = "../data/" + Path(input_csv).stem + ".parquet"
compression = "snappy"

## Convert

duckdb.sql(f"""
    COPY (SELECT * FROM read_csv_auto('{input_csv}'))
    TO '{OUTPUT_PARQUET}'
    (FORMAT PARQUET, COMPRESSION {compression})
""")


## Verifying with PyArrow
# At this point the Parquet file is ready to use with PyArrow in the project

pf = pq.ParquetFile(OUTPUT_PARQUET)
print(f"Rows: {pf.metadata.num_rows:,}")
print(f"Row groups: {pf.metadata.num_row_groups}")
print(f"Columns: {pf.metadata.num_columns}")
print(pf.schema_arrow)
