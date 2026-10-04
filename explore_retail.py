import duckdb

CSV_PATH = "Online_Retail.csv"

con = duckdb.connect()
con.execute(f"""
    CREATE VIEW retail AS
    SELECT * FROM read_csv_auto('{CSV_PATH}', sample_size=-1, encoding='latin-1')
""")

columns = [row[0] for row in con.execute("DESCRIBE retail").fetchall()]

row_count = con.execute("SELECT COUNT(*) FROM retail").fetchone()[0]
print(f"Total row count: {row_count}")

min_date, max_date = con.execute(
    """
    SELECT MIN(parsed), MAX(parsed)
    FROM (SELECT strptime(InvoiceDate, '%-m/%-d/%y %-H:%M') AS parsed FROM retail)
    """
).fetchone()
print(f"Min InvoiceDate: {min_date}")
print(f"Max InvoiceDate: {max_date}")

print("\nNull counts per column:")
null_exprs = ", ".join(
    f'SUM(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS "{col}"' for col in columns
)
null_counts = con.execute(f"SELECT {null_exprs} FROM retail").fetchone()
for col, count in zip(columns, null_counts):
    print(f"  {col}: {count}")

print("\n5 sample rows:")
sample = con.execute("SELECT * FROM retail LIMIT 5").fetchdf()
print(sample.to_string(index=False))
