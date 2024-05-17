import pandas as pd
import click


@click.command()
@click.option("--file", help="Parquet file to read.", required=True)
@click.option("--newfile", help="File to write out.", required=True)
@click.option("--rows", type=int, help="Number of rows to write out.", required=True)
def run(file: str, newfile: str, rows: int):

    df = pd.read_parquet(file)
    small_df = df.head(rows)
    small_df.to_parquet(newfile)
    print(f"finished writing {newfile}")


if __name__ == "__main__":
    run()
