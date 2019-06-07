import pandas as pd
import click

@click.command()
@click.option('-f', "--file-path", default="", help="file path to csv")

def main(file_path):
    data = pd.read_csv(
        file_path
    )
