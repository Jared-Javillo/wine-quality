from pathlib import Path
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm

from wine_quality.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


def load_raw_data(filename: str, delimiter=","):
    input_path: Path = RAW_DATA_DIR / filename

    return pd.read_csv(input_path, delimiter=delimiter)


if __name__ == "__main__":
    app()
