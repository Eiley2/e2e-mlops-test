# flake8: noqa
from dataclasses import dataclass

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import DataFrame as SparkDataFrame

from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


@dataclass
class FeaturizerConfig:
    """
    Attributes:
       label_col (str): Name of original label column in input data
       ohe (bool): Flag to indicate whether or not to one hot encode categorical columns
       cat_cols (list): List of categorical columns. Only required if ohe=True
       drop_missing (bool): Flag to indicate whether or not to drop missing values
    """
    label_col: str
    ohe: bool
    drop_missing: bool = True


class Featurizer:
    """
    Class containing featurization logic to apply to input Spark DataFrame
    """

    def __init__(self, cfg: FeaturizerConfig):
        self.cfg: FeaturizerConfig = cfg

    @staticmethod
    def drop_missing_values(
        psdf: SparkDataFrame,
    ) -> SparkDataFrame:
        """
        Remove missing values

        Parameters
        ----------
        psdf
        Returns
        -------
        pyspark.pandas.DataFrame
        """
        return psdf.select("product_name").dropna()

    @staticmethod
    def get_country_name(
        df: SparkDataFrame,
    ) -> SparkDataFrame:
        """
        Get short name for Colombia

        Parameters
        ----------
        psfg : SparkDataFrame
                Input Dataframe to preproccess

        Returns
        --------
        SparkDataFrame
            Preprocessed dataset of features and label column

        """

        return df.withColumn(
            "short_name",
            F.when(
                F.col("country_name") == "COLOMBIA", F.lit(self.cfg.value_name)
            ),
        )

    def run(self, df: SparkDataFrame) -> SparkDataFrame:
        """
        Run all data preprocessing steps. Consists of the following:

            1. Convert PySpark DataFrame to pandas_on_spark DataFrame
            2. Process the label column - converting to int and renaming col to 'churn'
            3. Apply OHE if specified in the config
            4. Drop any missing values if specified in the config
            5. Return resulting preprocessed dataset as a PySpark DataFrame

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input PySpark DataFrame to preprocess

        Returns
        -------
        pyspark.sql.DataFrame
            Preprocessed dataset of features and label column
        """
        _logger.info("Running Data Preprocessing steps...")

        # Get country short name
        _logger.info(f"Getting short name values")
        preproc_df = self.get_country_name(df)

        # Drop missing values
        if self.cfg.drop_missing:
            _logger.info(f"Dropping missing values")
            preproc_df = self.drop_missing_values(preproc_df)

        preproc_df.printSchema()
        preproc_df.show()

        return preproc_df
