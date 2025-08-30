# Databricks notebook source
# MAGIC %run /Workspace/Repos/tripathiaakriti355@gmail.com/projectcreditrisk/creditrisk1/bronzefinal

# COMMAND ----------

# MAGIC %run /Workspace/Repos/tripathiaakriti355@gmail.com/projectcreditrisk/creditrisk1/silverfinal

# COMMAND ----------

# MAGIC %run /Workspace/Repos/tripathiaakriti355@gmail.com/projectcreditrisk/creditrisk1/goldfinal

# COMMAND ----------

# MAGIC %pip install pytest

# COMMAND ----------

import pytest
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import IntegerType, DoubleType

# ---------------------------
# PySpark Fixture
# ---------------------------
@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("pytest_credit_pipeline")
        .getOrCreate()
    )
    yield spark
    spark.stop()

# ---------------------------
# Bronze Layer Tests
# ---------------------------
def test_bronze_layer(spark):
    data = [
        ("Alice", 25, 50000, 10000),
        ("Bob", 30, 60000, 15000),
    ]
    columns = ["Person Name", "Person Age", "Person Income", "Loan Amnt"]
    df = spark.createDataFrame(data, columns)

    # Transform: clean column names
    df_bronze = df.toDF(*[c.strip().replace(" ", "_").lower() for c in df.columns])

    # ✅ Test 1: Schema lowercase
    assert all(col.islower() for col in df_bronze.columns)

    # ✅ Test 2: Data not empty
    assert df_bronze.count() > 0

# ---------------------------
# Silver Layer Tests
# ---------------------------
def test_silver_layer(spark):
    data = [
        ("Alice", None, 50000.0, 10000.0, "home", 1),
        ("Bob", 30, None, 15000.0, None, 0),
        ("Bob", 30, None, 15000.0, None, 0),  # duplicate
    ]
    columns = ["person_name", "person_age", "person_income", "loan_amnt", "loan_intent", "loan_status"]

    df = spark.createDataFrame(data, columns)

    # Fill null numeric with mean
    numeric_cols = ["person_age", "person_income", "loan_amnt"]
    df_clean = df
    for col in numeric_cols:
        mean_value = df_clean.select(F.mean(F.col(col))).collect()[0][0]
        if mean_value is not None:
            df_clean = df_clean.fillna({col: mean_value})

    # Fill categorical nulls with "Unknown"
    df_clean = df_clean.fillna({"loan_intent": "Unknown"})

    # Drop duplicates
    df_clean = df_clean.dropDuplicates()

    # Cast types
    df_clean = df_clean.withColumn("person_age", F.col("person_age").cast(IntegerType()))
    df_clean = df_clean.withColumn("person_income", F.col("person_income").cast(DoubleType()))

    # ✅ Test 1: No NULLs remain in key columns
    assert df_clean.filter("person_age IS NULL OR person_income IS NULL").count() == 0

    # ✅ Test 2: Duplicates removed
    assert df_clean.count() == 2

    # ✅ Test 3: Data types
    assert dict(df_clean.dtypes)["person_income"] == "double"

# ---------------------------
# Gold Layer Tests
# ---------------------------
def test_gold_layer(spark):
    data = [
        (25, "education", "A", "RENT", 50000, 10000, 0),
        (30, "medical", "B", "OWN", 60000, 15000, 1),
    ]
    columns = ["person_age", "loan_intent", "loan_grade", "person_home_ownership",
               "person_income", "loan_amnt", "loan_status"]

    df = spark.createDataFrame(data, columns)

    # Risk Score transformation
    df_gold = df.withColumn(
        "risk_score",
        (F.col("loan_amnt") / (F.col("person_income") + F.lit(1))) *
        (F.when(F.col("loan_status") == 1, 2).otherwise(1))
    )

    # Aggregation
    group_cols = ["person_age", "loan_intent", "loan_grade", "person_home_ownership"]
    df_aggregated = (
        df_gold.groupBy(*group_cols)
        .agg(
            F.avg("risk_score").alias("avg_risk_score"),
            F.avg("person_income").alias("avg_income"),
            F.avg("loan_amnt").alias("avg_loan_amount"),
            F.sum("loan_status").alias("total_defaults"),
            F.count("*").alias("customer_count"),
            (F.sum("loan_status") / F.count("*")).alias("default_rate")
        )
    )

    # ✅ Test 1: Risk score column created
    assert "risk_score" in df_gold.columns

    # ✅ Test 2: Aggregation has required metrics
    expected_cols = {"avg_risk_score", "avg_income", "avg_loan_amount",
                     "total_defaults", "customer_count", "default_rate"}
    assert expected_cols.issubset(set(df_aggregated.columns))

    # ✅ Test 3: Default rate within 0–1
    assert df_aggregated.select(F.min("default_rate")).first()[0] >= 0
    assert df_aggregated.select(F.max("default_rate")).first()[0] <= 1
    print()

# COMMAND ----------

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------- Fixture: Initialize Spark ----------
@pytest.fixture(scope="session")
def spark():
    spark = (SparkSession.builder
             .appName("pytest_bronze_tests")
             .master("local[*]")
             .enableHiveSupport()
             .getOrCreate())
    return spark

# ---------- Test 1: Check Schema Creation ----------
def test_schema_exists(spark):
    dbs = [db.databaseName for db in spark.catalog.listDatabases()]
    assert "bronze1" in dbs, "❌ Glue catalog 'bronze1' not created"

# ---------- Test 2: Bronze Table Exists ----------
def test_bronze_table_exists(spark):
    tables = [t.name for t in spark.catalog.listTables("bronze1.credit_data")]
    assert "credit_bronze" in tables, "❌ Bronze table not found"

# ---------- Test 3: Check Row Count > 0 ----------
def test_bronze_row_count(spark):
    df = spark.table("bronze1.credit_data.credit_bronze")
    assert df.count() > 0, "❌ Bronze table is empty"

# ---------- Test 4: Column Naming Convention ----------
def test_bronze_column_names(spark):
    df = spark.table("bronze1.credit_data.credit_bronze")
    for col in df.columns:
        assert col == col.lower(), f"❌ Column {col} not standardized to lowercase"
        assert " " not in col, f"❌ Column {col} contains spaces"

# ---------- Test 5: No Nulls in Primary Key ----------
def test_primary_key_not_null(spark):
    df = spark.table("bronze1.credit_data.credit_bronze")
    assert df.filter(F.col("applicant_id").isNull()).count() == 0, "❌ Null applicant_id found"
