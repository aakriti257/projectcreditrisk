# Databricks notebook source
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM gold1.analytics.risk_features;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   person_age,
# MAGIC   default_rate
# MAGIC FROM gold1.analytics.risk_features
# MAGIC ORDER BY person_age;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   loan_grade,
# MAGIC   avg_risk_score
# MAGIC FROM gold1.analytics.risk_features
# MAGIC ORDER BY loan_grade;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   person_home_ownership,
# MAGIC   loan_intent,
# MAGIC   default_rate
# MAGIC FROM gold1.analytics.risk_features;