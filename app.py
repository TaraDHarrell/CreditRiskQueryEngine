"""Northbridge Bank Credit Risk Query Engine — Streamlit deployment.
Adapted from the UT Austin / Great Learning Project 3 notebook.
"""
import os
import re
import json
import sqlite3
import math
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd
import numpy as np
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Northbridge Credit Risk Query Engine", page_icon="🏦", layout="wide")


def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


def configure_credentials():
    """Use Streamlit secrets (deployment) or environment variables (local)."""
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        value = get_secret(name, os.environ.get(name))
        if value:
            os.environ[name] = str(value)
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY. Add it to Streamlit secrets or your environment.")
        st.stop()


configure_credentials()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def open_read_only_database():
    """Open the supplied portfolio SQLite file without write permissions."""
    db_path = Path(get_secret("DATABASE_PATH", os.environ.get("DATABASE_PATH", "credit_risk_portfolio.db"))).expanduser().resolve()
    if not db_path.is_file():
        st.error(f"Database not found: {db_path.name}. Add credit_risk_portfolio.db to the deployed app files.")
        st.stop()
    conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


# Database schema: unchanged from the notebook
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# Verified SQL from notebook cell 37
sql_1 = """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_m,
    ROUND(SUM(CASE
        WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
        THEN l.total_outstanding
        ELSE 0
    END) / 1000000.0, 2) AS npa_exposure_m
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_m DESC
"""


# Verified SQL from notebook cell 41
sql_2 = """
SELECT
    loan_category,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_m,
    COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_m DESC
"""


# Verified SQL from notebook cell 45
sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_m,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_m
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage
"""


# Verified SQL from notebook cell 49
sql_4 = """
SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_coverage_ratio
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_coverage_ratio DESC
"""


# Verified SQL from notebook cell 53
sql_5 = """
SELECT
    borrower_name,
    sector_code,
    ROUND(total_outstanding / 1000000.0, 2) AS outstanding_m,
    asset_classification
FROM loan_master
ORDER BY total_outstanding DESC
LIMIT 10
"""


# Verified SQL from notebook cell 57
sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_m
FROM loan_master
WHERE group_name IS NOT NULL AND TRIM(group_name) != ''
GROUP BY group_name
ORDER BY total_outstanding_m DESC
LIMIT 5
"""


# Verified SQL from notebook cell 61
sql_7 = """
SELECT
    loan_account_number,
    borrower_name,
    sector_code,
    ROUND(total_outstanding / 1000000.0, 2) AS outstanding_m,
    days_past_due,
    asset_classification
FROM loan_master
WHERE days_past_due > 0
ORDER BY days_past_due DESC
"""


# Verified SQL from notebook cell 65
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_m
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        ELSE 5
    END
"""


# Verified SQL from notebook cell 69
sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
    AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""


# Verified SQL from notebook cell 73
sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS stage3_book_size_m,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS stage3_ecl_m
FROM provisioning
WHERE ifrs9_stage = 3
AND reporting_date IN (
    '2024-12-31',
    '2025-03-31',
    '2025-06-30',
    '2025-09-30'
)
GROUP BY reporting_date
ORDER BY reporting_date
"""


# Verified query library: unchanged from the notebook
verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },


    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },


    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },


    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },


    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },


    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },


    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },


    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },


    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },


    'VQ10': {
    'description': 'Stage 3 book size and expected credit loss trend over the last four quarters',
    'sql': sql_10
}
}

print(f"Verified query library loaded with {len(verified_query_library)} templates")
for qid, entry in verified_query_library.items():
    print(f"  {qid}: {entry['description']}")


# Original notebook pipeline function
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

You are an intent classification assistant for a bank's credit risk query engine.

Your task is to determine whether the user's question can be answered
using one of the verified SQL query templates listed below.

### Instructions
1. Read the user's question carefully.
2. Compare it with the descriptions of the verified queries.
3. If a verified query can answer the question using its existing SQL
   without modification, select the "verified" route and return its query ID.
4. If no verified query can fully answer the question, select the
   "generated" route and set query_id to null.
5. Do not select a verified query based only on similar keywords.
6. Provide a short explanation for your decision.

### User Question

{user_question}

{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# Original notebook pipeline function
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

You are a SQL assistant for a bank's credit risk analytics system.

Your task is to generate an accurate, read-only SQLite query
that answers the user's question using the database schema provided.

### Instructions
1. Use only the tables and columns listed in the database schema.
2. Generate SQLite-compatible SQL.
3. Only use SELECT statements or WITH clauses that produce SELECT results.
4. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or other
   commands that modify the database.
5. Use appropriate JOIN conditions when combining tables.
6. Use WHERE, GROUP BY, ORDER BY, and aggregation functions as needed.
7. Use the definitions and reporting dates provided in the schema.
8. Do not invent tables, columns, or data.
9. Return only the SQL query, without explanations or markdown formatting.
10. Do not include multiple SQL statements.
11. If the question cannot be answered using the available data,
    return a SQL query that produces no rows rather than guessing.

### User Question

{user_question}

{schema_context}

"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# Original notebook pipeline function
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }



    # Check 1: Require one read-only SQL statement.
    statements = sqlparse.parse(candidate_sql)
    if len(statements) != 1 or statements[0].get_type() != 'SELECT':
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Only one SELECT query (including SELECT-based CTEs) is allowed.'
        return result
    # SQLite's query_only connection is an additional protection against writes.
    forbidden = ('INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'REPLACE',
                 'ATTACH', 'DETACH', 'PRAGMA', 'VACUUM', 'CREATE', 'REINDEX')
    flattened = [token for token in statements[0].flatten()
                 if not token.is_whitespace and not token.ttype in sqlparse.tokens.Comment]
    for token in flattened:
        if token.value.upper() in forbidden and token.ttype in sqlparse.tokens.Keyword:
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Disallowed SQL operation: {token.value}'
            return result

    # Check 2 and 3: Let SQLite validate real tables, columns and SQL syntax.
    # This is more reliable than scanning words (which mistakes aliases and
    # text literals for nonexistent columns).
    cur = db_connection.cursor()
    try:
        cur.execute(f"EXPLAIN QUERY PLAN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as exc:
        result['failed_check'] = 'schema_or_query_plan'
        result['details'] = f'SQL schema or query error: {exc}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )


    relevance_prompt = f"""

You are a SQL validation assistant for a bank's credit risk query engine.

Your task is to evaluate whether the candidate SQL query correctly
answers the user's question.

### Instructions
1. Compare the user's question with the candidate SQL query.
2. Check that the SQL uses the appropriate tables and columns.
3. Verify that the selected metrics, calculations, aggregations,
   filters, and JOIN conditions match the user's request.
4. Check that the query uses the correct reporting dates and
   credit risk definitions when applicable.
5. Consider the query context provided below.
6. For verified templates, allow broader results when the approved
   query returns all sectors, categories, or stages. The response
   generation step will identify the relevant information.
7. For generated queries, ensure the SQL directly answers the
   specific question without unnecessary or incorrect results.
8. Return a verdict of "yes" only when the SQL is relevant and
   logically appropriate.
9. Assign a confidence score between 0.0 and 1.0.
10. Provide a short explanation for your decision.

### Query Context

{track_context}

{user_question}

{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result



    # Check 5: Verified templates must be exactly the approved SQL.
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        normalized = lambda q: sqlparse.format(q, strip_comments=True).strip().rstrip(';').strip()
        if normalized(candidate_sql) != normalized(expected_sql):
            result['failed_check'] = 'template_integrity'
            result['details'] = 'Verified query differs from the approved template.'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# Original notebook pipeline function
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
You are a SQL correction assistant for a bank's credit risk query engine.

Your task is to correct a SQL query that failed validation while
preserving the user's original question and intent.

### Instructions
1. Review the original user question carefully.
2. Examine the failed SQL query and the validation error.
3. Identify the issue that caused the query to fail.
4. Use the provided database schema to correct the SQL.
5. Generate a valid, read-only SQLite query.
6. Use only tables and columns that exist in the database schema.
7. Correct any invalid JOIN conditions, filters, aggregations,
   or calculations.
8. Preserve the original business logic and reporting requirements.
9. Do not invent tables, columns, or data.
10. Return only one SQL query beginning with SELECT or WITH.
11. Never use INSERT, UPDATE, DELETE, DROP, ALTER, or other
    database-modifying commands.
12. Do not include explanations, comments, or markdown formatting.
13. If the requested information is unavailable, return a
    read-only query that produces no rows rather than inventing data.

### Original User Question

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# Original notebook pipeline function
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# Original notebook pipeline function
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

You are a credit risk analyst assistant for a bank.

Your task is to answer the user's question using only the SQL query
results provided below.

Instructions:
1. Answer the user's question directly and clearly.
2. Use a professional, concise, business-focused tone.
3. Include relevant figures, categories, and comparisons when available.
4. Only use information contained in the query results.
5. Do not invent numbers, assumptions, or unsupported conclusions.
6. If the results are empty, explain that no matching records were found.
7. Highlight important credit risk findings when relevant.
8. Do not include SQL code or unnecessary technical details.
9. Keep the response focused on the user's original question.

### User Question

{user_question}

{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    # Add deterministic extrema calculated in Python rather than trusting
    # the language model to identify the minimum and maximum correctly.
    if (not dataframe.empty and 'sector_name' in dataframe.columns
            and 'average_interest_rate' in dataframe.columns):
        values = pd.to_numeric(dataframe['average_interest_rate'], errors='coerce')
        if values.notna().any():
            lo, hi = values.idxmin(), values.idxmax()
            verified_summary = (
                f"\n\n**Verified from query data:** Highest average interest rate: "
                f"{dataframe.loc[hi, 'sector_name']} ({values.loc[hi]:.2f}%). "
                f"Lowest: {dataframe.loc[lo, 'sector_name']} ({values.loc[lo]:.2f}%)."
            )
            # For this query, avoid displaying an unverified LLM ranking.
            return verified_summary.strip()
    # Use the actual query results to calculate restructured loan impairment.
    # This prevents the AI response from reporting incorrect totals.
    if (
        not dataframe.empty
        and 'is_restructured' in dataframe.columns
        and 'is_impaired' in dataframe.columns
        and 'loan_account_number' in dataframe.columns
    ):
        restructured = dataframe[
            pd.to_numeric(
                dataframe['is_restructured'], errors='coerce'
            ) == 1
        ]

        if not restructured.empty:
            total_loans = restructured['loan_account_number'].nunique()

            impaired_loans = restructured.loc[
                pd.to_numeric(
                    restructured['is_impaired'], errors='coerce'
                ) == 1,
                'loan_account_number'
            ].nunique()

            impaired_pct = (
                impaired_loans / total_loans * 100
                if total_loans else 0
            )

            if 'restructur' in user_question.lower():
                return (
                    f"Of the {total_loans} restructured loans, "
                    f"{impaired_loans} are classified as impaired, "
                    f"representing {impaired_pct:.2f}% of the total.\n\n"
                    "The individual loan records and impairment flags "
                    "are available in the result data below."
                )

    return narrative


# Original notebook pipeline function
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, prints intermediate pipeline stages.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        print(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        print(f"    Reason: {log['match_reason']}")


    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] not in query_library:
        log['route'] = 'generated'
        log['query_id'] = None
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    if verbose:
        print(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")


    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    if verbose:
        print(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            print(f"    Failed check: {gate.get('failed_check')}")
            print(f"    Details: {gate.get('details')}")


    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            print(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        if verbose:
            print(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                print(f"    Retry failed check: {gate.get('failed_check')}")
                print(f"    Retry details: {gate.get('details')}")


    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            print(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}


    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    try:
        exec_result = execute_query(candidate_sql, db_connection)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        log['escalated'] = True
        log['narrative'] = f"The SQL passed validation but execution failed: {exc}"
        log['confidence'] = 'ESCALATED'
        return {'log': log, 'dataframe': None, **log}
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    if verbose:
        print(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            print(f"    Warnings: {exec_result['warnings']}")


    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative


    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    if verbose:
        print(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# Streamlit interface
st.title("Northbridge Bank | Credit Risk Query Engine")
st.caption("Ask questions about portfolio exposure, provisioning, credit ratings, and loan performance.")
with st.sidebar:
    st.header("Query options")
    st.write("Verified query templates: **10**")
    show_templates = st.checkbox("Show verified query library", value=False)
    if show_templates:
        for qid, entry in verified_query_library.items():
            st.markdown(f"**{qid}** — {entry['description']}")
    st.caption("Queries run against a read-only SQLite connection. Generated SQL is validated before execution.")

examples = [
    "Select an example question…",
    "How much do we have outstanding in each loan category?",
    "How has the Stage 3 book size and its expected credit loss moved over the last four quarters?",
    "Which sectors have the highest total outstanding exposure and NPA exposure?",
    "What is the average interest rate by sector?",
]
example = st.selectbox("Example questions (optional)", examples)
question = st.text_area(
    "Your credit risk question",
    value="" if example == examples[0] else example,
    height=110,
    placeholder="e.g., How have Stage 3 exposures changed over the last four quarters?",
)
run = st.button("Run query", type="primary", disabled=not question.strip())

if run:
    conn = open_read_only_database()
    try:
        trace = StringIO()
        with st.spinner("Classifying, validating, and executing your question…"):
            with redirect_stdout(trace):
                result = run_pipeline(question.strip(), conn, verified_query_library, database_schema)
        st.subheader("Answer")
        if result.get("escalated"):
            st.warning(result.get("narrative", "Query escalated for review."))
        else:
            st.write(result.get("narrative", "No narrative returned."))
        left, middle, right = st.columns(3)
        left.metric("Route", str(result.get("route", "—")).title())
        middle.metric("Query ID", str(result.get("query_id") or "Generated"))
        confidence = result.get("confidence")
        right.metric("Confidence", f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence or "—"))
        df = result.get("dataframe")
        if isinstance(df, pd.DataFrame):
            st.subheader("Result data")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download results as CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="credit_risk_results.csv", mime="text/csv")
        with st.expander("SQL, validation and execution details"):
            st.write("**Classification reason:**", result.get("match_reason"))
            st.write("**Validation gate:**", result.get("gate_result"))
            st.write("**Retry used:**", result.get("retry_used"))
            st.write("**Escalated:**", result.get("escalated"))
            st.write("**Rows returned:**", result.get("row_count"))
            st.code(result.get("executed_sql") or result.get("candidate_sql") or "No SQL generated", language="sql")
            st.text(trace.getvalue())
    except Exception as exc:
        st.error(f"The query could not be completed: {exc}")
    finally:
        conn.close()
