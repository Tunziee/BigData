WITH age_banded AS (
  SELECT
    CASE
      WHEN c.`age` IS NULL THEN 'UNKNOWN'
      WHEN CAST(c.`age` AS DOUBLE) < 25 THEN 'UNDER_25'
      WHEN CAST(c.`age` AS DOUBLE) BETWEEN 25 AND 34 THEN 'AGE_25_34'
      WHEN CAST(c.`age` AS DOUBLE) BETWEEN 35 AND 44 THEN 'AGE_35_44'
      WHEN CAST(c.`age` AS DOUBLE) BETWEEN 45 AND 54 THEN 'AGE_45_54'
      WHEN CAST(c.`age` AS DOUBLE) >= 55 THEN 'AGE_55_PLUS'
      ELSE 'UNKNOWN'
    END AS age_band,
    t.customer_id, t.article_id, t.price
  FROM transactions_clean t
  JOIN customers_clean c ON t.customer_id = c.customer_id
)
SELECT
  age_band, COUNT(DISTINCT customer_id) AS customer_count,
  COUNT(*) AS purchase_item_rows, COUNT(DISTINCT article_id) AS distinct_articles,
  SUM(price) AS monetary_sum, AVG(price) AS average_price_per_item_row,
  SUM(price) / COUNT(DISTINCT customer_id) AS average_monetary_per_customer,
  COUNT(*) / COUNT(DISTINCT customer_id) AS average_purchase_rows_per_customer
FROM age_banded
GROUP BY age_band
ORDER BY age_band
