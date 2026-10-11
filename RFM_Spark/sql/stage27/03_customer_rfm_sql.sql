WITH customer_rollup AS (
  SELECT
    customer_id,
    MAX(t_dat) AS LastPurchaseDate,
    DATEDIFF(DATE '2020-09-23', MAX(t_dat)) AS Recency,
    COUNT(DISTINCT t_dat) AS Frequency,
    SUM(price) AS Monetary,
    COUNT(*) AS TotalItems,
    COUNT(DISTINCT article_id) AS ProductVariety
  FROM transactions_clean
  GROUP BY customer_id
)
SELECT
  customer_id, LastPurchaseDate, Recency, Frequency, Monetary, TotalItems,
  ProductVariety, CAST(TotalItems AS DOUBLE) / Frequency AS AvgBasketSize
FROM customer_rollup
