SELECT
  DATE_TRUNC('month', t_dat) AS month,
  COUNT(*) AS purchase_item_rows,
  COUNT(DISTINCT customer_id) AS distinct_customers,
  COUNT(DISTINCT article_id) AS distinct_articles,
  SUM(price) AS monetary_sum,
  AVG(price) AS average_price_per_item_row
FROM transactions_clean
GROUP BY DATE_TRUNC('month', t_dat)
ORDER BY month
