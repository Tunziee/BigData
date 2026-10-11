SELECT
  COUNT(*) AS transaction_item_rows,
  COUNT(DISTINCT customer_id) AS distinct_customers,
  COUNT(DISTINCT article_id) AS distinct_articles,
  COUNT(DISTINCT t_dat) AS distinct_purchase_dates,
  MIN(t_dat) AS min_purchase_date,
  MAX(t_dat) AS max_purchase_date,
  SUM(price) AS monetary_sum,
  AVG(price) AS average_price_per_item_row,
  MIN(price) AS minimum_price,
  MAX(price) AS maximum_price
FROM transactions_clean
