SELECT
  a.`product_group_name` AS product_group,
  COUNT(*) AS purchase_item_rows,
  COUNT(DISTINCT t.customer_id) AS distinct_customers,
  COUNT(DISTINCT t.article_id) AS distinct_articles,
  SUM(t.price) AS monetary_sum,
  AVG(t.price) AS average_price_per_item_row
FROM transactions_clean t
JOIN articles_clean a ON t.article_id = a.article_id
GROUP BY a.`product_group_name`
ORDER BY monetary_sum DESC
