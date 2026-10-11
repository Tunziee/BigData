WITH article_stats AS (
  SELECT
    t.article_id, a.`product_group_name` AS product_group, a.`prod_name` AS product_name,
    COUNT(*) AS purchase_item_rows, COUNT(DISTINCT t.customer_id) AS distinct_customers,
    SUM(t.price) AS monetary_sum
  FROM transactions_clean t
  JOIN articles_clean a ON t.article_id = a.article_id
  GROUP BY t.article_id, a.`product_group_name`, a.`prod_name`
), ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY product_group ORDER BY monetary_sum DESC, article_id) AS product_group_rank
  FROM article_stats
)
SELECT * FROM ranked WHERE product_group_rank <= 5
ORDER BY product_group, product_group_rank
