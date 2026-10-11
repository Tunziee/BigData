WITH labeled AS (
  SELECT
    p.customer_id, CAST(p.prediction AS INT) AS prediction,
    r.Recency, r.Frequency, r.Monetary, r.AvgBasketSize, r.ProductVariety,
    CASE CAST(p.prediction AS INT)
      WHEN 0 THEN 'Khách hàng mua ít nhưng giỏ lớn'
      WHEN 1 THEN 'Khách hàng hoạt động giá trị cao'
      WHEN 2 THEN 'Khách hàng lâu chưa quay lại'
      WHEN 3 THEN 'Khách hàng ít tương tác'
      WHEN 4 THEN 'Khách hàng hoạt động thường xuyên'
    END AS segment_name_vi,
    CASE CAST(p.prediction AS INT)
      WHEN 0 THEN 'Infrequent Large-Basket Customers'
      WHEN 1 THEN 'Active High-Value Customers'
      WHEN 2 THEN 'Lapsed Customers'
      WHEN 3 THEN 'Low-Engagement Customers'
      WHEN 4 THEN 'Active Regular Customers'
    END AS segment_name_en
  FROM m2_predictions p
  JOIN rfm_customer r ON p.customer_id = r.customer_id
)
SELECT
  prediction, segment_name_vi, segment_name_en, COUNT(*) AS customer_count,
  COUNT(*) / CAST(SUM(COUNT(*)) OVER () AS DOUBLE) AS customer_ratio,
  AVG(Recency) AS mean_recency, percentile_approx(Recency, 0.5) AS median_recency,
  AVG(Frequency) AS mean_frequency, percentile_approx(Frequency, 0.5) AS median_frequency,
  AVG(Monetary) AS mean_monetary, percentile_approx(Monetary, 0.5) AS median_monetary,
  AVG(AvgBasketSize) AS mean_avg_basket_size, percentile_approx(AvgBasketSize, 0.5) AS median_avg_basket_size,
  AVG(ProductVariety) AS mean_product_variety, percentile_approx(ProductVariety, 0.5) AS median_product_variety
FROM labeled
GROUP BY prediction, segment_name_vi, segment_name_en
ORDER BY prediction
