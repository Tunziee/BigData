# Tóm tắt điều hành

Dự án đã hoàn tất pipeline Big Data từ dữ liệu giao dịch, RFM/eRFM, K-Means, benchmark hệ thống, Spark SQL, FP-Growth và lớp trực quan hóa. M2 cải thiện objective eRFM5D theo silhouette và SSE mỗi khách hàng; M1 vẫn mạnh hơn trong RFM3D. Parquet giảm dung lượng rõ ràng, cache phụ thuộc workload, và Spark SQL/FP-Growth bổ sung lớp giải thích kinh doanh.

Các kết luận được giới hạn bởi dữ liệu không có invoice/order ID, runtime local, proxy customer-day basket và việc chưa có A/B test kinh doanh.
