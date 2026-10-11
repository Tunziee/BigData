# Hướng dẫn tích hợp dashboard vào báo cáo

## Phạm vi và nguyên tắc

Dashboard là lớp trình bày và phân tích bổ sung trên các artifact đã khóa của Stage 1–30. Ứng dụng không tạo thí nghiệm mới, không retrain B0/M1/M2, không rerun FP-Growth, không rerun benchmark Spark và không thay đổi bất kỳ artifact authority nào.

## Kiến trúc

- `web_dashboard/app.py`: cấu hình Streamlit, tiêu đề, sidebar và điểm vào ứng dụng.
- `web_dashboard/pages/`: bảy trang tiếng Việt: Tổng quan; Data & RFM/eRFM; So sánh mô hình; Phân khúc M2; Spark benchmark; Spark SQL; FP-Growth.
- `web_dashboard/components/loaders.py`: loader có `st.cache_data`, chỉ đọc JSON/CSV/Parquet nhẹ.
- `web_dashboard/components/charts.py`: biểu đồ Plotly dùng chung.
- `web_dashboard/components/metrics.py` và `styles.py`: KPI, nhận xét rule-based và kiểu trình bày.
- `web_dashboard/assets/logo.png`: logo trung tính của lớp dashboard.

Ứng dụng không đọc `transactions_train.csv` hoặc full clean transaction Parquet khi mở. PyArrow chỉ được dùng để đọc selected association rules Parquet; nếu môi trường chưa có PyArrow, loader dùng `enriched_top_rules.csv` làm fallback nhẹ cho bảng top rules.

## Bảy trang

1. **Tổng quan**: KPI dataset, M2, FP-Growth và pipeline.
2. **Data & RFM/eRFM**: semantics Frequency là số ngày mua khác nhau; AvgBasketSize là proxy dòng sản phẩm/ngày mua; biểu đồ skewness.
3. **So sánh mô hình**: cấu hình B0/M1/M2, Silhouette, SSE/customer, Entropy, Cluster CV và min/max share; giữ đúng evaluation space.
4. **Phân khúc M2**: phân phối, heatmap, radar, bảng profile và selectbox cụm.
5. **Spark benchmark**: runtime Stage 23, CSV/Parquet Stage 24, cache Stage 25 và shuffle Stage 26.
6. **Spark SQL**: monthly trend, product group, age band và M2 segment analytics từ Stage 27.
7. **FP-Growth**: threshold sensitivity Stage 28 V2, top rules, scatter support–confidence và slider chỉ lọc rules đã sinh.

## Nguồn dữ liệu

- `output/comparison/stage22/`
- `output/benchmark/stage23/`, `stage24/`, `stage25/`
- `output/analysis/stage26/`
- `output/sql/stage27/`
- `output/fpgrowth/stage28_v2/`
- `output/visualization/stage29/`

Các con số dashboard được lấy từ artifact; các nhận xét là deterministic rule-based, không gọi LLM.

## Ảnh chụp đề nghị cho báo cáo

Chụp từ dashboard đang chạy, không sử dụng ảnh giả:

1. Overview với KPI và pipeline.
2. B0/M1/M2 comparison với bảng và improvement panel.
3. M2 segment analysis với heatmap/radar và cụm được chọn.
4. Spark benchmark với runtime và CSV/Parquet/cache/shuffle.
5. FP-Growth với threshold sensitivity, bộ lọc và top rule.

## Đề xuất tiểu mục báo cáo

### 4.x. Giao diện trực quan hóa và hỗ trợ phân tích kết quả

Dashboard web được xây dựng như một lớp trực quan hóa bổ sung cho các kết quả thực nghiệm đã khóa. Ứng dụng không tạo thí nghiệm mới; nó đọc và trực quan hóa các artifact JSON, CSV và Parquet nhẹ từ Stage 22–29. Nhờ đó, người trình bày có thể chuyển giữa tổng quan dữ liệu, semantics RFM/eRFM, so sánh B0/M1/M2, phân khúc M2, benchmark Spark, Spark SQL và FP-Growth trong cùng một giao diện.

Các bộ lọc trong trang FP-Growth chỉ lọc các association rules đã được sinh ở Stage 28 V2, không rerun mô hình. Các nhận xét trên giao diện được sinh bằng quy tắc xác định từ metric khóa, vì vậy không thay thế kiểm định khoa học và không diễn giải association như causality.
