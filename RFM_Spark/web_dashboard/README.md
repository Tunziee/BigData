# Dashboard phân tích khách hàng bán lẻ thời trang

Dashboard Streamlit tiếng Việt này là lớp trình bày bổ sung trên các artifact đã khóa của Stage 1–30. Nó không phải Stage 31, không huấn luyện lại mô hình, không chạy lại FP-Growth và không chạy benchmark Spark.

## Cài đặt

```powershell
cd D:\Doan_bigdata\BigData\BigData\RFM_Spark\web_dashboard
pip install -r requirements.txt
```

## Chạy

```powershell
streamlit run app.py
```

Mở trình duyệt tại: http://localhost:8501

Dashboard cần thư mục `web_dashboard` nằm trực tiếp trong project root `RFM_Spark` để loader tìm được `output/`. Ứng dụng chỉ đọc các artifact nhẹ từ `output/comparison/stage22`, `output/benchmark/stage23–25`, `output/analysis/stage26`, `output/sql/stage27`, `output/fpgrowth/stage28_v2` và `output/visualization/stage29`.

## Chụp màn hình cho báo cáo

1. **Tổng quan**: KPI và pipeline.
2. **So sánh mô hình**: bảng B0/M1/M2, silhouette và SSE/customer.
3. **Phân khúc M2**: phân phối, heatmap/radar và cụm được chọn.
4. **Benchmark Spark**: runtime, CSV/Parquet, cache và shuffle.
5. **FP-Growth**: threshold sensitivity, lọc rules và top rule.

Các ảnh chụp phải được tạo từ dashboard đang chạy; không dùng ảnh minh họa giả.
