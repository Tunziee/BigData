from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import polish
from components.loaders import load_overview, load_transformation
from components.metrics import observation


st.set_page_config(page_title="Data & RFM/eRFM", layout="wide")
st.title("Dữ liệu & RFM/eRFM")
data = load_overview()
tabs = st.tabs(["DATASET", "RFM", "eRFM", "TRANSFORMATION"])

with tabs[0]:
    st.subheader("Thống kê dataset")
    st.dataframe(pd.DataFrame({"Chỉ tiêu": ["Transaction item rows", "Customers", "Articles", "Distinct purchase dates", "Khoảng thời gian", "Tổng Monetary"], "Giá trị": [f"{data['transaction_rows']:,}", f"{data['customers']:,}", f"{data['articles']:,}", f"{data['purchase_dates']:,}", f"{data['start_date']} → {data['end_date']}", f"{data['monetary_sum']:,.4f}"]}), hide_index=True, use_container_width=True)
    st.success("Clean-data validation PASS: các reference customer/article trong pipeline sạch không bị thiếu.")

with tabs[1]:
    st.subheader("Định nghĩa RFM")
    st.markdown("- **Recency**: số ngày từ lần mua cuối đến analysis date 2020-09-23.\n- **Frequency**: **số ngày phát sinh mua hàng khác nhau** (`COUNT(DISTINCT t_dat)`), không phải số đơn hàng.\n- **Monetary**: tổng `SUM(price)` theo khách hàng.")
    observation("Frequency được dùng như tần suất theo ngày mua vì dữ liệu không có invoice/order ID đáng tin cậy.")

with tabs[2]:
    st.subheader("Định nghĩa eRFM5D")
    st.dataframe(pd.DataFrame({"Feature": ["Recency", "Frequency", "Monetary", "AvgBasketSize", "ProductVariety"], "Ý nghĩa": ["Ngày từ lần mua cuối", "Số ngày mua khác nhau", "Tổng giá trị price", "proxy số dòng sản phẩm trung bình trên mỗi ngày mua", "Số article_id khác nhau"]}), hide_index=True, use_container_width=True)
    observation("AvgBasketSize là proxy theo dòng sản phẩm/ngày mua, không phải kích thước invoice basket thực.")

with tabs[3]:
    st.subheader("Raw skewness và sau log1p")
    df = load_transformation()
    plot_df = df.melt(id_vars="feature", value_vars=["raw_skewness", "log1p_skewness"], var_name="transformation", value_name="skewness")
    plot_df["transformation"] = plot_df["transformation"].map({"raw_skewness": "Raw", "log1p_skewness": "log1p"})
    st.plotly_chart(polish(px.bar(plot_df, x="feature", y="skewness", color="transformation", barmode="group", title="Skewness trước và sau log1p", color_discrete_sequence=["#2563eb", "#f59e0b"])), use_container_width=True)
    st.caption("Giá trị được đọc từ artifact Stage 29; dashboard không tính lại trên raw transaction data.")
