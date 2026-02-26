import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Power BI + Azure Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');

    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    .hero-banner {
        background: linear-gradient(135deg, #1B2631 0%, #2C3E50 40%, #F2C811 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-banner h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .hero-banner p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem; }

    .kpi-card {
        background: linear-gradient(135deg, #232F3E, #2C3A4A);
        border: 1px solid #3A4A5C;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-top: 3px solid #F2C811;
    }
    .kpi-card .value {
        font-size: 1.8rem; font-weight: 700; color: #F2C811;
    }
    .kpi-card .label {
        font-size: 0.8rem; color: #8899AA; margin-top: 0.3rem;
    }
    .kpi-card .delta {
        font-size: 0.85rem; margin-top: 0.2rem;
    }
    .kpi-card .delta.positive { color: #2ECC71; }
    .kpi-card .delta.negative { color: #E74C3C; }

    .pbi-section {
        background: #232F3E;
        border: 1px solid #3A4A5C;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .azure-card {
        background: linear-gradient(135deg, #232F3E, #1A2535);
        border: 1px solid #0078D4;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .azure-card h4 { color: #0078D4; margin: 0 0 0.5rem 0; }

    .pipeline-step {
        background: #232F3E;
        border-left: 4px solid #F2C811;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .pipeline-step.success { border-left-color: #2ECC71; }
    .pipeline-step.running { border-left-color: #F2C811; }
    .pipeline-step.failed { border-left-color: #E74C3C; }

    .fabric-badge {
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin: 0.2rem;
    }
    .badge-lakehouse { background: #0078D4; color: white; }
    .badge-warehouse { background: #F2C811; color: #1B2631; }
    .badge-notebook { background: #7B2FF2; color: white; }
    .badge-pipeline { background: #2ECC71; color: white; }

    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #232F3E; border-radius: 8px; padding: 8px 16px;
        color: #ccc; border: 1px solid #3A4A5C;
    }
    .stTabs [aria-selected="true"] {
        background: #F2C81122; border-color: #F2C811; color: #F2C811;
    }
</style>
""", unsafe_allow_html=True)

# ── Synthetic Data ───────────────────────────────────────────────────────────
@st.cache_data
def generate_sales_data():
    np.random.seed(42)
    n = 365
    dates = pd.date_range("2025-03-01", periods=n, freq="D")
    regions = ["UAE", "KSA", "Qatar", "Oman", "Bahrain", "Kuwait"]
    products = ["Enterprise Suite", "Analytics Pro", "Cloud Services", "Support Plan", "Training", "Consulting"]

    rows = []
    for d in dates:
        for region in regions:
            for product in products:
                base = np.random.uniform(5000, 50000)
                seasonal = 1 + 0.15 * np.sin(2 * np.pi * d.dayofyear / 365)
                regional_mult = {"UAE": 1.8, "KSA": 1.5, "Qatar": 1.0, "Oman": 0.7, "Bahrain": 0.6, "Kuwait": 0.8}
                product_mult = {"Enterprise Suite": 2.0, "Analytics Pro": 1.5, "Cloud Services": 1.3, "Support Plan": 0.8, "Training": 0.5, "Consulting": 1.1}
                revenue = base * seasonal * regional_mult[region] * product_mult[product] * np.random.uniform(0.85, 1.15)
                cost = revenue * np.random.uniform(0.35, 0.55)
                units = max(1, int(revenue / np.random.uniform(200, 2000)))
                rows.append({
                    "date": d, "region": region, "product": product,
                    "revenue": round(revenue, 2), "cost": round(cost, 2),
                    "profit": round(revenue - cost, 2), "units": units,
                    "customers": max(1, int(units * np.random.uniform(0.3, 0.8))),
                })
    return pd.DataFrame(rows)


@st.cache_data
def generate_pipeline_data():
    np.random.seed(55)
    pipelines = [
        {"name": "Sales ETL Pipeline", "type": "ADF", "schedule": "Daily 02:00 UTC", "source": "Dynamics 365", "dest": "Azure SQL DW"},
        {"name": "CRM Sync Pipeline", "type": "ADF", "schedule": "Every 4 hours", "source": "Salesforce", "dest": "Fabric Lakehouse"},
        {"name": "Web Analytics Ingest", "type": "Fabric", "schedule": "Every 15 min", "source": "Google Analytics API", "dest": "Fabric Lakehouse"},
        {"name": "Financial Data Load", "type": "ADF", "schedule": "Daily 06:00 UTC", "source": "SAP", "dest": "Azure SQL DW"},
        {"name": "HR Data Sync", "type": "Fabric", "schedule": "Daily 01:00 UTC", "source": "Workday API", "dest": "Fabric Lakehouse"},
        {"name": "Inventory Refresh", "type": "ADF", "schedule": "Every 2 hours", "source": "Oracle EBS", "dest": "Azure SQL DW"},
    ]

    runs = []
    for pipe in pipelines:
        for i in range(30):
            date = datetime(2026, 2, 25) - timedelta(days=i)
            duration = np.random.randint(120, 900)
            status = np.random.choice(["Success", "Success", "Success", "Success", "Success", "Warning", "Failed"], p=[0.7, 0.1, 0.05, 0.05, 0.03, 0.05, 0.02])
            rows_processed = np.random.randint(10000, 5000000)
            runs.append({
                **pipe, "date": date, "duration_sec": duration,
                "status": status, "rows_processed": rows_processed,
                "cost_usd": round(duration * 0.002 + rows_processed * 0.000001, 2),
            })
    return pd.DataFrame(runs)


@st.cache_data
def generate_data_model():
    tables = {
        "FactSales": {
            "type": "Fact",
            "columns": ["SalesKey", "DateKey", "ProductKey", "CustomerKey", "RegionKey", "Revenue", "Cost", "Profit", "Units", "DiscountPct"],
            "rows": "12.4M",
            "relationships": ["DimDate", "DimProduct", "DimCustomer", "DimRegion"],
        },
        "FactInventory": {
            "type": "Fact",
            "columns": ["InventoryKey", "DateKey", "ProductKey", "WarehouseKey", "OnHand", "Ordered", "InTransit"],
            "rows": "3.2M",
            "relationships": ["DimDate", "DimProduct", "DimWarehouse"],
        },
        "DimDate": {
            "type": "Dimension",
            "columns": ["DateKey", "Date", "Year", "Quarter", "Month", "MonthName", "WeekOfYear", "DayOfWeek", "IsWeekend", "FiscalYear", "FiscalQuarter"],
            "rows": "3,652",
            "relationships": [],
        },
        "DimProduct": {
            "type": "Dimension",
            "columns": ["ProductKey", "ProductName", "Category", "SubCategory", "Brand", "ListPrice", "StandardCost", "LaunchDate"],
            "rows": "1,240",
            "relationships": [],
        },
        "DimCustomer": {
            "type": "Dimension",
            "columns": ["CustomerKey", "CustomerName", "Company", "Segment", "Industry", "Country", "City", "Tier"],
            "rows": "45,832",
            "relationships": [],
        },
        "DimRegion": {
            "type": "Dimension",
            "columns": ["RegionKey", "Region", "Country", "City", "Territory", "Manager"],
            "rows": "156",
            "relationships": [],
        },
        "DimWarehouse": {
            "type": "Dimension",
            "columns": ["WarehouseKey", "WarehouseName", "Location", "Capacity", "Type"],
            "rows": "24",
            "relationships": [],
        },
    }
    return tables


@st.cache_data
def generate_streaming_data():
    np.random.seed(77)
    n = 200
    timestamps = pd.date_range(end=datetime(2026, 2, 25, 14, 30), periods=n, freq="1min")
    data = pd.DataFrame({
        "timestamp": timestamps,
        "transactions_per_sec": np.random.poisson(450, n) + np.sin(np.linspace(0, 4*np.pi, n)) * 80,
        "avg_latency_ms": np.random.exponential(15, n) + 5,
        "error_rate_pct": np.clip(np.random.exponential(0.3, n), 0, 5),
        "active_sessions": np.random.poisson(12000, n) + np.sin(np.linspace(0, 2*np.pi, n)) * 2000,
        "cpu_pct": np.clip(np.random.normal(45, 12, n), 5, 95),
        "memory_gb": np.clip(np.random.normal(28, 4, n), 10, 60),
    })
    return data


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Power BI + Azure")
    st.markdown("##### Analytics Dashboard")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "1. Executive Dashboard",
            "2. Azure Pipeline Monitor",
            "3. Real-Time Streaming",
            "4. Data Model Explorer",
            "5. AI / ML Insights",
            "6. Architecture & Fabric",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Stack:**")
    st.markdown("- Microsoft Fabric")
    st.markdown("- Azure Data Factory")
    st.markdown("- Power BI Premium")
    st.markdown("- Azure ML")
    st.markdown("- Synapse Analytics")

sales_df = generate_sales_data()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "1. Executive Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <h1>Executive Sales Dashboard</h1>
        <p>Power BI-style analytics — Revenue, profitability, and regional performance across the GCC</p>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_regions = st.multiselect("Region", sales_df["region"].unique(), default=list(sales_df["region"].unique()))
    with col_f2:
        selected_products = st.multiselect("Product", sales_df["product"].unique(), default=list(sales_df["product"].unique()))
    with col_f3:
        date_range = st.date_input("Date Range", value=(sales_df["date"].min(), sales_df["date"].max()))

    # Filter data
    mask = (
        sales_df["region"].isin(selected_regions) &
        sales_df["product"].isin(selected_products)
    )
    if len(date_range) == 2:
        mask &= (sales_df["date"] >= pd.Timestamp(date_range[0])) & (sales_df["date"] <= pd.Timestamp(date_range[1]))
    fdf = sales_df[mask]

    # KPIs
    total_rev = fdf["revenue"].sum()
    total_profit = fdf["profit"].sum()
    total_units = fdf["units"].sum()
    total_customers = fdf["customers"].sum()
    margin = (total_profit / total_rev * 100) if total_rev > 0 else 0
    avg_order = total_rev / total_customers if total_customers > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Revenue", f"${total_rev/1e6:,.1f}M", "+12.3%")
    c2.metric("Gross Profit", f"${total_profit/1e6:,.1f}M", "+8.7%")
    c3.metric("Profit Margin", f"{margin:.1f}%", "+2.1pp")
    c4.metric("Units Sold", f"{total_units:,}", "+15.2%")
    c5.metric("Customers", f"{total_customers:,}", "+9.8%")
    c6.metric("Avg Order Value", f"${avg_order:,.0f}", "+4.5%")

    st.markdown("---")

    # Revenue trend
    col_trend, col_region = st.columns([3, 2])

    with col_trend:
        st.subheader("Revenue & Profit Trend")
        daily = fdf.groupby("date").agg({"revenue": "sum", "profit": "sum", "cost": "sum"}).reset_index()
        daily["revenue_7d"] = daily["revenue"].rolling(7, min_periods=1).mean()
        daily["profit_7d"] = daily["profit"].rolling(7, min_periods=1).mean()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily["date"], y=daily["revenue"], mode="lines",
            name="Revenue", line=dict(color="#F2C811", width=1), opacity=0.3,
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily["date"], y=daily["revenue_7d"], mode="lines",
            name="Revenue (7d avg)", line=dict(color="#F2C811", width=3),
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily["date"], y=daily["profit_7d"], mode="lines",
            name="Profit (7d avg)", line=dict(color="#2ECC71", width=3),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
        ))
        fig_trend.update_layout(
            height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50", tickprefix="$"),
            legend=dict(orientation="h", y=1.1), margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_region:
        st.subheader("Revenue by Region")
        region_rev = fdf.groupby("region")["revenue"].sum().reset_index().sort_values("revenue", ascending=True)
        fig_region = px.bar(
            region_rev, x="revenue", y="region", orientation="h",
            color="revenue", color_continuous_scale=["#232F3E", "#F2C811"],
            labels={"revenue": "Revenue ($)", "region": ""},
        )
        fig_region.update_layout(
            height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"), showlegend=False, coloraxis_showscale=False,
            xaxis=dict(gridcolor="#2C3E50", tickprefix="$"),
            yaxis=dict(gridcolor="#2C3E50"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_region.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
        st.plotly_chart(fig_region, use_container_width=True)

    # Product mix & monthly heatmap
    col_prod, col_heat = st.columns(2)

    with col_prod:
        st.subheader("Product Revenue Mix")
        prod_rev = fdf.groupby("product")["revenue"].sum().reset_index()
        fig_prod = px.pie(
            prod_rev, values="revenue", names="product",
            color_discrete_sequence=["#F2C811", "#0078D4", "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12"],
            hole=0.45,
        )
        fig_prod.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_prod.update_traces(textinfo="label+percent", textfont_size=11)
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_heat:
        st.subheader("Monthly Revenue Heatmap (Region)")
        fdf_copy = fdf.copy()
        fdf_copy["month"] = fdf_copy["date"].dt.to_period("M").astype(str)
        heat = fdf_copy.groupby(["region", "month"])["revenue"].sum().reset_index()
        heat_pivot = heat.pivot(index="region", columns="month", values="revenue").fillna(0)

        fig_heat = px.imshow(
            heat_pivot.values,
            x=list(heat_pivot.columns), y=list(heat_pivot.index),
            color_continuous_scale=["#1B2631", "#0078D4", "#F2C811"],
            labels=dict(color="Revenue"),
        )
        fig_heat.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Top customers table
    st.markdown("---")
    st.subheader("Top Performing Segments")
    seg = fdf.groupby(["region", "product"]).agg(
        revenue=("revenue", "sum"),
        profit=("profit", "sum"),
        units=("units", "sum"),
        customers=("customers", "sum"),
    ).reset_index()
    seg["margin"] = (seg["profit"] / seg["revenue"] * 100).round(1)
    seg = seg.sort_values("revenue", ascending=False).head(15)
    seg["revenue"] = seg["revenue"].apply(lambda x: f"${x:,.0f}")
    seg["profit"] = seg["profit"].apply(lambda x: f"${x:,.0f}")
    seg["margin"] = seg["margin"].apply(lambda x: f"{x}%")
    st.dataframe(seg.rename(columns={"region": "Region", "product": "Product", "revenue": "Revenue", "profit": "Profit", "units": "Units", "customers": "Customers", "margin": "Margin"}), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: AZURE PIPELINE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "2. Azure Pipeline Monitor":
    st.markdown("""
    <div class="hero-banner">
        <h1>Azure Data Factory Pipeline Monitor</h1>
        <p>Real-time monitoring of ETL pipelines, data freshness, and processing costs</p>
    </div>
    """, unsafe_allow_html=True)

    pipe_df = generate_pipeline_data()

    # Summary KPIs
    recent = pipe_df[pipe_df["date"] >= datetime(2026, 2, 20)]
    total_runs = len(recent)
    success_rate = (recent["status"] == "Success").sum() / total_runs * 100
    total_rows = recent["rows_processed"].sum()
    total_cost = recent["cost_usd"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline Runs (7d)", total_runs)
    c2.metric("Success Rate", f"{success_rate:.1f}%")
    c3.metric("Rows Processed", f"{total_rows/1e6:,.1f}M")
    c4.metric("Pipeline Cost", f"${total_cost:,.2f}")

    st.markdown("---")

    # Pipeline status grid
    st.subheader("Pipeline Status Overview")

    pipeline_names = pipe_df["name"].unique()
    cols = st.columns(3)

    for i, pname in enumerate(pipeline_names):
        p_recent = pipe_df[(pipe_df["name"] == pname)].sort_values("date", ascending=False).iloc[0]
        with cols[i % 3]:
            status_color = {"Success": "#2ECC71", "Warning": "#F2C811", "Failed": "#E74C3C"}.get(p_recent["status"], "#888")
            st.markdown(f"""
            <div class="pipeline-step {'success' if p_recent['status']=='Success' else ('running' if p_recent['status']=='Warning' else 'failed')}">
                <strong>{pname}</strong><br>
                <span style="font-size: 0.8rem; color: #8899AA;">
                    {p_recent['type']} | {p_recent['schedule']}<br>
                    Last: <span style="color: {status_color};">{p_recent['status']}</span> |
                    {p_recent['duration_sec']}s |
                    {p_recent['rows_processed']:,} rows
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline run history
    st.subheader("Run History (Last 30 Days)")

    selected_pipe = st.selectbox("Select Pipeline", pipeline_names)
    pipe_hist = pipe_df[pipe_df["name"] == selected_pipe].sort_values("date")

    fig_hist = go.Figure()
    color_map = {"Success": "#2ECC71", "Warning": "#F2C811", "Failed": "#E74C3C"}
    for status in ["Success", "Warning", "Failed"]:
        mask = pipe_hist["status"] == status
        fig_hist.add_trace(go.Bar(
            x=pipe_hist[mask]["date"], y=pipe_hist[mask]["duration_sec"],
            name=status, marker_color=color_map[status],
            hovertemplate="Date: %{x}<br>Duration: %{y}s<br>Status: " + status,
        ))

    fig_hist.update_layout(
        barmode="stack", height=350,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAECEE"),
        xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50", title="Duration (seconds)"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Throughput and cost
    col_tp, col_cost = st.columns(2)

    with col_tp:
        st.subheader("Rows Processed per Run")
        fig_rows = px.scatter(
            pipe_hist, x="date", y="rows_processed", color="status",
            size="duration_sec", color_discrete_map=color_map,
            labels={"rows_processed": "Rows", "date": "Date"},
        )
        fig_rows.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50"),
        )
        st.plotly_chart(fig_rows, use_container_width=True)

    with col_cost:
        st.subheader("Pipeline Cost Trend")
        cost_daily = pipe_df.groupby("date")["cost_usd"].sum().reset_index().sort_values("date")
        fig_cost = px.area(
            cost_daily, x="date", y="cost_usd",
            color_discrete_sequence=["#F2C811"],
            labels={"cost_usd": "Cost ($)", "date": "Date"},
        )
        fig_cost.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50", tickprefix="$"),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    # ADF Pipeline JSON definition
    st.markdown("---")
    st.subheader("ADF Pipeline Definition (ARM Template)")
    st.code("""{
  "name": "SalesETLPipeline",
  "properties": {
    "activities": [
      {
        "name": "CopyFromDynamics365",
        "type": "Copy",
        "inputs": [{"referenceName": "Dynamics365Source", "type": "DatasetReference"}],
        "outputs": [{"referenceName": "ADLSRawLanding", "type": "DatasetReference"}],
        "typeProperties": {
          "source": {"type": "DynamicsCrmSource", "query": "fetchxml_query"},
          "sink": {"type": "ParquetSink", "storeSettings": {"type": "AzureBlobFSWriteSettings"}},
          "enableStaging": true,
          "parallelCopies": 8
        }
      },
      {
        "name": "TransformInDatabricks",
        "type": "DatabricksSparkPython",
        "dependsOn": [{"activity": "CopyFromDynamics365", "dependencyConditions": ["Succeeded"]}],
        "typeProperties": {
          "pythonFile": "dbfs:/etl/transform_sales.py",
          "parameters": ["--date", "@pipeline().TriggerTime"]
        }
      },
      {
        "name": "LoadToSynapseWarehouse",
        "type": "Copy",
        "dependsOn": [{"activity": "TransformInDatabricks", "dependencyConditions": ["Succeeded"]}],
        "typeProperties": {
          "source": {"type": "ParquetSource"},
          "sink": {"type": "SqlDWSink", "writeBehavior": "Upsert", "upsertSettings": {"keys": ["SalesKey"]}},
          "enableStaging": true
        }
      },
      {
        "name": "RefreshPowerBIDataset",
        "type": "WebActivity",
        "dependsOn": [{"activity": "LoadToSynapseWarehouse", "dependencyConditions": ["Succeeded"]}],
        "typeProperties": {
          "url": "https://api.powerbi.com/v1.0/myorg/datasets/{datasetId}/refreshes",
          "method": "POST",
          "authentication": {"type": "MSI", "resource": "https://analysis.windows.net/powerbi/api"}
        }
      }
    ],
    "annotations": ["production", "sales", "daily"]
  }
}""", language="json")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: REAL-TIME STREAMING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "3. Real-Time Streaming":
    st.markdown("""
    <div class="hero-banner">
        <h1>Real-Time Streaming Monitor</h1>
        <p>Live KPIs, system health, and streaming analytics from Azure Event Hub / Stream Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    stream_df = generate_streaming_data()

    # Real-time KPIs
    latest = stream_df.iloc[-1]
    avg_30m = stream_df.tail(30).mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("TPS", f"{latest['transactions_per_sec']:.0f}", f"{latest['transactions_per_sec'] - avg_30m['transactions_per_sec']:+.0f}")
    c2.metric("Avg Latency", f"{latest['avg_latency_ms']:.1f}ms", f"{latest['avg_latency_ms'] - avg_30m['avg_latency_ms']:+.1f}ms", delta_color="inverse")
    c3.metric("Error Rate", f"{latest['error_rate_pct']:.2f}%", f"{latest['error_rate_pct'] - avg_30m['error_rate_pct']:+.2f}%", delta_color="inverse")
    c4.metric("Active Sessions", f"{latest['active_sessions']:,.0f}", f"{latest['active_sessions'] - avg_30m['active_sessions']:+,.0f}")
    c5.metric("CPU Usage", f"{latest['cpu_pct']:.1f}%")
    c6.metric("Memory", f"{latest['memory_gb']:.1f} GB")

    st.markdown("---")

    # Live charts
    col_tps, col_lat = st.columns(2)

    with col_tps:
        st.subheader("Transactions Per Second")
        fig_tps = go.Figure()
        fig_tps.add_trace(go.Scatter(
            x=stream_df["timestamp"], y=stream_df["transactions_per_sec"],
            mode="lines", fill="tozeroy",
            line=dict(color="#F2C811", width=2),
            fillcolor="rgba(242,200,17,0.15)",
        ))
        fig_tps.add_hline(y=500, line_dash="dash", line_color="#E74C3C", annotation_text="SLA Threshold")
        fig_tps.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50", title="TPS"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_tps, use_container_width=True)

    with col_lat:
        st.subheader("Response Latency (ms)")
        fig_lat = go.Figure()
        fig_lat.add_trace(go.Scatter(
            x=stream_df["timestamp"], y=stream_df["avg_latency_ms"],
            mode="lines", fill="tozeroy",
            line=dict(color="#0078D4", width=2),
            fillcolor="rgba(0,120,212,0.15)",
        ))
        fig_lat.add_hline(y=50, line_dash="dash", line_color="#E74C3C", annotation_text="SLA: 50ms")
        fig_lat.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50", title="Latency (ms)"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    # System health gauges
    st.subheader("System Health")
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)

    gauges = [
        (col_g1, "CPU Utilization", latest["cpu_pct"], "%"),
        (col_g2, "Memory Usage", latest["memory_gb"] / 64 * 100, "%"),
        (col_g3, "Error Rate", latest["error_rate_pct"], "%"),
        (col_g4, "SLA Compliance", 99.7, "%"),
    ]

    for col, title, val, suffix in gauges:
        with col:
            color = "#2ECC71" if val < 70 else ("#F2C811" if val < 85 else "#E74C3C")
            if title == "Error Rate":
                color = "#2ECC71" if val < 1 else ("#F2C811" if val < 3 else "#E74C3C")
            if title == "SLA Compliance":
                color = "#2ECC71" if val > 99 else ("#F2C811" if val > 95 else "#E74C3C")

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": title, "font": {"size": 14, "color": "#EAECEE"}},
                number={"suffix": suffix, "font": {"size": 20, "color": "#EAECEE"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#555"},
                    "bar": {"color": color},
                    "bgcolor": "#232F3E",
                    "bordercolor": "#3A4A5C",
                    "steps": [
                        {"range": [0, 70], "color": "#1B2631"},
                        {"range": [70, 85], "color": "#2C3E50"},
                        {"range": [85, 100], "color": "#3A4A5C"},
                    ],
                },
            ))
            fig_g.update_layout(
                height=200, paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EAECEE"), margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_g, use_container_width=True)

    # Error rate and sessions
    st.markdown("---")
    col_err, col_sess = st.columns(2)

    with col_err:
        st.subheader("Error Rate Over Time")
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=stream_df["timestamp"], y=stream_df["error_rate_pct"],
            mode="lines", fill="tozeroy",
            line=dict(color="#E74C3C", width=2),
            fillcolor="rgba(231,76,60,0.15)",
        ))
        fig_err.update_layout(
            height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50", title="Error %"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_err, use_container_width=True)

    with col_sess:
        st.subheader("Active Sessions")
        fig_sess = go.Figure()
        fig_sess.add_trace(go.Scatter(
            x=stream_df["timestamp"], y=stream_df["active_sessions"],
            mode="lines", fill="tozeroy",
            line=dict(color="#9B59B6", width=2),
            fillcolor="rgba(155,89,182,0.15)",
        ))
        fig_sess.update_layout(
            height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50", title="Sessions"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_sess, use_container_width=True)

    # Stream Analytics query
    st.subheader("Azure Stream Analytics Query")
    st.code("""-- Real-time anomaly detection on streaming data
WITH AnomalyDetection AS (
    SELECT
        System.Timestamp() AS WindowEnd,
        IoTHub.ConnectionDeviceId AS DeviceId,
        AVG(transactions_per_sec) AS AvgTPS,
        MAX(avg_latency_ms) AS MaxLatency,
        AVG(error_rate_pct) AS AvgErrorRate,
        COUNT(*) AS EventCount,
        AnomalyDetection_SpikeAndDip(
            CAST(error_rate_pct AS float), 95, 120, 'spikesanddips'
        ) OVER (LIMIT DURATION(minute, 10)) AS SpikeScore
    FROM
        [eventhub-input] TIMESTAMP BY EventEnqueuedUtcTime
    GROUP BY
        IoTHub.ConnectionDeviceId,
        TumblingWindow(minute, 1)
)
SELECT * INTO [powerbi-output]
FROM AnomalyDetection
WHERE SpikeScore.IsAnomaly = 1
   OR AvgErrorRate > 2.0
   OR MaxLatency > 100;""", language="sql")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: DATA MODEL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "4. Data Model Explorer":
    st.markdown("""
    <div class="hero-banner">
        <h1>Power BI Data Model Explorer</h1>
        <p>Star schema design, DAX measures, and relationship mapping</p>
    </div>
    """, unsafe_allow_html=True)

    model = generate_data_model()

    # Star schema visualization
    st.subheader("Star Schema Diagram")

    # Build the star schema as a Sankey-style diagram
    fact_tables = [k for k, v in model.items() if v["type"] == "Fact"]
    dim_tables = [k for k, v in model.items() if v["type"] == "Dimension"]

    labels = fact_tables + dim_tables
    source_idx, target_idx, values = [], [], []

    for i, fact in enumerate(fact_tables):
        for rel in model[fact]["relationships"]:
            if rel in labels:
                j = labels.index(rel)
                source_idx.append(j)
                target_idx.append(i)
                values.append(1)

    colors = ["#F2C811"] * len(fact_tables) + ["#0078D4"] * len(dim_tables)

    fig_star = go.Figure(go.Sankey(
        node=dict(pad=20, thickness=30, label=labels, color=colors),
        link=dict(source=source_idx, target=target_idx, value=values, color="rgba(242,200,17,0.2)"),
    ))
    fig_star.update_layout(
        height=400, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAECEE", size=13),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_star, use_container_width=True)

    # Table explorer
    st.markdown("---")
    st.subheader("Table Details")

    sel_table = st.selectbox("Select Table", list(model.keys()))
    tbl = model[sel_table]

    col_meta, col_cols = st.columns([1, 2])

    with col_meta:
        tbl_color = "#F2C811" if tbl["type"] == "Fact" else "#0078D4"
        st.markdown(f"""
        <div style="background: #232F3E; border: 2px solid {tbl_color}; border-radius: 12px; padding: 1.5rem;">
            <h3 style="color: {tbl_color}; margin-top: 0;">{sel_table}</h3>
            <p><strong>Type:</strong> <span style="color: {tbl_color};">{tbl['type']} Table</span></p>
            <p><strong>Rows:</strong> {tbl['rows']}</p>
            <p><strong>Columns:</strong> {len(tbl['columns'])}</p>
            <p><strong>Relationships:</strong> {len(tbl['relationships'])}</p>
        </div>
        """, unsafe_allow_html=True)

        if tbl["relationships"]:
            st.markdown("**Related Tables:**")
            for rel in tbl["relationships"]:
                st.markdown(f"- `{sel_table}` → `{rel}`")

    with col_cols:
        col_df = pd.DataFrame({"Column": tbl["columns"]})
        col_df["Key Type"] = col_df["Column"].apply(
            lambda c: "PK" if c.endswith("Key") and c == tbl["columns"][0]
            else ("FK" if c.endswith("Key") else "Attribute")
        )
        col_df["Data Type"] = col_df["Column"].apply(
            lambda c: "INT/BIGINT" if c.endswith("Key") or c.endswith("Id")
            else ("DECIMAL" if c in ["Revenue", "Cost", "Profit", "ListPrice", "StandardCost", "UnitPrice", "DiscountPct"]
                  else ("DATE" if c in ["Date", "LaunchDate"] else ("BIT" if c.startswith("Is") else "NVARCHAR")))
        )
        st.dataframe(col_df, use_container_width=True, hide_index=True, height=400)

    # DAX Measures
    st.markdown("---")
    st.subheader("DAX Measures Library")

    tabs = st.tabs(["Revenue Measures", "Time Intelligence", "KPI Measures", "Advanced"])

    with tabs[0]:
        st.code("""// Total Revenue
Total Revenue =
    SUM(FactSales[Revenue])

// Total Profit
Total Profit =
    SUM(FactSales[Profit])

// Profit Margin %
Profit Margin % =
    DIVIDE([Total Profit], [Total Revenue], 0)

// Revenue per Customer
Revenue per Customer =
    DIVIDE(
        [Total Revenue],
        DISTINCTCOUNT(FactSales[CustomerKey]),
        0
    )

// Average Order Value
Avg Order Value =
    DIVIDE(
        [Total Revenue],
        DISTINCTCOUNT(FactSales[SalesKey]),
        0
    )""", language="dax")

    with tabs[1]:
        st.code("""// YTD Revenue
YTD Revenue =
    TOTALYTD([Total Revenue], DimDate[Date])

// Same Period Last Year
Revenue SPLY =
    CALCULATE(
        [Total Revenue],
        SAMEPERIODLASTYEAR(DimDate[Date])
    )

// Year-over-Year Growth %
YoY Growth % =
    VAR CurrentPeriod = [Total Revenue]
    VAR PriorPeriod = [Revenue SPLY]
    RETURN
        DIVIDE(CurrentPeriod - PriorPeriod, PriorPeriod, 0)

// Moving Average (3 Month)
Revenue 3M MA =
    AVERAGEX(
        DATESINPERIOD(DimDate[Date], MAX(DimDate[Date]), -3, MONTH),
        [Total Revenue]
    )

// Quarter-to-Date
QTD Revenue =
    TOTALQTD([Total Revenue], DimDate[Date])""", language="dax")

    with tabs[2]:
        st.code("""// Revenue Target Achievement
Target Achievement % =
    DIVIDE([Total Revenue], [Revenue Target], 0)

// Revenue Traffic Light
Revenue Status =
    VAR Achievement = [Target Achievement %]
    RETURN
        SWITCH(
            TRUE(),
            Achievement >= 1.0, "Green",
            Achievement >= 0.85, "Amber",
            "Red"
        )

// Customer Churn Rate
Churn Rate =
    VAR ActiveLastPeriod =
        CALCULATE(
            DISTINCTCOUNT(FactSales[CustomerKey]),
            PREVIOUSMONTH(DimDate[Date])
        )
    VAR LostCustomers =
        ActiveLastPeriod -
        CALCULATE(
            DISTINCTCOUNT(FactSales[CustomerKey]),
            INTERSECT(
                VALUES(FactSales[CustomerKey]),
                CALCULATETABLE(VALUES(FactSales[CustomerKey]), PREVIOUSMONTH(DimDate[Date]))
            )
        )
    RETURN
        DIVIDE(LostCustomers, ActiveLastPeriod, 0)""", language="dax")

    with tabs[3]:
        st.code("""// Pareto Analysis - Top 20% of customers
Top 20% Customer Revenue =
    VAR CustomerRevenue =
        ADDCOLUMNS(
            VALUES(DimCustomer[CustomerKey]),
            "@Rev", [Total Revenue]
        )
    VAR RankedCustomers =
        ADDCOLUMNS(
            CustomerRevenue,
            "@Rank", RANKX(CustomerRevenue, [@Rev])
        )
    VAR TotalCustomers = COUNTROWS(CustomerRevenue)
    VAR Top20Pct = ROUNDUP(TotalCustomers * 0.2, 0)
    RETURN
        SUMX(
            FILTER(RankedCustomers, [@Rank] <= Top20Pct),
            [@Rev]
        )

// Dynamic Segmentation
Customer Segment =
    VAR CustRevenue = [Revenue per Customer]
    RETURN
        SWITCH(
            TRUE(),
            CustRevenue >= PERCENTILEX.INC(ALL(DimCustomer), [Revenue per Customer], 0.9), "Platinum",
            CustRevenue >= PERCENTILEX.INC(ALL(DimCustomer), [Revenue per Customer], 0.7), "Gold",
            CustRevenue >= PERCENTILEX.INC(ALL(DimCustomer), [Revenue per Customer], 0.4), "Silver",
            "Bronze"
        )""", language="dax")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: AI / ML INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "5. AI / ML Insights":
    st.markdown("""
    <div class="hero-banner">
        <h1>AI & ML-Powered Insights</h1>
        <p>Forecasting, anomaly detection, and customer segmentation powered by Azure ML</p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Revenue Forecast", "Anomaly Detection", "Customer Segmentation", "Key Influencers"])

    with tabs[0]:
        st.subheader("Revenue Forecast (Next 90 Days)")

        daily = sales_df.groupby("date")["revenue"].sum().reset_index()

        # Simple forecast: trend + seasonality
        np.random.seed(42)
        last_date = daily["date"].max()
        forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=90, freq="D")
        trend = np.linspace(daily["revenue"].iloc[-7:].mean(), daily["revenue"].iloc[-7:].mean() * 1.08, 90)
        seasonality = np.sin(np.linspace(0, 2 * np.pi, 90)) * daily["revenue"].std() * 0.3
        noise = np.random.normal(0, daily["revenue"].std() * 0.15, 90)
        forecast = trend + seasonality + noise
        upper = forecast * 1.15
        lower = forecast * 0.85

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=daily["date"], y=daily["revenue"],
            mode="lines", name="Actual",
            line=dict(color="#F2C811", width=2),
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast_dates, y=forecast,
            mode="lines", name="Forecast",
            line=dict(color="#0078D4", width=2, dash="dash"),
        ))
        fig_fc.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself", name="95% CI",
            fillcolor="rgba(0,120,212,0.15)",
            line=dict(width=0),
        ))
        fig_fc.update_layout(
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50", tickprefix="$"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("30-Day Forecast", f"${forecast[:30].sum()/1e6:,.1f}M")
        fc2.metric("60-Day Forecast", f"${forecast[:60].sum()/1e6:,.1f}M")
        fc3.metric("90-Day Forecast", f"${forecast.sum()/1e6:,.1f}M")

    with tabs[1]:
        st.subheader("Revenue Anomaly Detection")

        daily = sales_df.groupby("date")["revenue"].sum().reset_index()
        daily["rolling_mean"] = daily["revenue"].rolling(14, min_periods=1).mean()
        daily["rolling_std"] = daily["revenue"].rolling(14, min_periods=1).std().fillna(0)
        daily["upper"] = daily["rolling_mean"] + 2 * daily["rolling_std"]
        daily["lower"] = daily["rolling_mean"] - 2 * daily["rolling_std"]
        daily["anomaly"] = (daily["revenue"] > daily["upper"]) | (daily["revenue"] < daily["lower"])

        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=daily["date"], y=daily["revenue"],
            mode="lines", name="Revenue",
            line=dict(color="#F2C811", width=1.5),
        ))
        fig_anom.add_trace(go.Scatter(
            x=daily["date"], y=daily["rolling_mean"],
            mode="lines", name="14-Day Mean",
            line=dict(color="#0078D4", width=2),
        ))
        fig_anom.add_trace(go.Scatter(
            x=list(daily["date"]) + list(daily["date"][::-1]),
            y=list(daily["upper"]) + list(daily["lower"][::-1]),
            fill="toself", name="2-sigma band",
            fillcolor="rgba(0,120,212,0.1)",
            line=dict(width=0),
        ))
        anomalies = daily[daily["anomaly"]]
        fig_anom.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies["revenue"],
            mode="markers", name="Anomaly",
            marker=dict(color="#E74C3C", size=10, symbol="diamond"),
        ))
        fig_anom.update_layout(
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50"), yaxis=dict(gridcolor="#2C3E50", tickprefix="$"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_anom, use_container_width=True)

        st.info(f"Detected **{len(anomalies)}** anomalous days out of {len(daily)} ({len(anomalies)/len(daily)*100:.1f}%)")

    with tabs[2]:
        st.subheader("Customer Segmentation (K-Means Clustering)")

        np.random.seed(42)
        n_cust = 500
        segments = {
            "Platinum": {"n": 50, "rev_mean": 80000, "freq_mean": 25, "recency_mean": 5},
            "Gold": {"n": 120, "rev_mean": 40000, "freq_mean": 15, "recency_mean": 15},
            "Silver": {"n": 180, "rev_mean": 15000, "freq_mean": 8, "recency_mean": 30},
            "Bronze": {"n": 150, "rev_mean": 3000, "freq_mean": 2, "recency_mean": 60},
        }

        cust_rows = []
        for seg_name, params in segments.items():
            for _ in range(params["n"]):
                cust_rows.append({
                    "segment": seg_name,
                    "total_revenue": max(100, np.random.normal(params["rev_mean"], params["rev_mean"] * 0.3)),
                    "order_frequency": max(1, int(np.random.normal(params["freq_mean"], params["freq_mean"] * 0.3))),
                    "days_since_last": max(1, int(np.random.normal(params["recency_mean"], params["recency_mean"] * 0.4))),
                    "avg_order_value": max(50, np.random.normal(params["rev_mean"] / max(1, params["freq_mean"]), 500)),
                })
        cust_df = pd.DataFrame(cust_rows)

        fig_seg = px.scatter(
            cust_df, x="total_revenue", y="order_frequency",
            color="segment", size="avg_order_value",
            color_discrete_map={"Platinum": "#F2C811", "Gold": "#F39C12", "Silver": "#BDC3C7", "Bronze": "#8B4513"},
            labels={"total_revenue": "Total Revenue ($)", "order_frequency": "Order Frequency", "segment": "Segment"},
        )
        fig_seg.update_layout(
            height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50", tickprefix="$"), yaxis=dict(gridcolor="#2C3E50"),
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        # Segment summary
        seg_summary = cust_df.groupby("segment").agg(
            customers=("segment", "count"),
            avg_revenue=("total_revenue", "mean"),
            avg_frequency=("order_frequency", "mean"),
            avg_recency=("days_since_last", "mean"),
        ).reset_index()
        seg_summary["avg_revenue"] = seg_summary["avg_revenue"].apply(lambda x: f"${x:,.0f}")
        seg_summary["avg_frequency"] = seg_summary["avg_frequency"].apply(lambda x: f"{x:.1f}")
        seg_summary["avg_recency"] = seg_summary["avg_recency"].apply(lambda x: f"{x:.0f} days")
        st.dataframe(seg_summary.rename(columns={
            "segment": "Segment", "customers": "Count",
            "avg_revenue": "Avg Revenue", "avg_frequency": "Avg Frequency",
            "avg_recency": "Avg Recency",
        }), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Key Influencers Analysis")
        st.markdown("*Inspired by Power BI's AI-powered Key Influencers visual*")

        influencers = pd.DataFrame({
            "Factor": [
                "Region = UAE",
                "Product = Enterprise Suite",
                "Customer Segment = Platinum",
                "Quarter = Q4",
                "Sales Channel = Direct",
                "Deal Size > $50k",
                "Region = KSA",
                "Product = Analytics Pro",
            ],
            "Impact on Revenue": ["+42%", "+35%", "+28%", "+18%", "+15%", "+12%", "+10%", "+8%"],
            "Confidence": [0.95, 0.92, 0.89, 0.85, 0.82, 0.78, 0.75, 0.71],
            "Direction": ["Increases", "Increases", "Increases", "Increases", "Increases", "Increases", "Increases", "Increases"],
        })

        fig_inf = px.bar(
            influencers, x="Confidence", y="Factor",
            orientation="h", color="Confidence",
            color_continuous_scale=["#232F3E", "#F2C811"],
            text="Impact on Revenue",
        )
        fig_inf.update_layout(
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#EAECEE"),
            xaxis=dict(gridcolor="#2C3E50", title="Confidence Score"),
            yaxis=dict(gridcolor="#2C3E50", autorange="reversed"),
            showlegend=False, coloraxis_showscale=False,
        )
        fig_inf.update_traces(textposition="outside")
        st.plotly_chart(fig_inf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: ARCHITECTURE & FABRIC
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "6. Architecture & Fabric":
    st.markdown("""
    <div class="hero-banner">
        <h1>Architecture & Microsoft Fabric</h1>
        <p>End-to-end data platform architecture with Microsoft Fabric, Azure Data Factory, and Power BI</p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture diagram (Sankey)
    st.subheader("End-to-End Architecture")

    arch_labels = [
        # Sources (0-4)
        "Dynamics 365", "Salesforce", "SAP", "Google Analytics", "Workday",
        # Ingestion (5-6)
        "Azure Data Factory", "Event Hub",
        # Storage (7-9)
        "ADLS Gen2 (Raw)", "Fabric Lakehouse", "Azure SQL DW",
        # Transform (10-11)
        "Databricks", "Fabric Notebooks",
        # Serve (12-14)
        "Power BI Premium", "Synapse Serverless", "REST APIs",
        # Consume (15-17)
        "Executive Dashboards", "Self-Service Analytics", "Mobile Reports",
    ]

    arch_source = [0, 1, 2, 3, 4, 5, 5, 6, 7, 7, 8, 8, 10, 11, 9, 9, 12, 12, 12]
    arch_target = [5, 5, 5, 6, 5, 7, 8, 8, 10, 11, 9, 12, 9, 9, 12, 14, 15, 16, 17]
    arch_values = [3, 2, 3, 1, 1, 4, 3, 1, 3, 2, 5, 3, 5, 2, 5, 1, 4, 3, 2]

    node_colors = (
        ["#E74C3C"] * 5 +  # Sources
        ["#0078D4"] * 2 +  # Ingestion
        ["#2ECC71"] * 3 +  # Storage
        ["#9B59B6"] * 2 +  # Transform
        ["#F2C811"] * 3 +  # Serve
        ["#F39C12"] * 3    # Consume
    )

    fig_arch = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=25,
            label=arch_labels,
            color=node_colors,
        ),
        link=dict(
            source=arch_source, target=arch_target, value=arch_values,
            color="rgba(242,200,17,0.15)",
        ),
    ))
    fig_arch.update_layout(
        height=500, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAECEE", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_arch, use_container_width=True)

    # Layer descriptions
    st.markdown("---")
    st.subheader("Architecture Layers")

    layer_cols = st.columns(3)

    with layer_cols[0]:
        st.markdown("""
        <div class="azure-card">
            <h4>Data Sources</h4>
            <ul>
                <li><strong>Dynamics 365:</strong> CRM & ERP data</li>
                <li><strong>Salesforce:</strong> Sales pipeline data</li>
                <li><strong>SAP:</strong> Financial transactions</li>
                <li><strong>Google Analytics:</strong> Web traffic</li>
                <li><strong>Workday:</strong> HR & payroll</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="azure-card">
            <h4>Ingestion Layer</h4>
            <ul>
                <li><strong>ADF Pipelines:</strong> Batch ETL (daily, hourly)</li>
                <li><strong>Event Hub:</strong> Real-time streaming</li>
                <li><strong>Fabric Shortcuts:</strong> Zero-copy references</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with layer_cols[1]:
        st.markdown("""
        <div class="azure-card">
            <h4>Storage & Processing</h4>
            <ul>
                <li><strong>ADLS Gen2:</strong> Raw data lake (Parquet/Delta)</li>
                <li><strong>Fabric Lakehouse:</strong> Unified analytics</li>
                <li><strong>Azure SQL DW:</strong> Star schema warehouse</li>
                <li><strong>Databricks:</strong> Complex transformations</li>
                <li><strong>Fabric Notebooks:</strong> PySpark processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with layer_cols[2]:
        st.markdown("""
        <div class="azure-card">
            <h4>Serving & Consumption</h4>
            <ul>
                <li><strong>Power BI Premium:</strong> Enterprise dashboards</li>
                <li><strong>Synapse Serverless:</strong> Ad-hoc queries</li>
                <li><strong>REST APIs:</strong> External integrations</li>
                <li><strong>Mobile Reports:</strong> On-the-go access</li>
                <li><strong>Embedded Analytics:</strong> Customer-facing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Microsoft Fabric components
    st.markdown("---")
    st.subheader("Microsoft Fabric Workloads")

    fabric_items = [
        ("Lakehouse", "Unified storage for structured and unstructured data with T-SQL and Spark access", "badge-lakehouse"),
        ("Data Warehouse", "Enterprise SQL warehouse with full T-SQL support and automatic optimization", "badge-warehouse"),
        ("Notebooks", "PySpark/Python notebooks for data engineering and data science", "badge-notebook"),
        ("Data Pipelines", "Orchestration with Copy, Dataflow Gen2, and Notebook activities", "badge-pipeline"),
    ]

    for name, desc, badge_class in fabric_items:
        st.markdown(f"""
        <div style="background: #232F3E; border: 1px solid #3A4A5C; border-radius: 8px; padding: 1rem; margin-bottom: 0.6rem;">
            <span class="fabric-badge {badge_class}">{name}</span>
            <span style="margin-left: 1rem; color: #AABBCC;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    # Fabric Notebook example
    st.markdown("---")
    st.subheader("Fabric Notebook — Sales Transformation")
    st.code("""# Microsoft Fabric Notebook — PySpark
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Read from Lakehouse (Delta format)
raw_sales = spark.read.format("delta").load("Tables/raw_sales")

# Bronze → Silver: Clean and standardize
silver_sales = (
    raw_sales
    .filter(F.col("revenue").isNotNull() & (F.col("revenue") > 0))
    .withColumn("region", F.upper(F.trim(F.col("region"))))
    .withColumn("order_date", F.to_date(F.col("order_date"), "yyyy-MM-dd"))
    .withColumn("fiscal_year",
        F.when(F.month("order_date") >= 7, F.year("order_date") + 1)
        .otherwise(F.year("order_date"))
    )
    .withColumn("fiscal_quarter",
        F.concat(F.lit("Q"),
            F.ceil((F.month("order_date") - 6) % 12 / 3).cast("int") + 1
        )
    )
    .dropDuplicates(["order_id"])
)

# Silver → Gold: Aggregations and KPIs
window_ytd = Window.partitionBy("region", "fiscal_year").orderBy("order_date").rowsBetween(Window.unboundedPreceding, Window.currentRow)

gold_sales = (
    silver_sales
    .groupBy("order_date", "region", "product", "fiscal_year", "fiscal_quarter")
    .agg(
        F.sum("revenue").alias("daily_revenue"),
        F.sum("profit").alias("daily_profit"),
        F.countDistinct("customer_id").alias("unique_customers"),
        F.sum("units").alias("units_sold"),
    )
    .withColumn("ytd_revenue", F.sum("daily_revenue").over(window_ytd))
    .withColumn("margin_pct", F.round(F.col("daily_profit") / F.col("daily_revenue") * 100, 2))
)

# Write to Fabric Warehouse for Power BI
gold_sales.write.format("delta").mode("overwrite").saveAsTable("GoldSalesAggregated")

# Refresh Power BI semantic model
import notebookutils
notebookutils.fabric.refresh_dataset("Sales_Analytics_Model")
print(f"Processed {gold_sales.count():,} records → GoldSalesAggregated")""", language="python")

    # Power BI embedded config
    st.markdown("---")
    st.subheader("Power BI Embedding Configuration")
    st.code("""{
  "reportId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "embedUrl": "https://app.powerbi.com/reportEmbed",
  "accessToken": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIs...",
  "tokenType": "Aad",
  "settings": {
    "panes": {
      "filters": { "expanded": false, "visible": true },
      "pageNavigation": { "visible": true }
    },
    "background": "transparent",
    "layoutType": "Custom",
    "customLayout": {
      "displayOption": "FitToWidth",
      "pageSize": { "type": "Custom", "width": 1600, "height": 900 }
    },
    "localeSettings": { "language": "en", "formatLocale": "en-AE" }
  },
  "datasetBinding": {
    "datasetId": "dataset-guid",
    "paginatedReportDefaultValues": {
      "Region": ["UAE", "KSA"],
      "FiscalYear": [2026]
    }
  }
}""", language="json")
