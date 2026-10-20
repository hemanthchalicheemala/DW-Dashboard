
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_InsuranceData.csv")

    # Convert date column
    if "PolicyStartDate" in df.columns:
        df["PolicyStartDate"] = pd.to_datetime(
            df["PolicyStartDate"], errors="coerce"
        )
        df["Year"] = df["PolicyStartDate"].dt.year

    # Convert numeric columns
    for col in ["PremiumAmount", "ClaimAmount", "CoverageAmount", "Age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


df = load_data()

# --------------------------------------------------
# 3. DASHBOARD TITLE
# --------------------------------------------------
st.title("📊 Insurance Analytics Dashboard")
st.write(
    "Analyze insurance policies, premiums, claims, "
    "and customer segments."
)

# --------------------------------------------------
# 4. SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Filters")

filtered_df = df.copy()

if "PolicyType" in df.columns:
    policy_types = sorted(df["PolicyType"].dropna().unique())

    selected_policy = st.sidebar.multiselect(
        "Select Policy Type",
        policy_types,
        default=policy_types
    )

    filtered_df = filtered_df[
        filtered_df["PolicyType"].isin(selected_policy)
    ]

if "Gender" in df.columns:
    genders = sorted(df["Gender"].dropna().unique())

    selected_gender = st.sidebar.multiselect(
        "Select Gender",
        genders,
        default=genders
    )

    filtered_df = filtered_df[
        filtered_df["Gender"].isin(selected_gender)
    ]

if "ClaimStatus" in df.columns:
    claim_statuses = sorted(df["ClaimStatus"].dropna().unique())

    selected_status = st.sidebar.multiselect(
        "Select Claim Status",
        claim_statuses,
        default=claim_statuses
    )

    filtered_df = filtered_df[
        filtered_df["ClaimStatus"].isin(selected_status)
    ]

if "Year" in df.columns:
    years = sorted(df["Year"].dropna().astype(int).unique())

    if years:
        selected_years = st.sidebar.multiselect(
            "Select Policy Year",
            years,
            default=years
        )

        filtered_df = filtered_df[
            filtered_df["Year"].isin(selected_years)
        ]

# --------------------------------------------------
# 5. KEY PERFORMANCE INDICATORS
# --------------------------------------------------
st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{len(filtered_df):,}"
)

if "PremiumAmount" in filtered_df.columns:
    total_premium = filtered_df["PremiumAmount"].sum()
    average_premium = filtered_df["PremiumAmount"].mean()

    col2.metric("Total Premium", f"{total_premium:,.2f}")
    col3.metric("Average Premium", f"{average_premium:,.2f}")
else:
    col2.metric("Total Premium", "N/A")
    col3.metric("Average Premium", "N/A")

if "ClaimStatus" in filtered_df.columns:
    settled_claims = (
        filtered_df["ClaimStatus"].astype(str).str.strip().str.lower()
        == "settled"
    ).sum()

    col4.metric("Settled Claims", f"{settled_claims:,}")
else:
    col4.metric("Settled Claims", "N/A")

st.divider()

# --------------------------------------------------
# 6. PREMIUM ANALYSIS
# --------------------------------------------------
st.subheader("Premium Analysis")

left, right = st.columns(2)

with left:
    if {"PolicyType", "PremiumAmount"}.issubset(filtered_df.columns):
        premium_by_type = (
            filtered_df.groupby("PolicyType", as_index=False)
            ["PremiumAmount"].sum()
        )

        fig = px.bar(
            premium_by_type,
            x="PolicyType",
            y="PremiumAmount",
            color="PolicyType",
            title="Total Premium by Policy Type",
            labels={
                "PolicyType": "Policy Type",
                "PremiumAmount": "Total Premium"
            }
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Policy type or premium data is unavailable.")

with right:
    if {"Year", "PremiumAmount"}.issubset(filtered_df.columns):
        yearly_premium = (
            filtered_df.dropna(subset=["Year"])
            .groupby("Year", as_index=False)["PremiumAmount"].sum()
        )

        fig = px.line(
            yearly_premium,
            x="Year",
            y="PremiumAmount",
            markers=True,
            title="Year-wise Premium Trend",
            labels={
                "Year": "Policy Year",
                "PremiumAmount": "Total Premium"
            }
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Year-wise premium data is unavailable.")

st.divider()

# --------------------------------------------------
# 7. CLAIMS ANALYSIS
# --------------------------------------------------
st.subheader("Claims Analysis")

left, right = st.columns(2)

with left:
    if "ClaimStatus" in filtered_df.columns:
        claim_counts = (
            filtered_df["ClaimStatus"]
            .value_counts()
            .rename_axis("ClaimStatus")
            .reset_index(name="Count")
        )

        if not claim_counts.empty:
            fig = px.pie(
                claim_counts,
                names="ClaimStatus",
                values="Count",
                title="Distribution of Claim Status",
                hole=0.35
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No claim status records match the filters.")
    else:
        st.info("Claim status data is unavailable.")

with right:
    if {"ClaimStatus", "ClaimAmount"}.issubset(filtered_df.columns):
        claims_by_status = (
            filtered_df.groupby("ClaimStatus", as_index=False)
            ["ClaimAmount"].sum()
        )

        fig = px.bar(
            claims_by_status,
            x="ClaimStatus",
            y="ClaimAmount",
            color="ClaimStatus",
            title="Total Claim Amount by Status",
            labels={
                "ClaimStatus": "Claim Status",
                "ClaimAmount": "Total Claim Amount"
            }
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Claim amount or claim status data is unavailable.")

st.divider()

# --------------------------------------------------
# 8. RFM CUSTOMER SEGMENTATION
# --------------------------------------------------
st.subheader("RFM-Based Customer Segmentation")

required_columns = [
    "CustomerID",
    "PolicyNumber",
    "PolicyStartDate",
    "PremiumAmount"
]

if all(col in df.columns for col in required_columns):

    # Prepare RFM data
    rfm = df[required_columns].copy()

    rfm["PolicyStartDate"] = pd.to_datetime(
        rfm["PolicyStartDate"], errors="coerce"
    )

    rfm["PremiumAmount"] = pd.to_numeric(
        rfm["PremiumAmount"], errors="coerce"
    )

    rfm = rfm.dropna(
        subset=[
            "CustomerID",
            "PolicyNumber",
            "PolicyStartDate",
            "PremiumAmount"
        ]
    )

    if not rfm.empty:

        latest_date = rfm["PolicyStartDate"].max()

        rfm_data = rfm.groupby("CustomerID").agg(
            Recency=(
                "PolicyStartDate",
                lambda dates: (latest_date - dates.max()).days
            ),
            Frequency=("PolicyNumber", "nunique"),
            Monetary=("PremiumAmount", "sum")
        ).reset_index()

        if len(rfm_data) >= 3:

            # Standardize recency and monetary values
            features = rfm_data[["Recency", "Monetary"]]

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Apply K-Means clustering
            model = KMeans(
                n_clusters=3,
                random_state=42,
                n_init=10
            )

            rfm_data["Cluster"] = model.fit_predict(
                scaled_features
            )

            # RFM KPI cards
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Customers Segmented",
                f"{len(rfm_data):,}"
            )

            col2.metric(
                "Number of Clusters",
                rfm_data["Cluster"].nunique()
            )

            col3.metric(
                "Average Customer Premium",
                f"{rfm_data['Monetary'].mean():,.2f}"
            )

            # Cluster visualization
            fig = px.scatter(
                rfm_data,
                x="Recency",
                y="Monetary",
                color=rfm_data["Cluster"].astype(str),
                hover_data=["CustomerID", "Frequency"],
                title="Customer Segments: Recency vs Premium",
                labels={
                    "Recency": "Recency (Days)",
                    "Monetary": "Total Premium",
                    "color": "Cluster"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

            # Cluster summary
            st.write("### Cluster Summary")

            cluster_summary = rfm_data.groupby("Cluster").agg(
                Customers=("CustomerID", "count"),
                Average_Recency=("Recency", "mean"),
                Average_Premium=("Monetary", "mean")
            ).reset_index()

            cluster_summary["Average_Recency"] = (
                cluster_summary["Average_Recency"].round(2)
            )

            cluster_summary["Average_Premium"] = (
                cluster_summary["Average_Premium"].round(2)
            )

            st.dataframe(
                cluster_summary,
                use_container_width=True,
                hide_index=True
            )

            # Customer segment interpretation
            st.write("### Customer Segment Interpretation")

            median_recency = rfm_data["Recency"].median()
            median_premium = rfm_data["Monetary"].median()

            for _, row in cluster_summary.iterrows():

                cluster = int(row["Cluster"])

                st.markdown(f"**Cluster {cluster}**")

                st.write(
                    f"- Customers: {int(row['Customers']):,}"
                )

                st.write(
                    f"- Average recency: "
                    f"{row['Average_Recency']} days"
                )

                st.write(
                    f"- Average premium: "
                    f"{row['Average_Premium']:,.2f}"
                )

                if (
                    row["Average_Recency"] <= median_recency
                    and row["Average_Premium"] >= median_premium
                ):
                    st.write(
                        "Interpretation: More recent policy starts "
                        "and relatively higher premiums."
                    )

                elif (
                    row["Average_Recency"] > median_recency
                    and row["Average_Premium"] < median_premium
                ):
                    st.write(
                        "Interpretation: Older policy starts "
                        "and relatively lower premiums."
                    )

                else:
                    st.write(
                        "Interpretation: A mixed segment based "
                        "on policy start dates and premium values."
                    )

                st.divider()

            # Full RFM customer table
            st.write("### RFM Customer Data")

            st.dataframe(
                rfm_data,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning(
                "At least three valid customers are required "
                "for K-Means clustering."
            )

    else:
        st.warning("No valid records are available for RFM analysis.")

else:
    st.warning(
        "RFM analysis requires CustomerID, PolicyNumber, "
        "PolicyStartDate, and PremiumAmount columns."
    )

# --------------------------------------------------
# 9. FILTERED DATASET PREVIEW
# --------------------------------------------------
st.divider()

st.subheader("Filtered Dataset Preview")

st.write(f"Showing {len(filtered_df):,} records.")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------
st.caption(
    "Insurance Analytics Dashboard | "
    "Data Warehouse and Data Mining Project"
)
