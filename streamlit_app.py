
import os
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Insurance Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Insurance Dashboard")
st.caption("Insurance Analytics and Prediction Platform")

# ==================================================
# 2. LOAD DATASET
# ==================================================

DATA_PATH = "Cleaned_InsuranceData.csv"


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset not found: {DATA_PATH}. "
        "Ensure the CSV file is in your GitHub repository "
        "alongside streamlit_app.py."
    )
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

df.columns = df.columns.str.strip()


def find_column(possible_names):
    column_map = {column.lower(): column for column in df.columns}

    for name in possible_names:
        if name.lower() in column_map:
            return column_map[name.lower()]

    return None


age_col = find_column(["Age"])
gender_col = find_column(["Gender"])
policy_col = find_column(["PolicyType", "Policy Type"])
coverage_col = find_column(["CoverageAmount", "Coverage Amount"])
claim_status_col = find_column(["ClaimStatus", "Claim Status"])

# ==================================================
# 3. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🛡️ Insurance Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Analysis", "Prediction"]
)

st.sidebar.divider()
st.sidebar.caption("Powered by Python and Streamlit")

# ==================================================
# 4. DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    st.header("Insurance Overview")

    col1, col2, col3, col4 = st.columns(4)

    customer_col = find_column(["CustomerID", "Customer ID"])

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        customers = (
            df[customer_col].nunique()
            if customer_col
            else len(df)
        )
        st.metric("Unique Customers", f"{customers:,}")

    with col3:
        if coverage_col:
            total_coverage = pd.to_numeric(
                df[coverage_col], errors="coerce"
            ).sum()

            st.metric("Total Coverage", f"₹{total_coverage:,.0f}")
        else:
            st.metric("Total Coverage", "Unavailable")

    with col4:
        if claim_status_col:
            st.metric(
                "Claim Categories",
                df[claim_status_col].nunique()
            )
        else:
            st.metric("Claim Categories", "Unavailable")

    st.divider()

    left, right = st.columns(2)

    with left:
        if policy_col:
            policy_counts = (
                df[policy_col]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Policy Type")
                .reset_index(name="Count")
            )

            fig = px.bar(
                policy_counts,
                x="Policy Type",
                y="Count",
                color="Policy Type",
                title="Policies by Type"
            )

            st.plotly_chart(fig, use_container_width=True)

    with right:
        if gender_col:
            gender_counts = (
                df[gender_col]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Gender")
                .reset_index(name="Count")
            )

            fig = px.pie(
                gender_counts,
                names="Gender",
                values="Count",
                title="Gender Distribution",
                hole=0.4
            )

            st.plotly_chart(fig, use_container_width=True)

    if age_col:
        age_data = pd.to_numeric(
            df[age_col], errors="coerce"
        ).dropna()

        fig = px.histogram(
            x=age_data,
            nbins=20,
            title="Customer Age Distribution",
            labels={"x": "Age"}
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ==================================================
# 5. DATA ANALYSIS PAGE
# ==================================================

elif page == "Data Analysis":

    st.header("📊 Data Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", f"{df.shape[1]:,}")

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.subheader("Missing Values by Column")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    fig = px.bar(
        missing,
        x="Column",
        y="Missing Values",
        title="Missing Values Analysis"
    )

    st.plotly_chart(fig, use_container_width=True)

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if numeric_cols:
        selected_numeric = st.selectbox(
            "Choose a numerical column",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=selected_numeric,
            title=f"Distribution of {selected_numeric}"
        )

        st.plotly_chart(fig, use_container_width=True)

    if categorical_cols:
        selected_category = st.selectbox(
            "Choose a categorical column",
            categorical_cols
        )

        category_counts = (
            df[selected_category]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis(selected_category)
            .reset_index(name="Count")
        )

        fig = px.bar(
            category_counts,
            x=selected_category,
            y="Count",
            title=f"Distribution of {selected_category}"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Complete Dataset")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="insurance_data.csv",
        mime="text/csv"
    )

# ==================================================
# 6. PREDICTION PAGE
# ==================================================

elif page == "Prediction":

    st.header("🔮 Insurance Prediction")

    st.write(
        "Enter customer details to estimate the historical "
        "claim-status category."
    )

    required_columns = {
        "Age": age_col,
        "Gender": gender_col,
        "Policy Type": policy_col,
        "Coverage Amount": coverage_col,
        "Claim Status": claim_status_col
    }

    missing_columns = [
        name
        for name, column in required_columns.items()
        if column is None
    ]

    if missing_columns:

        st.error(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

        st.write("Available columns:", df.columns.tolist())

    else:

        feature_cols = [
            age_col,
            gender_col,
            policy_col,
            coverage_col
        ]

        target_col = claim_status_col

        model_df = df[feature_cols + [target_col]].copy()

        # Convert numeric fields.
        model_df[age_col] = pd.to_numeric(
            model_df[age_col], errors="coerce"
        )

        model_df[coverage_col] = pd.to_numeric(
            model_df[coverage_col], errors="coerce"
        )

        # Remove invalid target and numeric records.
        model_df = model_df.dropna(
            subset=[target_col, age_col, coverage_col]
        )

        model_df[target_col] = model_df[target_col].astype(str)

        if model_df[target_col].nunique() < 2:

            st.error(
                "At least two target classes are required "
                "to train the classification model."
            )

        elif len(model_df) < 10:

            st.error(
                "There are too few valid records to train the model."
            )

        else:

            X = model_df[feature_cols]
            y = model_df[target_col]

            numeric_features = [age_col, coverage_col]
            categorical_features = [gender_col, policy_col]

            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            model = Pipeline([
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ])

            class_counts = y.value_counts()

            can_stratify = (
                len(class_counts) >= 2
                and class_counts.min() >= 2
            )

            stratify_arg = y if can_stratify else None

            # ------------------------------------------
            # TRAIN AND EVALUATE MODEL
            # ------------------------------------------

            try:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=stratify_arg
                )

                model.fit(X_train, y_train)

                test_predictions = model.predict(X_test)

                accuracy = accuracy_score(
                    y_test,
                    test_predictions
                )

                st.success("Model trained successfully.")

                st.metric(
                    "Holdout Test Accuracy",
                    f"{accuracy:.1%}"
                )

                st.caption(
                    "Accuracy is measured on a held-out test set. "
                    "It does not guarantee an individual outcome."
                )

                st.divider()
                st.subheader("Enter Customer Information")

                # ------------------------------------------
                # PREDICTION INPUT FORM
                # ------------------------------------------

                with st.form("insurance_prediction_form"):

                    col1, col2 = st.columns(2)

                    with col1:

                        age = st.number_input(
                            "Age",
                            min_value=18,
                            max_value=100,
                            value=25,
                            step=1
                        )

                        gender_options = sorted(
                            df[gender_col]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        if not gender_options:
                            gender_options = ["Unknown"]

                        gender = st.selectbox(
                            "Gender",
                            gender_options
                        )

                    with col2:

                        policy_options = sorted(
                            df[policy_col]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        if not policy_options:
                            policy_options = ["Unknown"]

                        policy_type = st.selectbox(
                            "Policy Type",
                            policy_options
                        )

                        coverage_values = pd.to_numeric(
                            model_df[coverage_col],
                            errors="coerce"
                        ).dropna()

                        # Permit values above the training-data maximum.
                        # This is an input range, not evidence that the
                        # model is reliable at those higher amounts.
                        observed_max = float(coverage_values.max())

                        coverage_min = 0.0
                        coverage_max = max(
                            50_000_000.0,
                            observed_max
                        )

                        default_coverage = min(
                            max(500_000.0, coverage_min),
                            coverage_max
                        )

                        coverage_amount = st.number_input(
                            "Coverage Amount (₹)",
                            min_value=coverage_min,
                            max_value=coverage_max,
                            value=default_coverage,
                            step=10_000.0,
                            format="%.2f"
                        )

                    submitted = st.form_submit_button(
                        "Predict Claim Status",
                        use_container_width=True
                    )

                # ------------------------------------------
                # DISPLAY PREDICTION
                # ------------------------------------------

                if submitted:

                    input_df = pd.DataFrame([{
                        age_col: age,
                        gender_col: gender,
                        policy_col: policy_type,
                        coverage_col: coverage_amount
                    }])

                    prediction = model.predict(input_df)[0]

                    st.subheader("Prediction Result")

                    st.info(
                        f"Predicted Claim Status: {prediction}"
                    )

                    if coverage_amount > observed_max:
                        st.warning(
                            "This coverage amount exceeds the maximum "
                            "observed in the training data. The model "
                            "is extrapolating, so this prediction may "
                            "be unreliable."
                        )

                    st.caption(
                        "This is an estimate of the dataset's "
                        "ClaimStatus category, not an actual claim "
                        "approval or rejection decision."
                    )

                    # Show model class probabilities.
                    classifier = model.named_steps["classifier"]

                    probabilities = model.predict_proba(
                        input_df
                    )[0]

                    probability_df = pd.DataFrame({
                        "Claim Status": classifier.classes_,
                        "Probability (%)": probabilities * 100
                    })

                    fig = px.bar(
                        probability_df,
                        x="Claim Status",
                        y="Probability (%)",
                        title="Model Class Probabilities",
                        text_auto=".1f"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.caption(
                        "Model probabilities are estimates and may "
                        "not be calibrated to real-world probabilities."
                    )

            except (ValueError, TypeError) as e:

                st.error(
                    f"Model training or prediction failed: {e}"
                )

            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )
