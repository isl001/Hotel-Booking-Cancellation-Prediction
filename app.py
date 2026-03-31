from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).resolve().parent / "hotel_bookings.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(
        columns={
            "Booking_ID": "booking_id",
            "number of adults": "number_of_adults",
            "number of children": "number_of_children",
            "number of weekend nights": "number_of_weekend_nights",
            "number of week nights": "number_of_week_nights",
            "type of meal": "type_of_meal",
            "car parking space": "car_parking_space",
            "room type": "room_type",
            "lead time": "lead_time",
            "market segment type": "market_segment_type",
            "repeated": "repeated",
            "P-C": "previous_cancellations",
            "P-not-C": "previous_non_cancellations",
            "average price ": "average_price",
            "special requests": "special_requests",
            "date of reservation": "reservation_date",
            "booking status": "booking_status",
        }
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.drop_duplicates(inplace=True, ignore_index=True)
    cleaned["reservation_date"] = pd.to_datetime(
        cleaned["reservation_date"], format="%m/%d/%Y", errors="coerce"
    )

    numeric_columns = [
        "number_of_adults",
        "number_of_children",
        "number_of_weekend_nights",
        "number_of_week_nights",
        "car_parking_space",
        "lead_time",
        "repeated",
        "previous_cancellations",
        "previous_non_cancellations",
        "average_price",
        "special_requests",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["type_of_meal"] = cleaned["type_of_meal"].fillna("Unknown")
    cleaned["room_type"] = cleaned["room_type"].fillna("Unknown")
    cleaned["market_segment_type"] = cleaned["market_segment_type"].fillna("Unknown")
    cleaned.dropna(subset=["reservation_date", "booking_status"], inplace=True)
    return cleaned


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["reservation_month"] = featured["reservation_date"].dt.month
    featured["reservation_day"] = featured["reservation_date"].dt.day
    featured["reservation_weekday"] = featured["reservation_date"].dt.dayofweek
    featured["total_guests"] = featured["number_of_adults"] + featured["number_of_children"]
    featured["total_nights"] = (
        featured["number_of_weekend_nights"] + featured["number_of_week_nights"]
    )
    history_total = (
        featured["previous_cancellations"] + featured["previous_non_cancellations"]
    )
    featured["cancellation_ratio"] = featured["previous_cancellations"].div(
        history_total.where(history_total > 0, 1)
    ).fillna(0.0)
    featured["arrival_is_weekend"] = featured["reservation_weekday"].isin([4, 5]).astype(int)
    featured["lead_time_level"] = pd.cut(
        featured["lead_time"],
        bins=[-1, 1, 7, 30, 90, featured["lead_time"].max()],
        labels=["same_day", "short", "medium", "planned", "long"],
    )
    return featured


def get_feature_lists() -> tuple[list[str], list[str], list[str]]:
    numeric_features = [
        "number_of_adults",
        "number_of_children",
        "number_of_weekend_nights",
        "number_of_week_nights",
        "car_parking_space",
        "lead_time",
        "repeated",
        "previous_cancellations",
        "previous_non_cancellations",
        "average_price",
        "special_requests",
        "reservation_month",
        "reservation_day",
        "reservation_weekday",
        "total_guests",
        "total_nights",
        "cancellation_ratio",
        "arrival_is_weekend",
    ]
    categorical_features = [
        "type_of_meal",
        "room_type",
        "market_segment_type",
        "lead_time_level",
    ]
    return numeric_features + categorical_features, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def evaluate_model(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
    }


@st.cache_data
def prepare_dataset() -> pd.DataFrame:
    return add_features(clean_data(load_data()))


@st.cache_resource
def train_models():
    df = prepare_dataset()
    feature_columns, numeric_features, categorical_features = get_feature_lists()

    X = df[feature_columns].copy()
    y = (df["booking_status"] == "Canceled").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_test_processed)
        row = evaluate_model(y_test, predictions)
        row["model"] = name
        results.append(row)

    comparison_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=1),
        param_grid={
            "n_estimators": [150, 200],
            "max_depth": [10, None],
            "min_samples_split": [2, 5],
        },
        scoring="f1",
        cv=3,
        n_jobs=1,
    )
    grid_search.fit(X_train_processed, y_train)
    best_model = grid_search.best_estimator_
    final_predictions = best_model.predict(X_test_processed)
    final_metrics = evaluate_model(y_test, final_predictions)

    return {
        "data": df,
        "feature_columns": feature_columns,
        "preprocessor": preprocessor,
        "best_model": best_model,
        "comparison_df": comparison_df,
        "final_metrics": final_metrics,
        "best_params": grid_search.best_params_,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def get_lead_time_level(lead_time: int) -> str:
    if lead_time <= 1:
        return "same_day"
    if lead_time <= 7:
        return "short"
    if lead_time <= 30:
        return "medium"
    if lead_time <= 90:
        return "planned"
    return "long"


def build_input_frame(values: dict, feature_columns: list[str]) -> pd.DataFrame:
    reservation_date = pd.to_datetime(values["reservation_date"], errors="coerce")
    history_total = values["previous_cancellations"] + values["previous_non_cancellations"]
    return pd.DataFrame(
        [
            {
                "number_of_adults": values["number_of_adults"],
                "number_of_children": values["number_of_children"],
                "number_of_weekend_nights": values["number_of_weekend_nights"],
                "number_of_week_nights": values["number_of_week_nights"],
                "car_parking_space": values["car_parking_space"],
                "lead_time": values["lead_time"],
                "repeated": values["repeated"],
                "previous_cancellations": values["previous_cancellations"],
                "previous_non_cancellations": values["previous_non_cancellations"],
                "average_price": values["average_price"],
                "special_requests": values["special_requests"],
                "reservation_month": reservation_date.month,
                "reservation_day": reservation_date.day,
                "reservation_weekday": reservation_date.dayofweek,
                "total_guests": values["number_of_adults"] + values["number_of_children"],
                "total_nights": values["number_of_weekend_nights"] + values["number_of_week_nights"],
                "cancellation_ratio": (
                    0 if history_total == 0 else values["previous_cancellations"] / history_total
                ),
                "arrival_is_weekend": int(reservation_date.dayofweek in [4, 5]),
                "type_of_meal": values["type_of_meal"],
                "room_type": values["room_type"],
                "market_segment_type": values["market_segment_type"],
                "lead_time_level": get_lead_time_level(values["lead_time"]),
            }
        ],
        columns=feature_columns,
    )


st.set_page_config(page_title="Hotel Cancellation Prediction", layout="wide")
st.title("Hotel Booking Cancellation Prediction")
st.caption(
    "3-file final project: dataset, notebook, and Streamlit app with feature engineering, model comparison, tuning, and prediction."
)

bundle = train_models()
metrics = bundle["final_metrics"]

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Accuracy", metrics["accuracy"])
metric_col2.metric("Precision", metrics["precision"])
metric_col3.metric("Recall", metrics["recall"])
metric_col4.metric("F1 Score", metrics["f1_score"])

left_col, right_col = st.columns([1.35, 1])

with left_col:
    st.subheader("Booking Data")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            number_of_adults = st.number_input("Number of Adults", min_value=1, value=2)
            number_of_children = st.number_input("Number of Children", min_value=0, value=0)
            number_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=1)
            number_of_week_nights = st.number_input("Week Nights", min_value=0, value=2)
            type_of_meal = st.selectbox(
                "Meal Plan",
                ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
            )
            car_parking_space = st.selectbox("Car Parking Space", [0, 1])
            room_type = st.selectbox(
                "Room Type",
                [
                    "Room_Type 1",
                    "Room_Type 2",
                    "Room_Type 3",
                    "Room_Type 4",
                    "Room_Type 5",
                    "Room_Type 6",
                    "Room_Type 7",
                ],
            )
        with c2:
            lead_time = st.number_input("Lead Time", min_value=0, value=20)
            market_segment_type = st.selectbox(
                "Market Segment",
                ["Online", "Offline", "Corporate", "Aviation", "Complementary"],
            )
            repeated = st.selectbox("Repeated Guest", [0, 1])
            previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
            previous_non_cancellations = st.number_input(
                "Previous Non-Cancellations", min_value=0, value=0
            )
            average_price = st.number_input("Average Price", min_value=0.0, value=110.0, step=1.0)
            special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=1)
            reservation_date = st.date_input("Reservation Date")
        submitted = st.form_submit_button("Predict")

with right_col:
    st.subheader("Model Comparison")
    st.dataframe(bundle["comparison_df"], use_container_width=True)


if submitted:
    sample = build_input_frame(
        {
            "number_of_adults": int(number_of_adults),
            "number_of_children": int(number_of_children),
            "number_of_weekend_nights": int(number_of_weekend_nights),
            "number_of_week_nights": int(number_of_week_nights),
            "car_parking_space": int(car_parking_space),
            "lead_time": int(lead_time),
            "repeated": int(repeated),
            "previous_cancellations": int(previous_cancellations),
            "previous_non_cancellations": int(previous_non_cancellations),
            "average_price": float(average_price),
            "special_requests": int(special_requests),
            "reservation_date": reservation_date,
            "type_of_meal": type_of_meal,
            "room_type": room_type,
            "market_segment_type": market_segment_type,
        },
        bundle["feature_columns"],
    )

    sample_processed = bundle["preprocessor"].transform(sample)
    prediction = int(bundle["best_model"].predict(sample_processed)[0])
    probabilities = bundle["best_model"].predict_proba(sample_processed)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Prediction: Canceled")
    else:
        st.success("Prediction: Not Canceled")

    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Cancellation Probability", f"{round(float(probabilities[1] * 100), 2)}%")
    result_col2.metric("Not Canceled Probability", f"{round(float(probabilities[0] * 100), 2)}%")
