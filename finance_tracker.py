import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.ensemble import IsolationForest
import plotly.express as px
from pmdarima import auto_arima


# Streamlit App Title
st.title("📊 Personal Finance Tracker")

# File Uploader
uploaded_file = st.file_uploader("📂 Upload Your Expense Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')
    df = df.dropna(subset=["Date"])

    st.subheader("📜 Uploaded Data Preview")
    st.dataframe(df.head())

    tab1, tab2, tab3, tab4 = st.tabs(["🏷 Expense Categorization", "📊 Spending Analysis", "📈 Budget Planning", "🔮 Budget Forecasting"])

    with tab1:
        st.header("🏷 Expense Categorization")
        with open("random_forest_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("category_encoder.pkl", "rb") as f:
            category_encoder = pickle.load(f)

        description = st.text_input("Enter Transaction Description:")
        amount = st.number_input("Enter Transaction Amount:", min_value=0.0)
        transaction_type = st.selectbox("Transaction Type:", ["debit", "credit"])
        if st.button("Predict Category"):
            if description and amount:
                input_features = vectorizer.transform([description]).toarray()
                transaction_type_num = 0 if transaction_type == "debit" else 1
                input_data = [[*input_features[0], amount, transaction_type_num]]
                pred = model.predict(input_data)[0]
                category = category_encoder.inverse_transform([pred])[0]
                st.success(f"Predicted Category: {category}")
            else:
                st.warning("Please enter all details.")

    with tab2:
        # Your full tab2 Spending Analysis code (unchanged)
        pass

    with tab3:
        # Your full tab3 Budget Planning code (unchanged)
        pass

    with tab4:
        st.header("🔮 Budget Forecasting")
        st.write("Predict your future expenses using ML.")
        
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True)
        df["Month"] = df["Date"].dt.to_period("M")

        monthly_expenses = df.groupby("Month")["Amount"].sum().reset_index()
        monthly_expenses["Month"] = monthly_expenses["Month"].astype(str)

        st.subheader("📊 Past Monthly Expenses")
        st.line_chart(monthly_expenses.set_index("Month"))

        st.subheader("🔮 Forecasting Future Expenses")
        past_expense_series = monthly_expenses.set_index("Month")["Amount"]
        past_expense_series.index = pd.RangeIndex(start=1, stop=len(past_expense_series) + 1)

        best_model = auto_arima(past_expense_series, seasonal=False, stepwise=True, suppress_warnings=True)
        forecast_steps = 3
        forecast_index = [f"Month {i}" for i in range(len(past_expense_series) + 1, len(past_expense_series) + forecast_steps + 1)]
        forecast_values = best_model.predict(n_periods=forecast_steps)

        forecast_df = pd.DataFrame({"Month": forecast_index, "Predicted Expense": forecast_values})
        st.write("📌 **Predicted Expenses for the Next 3 Months:**")
        st.dataframe(forecast_df)

        plt.figure(figsize=(10, 5))
        plt.plot(past_expense_series.index, past_expense_series, marker="o", label="Actual Expense", color="blue")
        plt.plot(range(len(past_expense_series) + 1, len(past_expense_series) + forecast_steps + 1),
                 forecast_values, marker="o", linestyle="dashed", label="Forecasted Expense", color="red")
        plt.xlabel("Month")
        plt.ylabel("Expense Amount (₹)")
        plt.title("Past & Forecasted Monthly Expenses")
        plt.legend()
        st.pyplot(plt)
