# app.py – Customer Churn GenAI App

# ---------- 1. Import all the tools we need ----------
import streamlit as st          # for the web interface
import pandas as pd             # for handling data
import numpy as np
import joblib                   # to load saved model and scaler
import plotly.express as px     # for nice interactive charts
from transformers import pipeline  # this is the free AI model

# ---------- 2. Load the saved files (only once, to make it fast) ----------
@st.cache_resource
def load_all():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_columns.pkl")
    data = pd.read_csv("processed_data.csv")
    return model, scaler, features, data

model, scaler, features, df = load_all()

# ---------- 3. Load the free AI model (also only once) ----------
@st.cache_resource
def load_ai():
    # This downloads the model the first time you run the app.
    # It's about 1 GB – be patient.
    try:
       return pipeline("text2text-generation", model="google/flan-t5-base", framework="pt")
    except Exception as e:
        st.error(f"Could not load AI model: {e}. We'll use a simple fallback.")
        return None

ai_pipeline = load_ai()

# ---------- 4. Helper function to predict churn for a new customer ----------
def predict_churn(input_dict):
    # Turn the dictionary into a DataFrame with the correct column order
    input_df = pd.DataFrame([input_dict])
    # Keep only the features that the model knows
    input_df = input_df[features]
    # Scale the numbers (same scaling as during training)
    scaled = scaler.transform(input_df)
    # Get probability of churn (class 1)
    prob = model.predict_proba(scaled)[0, 1]
    return prob

# ---------- 5. Helper function to answer questions using AI ----------
def ask_ai(question, context):
    if ai_pipeline is None:
        # Fallback: simple template answer
        return "I'm sorry, the AI model isn't available. But here is the data: " + context
    # Build a clear prompt for the AI
    prompt = f"""
You are a business analyst. Use the following facts to answer the question.

Facts:
{context}

Question: {question}

Answer concisely with numbers where possible.
"""
    # Generate the answer
    result = ai_pipeline(prompt, max_length=150, do_sample=False)[0]['generated_text']
    return result

# ---------- 6. Build the Streamlit user interface ----------
st.set_page_config(page_title="Churn Analyst", layout="wide")
st.title("📊 Customer Churn Predictor & AI Analyst")

# Sidebar: choose mode
mode = st.sidebar.radio("Choose what you want to do:", 
                        ["🔮 Predict Churn Risk", "💬 Ask a Question"])

# ---------- MODE 1: PREDICT ----------
if mode == "🔮 Predict Churn Risk":
    st.header("Fill in customer details")
    st.markdown("These are the features our model uses. Just type the numbers.")

    # We create input boxes for each feature.
    # For simplicity, we treat all features as numbers (you can change this later)
    input_data = {}
    cols = st.columns(4)   # split into 4 columns to save space
    for i, col in enumerate(features):
        # Use number_input for numeric features
        input_data[col] = cols[i % 4].number_input(col, value=0.0, step=0.1)

    if st.button("Predict Churn Probability"):
        prob = predict_churn(input_data)
        st.metric("Churn Risk", f"{prob:.1%}")
        if prob > 0.5:
            st.error("⚠️ This customer is likely to churn. Consider a retention offer.")
        else:
            st.success("✅ This customer is likely to stay. Good job!")

# ---------- MODE 2: ASK A QUESTION ----------
else:
    st.header("Ask me anything about customer churn")
    user_question = st.text_input("Type your question here:", 
                                  placeholder="e.g. Why are customers churning?")

    if user_question:
        with st.spinner("Thinking... (this may take a few seconds)"):
            # --- Step A: Gather useful facts from the data ---
            overall_churn = df["Churn Value"].mean()
            high_risk = df[df["Monthly Charges"] > 70]
            high_risk_count = len(high_risk)
            avg_monthly = df["Monthly Charges"].mean()

            # If we have a 'Plan' column (you might have a different name)
            plan_col = None
            for col in df.columns:
                if "plan" in col.lower() or "subscription" in col.lower():
                    plan_col = col
                    break
            if plan_col:
                churn_by_plan = df.groupby(plan_col)["Churn Value"].mean().to_dict()
                plan_str = ", ".join([f"{k}: {v:.1%}" for k, v in churn_by_plan.items()])
            else:
                plan_str = "No plan column found."

            # Tenure group if exists
            if "TenureGroup" in df.columns:
                churn_by_tenure = df.groupby("TenureGroup")["Churn Value"].mean().to_dict()
                tenure_str = ", ".join([f"{k}: {v:.1%}" for k, v in churn_by_tenure.items()])
            else:
                tenure_str = "No tenure group data."

            # Build a short context text for the AI
            context = f"""
- Overall churn rate: {overall_churn:.1%}
- Average monthly charge: ${avg_monthly:.2f}
- Number of customers with high monthly charges (>$70): {high_risk_count}
- Churn by plan: {plan_str}
- Churn by tenure group: {tenure_str}
"""
            # --- Step B: Ask the AI ---
            answer = ask_ai(user_question, context)

            # --- Step C: Show the answer ---
            st.subheader("💡 Answer")
            st.write(answer)

            # --- Step D: Show a relevant chart automatically ---
            st.subheader("📈 Supporting Chart")
            # Try to guess what the user asked about
            if "plan" in user_question.lower() and plan_col:
                churn_by_plan_df = df.groupby(plan_col)["Churn Value"].mean().reset_index()
                fig = px.bar(churn_by_plan_df, x=plan_col, y="Churn Value", 
                             title="Churn Rate by Plan", color="Churn Value")
                st.plotly_chart(fig, use_container_width=True)
            elif "tenure" in user_question.lower() and "TenureGroup" in df.columns:
                churn_by_tenure_df = df.groupby("TenureGroup")["Churn Value"].mean().reset_index()
                fig = px.bar(churn_by_tenure_df, x="TenureGroup", y="Churn Value",
                             title="Churn Rate by Tenure Group")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Default: show churn distribution
                fig = px.histogram(df, x="Churn Value", title="Overall Churn Distribution",
                                   labels={"Churn Value": "Churn (0=No, 1=Yes)"})
                st.plotly_chart(fig, use_container_width=True)

            # Show the raw context (optional, for transparency)
            with st.expander("See the data facts used"):
                st.text(context)