
import streamlit as st
import pandas as pd
from graph import build_graph
import openai

st.set_page_config(page_title="Lease Loss Analyzer", layout="wide")
st.title("🏢 Lease Loss & Re-Leasing Potential Analyzer")

uploaded_file = st.file_uploader("Upload Lease Data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Lease Data")
    st.dataframe(df)

    run_button = st.button("Run Analysis")

    if run_button:
        graph = build_graph()
        final_state = graph.invoke({"lease_df": df})

        st.subheader("📆 Expiring Leases")
        expiring_df = pd.DataFrame(final_state["expiring_leases"])
        st.dataframe(expiring_df)

        st.subheader("💸 Financial Impact (If Not Renewed)")
        fin = final_state["financial_impact"]
        st.metric("Total Potential Loss", f"${fin['total_loss']:,.0f}")
        st.dataframe(pd.DataFrame(fin["loss_by_property"]))

        st.subheader("🔁 Expected Re-Leasing Recovery")
        recovery = final_state["re_leasing_potential"]
        st.metric("Total Expected Recovery", f"${recovery['total_expected_recovery']:,.0f}")
        st.dataframe(pd.DataFrame(recovery["recovery_estimates"]))

        st.subheader("🧠 AI-Generated Executive Summary")
        st.markdown(final_state["genai_summary"])

        st.subheader("🎯 Strategic Recommendations")
        st.markdown(final_state["strategy_recommendation"])

        st.sidebar.title("🧠 Ask the Agent")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_q = st.sidebar.text_input("Ask about the leases, risks, or actions:")
        if user_q:
            chat_prompt = f"""
You are a helpful AI analyst. Answer based on this context:
Expiring leases: {final_state['expiring_leases']}
Loss data: {final_state['financial_impact']}
Recovery estimates: {final_state['re_leasing_potential']}
Strategy: {final_state['strategy_recommendation']}

User: {user_q}
"""
            chat_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": chat_prompt}],
                temperature=0.3
            )
            answer = chat_response.choices[0].message["content"]
            st.session_state.chat_history.append((user_q, answer))

        for q, a in st.session_state.chat_history[::-1]:
            st.sidebar.markdown(f"**You:** {q}")
            st.sidebar.markdown(f"**AI:** {a}")
