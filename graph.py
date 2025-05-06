from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import pandas as pd
from openai import OpenAI

client = OpenAI()
State = dict

def lease_analyzer_agent(state: State) -> State:
    df = state["lease_df"]
    df["lease_end_date"] = pd.to_datetime(df["lease_end_date"])
    six_months = pd.Timestamp.now() + pd.DateOffset(months=6)
    expiring = df[df["lease_end_date"] <= six_months]
    leases = expiring.to_dict(orient="records")
    state["expiring_leases"] = leases
    return state

def financial_impact_agent(state: State) -> State:
    leases = state.get("expiring_leases", [])
    total_loss = 0
    for lease in leases:
        months_remaining = 6
        lease["estimated_loss"] = lease["monthly_rent"] * months_remaining
        total_loss += lease["estimated_loss"]
    state["financial_impact"] = {
        "loss_by_property": leases,
        "total_loss": total_loss
    }
    return state

def re_leasing_potential_agent(state: State) -> State:
    leases = state["financial_impact"]["loss_by_property"]
    recovery_estimates = []
    total_expected_recovery = 0
    for lease in leases:
        market_rent = lease["monthly_rent"] * 1.1
        re_lease_probability = 0.6
        downtime_months = 1
        months_occupied = 6 - downtime_months
        recovered = (market_rent * months_occupied) * re_lease_probability
        recovery = recovered - (market_rent * downtime_months)
        lease.update({
            "expected_recovery": round(recovery, 2),
            "market_rent": market_rent,
            "re_lease_probability": re_lease_probability
        })
        total_expected_recovery += recovery
        recovery_estimates.append(lease)
    state["re_leasing_potential"] = {
        "recovery_estimates": recovery_estimates,
        "total_expected_recovery": round(total_expected_recovery, 2)
    }
    return state

def summarize_results(state: State) -> State:
    leases = state.get("expiring_leases", [])
    impact = state.get("financial_impact", {})
    recovery = state.get("re_leasing_potential", {})

    prompt = f"""
You are a commercial real estate advisor.

Based on:
- {len(leases)} leases expiring
- Loss risk: ${impact.get('total_loss', 0):,.0f}
- Expected recovery: ${recovery.get('total_expected_recovery', 0):,.0f}

Write a 3-4 point strategic summary for executive stakeholders.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    state["genai_summary"] = response.choices[0].message.content
    return state

def strategic_recommendation_agent(state: State) -> State:
    leases = state.get("expiring_leases", [])
    loss = state["financial_impact"]["total_loss"]
    recovery = state["re_leasing_potential"]["total_expected_recovery"]

    prompt = f"""
You are a strategic real estate investment advisor.

Given:
- {len(leases)} leases expiring
- Estimated loss: ${loss:,.0f}
- Expected recovery: ${recovery:,.0f}

Recommend 3 actions to optimize asset value and reduce risk.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    state["strategy_recommendation"] = response.choices[0].message.content
    return state

def build_graph():
    lease_node = RunnableLambda(lease_analyzer_agent)
    impact_node = RunnableLambda(financial_impact_agent)
    recovery_node = RunnableLambda(re_leasing_potential_agent)
    summary_node = RunnableLambda(summarize_results)
    strategy_node = RunnableLambda(strategic_recommendation_agent)

    builder = StateGraph(State)
    builder.add_node("lease_analyzer", lease_node)
    builder.add_node("financial_impact", impact_node)
    builder.add_node("re_leasing_potential", recovery_node)
    builder.add_node("genai_summary", summary_node)
    builder.add_node("strategic_advisor", strategy_node)

    builder.set_entry_point("lease_analyzer")
    builder.add_edge("lease_analyzer", "financial_impact")
    builder.add_edge("financial_impact", "re_leasing_potential")
    builder.add_edge("re_leasing_potential", "genai_summary")
    builder.add_edge("genai_summary", "strategic_advisor")
    builder.add_edge("strategic_advisor", END)

    return builder.compile()