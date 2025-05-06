import pandas as pd
from graph import build_graph

lease_df = pd.read_csv("lease_data.csv")
graph = build_graph()
final_state = graph.invoke({"lease_df": lease_df})

print("\n📆 Expiring Leases:")
print(pd.DataFrame(final_state["expiring_leases"]).to_markdown(index=False))

print("\n💸 Financial Impact:")
print(f"Total Potential Loss: ${final_state['financial_impact']['total_loss']:,.0f}")

print("\n🔁 Re-Leasing Potential:")
print(f"Expected Recovery: ${final_state['re_leasing_potential']['total_expected_recovery']:,.0f}")

print("\n🧠 GenAI Summary:")
print(final_state["genai_summary"])

print("\n🎯 Strategic Recommendations:")
print(final_state["strategy_recommendation"])