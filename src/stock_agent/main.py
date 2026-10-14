from .graph import build_graph
from pprint import pprint

graph = build_graph()


result = graph.invoke({
    "symbol": "خودرو",
    "data_until": "2026-10-06",
})


for analysis in result["analyses"]:
    pprint(analysis)

print("Proposed decision:")
pprint(result["decision"])

print("Risk review:")
pprint(result["risk_review"])

print("Final, risk-checked decision:")
pprint(result["final_decision"])

pprint("decision review:")
pprint(result["decision_review"])