from .graph import build_graph


graph = build_graph()


result = graph.invoke({
    "symbol": "خودرو",
    "data_until": "2026-10-06",
})


print(
    result["technical_analysis"]
)

print(
    result["decision"]
)