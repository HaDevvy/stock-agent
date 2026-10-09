from .graph import build_graph
from pprint import pprint

graph = build_graph()


result = graph.invoke({
    "symbol": "خودرو",
    "data_until": "2026-10-06",
})


for analysis in result["analyses"]:
    pprint(
        analysis
    )

pprint(
    result["decision"]
)