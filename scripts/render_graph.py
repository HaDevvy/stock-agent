from pathlib import Path

from stock_agent.graph import build_graph


OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

graph = build_graph()

# Mermaid source
mermaid = graph.get_graph().draw_mermaid()

(OUTPUT_DIR / "graph.mmd").write_text(
    mermaid,
    encoding="utf-8",
)

# PNG
png = graph.get_graph().draw_mermaid_png()

(OUTPUT_DIR / "graph.png").write_bytes(png)

print("Graph generated:")
print("  artifacts/graph.mmd")
print("  artifacts/graph.png")