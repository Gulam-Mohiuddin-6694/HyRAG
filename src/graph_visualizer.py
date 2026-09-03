"""
graph_visualizer.py - Interactive Knowledge Graph Explorer Component for HyRAG.

Generates an embedded, interactive Vis.js Network Graph with physics simulation,
entity-type color themes, provenance inspection, document filtering, and full Light/Dark theme support.
"""

import json
from typing import Dict, Any, List, Optional
from src.graph_store import BaseGraphStore

# Color mapping by entity type
ENTITY_TYPE_COLORS = {
    "SecurityControl": "#146EB4",       # AWS Blue
    "SecurityAccount": "#0284C7",       # Sky Blue
    "Policy": "#FF9900",                # AWS Orange
    "Organization": "#8B5CF6",          # Violet
    "FrameworkPillar": "#10B981",       # Emerald
    "Framework": "#059669",             # Teal
    "Person": "#EC4899",                # Pink
    "Department": "#F59E0B",            # Amber
    "Entity": "#64748B",                # Slate
    "Concept": "#475569",               # Dark Slate
    "Unknown": "#94A3B8"                # Light Slate
}


def generate_interactive_graph_html(
    graph_store: BaseGraphStore,
    filter_doc_id: Optional[str] = None,
    filter_entity_type: Optional[str] = None,
    height_px: int = 600,
    is_dark_theme: bool = False
) -> str:
    """
    Constructs an interactive Vis.js standalone HTML component for the Knowledge Graph
    adapted to the current Light/Dark theme.
    """
    text_color = "#94A3B8" if is_dark_theme else "#64748B"
    if not hasattr(graph_store, "graph") or graph_store.graph.number_of_nodes() == 0:
        return f"""
        <div style="padding: 40px; text-align: center; color: {text_color}; font-family: sans-serif;">
            <h3>🕸️ Knowledge Graph is currently empty</h3>
            <p>Upload enterprise documents to automatically extract and visualize entities and relationships.</p>
        </div>
        """

    nodes_data = []
    edges_data = []
    node_set = set()

    node_border_color = "#334155" if is_dark_theme else "#232F3E"
    edge_font_bg = "#1E293B" if is_dark_theme else "#FFFFFF"
    edge_font_color = "#F8FAFC" if is_dark_theme else "#232F3E"
    edge_line_color = "#64748B" if is_dark_theme else "#94A3B8"

    for node_id, data in graph_store.graph.nodes(data=True):
        etype = data.get("entity_type", "Concept")
        prov_list = data.get("provenance", [])

        # Apply Document Filter
        if filter_doc_id and filter_doc_id != "ALL":
            if not any(p.get("doc_id") == filter_doc_id for p in prov_list):
                continue

        # Apply Entity Type Filter
        if filter_entity_type and filter_entity_type != "ALL":
            if etype != filter_entity_type:
                continue

        node_set.add(node_id)
        name = data.get("name", node_id)
        color = ENTITY_TYPE_COLORS.get(etype, "#64748B")
        
        # Build hover tooltip HTML
        prov_sources = ", ".join(list(set([p.get("file_name", "doc") for p in prov_list]))) or "None"
        aliases_str = ", ".join(data.get("aliases", [])) or name
        
        tooltip = f"<b>{name}</b><br/>Type: {etype}<br/>Aliases: {aliases_str}<br/>Sources: {prov_sources}"

        nodes_data.append({
            "id": node_id,
            "label": name,
            "title": tooltip,
            "color": {
                "background": color,
                "border": node_border_color,
                "highlight": {"background": "#FF9900", "border": node_border_color}
            },
            "font": {"color": "#FFFFFF", "size": 13, "face": "Inter, sans-serif"},
            "shape": "box",
            "margin": 10,
            "shadow": True
        })

    for u, v, k, data in graph_store.graph.edges(keys=True, data=True):
        if u in node_set and v in node_set:
            rel = data.get("relation_type", "RELATED_TO")
            conf = data.get("confidence", 1.0)
            prov_list = data.get("provenance", [])
            p_source = prov_list[0].get("file_name", "doc") if prov_list else "Unknown"
            
            tooltip = f"<b>Relation:</b> {rel}<br/>Confidence: {conf:.2f}<br/>Source: {p_source}"

            edges_data.append({
                "from": u,
                "to": v,
                "label": rel,
                "title": tooltip,
                "arrows": "to",
                "color": {"color": edge_line_color, "highlight": "#FF9900"},
                "font": {"color": edge_font_color, "size": 10, "align": "middle", "background": edge_font_bg},
                "smooth": {"type": "curvedCW", "roundness": 0.15},
                "width": max(1, int(conf * 2.5))
            })

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)

    bg_color = "#131A22" if is_dark_theme else "#FFFFFF"
    container_bg = "#1E293B" if is_dark_theme else "#FAFAFA"
    container_border = "#334155" if is_dark_theme else "#E2E8F0"
    legend_bg = "#1E293B" if is_dark_theme else "#FFFFFF"
    legend_text = "#F8FAFC" if is_dark_theme else "#232F3E"

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                background-color: {bg_color};
                overflow: hidden;
            }}
            #network-container {{
                width: 100%;
                height: {height_px}px;
                border: 1px solid {container_border};
                border-radius: 10px;
                background-color: {container_bg};
            }}
            .legend-bar {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                padding: 8px 14px;
                background: {legend_bg};
                border: 1px solid {container_border};
                border-bottom: none;
                border-radius: 8px 8px 0 0;
                font-size: 11px;
                font-weight: 600;
                color: {legend_text};
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            .legend-dot {{
                width: 10px;
                height: 10px;
                border-radius: 2px;
            }}
        </style>
    </head>
    <body>
        <div class="legend-bar">
            <div class="legend-item"><div class="legend-dot" style="background: #146EB4;"></div> Security Control</div>
            <div class="legend-item"><div class="legend-dot" style="background: #FF9900;"></div> Policy</div>
            <div class="legend-item"><div class="legend-dot" style="background: #8B5CF6;"></div> Organization</div>
            <div class="legend-item"><div class="legend-dot" style="background: #10B981;"></div> Framework Pillar</div>
            <div class="legend-item"><div class="legend-dot" style="background: #64748B;"></div> Concept / Entity</div>
        </div>
        <div id="network-container"></div>

        <script type="text/javascript">
            const nodes = new vis.DataSet({nodes_json});
            const edges = new vis.DataSet({edges_json});
            const container = document.getElementById('network-container');

            const data = {{ nodes: nodes, edges: edges }};
            const options = {{
                nodes: {{
                    borderWidth: 1,
                    shadow: true
                }},
                edges: {{
                    shadow: false
                }},
                physics: {{
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {{
                        gravitationalConstant: -35,
                        centralGravity: 0.005,
                        springLength: 130,
                        springConstant: 0.18
                    }},
                    maxVelocity: 30,
                    minVelocity: 0.1,
                    stabilization: {{ iterations: 120 }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 100,
                    zoomView: true,
                    dragView: true,
                    navigationButtons: true
                }}
            }};

            const network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    return html_template
