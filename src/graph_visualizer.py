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
    text_color = "#94A3B8" if is_dark_theme else "#475569"
    card_bg = "#1E293B" if is_dark_theme else "#FFFFFF"
    card_border = "#334155" if is_dark_theme else "#E2E8F0"

    if not hasattr(graph_store, "graph") or graph_store.graph.number_of_nodes() == 0:
        return f"""
        <div style="padding: 48px; text-align: center; background: {card_bg}; border: 1px dashed {card_border}; border-radius: 10px; color: {text_color}; font-family: 'Inter', system-ui, sans-serif;">
            <div style="font-size: 32px; margin-bottom: 8px;">🕸️</div>
            <div style="font-size: 16px; font-weight: 700; color: {'#F8FAFC' if is_dark_theme else '#0F172A'}; margin-bottom: 4px;">Knowledge Graph is currently empty</div>
            <div style="font-size: 13px;">Upload enterprise documents in Document Management to extract and visualize entities and relationships.</div>
        </div>
        """

    nodes_data = []
    edges_data = []
    node_set = set()

    node_border_color = "#1E293B" if is_dark_theme else "#0F172A"
    edge_font_bg = "#1E293B" if is_dark_theme else "#FFFFFF"
    edge_font_color = "#F8FAFC" if is_dark_theme else "#1E293B"
    edge_line_color = "#64748B" if is_dark_theme else "#94A3B8"

    type_counts = {}

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

        type_counts[etype] = type_counts.get(etype, 0) + 1
        node_set.add(node_id)
        name = data.get("name", node_id)
        color = ENTITY_TYPE_COLORS.get(etype, "#64748B")
        
        # Build hover tooltip HTML
        prov_sources = ", ".join(list(set([p.get("file_name", "doc") for p in prov_list]))) or "None"
        aliases_str = ", ".join(data.get("aliases", [])) or name
        
        tooltip = f"<div style='font-family:sans-serif;font-size:12px;padding:4px;'><b>{name}</b><br/><span style='color:#38BDF8;'>Type:</span> {etype}<br/><span style='color:#94A3B8;'>Aliases:</span> {aliases_str}<br/><span style='color:#10B981;'>Sources:</span> {prov_sources}</div>"

        nodes_data.append({
            "id": node_id,
            "label": name,
            "title": tooltip,
            "color": {
                "background": color,
                "border": node_border_color,
                "highlight": {"background": "#FF9900", "border": "#FFFFFF"}
            },
            "font": {"color": "#FFFFFF", "size": 12, "face": "Inter, sans-serif", "weight": "600"},
            "shape": "box",
            "margin": {"top": 7, "bottom": 7, "left": 10, "right": 10},
            "shadow": {"enabled": True, "color": "rgba(0,0,0,0.15)", "size": 4, "x": 1, "y": 2}
        })

    for u, v, k, data in graph_store.graph.edges(keys=True, data=True):
        if u in node_set and v in node_set:
            rel = data.get("relation_type", "RELATED_TO")
            conf = data.get("confidence", 1.0)
            prov_list = data.get("provenance", [])
            p_source = prov_list[0].get("file_name", "doc") if prov_list else "Unknown"
            
            tooltip = f"<div style='font-family:sans-serif;font-size:12px;padding:4px;'><b>Relation:</b> {rel}<br/><b>Confidence:</b> {conf:.2f}<br/><b>Source:</b> {p_source}</div>"

            edges_data.append({
                "from": u,
                "to": v,
                "label": rel,
                "title": tooltip,
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
                "color": {"color": edge_line_color, "highlight": "#FF9900"},
                "font": {"color": edge_font_color, "size": 10, "align": "middle", "background": edge_font_bg, "strokeWidth": 2, "strokeColor": card_border},
                "smooth": {"type": "curvedCW", "roundness": 0.15},
                "width": max(1, int(conf * 2.2))
            })

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)

    bg_color = "#0F172A" if is_dark_theme else "#F8FAFC"
    container_bg = "#131A22" if is_dark_theme else "#FFFFFF"
    container_border = "#334155" if is_dark_theme else "#E2E8F0"
    legend_bg = "#1E293B" if is_dark_theme else "#F1F5F9"
    legend_text = "#F8FAFC" if is_dark_theme else "#0F172A"

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&display=swap');
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
                border-radius: 0 0 10px 10px;
                background-color: {container_bg};
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, {'0.3' if is_dark_theme else '0.05'});
            }}
            .legend-bar {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 12px;
                padding: 10px 16px;
                background: {legend_bg};
                border: 1px solid {container_border};
                border-bottom: none;
                border-radius: 10px 10px 0 0;
                font-size: 11.5px;
                font-weight: 600;
                color: {legend_text};
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 6px;
                background: {'#0F172A' if is_dark_theme else '#FFFFFF'};
                padding: 3px 8px;
                border-radius: 12px;
                border: 1px solid {container_border};
            }}
            .legend-dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
            }}
            .legend-count {{
                color: {'#94A3B8' if is_dark_theme else '#64748B'};
                font-size: 10.5px;
                font-weight: 700;
            }}
        </style>
    </head>
    <body>
        <div class="legend-bar">
            <span style="font-weight: 700; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: {'#94A3B8' if is_dark_theme else '#475569'};">Entity Categories:</span>
            <div class="legend-item"><div class="legend-dot" style="background: #146EB4;"></div> Security Control</div>
            <div class="legend-item"><div class="legend-dot" style="background: #FF9900;"></div> Policy</div>
            <div class="legend-item"><div class="legend-dot" style="background: #8B5CF6;"></div> Organization</div>
            <div class="legend-item"><div class="legend-dot" style="background: #10B981;"></div> Framework Pillar</div>
            <div class="legend-item"><div class="legend-dot" style="background: #64748B;"></div> Concept / Entity</div>
            <span style="margin-left: auto; font-size: 11px; color: {'#94A3B8' if is_dark_theme else '#64748B'}; font-weight: 600;">Showing <b>{len(node_set)}</b> entities • <b>{len(edges_data)}</b> relations</span>
        </div>
        <div id="network-container"></div>

        <script type="text/javascript">
            const nodes = new vis.DataSet({nodes_json});
            const edges = new vis.DataSet({edges_json});
            const container = document.getElementById('network-container');

            const data = {{ nodes: nodes, edges: edges }};
            const options = {{
                nodes: {{
                    borderWidth: 1.5,
                    shadow: true
                }},
                edges: {{
                    shadow: false
                }},
                physics: {{
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: {{
                        gravitationalConstant: -38,
                        centralGravity: 0.006,
                        springLength: 120,
                        springConstant: 0.16
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
                    navigationButtons: true,
                    keyboard: false
                }}
            }};

            const network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    return html_template
