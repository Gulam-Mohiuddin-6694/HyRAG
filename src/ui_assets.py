"""
ui_assets.py - Authoritative Vector Logos, Icons & Design System Assets for HyRAG.

Provides scalable SVG vector graphics and HTML helpers compliant with DESIGN.md
supporting both Light and Dark themes seamlessly.
"""

import base64


def get_logo_svg(is_dark: bool = False) -> str:
    """Generates the official HyRAG vector logo SVG for Light or Dark theme."""
    text_color = "#F8FAFC" if is_dark else "#232F3E"
    subtitle_bg = "#1E293B" if is_dark else "#F1F5F9"
    subtitle_color = "#94A3B8" if is_dark else "#64748B"
    edge_color = "#475569" if is_dark else "#CBD5E1"
    center_hub_bg = "#0F172A" if is_dark else "#232F3E"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 90" width="100%" height="100%">
  <defs>
    <linearGradient id="hyragGradOrange_{int(is_dark)}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFB800" />
      <stop offset="100%" stop-color="#FF7A00" />
    </linearGradient>
    <linearGradient id="hyragGradBlue_{int(is_dark)}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00C0F3" />
      <stop offset="100%" stop-color="#006699" />
    </linearGradient>
    <filter id="hyragGlow_{int(is_dark)}" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#FF9900" flood-opacity="0.35"/>
    </filter>
  </defs>

  <!-- Logo Icon: Tri-Hybrid Connected Graph Nodes -->
  <g transform="translate(10, 10)">
    <!-- Node Connectors (Graph Edges) -->
    <line x1="35" y1="18" x2="16" y2="52" stroke="{edge_color}" stroke-width="3" stroke-linecap="round" />
    <line x1="35" y1="18" x2="54" y2="52" stroke="{edge_color}" stroke-width="3" stroke-linecap="round" />
    <line x1="16" y1="52" x2="54" y2="52" stroke="{edge_color}" stroke-width="3" stroke-linecap="round" />
    
    <!-- Pulse Ring around central node -->
    <circle cx="35" cy="35" r="24" fill="none" stroke="#FF9900" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.6"/>

    <!-- Left Node: Dense Vector (Blue) -->
    <circle cx="16" cy="52" r="10" fill="url(#hyragGradBlue_{int(is_dark)})" />
    <circle cx="16" cy="52" r="4" fill="#FFFFFF" />

    <!-- Right Node: BM25 Sparse (Blue) -->
    <circle cx="54" cy="52" r="10" fill="url(#hyragGradBlue_{int(is_dark)})" />
    <circle cx="54" cy="52" r="4" fill="#FFFFFF" />

    <!-- Top Center Node: Knowledge Graph Apex (Orange) -->
    <circle cx="35" cy="18" r="12" fill="url(#hyragGradOrange_{int(is_dark)})" filter="url(#hyragGlow_{int(is_dark)})" />
    <circle cx="35" cy="18" r="5" fill="#FFFFFF" />

    <!-- Center Core Hub: Hallucination-Aware Fusion -->
    <circle cx="35" cy="35" r="7" fill="{center_hub_bg}" stroke="#FF9900" stroke-width="2"/>
  </g>

  <!-- Brand Typography -->
  <g transform="translate(85, 20)">
    <!-- "HyRAG" Wordmark -->
    <text x="0" y="32" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="34" font-weight="900" fill="{text_color}" letter-spacing="-0.8">Hy<tspan fill="#FF9900">RAG</tspan></text>
    
    <!-- Enterprise Tagline / Subtitle Pill -->
    <rect x="0" y="42" width="220" height="18" rx="4" fill="{subtitle_bg}" />
    <text x="8" y="55" font-family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="9.5" font-weight="700" fill="{subtitle_color}" letter-spacing="0.8">ENTERPRISE GRAPH RAG</text>
  </g>
</svg>"""


def get_icon_svg(is_dark: bool = False) -> str:
    """Generates the official HyRAG compact icon."""
    box_bg = "#1E293B" if is_dark else "#232F3E"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 70 70" width="100%" height="100%">
  <defs>
    <linearGradient id="hyragGradOrangeSmall_{int(is_dark)}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFB800" />
      <stop offset="100%" stop-color="#FF7A00" />
    </linearGradient>
    <linearGradient id="hyragGradBlueSmall_{int(is_dark)}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00C0F3" />
      <stop offset="100%" stop-color="#006699" />
    </linearGradient>
  </defs>

  <rect width="70" height="70" rx="14" fill="{box_bg}"/>
  
  <!-- Graph Edges -->
  <line x1="35" y1="18" x2="18" y2="48" stroke="#64748B" stroke-width="2.5" stroke-linecap="round" />
  <line x1="35" y1="18" x2="52" y2="48" stroke="#64748B" stroke-width="2.5" stroke-linecap="round" />
  <line x1="18" y1="48" x2="52" y2="48" stroke="#64748B" stroke-width="2.5" stroke-linecap="round" />

  <!-- Nodes -->
  <circle cx="18" cy="48" r="8" fill="url(#hyragGradBlueSmall_{int(is_dark)})" />
  <circle cx="52" cy="48" r="8" fill="url(#hyragGradBlueSmall_{int(is_dark)})" />
  <circle cx="35" cy="18" r="10" fill="url(#hyragGradOrangeSmall_{int(is_dark)})" />
  <circle cx="35" cy="18" r="4" fill="#FFFFFF" />
  <circle cx="35" cy="35" r="5" fill="#FFFFFF" stroke="#FF9900" stroke-width="2"/>
</svg>"""


def get_logo_svg_base64(is_dark: bool = False) -> str:
    """Returns Base64 encoded Data URI of the primary full SVG logo."""
    svg_str = get_logo_svg(is_dark)
    encoded = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def get_icon_svg_base64(is_dark: bool = False) -> str:
    """Returns Base64 encoded Data URI of the compact square icon."""
    svg_str = get_icon_svg(is_dark)
    encoded = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def render_logo_html(width: int = 190, height: int = 42, is_dark: bool = False) -> str:
    """Generates an inline HTML element embedding the crisp vector logo."""
    b64 = get_logo_svg_base64(is_dark)
    return f'<img src="{b64}" width="{width}" height="{height}" alt="HyRAG Logo" style="display: block; object-fit: contain; vertical-align: middle;" />'


def render_icon_html(size: int = 34, is_dark: bool = False) -> str:
    """Generates an inline HTML element embedding the compact icon."""
    b64 = get_icon_svg_base64(is_dark)
    return f'<img src="{b64}" width="{size}" height="{size}" alt="HyRAG Icon" style="display: inline-block; object-fit: contain; vertical-align: middle; border-radius: 8px;" />'
