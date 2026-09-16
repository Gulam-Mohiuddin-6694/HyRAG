import os
import base64
from typing import Optional

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
LOGO_SVG_PATH = os.path.join(ASSETS_DIR, "HyRAG_logo.svg")


def get_logo_svg(is_dark: bool = False) -> str:
    """Loads and returns the authoritative HyRAG vector logo SVG from assets/HyRAG_logo.svg."""
    for path in [LOGO_SVG_PATH, os.path.join("assets", "HyRAG_logo.svg")]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass

    # Fallback SVG if file is unreachable
    text_color = "#F8FAFC" if is_dark else "#0F172A"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 90" width="100%" height="100%">
  <text x="10" y="55" font-family="'Inter', sans-serif" font-size="34" font-weight="900" fill="{text_color}">Hy<tspan fill="#FF9900">RAG</tspan></text>
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
    """Returns Base64 encoded Data URI of the updated HyRAG logo from assets/HyRAG_logo.svg."""
    for path in [LOGO_SVG_PATH, os.path.join("assets", "HyRAG_logo.svg")]:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                return f"data:image/svg+xml;base64,{encoded}"
            except Exception:
                pass

    svg_str = get_logo_svg(is_dark)
    encoded = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def get_icon_svg_base64(is_dark: bool = False) -> str:
    """Returns Base64 encoded Data URI of the compact square icon."""
    svg_str = get_icon_svg(is_dark)
    encoded = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{encoded}"


def render_logo_html(width: int = 180, height: Optional[int] = None, is_dark: bool = False) -> str:
    """Generates an inline HTML element embedding the crisp vector logo from assets/."""
    b64 = get_logo_svg_base64(is_dark)
    h_style = f"height: {height}px;" if height else "height: auto;"
    w_style = f"width: {width}px;" if width else "width: auto;"
    return f'<img src="{b64}" alt="HyRAG Logo" style="display: block; object-fit: contain; vertical-align: middle; {w_style} {h_style}" />'


def render_icon_html(size: int = 34, is_dark: bool = False) -> str:
    """Generates an inline HTML element embedding the compact icon."""
    b64 = get_icon_svg_base64(is_dark)
    return f'<img src="{b64}" width="{size}" height="{size}" alt="HyRAG Icon" style="display: inline-block; object-fit: contain; vertical-align: middle; border-radius: 8px;" />'
