"""
pdf_generator.py - Pure Python Minimal Enterprise PDF Generator.

Generates standard, valid PDF documents without requiring reportlab or external C-extensions.
"""

import os

def create_simple_pdf(file_path: str, title: str, sections: list) -> None:
    """
    Creates a standard valid PDF 1.4 document with text content.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Build text content lines
    lines = [f"{title}", ""]
    for heading, body in sections:
        lines.append(f"--- {heading} ---")
        # Split body into lines of max ~80 chars
        words = body.split()
        current_line = []
        for w in words:
            current_line.append(w)
            if len(" ".join(current_line)) > 75:
                lines.append(" ".join(current_line))
                current_line = []
        if current_line:
            lines.append(" ".join(current_line))
        lines.append("")

    # Construct PDF stream
    text_ops = ["BT", "/F1 12 Tf", "50 750 Td", "14 TL"]
    for idx, line in enumerate(lines):
        # Escape parenthesis
        safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if idx == 0:
            text_ops.append(f"/F1 16 Tf ({safe_line}) Tj /F1 12 Tf T*")
        elif line.startswith("---"):
            text_ops.append(f"/F1 13 Tf ({safe_line}) Tj /F1 11 Tf T*")
        else:
            text_ops.append(f"({safe_line}) Tj T*")
    text_ops.append("ET")

    stream_content = "\n".join(text_ops)
    stream_bytes = stream_content.encode("latin-1", "replace")
    stream_len = len(stream_bytes)

    pdf_parts = [
        b"%PDF-1.4\n",
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("latin-1") + stream_bytes + b"\nendstream\nendobj\n",
    ]

    # Calculate xref
    xref_offsets = [0]
    total_len = 0
    for part in pdf_parts:
        total_len += len(part)
        xref_offsets.append(total_len)

    xref_pos = total_len
    xref_section = [
        b"xref\n0 6\n0000000000 65535 f \n",
        f"{0:010d} 00000 n \n".encode("latin-1"),
        f"{len(pdf_parts[0]):010d} 00000 n \n".encode("latin-1"),
        f"{(len(pdf_parts[0]) + len(pdf_parts[1])):010d} 00000 n \n".encode("latin-1"),
        f"{(len(pdf_parts[0]) + len(pdf_parts[1]) + len(pdf_parts[2])):010d} 00000 n \n".encode("latin-1"),
        f"{(len(pdf_parts[0]) + len(pdf_parts[1]) + len(pdf_parts[2]) + len(pdf_parts[3])):010d} 00000 n \n".encode("latin-1"),
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n",
        f"{xref_pos}\n%%EOF\n".encode("latin-1")
    ]

    with open(file_path, "wb") as f:
        for p in pdf_parts:
            f.write(p)
        for x in xref_section:
            f.write(x)
