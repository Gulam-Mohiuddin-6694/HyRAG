# HyRAG Enterprise Design System (DESIGN.md)

## 1. Brand Identity & Visual Language
HyRAG is an enterprise-grade, hallucination-aware Hybrid Graph RAG platform. The visual identity draws inspiration from top-tier enterprise cloud architectures (AWS Management Console, Stripe, Datadog), prioritizing clarity, legibility, restrained sophistication, and instant data scannability.

### 1.1 Color Palette
* **Primary Brand / Action:** `#FF9900` (AWS Enterprise Orange / Amber)
  - Hover: `#EC7211`
  - Active / Focus Ring: `rgba(255, 153, 0, 0.25)`
* **Primary Surface / Headers:** `#232F3E` (AWS Squid Ink Charcoal)
* **Secondary Surface:** `#131A22` (Deep Charcoal Navy)
* **Accent Tech / Graph Blue:** `#146EB4` (AWS Console Blue), `#0284C7` (Sky 600)
* **Status Success (Grounded / High Confidence):** `#10B981` (Emerald 500)
* **Status Warning (Low Grounding / Conflict):** `#F59E0B` (Amber 500)
* **Status Danger (Hallucination Risk / Deletion):** `#EF4444` (Rose 500)
* **Background Neutrals:**
  - Page Background: `#F8FAFC` (Slate 50)
  - Card & Container Surface: `#FFFFFF` (Pure White)
  - Subtle Fill / Hover: `#F1F5F9` (Slate 100)
  - Border Primary: `#E2E8F0` (Slate 200)
  - Border Subtle: `#F1F5F9` (Slate 100)
* **Typography Colors:**
  - Headings & Primary Text: `#0F172A` (Slate 900)
  - Body Text: `#1E293B` (Slate 800)
  - Muted / Secondary: `#64748B` (Slate 500)
  - Inverted Text: `#FFFFFF` (Pure White)

### 1.2 Typography
* **Font Family:** `'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
* **Hierarchy:**
  - Page Title: `24px / 1.3`, Weight 800 (Bold)
  - Section Heading: `18px / 1.4`, Weight 700 (SemiBold)
  - Card Heading: `14px / 1.4`, Weight 600
  - Body Regular: `14px / 1.5`, Weight 400
  - Small / Metadata / Captions: `12px / 1.4`, Weight 500
  - Monospace (Entities, Doc IDs): `'JetBrains Mono', 'Fira Code', Consolas, monospace`, `12.5px`

### 1.3 Spacing & Radius System
* **Radius:**
  - Inputs & Small Buttons: `6px`
  - Standard Cards & Containers: `10px`
  - Badges & Pills: `20px` (Full pill)
* **Shadows:**
  - Card Elevation 1: `0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05)`
  - Card Elevation 2 (Hover/Modal): `0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.07)`
  - Modal / Login Dialog: `0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.04)`

---

## 2. Component Specifications

### 2.1 Logo & Top Navigation Bar
* **Height:** `56px`
* **Layout:** Fixed flexbar, `align-items: center; justify-content: space-between;`
* **Branding:** Vector SVG symbol (tri-hybrid node cluster) + crisp `HyRAG` logotype with subtitle `Enterprise Knowledge Engine`.
* **User Badge:** Clean badge displaying user full name, ID, department, and role pill.
* **Logout Button:** Compact secondary outlined button with icon and hover effect.

### 2.2 Login Experience
* **Card Dimensions:** Width `420px`, Auto height, zero unnecessary scroll, horizontally and vertically centered in a clean viewport.
* **Layout Structure:**
  1. Top Branding: Centered vector logo with platform title and clean subtitle.
  2. Input Fields:
     - Employee ID / Username (Clean icon prefix, height `42px`, crisp border, active focus ring).
     - Password (Clean icon prefix, height `42px`, show/hide password toggle).
  3. Action CTA: Full-width high-contrast orange button (`#FF9900`), height `44px`, bold font, subtle hover brightness.
  4. Security Note: Clean minimal trust badge (`256-bit Encrypted Session • Role-Based Access`).
  5. Test Accounts: Collapsible clean drawer with formatted credentials table.

### 2.3 KPI & Metric Cards
* **Grid:** 4-column responsive grid.
* **Structure:**
  - Header: Icon + Metric Label (`12.5px`, SemiBold, `#64748B`)
  - Value: Large readable number (`24px`, ExtraBold, `#0F172A`)
  - Subtitle: Grounding/Lineage indicator (`12px`, `#10B981` / `#146EB4`)
  - Progress Indicator: Bottom 3px color bar corresponding to status.

### 2.4 Knowledge Assistant & Query Results
* **Input Box:** Enterprise search bar with clean rounded borders, placeholder text, and prominent search action.
* **Answer Card:** Clean white card with crisp border, structured markdown typography, callout alerts, and highlighted inline source tags.
* **Graph Evidence Card:** Left-bordered AWS Blue callout (`#146EB4`) presenting multi-hop traversal paths with monospace node badges.
* **Source Citations:** Modular document source rows with file type icon, document name, page badge, and verification status.

### 2.5 Admin Console
* **Navigation:** Clean unified tabs (`Overview`, `Enterprise Search`, `Document Management`, `Knowledge Graph`, `Audit Logs`).
* **Document Ingestion:** Drag-and-drop zone with supported formats pill (`PDF, TXT, MD, DOCX, CSV`), real-time status bar, and active registry table with delete actions.
* **Graph Explorer:** Vis.js interactive network container with color-coded entity legend, zoom/pan controls, and document filter dropdowns.
* **Audit History:** Data table with search filter and expandable execution details.
