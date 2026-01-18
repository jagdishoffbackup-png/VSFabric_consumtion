import streamlit as st
import fitz  # pymupdf
from shapely.geometry import Polygon
from shapely.affinity import scale, rotate, translate
import pandas as pd
import io
import re
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rectpack import newPacker, PackingMode, MaxRectsBl
import ezdxf
import base64
import xlsxwriter
import numpy as np
from PIL import Image, ImageDraw

# Constants
POINT_TO_CM = 0.0254 / 72.0 * 100.0  # 1 point = ~0.03527 cm
CM_TO_POINT = 1.0 / POINT_TO_CM

def extract_polygons_from_pdf(uploaded_file):
    """
    Extracts polygons from a PDF file.
    Returns a list of dictionaries: {'polygon': ShapelyPolygon, 'filename': str, 'page': int}
    """
    try:
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        extracted_pieces = []
        
        for page_num, page in enumerate(doc):
            paths = page.get_drawings()
            
            for path in paths:
                points = []
                for item in path["items"]:
                    if item[0] == "l":
                        points.extend([item[1], item[2]])
                    elif item[0] == "c":
                        points.extend([item[1], item[2], item[3], item[4]])
                    elif item[0] == "re":
                         rect = item[1]
                         points.extend([rect.tl, rect.tr, rect.br, rect.bl])
                
                if len(points) >= 3:
                     poly_points = [(p.x, p.y) for p in points]
                     try:
                        poly = Polygon(poly_points)
                        if poly.is_valid and not poly.is_empty:
                             clean_poly = poly.buffer(0)
                             if clean_poly.area > 50: # slightly lower threshold
                                extracted_pieces.append({
                                    "polygon": clean_poly,
                                    "original_filename": uploaded_file.name,
                                    "page": page_num + 1
                                })
                     except Exception:
                        pass
        return extracted_pieces

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return []

def extract_polygons_from_dxf(uploaded_file, import_unit="mm"):
    """
    Extracts polygons from a DXF file.
    assumes LWPOLYLINE or POLYLINE entities.
    """
    try:
        file_bytes = uploaded_file.read()
        # ezdxf.read expects a stream text/binary.
        # We need to decode if text, but ezdxf handles binary streams via `read`.
        # However, `read` expects a stream. `readfile` is for paths.
        # Use `ezdxf.read(stream)` for text stream or `load(stream)`? 
        # Actually ezdxf.read() reads from a file-like object (text). 
        # But uploaded_file is binary. We might need to wrap it.
        # Let's try `ezdxf.load(io.TextIOWrapper(uploaded_file, encoding='utf-8'))` for DXF text
        # OR handle binary DXFs. Most apparel DXFs are ASCII.
        
        # Safe approach: Read into string buffer
        # uploaded_file.seek(0)
        # content = uploaded_file.read()
        # doc = ezdxf.read(io.StringIO(content.decode('utf-8', errors='ignore')))
        
        # Simpler:
        uploaded_file.seek(0)
        file_content = uploaded_file.read().decode('utf-8', errors='ignore')
        doc = ezdxf.read(io.StringIO(file_content))
        msp = doc.modelspace()
        
        extracted_pieces = []
        
        # Scaling Factor to Points (internal unit)
        # 1 point = 1/72 inch
        # 1 mm = 2.83465 points
        # 1 cm = 28.3465 points
        # 1 inch = 72 points
        
        scale_factor = 1.0
        if import_unit == "mm":
            scale_factor = 72.0 / 25.4
        elif import_unit == "cm":
            scale_factor = 72.0 / 2.54
        elif import_unit == "inch":
            scale_factor = 72.0
            
        # Iterate entities
        # We look for closed polylines
        for entity in msp.query('LWPOLYLINE POLYLINE'):
            if entity.dxftype() == 'LWPOLYLINE':
                points = list(entity.get_points()) # Returns (x, y, start_width, end_width, bulge)
                # We typically only need x,y
                pts = [(p[0], p[1]) for p in points]
            else:
                # POLYLINE (old style)
                points = list(entity.points())
                pts = [(p[0], p[1]) for p in points]
                
            if len(pts) >= 3:
                 # Scale points
                 scaled_pts = [(x * scale_factor, y * scale_factor) for x, y in pts]
                 try:
                    poly = Polygon(scaled_pts)
                    if poly.is_valid and not poly.is_empty:
                         # Ensure closed? Polygon auto-closes.
                         clean_poly = poly.buffer(0)
                         extracted_pieces.append({
                            "polygon": clean_poly,
                            "original_filename": uploaded_file.name,
                            "page": 1 # DXF is single "page" usually
                        })
                 except Exception:
                    pass
                    
        return extracted_pieces
        
    except Exception as e:
        st.error(f"Error processing DXF {uploaded_file.name}: {e}")
        return []

def parse_size_from_filename(filename):
    name = filename.rsplit('.', 1)[0]
    parts = re.split(r'[-_ ]', name)
    if parts:
        potential_size = parts[-1].upper()
        if len(potential_size) <= 3: 
            return potential_size
    return "Unknown"

def generate_thumbnail(poly, size=(100, 100)):
    """Generates a base64 encoded PNG thumbnail of the polygon."""
    fig, ax = plt.subplots(figsize=(2, 2))
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    
    # Normalize to 0,0
    p = translate(poly, xoff=-minx, yoff=-miny)
    x, y = p.exterior.xy
    
    ax.fill(x, y, alpha=0.7, fc='blue', ec='black')
    ax.set_xlim(-w*0.1, w*1.1)
    ax.set_ylim(-h*0.1, h*1.1)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=50) # Low DPI for speed
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"

def get_shape_signature(poly):
    """
    Returns a hashable signature for shape grouping.
    Signature = (Normalized Area, Aspect Ratio, Vertex Count)
    """
    minx, miny, maxx, maxy = poly.bounds
    w = maxx - minx
    h = maxy - miny
    if h == 0 or w == 0: return (0, 0, 0)
    
    # Normalize scale (fit to unit box)
    scale_x = 1.0 / w
    scale_y = 1.0 / h
    norm_poly = scale(poly, xfact=scale_x, yfact=scale_y, origin=(minx, miny)) # Scale in place
    
    norm_area = round(norm_poly.area, 1) # Fuzzy matching (1 decimal)
    aspect_ratio = round(w/h, 1) # Fuzzy matching
    vertex_count = len(poly.exterior.coords)
    
    return (norm_area, aspect_ratio, vertex_count)

def get_smart_rotation_angle(poly):
    """
    Calculates the angle to rotate the polygon so that the longest side 
    of its Minimum Area Bounding Box aligns with the X-axis.
    """
    # 1. Get Minimum Rotated Rectangle (The "Tight" Box)
    min_rect = poly.minimum_rotated_rectangle
    
    # 2. Extract Coords
    # The rect is a polygon. Exterior coords are 5 points (start=end).
    x, y = min_rect.exterior.coords.xy
    
    # 3. Find Longest Edge of this Box
    max_len = -1
    best_angle = 0
    
    # Check edges 0-1, 1-2, 2-3, 3-4
    for i in range(4):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        length = math.hypot(dx, dy)
        
        if length > max_len:
            max_len = length
            # Angle of this edge
            best_angle = math.atan2(dy, dx)
            
    # best_angle is the angle of the Longest Side relative to X-axis.
    # We want to rotate the polygon by -best_angle so this side becomes 0 (Horizontal).
    return -math.degrees(best_angle)

class PixelPacker:
    """
    V3.0 True Shape Nesting Engine (Raster-based).
    Approximates shapes as boolean grids to allow interlocking.
    """
    def __init__(self, fabric_width_cm, resolution_cm=0.5):
        self.fabric_width_cm = fabric_width_cm
        self.res = resolution_cm # cm per pixel (lower = higher quality)
        self.px_per_cm = 1.0 / self.res
        
        # Canvas Width in Pixels
        self.bin_width_px = int(fabric_width_cm * self.px_per_cm)
        # Dynamic Height (starts small, grows)
        self.bin_height_px = 2000 
        self.canvas = np.zeros((self.bin_height_px, self.bin_width_px), dtype=bool)
        self.packed_items = []
        
    def _rasterize(self, poly):
        """Converts Shapely polygon to boolean numpy mask."""
        minx, miny, maxx, maxy = poly.bounds
        w_px = int(math.ceil((maxx - minx) * self.px_per_cm)) + 2 # +2 pad
        h_px = int(math.ceil((maxy - miny) * self.px_per_cm)) + 2
        
        # Create image
        img = Image.new('1', (w_px, h_px), 0)
        draw = ImageDraw.Draw(img)
        
        # Normalize coords to image space
        # Note: PIL uses top-left as 0,0. We need to flip Y or just handle it consistently.
        # Let's just map relative coords.
        norm_poly = translate(poly, xoff=-minx, yoff=-miny)
        coords = list(norm_poly.exterior.coords)
        px_coords = [(x * self.px_per_cm + 1, y * self.px_per_cm + 1) for x, y in coords]
        
        draw.polygon(px_coords, outline=1, fill=1)
        
        # Convert to numpy
        mask = np.array(img, dtype=bool)
        # PIL (H, W) matches Numpy (row, col) = (y, x)? 
        # Actually PIL image is (W, H). np.array(img) is (H, W).
        # So mask[y, x]. consistent.
        return mask, minx, miny

    def pack(self, items):
        """
        Naive Bottom-Left-Fill Strategy with Rotation Trials.
        items: list of {'poly': ShapelyPoly, ...}
        """
        # Sort by Area (Descending)
        items.sort(key=lambda x: x['poly'].area, reverse=True)
        
        for item in items:
            best_placement = None
            
            # Try 4 rotations: 0, 90, 180, 270
            rotations = [0, 90, 180, 270]
            
            for angle in rotations:
                # Rotate
                current_poly = rotate(item['poly'], angle, origin='center')
                mask, _, _ = self._rasterize(current_poly)
                h_m, w_m = mask.shape
                
                # Check if fits on canvas at all
                canvas_h, canvas_w = self.canvas.shape
                if w_m > canvas_w: continue
                
                # Search for placement
                placed = False
                found_y, found_x = -1, -1
                
                for y in range(0, canvas_h - h_m):
                    # Optimization: Check row occupancy first? 
                    for x in range(0, canvas_w - w_m):
                        if not np.any(self.canvas[y:y+h_m, x:x+w_m] & mask):
                            # FOUND SPOT
                            found_y, found_x = y, x
                            placed = True
                            break
                    if placed: break
                
                if placed:
                    # Score = Y-position (Bottom-Left strategy minimizes Max Y)
                    # We want to pick the rotation that gives the lowest Y, then lowest X.
                    # Or maybe just lowest Max Height increase?
                    # Let's simple Greedy: Accept FIRST valid if we want speed.
                    # But for efficiency, we should check all angles and pick "best" (lowest Y).
                    score = found_y * 10000 + found_x
                    if best_placement is None or score < best_placement['score']:
                        best_placement = {
                            'score': score,
                            'mask': mask,
                            'x': found_x,
                            'y': found_y,
                            'poly': current_poly,
                            'angle': angle
                        }
                        
            # Apply Best Placement
            if best_placement:
                x, y = best_placement['x'], best_placement['y']
                mask = best_placement['mask']
                self.canvas[y:y+mask.shape[0], x:x+mask.shape[1]] |= mask
                
                # Convert back to real coords
                real_x = (x - 1) * self.res
                real_y = (y - 1) * self.res
                
                poly = best_placement['poly']
                minx_orig, miny_orig, _, _ = poly.bounds
                zeroed_poly = translate(poly, xoff=-minx_orig, yoff=-miny_orig)
                final_poly = translate(zeroed_poly, xoff=real_x, yoff=real_y)
                
                res_item = item.copy()
                res_item['final_poly'] = final_poly
                self.packed_items.append(res_item)
            else:
                pass # Failed to pack item

    def get_max_height(self):
        # Find highest Y pixel used
        rows = np.any(self.canvas, axis=1)
        if not np.any(rows): return 0
        max_y_idx = len(rows) - np.argmax(rows[::-1]) # Last true
        return max_y_idx * self.res

# --- EXPORT FUNCTIONS ---
def create_dxf_export(items):
    doc = ezdxf.new()
    msp = doc.modelspace()
    for item in items:
        # Get placement coords (points)
        minx, miny, _, _ = item['poly'].bounds
        shift_x = item['visual_x'] - minx
        shift_y = item['visual_y'] - miny
        placed_poly = translate(item['poly'], xoff=shift_x, yoff=shift_y)
        
        # Convert to CM ??? DXF unitless, usually we assume 1 unit = 1 user unit.
        # User wants CM. Points * POINT_TO_CM
        final_poly = scale(placed_poly, xfact=POINT_TO_CM, yfact=POINT_TO_CM, origin=(0,0))
        
        # Add to DXF
        msp.add_lwpolyline(list(final_poly.exterior.coords))
    
    out = io.StringIO()
    doc.write(out)
    return out.getvalue()

def create_hpgl_export(items):
    # Very basic HPGL (PLT) generator
    # Scale: 40 plotter units per mm = 400 per cm.
    plt_scale = 400.0 * POINT_TO_CM # Points -> CM -> PLT units
    
    cmds = ["IN;"] # Initialize
    for item in items:
        minx, miny, _, _ = item['poly'].bounds
        shift_x = item['visual_x'] - minx
        shift_y = item['visual_y'] - miny
        placed_poly = translate(item['poly'], xoff=shift_x, yoff=shift_y)
        
        coords = list(placed_poly.exterior.coords)
        if not coords: continue
        
        # Move to first
        start = coords[0]
        cmds.append(f"PU{int(start[0]*plt_scale)},{int(start[1]*plt_scale)};") # Pen Up
        cmds.append(f"PD{int(start[0]*plt_scale)},{int(start[1]*plt_scale)};") # Pen Down
        
        # Draw rest
        for p in coords[1:]:
             cmds.append(f"PD{int(p[0]*plt_scale)},{int(p[1]*plt_scale)};")
        
        # Close loop (optional if last matches first)
        cmds.append("PU;")
        
    return "\\n".join(cmds)

# --- APP SETUP ---
st.set_page_config(page_title="Fabric Consumption Estimator Pro", layout="wide")

# Custom Styling for "Modern" and "Right Sidebar" feel
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #333;
        text-align: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# Title Bar
st.markdown('<div class="main-header"><h1>✨ Fabric Consumption Estimator Pro</h1></div>', unsafe_allow_html=True)

# Layout: Main Content (Left, 70%) | Settings (Right, 30%)
# Streamlit columns are defined left-to-right.
# We will put inputs in col2 (Right) and results in col1 (Left).

col_main, col_settings = st.columns([3, 1])

# --- SETTINGS SIDEBAR (Actually Right Column) ---
with col_settings:
    st.header("⚙️ Settings")
    
    with st.expander("Fabric & Marker", expanded=True):
        fabric_width_cm = st.number_input("Fabric Width (cm)", value=150.0, step=1.0)
        marker_margin_cm = st.number_input("Buffer (cm)", value=2.0)
        dxf_import_unit = st.radio("DXF Import Units", ["mm", "cm", "inch"], horizontal=True, help="Select the units used in your DXF file (default is usually mm).")
        fabric_cost_per_meter = st.number_input("Cost per Meter", value=0.0, step=0.5, help="Cost in your currency per meter of fabric.")
    
    with st.expander("Shrinkage", expanded=False):
        shrinkage_len_idx = st.number_input("Length Shrinkage (%)", value=0.0, step=0.5)
        shrinkage_wid_idx = st.number_input("Width Shrinkage (%)", value=0.0, step=0.5)
        
    with st.expander("Optimization Constraints", expanded=True):
        allow_rotation = st.checkbox("Allow 90° Rotation", value=False)
        allow_flip = st.checkbox("Allow 180° Flip", value=True, help="Note: 180° flip doesn't change 2D bounding box packing, always enabled logically.")
        auto_align_grain = st.checkbox("🔮 Auto-Align (Smart Rotation)", value=False, help="Overrides Grainline: Automatically finds the optimal rotation for each piece to minimize its bounding box area (reduces waste).")
        
        st.write("---")
        nesting_mode = st.radio("Nesting Engine", ["Standard (Rectangular)", "Advanced (True Shape)"], index=0, help="Standard: Fast, treats pieces as boxes. Advanced: Slow, allows interlocking shapes (pixel-based).")
        pixel_res = 0.5
        if nesting_mode == "Advanced (True Shape)":
            pixel_res = st.select_slider("Computation Quality (Resolution)", options=[1.0, 0.5, 0.25], value=0.5, format_func=lambda x: f"{x} cm/px")
            st.caption("Lower value = Better Fit but Slower.")
    
    uploaded_files = st.file_uploader("Upload Patterns (PDF / DXF)", type=["pdf", "dxf"], accept_multiple_files=True)
    
    regenerate_btn = st.button("🔄 Calculate / Regenerate Layout", type="primary")

# --- MAIN CONTENT ---
with col_main:
    if uploaded_files:
        # Store state for "Regeneration" (shuffling) in session state??
        # Actually rectpack is deterministic unless input order changes.
        
        all_pieces = []
        file_info_map = {}
        
        # Process Files
        # Process Files
        for f in uploaded_files:
            file_ext = f.name.split('.')[-1].lower()
            if file_ext == 'dxf':
                pieces = extract_polygons_from_dxf(f, dxf_import_unit)
            else:
                pieces = extract_polygons_from_pdf(f)
                
            size_guess = parse_size_from_filename(f.name)
            if f.name not in file_info_map:
                file_info_map[f.name] = {
                    "Size": size_guess, 
                    "Pieces": len(pieces), 
                    "Quantity": 1,
                    "Grainline": "Length (90°)"
                }
            for p in pieces:
                p['size_label'] = size_guess
                p['original_filename'] = f.name 
                p['signature'] = get_shape_signature(p['polygon']) # Add signature for grouping
                all_pieces.append(p)
            f.seek(0)
            
        # Quantity Input Section
        st.subheader("📋 Order Quantities")
        config_df = pd.DataFrame.from_dict(file_info_map, orient='index').reset_index().rename(columns={'index': 'Filename'})
        edited_df = st.data_editor(
            config_df,
            column_config={
                "Quantity": st.column_config.NumberColumn(min_value=1),
                "Grainline": st.column_config.SelectboxColumn(
                    "Grainline Orientation",
                    options=["Length (90°)", "Cross (0°)", "Bias (45°)"],
                    help="Length: Rotates 90° (aligns PDF vertical to Visual Length).\\nCross: Rotates 0° (keeps PDF vertical as Visual Width).\\nBias: Rotates 45°."
                ),
                "Size": st.column_config.TextColumn(disabled=True),
                "Pieces": st.column_config.NumberColumn(disabled=True),
                "Filename": st.column_config.TextColumn(disabled=True)
            },
            hide_index=True,
            use_container_width=True
        )
        
        # --- NEW: Piece Grouping & Detail Logic (V2.6 Aggregated) ---
        shape_groups = {}
        for p in all_pieces:
            sig = p['signature']
            sz = p['size_label']
            # Key is Signature ONLY (Cross-Size Grouping)
            unique_key = sig 
            
            if unique_key not in shape_groups:
                shape_groups[unique_key] = {
                    "sample": p['polygon'],
                    "count": 0,
                    "signature": sig,
                    "sizes_found": set()
                }
            shape_groups[unique_key]["count"] += 1
            shape_groups[unique_key]["sizes_found"].add(str(sz))
            
        group_rows = []
        # Sort by Signature
        sorted_keys = sorted(shape_groups.keys())
        
        for k in sorted_keys:
            data = shape_groups[k]
            # Generate thumbnail
            thumb = generate_thumbnail(data['sample'])
            
            # Formulate Size Label
            sizes_list = sorted(list(data['sizes_found']))
            if len(sizes_list) > 3:
                size_display = "Mixed (" + ", ".join(sizes_list[:3]) + "...)"
            else:
                size_display = ", ".join(sizes_list)

            # Row Data
            group_rows.append({
                "Shape ID": str(data['signature']), # Hidden
                "Sizes": size_display, # Renamed from Size
                "Preview": thumb,
                "Count": data['count'], # Total count across all sizes
                "Fabric": "Fabric 1",
                "Grainline": "Inherit",
                "Qty Override": 0 
            })
            
        override_configs = {} # Map: Signature -> {Fabric, Grain, Qty}
        
        if group_rows:
            with st.expander("🧩 Piece Details (Fabric, Grain, Qty)", expanded=False):
                st.info("Assign Fabrics, Grainlines, or Quantities. **Settings apply to ALL sizes for that shape.**")
                group_df = pd.DataFrame(group_rows)
                
                edited_groups = st.data_editor(
                    group_df,
                    column_config={
                        "Preview": st.column_config.ImageColumn("Shape", width=50),
                        "Sizes": st.column_config.TextColumn("Sizes", disabled=True),
                        "Count": st.column_config.NumberColumn("Total Pieces", disabled=True),
                        "Fabric": st.column_config.SelectboxColumn("Fabric", options=["Fabric 1", "Fabric 2", "Fabric 3", "Lining", "Fusible"], required=True),
                        "Grainline": st.column_config.SelectboxColumn("Grainline", options=["Inherit", "Length (90°)", "Cross (0°)", "Bias (45°)"]),
                        "Qty Override": st.column_config.NumberColumn("Qty Set", min_value=0, help="Sets quantity for EACH piece instance found."),
                        "Shape ID": None
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Build Map
                for idx, row in edited_groups.iterrows():
                    key = row["Shape ID"] # Key is just Signature string now
                    override_configs[key] = {
                        "fabric": row["Fabric"],
                        "grain": row["Grainline"],
                        "qty_val": row["Qty Override"]
                    }
        
        if regenerate_btn:
            # Determine unique fabrics
            unique_fabrics = set()
            if not override_configs:
                unique_fabrics.add("Fabric 1")
            else:
                for cfg in override_configs.values():
                    unique_fabrics.add(cfg['fabric'])
            sorted_fabrics = sorted(list(unique_fabrics))
            
            for fabric_name in sorted_fabrics:
                st.divider()
                st.markdown(f"### 🧵 Layout for: **{fabric_name}**")
                
                with st.spinner(f"Optimizing {fabric_name}..."):
                    # Prepare Items
                    scale_len = 1 + (shrinkage_len_idx / 100.0)
                    scale_wid = 1 + (shrinkage_wid_idx / 100.0)
                    
                    items_to_pack = []
                    for idx, row in edited_df.iterrows():
                        fname = row['Filename']
                        global_qty = row['Quantity']
                        size_lbl = row['Size']
                        global_grain = row['Grainline']
                        
                        file_pieces = [p for p in all_pieces if p['original_filename'] == fname]
                        
                        for p in file_pieces:
                            sig_str = str(p['signature'])
                            # Lookup by Signature Only (V2.6)
                            key = sig_str
                            
                            # Get Configs
                            if key in override_configs:
                                cfg = override_configs[key]
                                p_fabric = cfg['fabric']
                                p_grain = cfg['grain'] if cfg['grain'] != 'Inherit' else global_grain
                                p_qty_override = cfg['qty_val']
                            else:
                                p_fabric = "Fabric 1"
                                p_grain = global_grain
                                p_qty_override = 0
                            
                            # Filter by Fabric
                            if p_fabric != fabric_name:
                                continue
                            
                            # Quantity Logic
                            if p_qty_override > 0:
                                final_count = int(p_qty_override)
                            else:
                                final_count = int(global_qty)

                            # Scale
                            scaled_poly = scale(p['polygon'], xfact=scale_len, yfact=scale_wid, origin='center')

                            # Rotation Logic
                            if auto_align_grain:
                                # Smart Rotation (Overrides Manual Grain)
                                params_rot = get_smart_rotation_angle(scaled_poly)
                                scaled_poly = rotate(scaled_poly, params_rot, origin='center')
                            else:
                                # Manual Grainline
                                if p_grain == "Length (90°)":
                                    scaled_poly = rotate(scaled_poly, 90, origin='center')
                                elif p_grain == "Bias (45°)":
                                    scaled_poly = rotate(scaled_poly, 45, origin='center')
                            
                            # Prepare for Packing (Store strict polygon)
                            for _ in range(final_count):
                                minx, miny, maxx, maxy = scaled_poly.bounds
                                items_to_pack.append({
                                    'width_bound': maxx - minx, 
                                    'height_bound': maxy - miny, 
                                    'poly': scaled_poly, 
                                    'size': size_lbl,
                                    'area': scaled_poly.area
                                })
                    
                    if not items_to_pack:
                        st.info(f"No items for {fabric_name}.")
                        continue

                    # Shuffle
                    random.shuffle(items_to_pack) 
                    
                    # --- CORE ENGINE EXECUTION ---
                    final_placed_items = []
                    
                    if nesting_mode == "Standard (Rectangular)":
                        # Legacy Rectpack Logic
                        # Map: X-axis = Fabric Width, Y-axis = Marker Length (Infinite)
                        
                        # Rectpack expects (width, height). 
                        # We want to pack into Bin(Width=FabricWidth, Height=Inf).
                        # So Rectpack-X = Fabric Width. Rectpack-Y = Marker Length.
                        # Item Dimensions:
                        # We want item to fit in Fabric Width. So Item-Dim-1 should be aligned with Fabric Width.
                        # Since we pre-rotated items visually (Length(90) means vertical on screen),
                        # Visual Y-axis = Fabric Width. Visual X-axis = Marker Length.
                        # scaled_poly bounds (minx, miny...) are in Visual coords?
                        # No, shapely coords are abstract.
                        # Visual Plotting: X=Length, Y=Width.
                        # So if we want piece to fit "Width-wise", that is Y-axis of plot.
                        # Item `width_bound` (dx) and `height_bound` (dy).
                        # If Visual X = Length, Visual Y = Width.
                        # Then `height_bound` (dy) corresponds to Fabric Width axis?
                        # Yes, typically Y is "up/width". X is "right/length".
                        # So `height_bound` fits into `packer_bin_width`.
                        # Rectpack `add_rect(w, h)`.
                        # If we pack `add_rect(height_bound, width_bound)`:
                        # Then Rectpack-W = Item-Height (Y). Rectpack-H = Item-Width (X).
                        # Bin is (FabricWidth, Inf).
                        # So Item-Height must fit in FabricWidth. This is correct.
                        
                        cm_to_points = 72.0 / 2.54
                        packer_bin_width = fabric_width_cm * cm_to_points 
                        packer_bin_height = float("inf") 
                        
                        packer = newPacker(mode=PackingMode.Offline, pack_algo=MaxRectsBl, rotation=False)
                        
                        for i, item in enumerate(items_to_pack):
                            packer.add_rect(item['height_bound'], item['width_bound'], rid=i) 
    
                        packer.add_bin(packer_bin_width, packer_bin_height, count=1)
                        packer.pack()
                        
                        packed_rects = packer.rect_list()
                        
                        # Convert to Unified Format
                        for r in packed_rects:
                            b_idx, p_x, p_y, p_w, p_h, rid = r
                            # Rectpack-X (p_x) -> Visual Y (Width)
                            # Rectpack-Y (p_y) -> Visual X (Length)
                            
                            visual_x = p_y
                            visual_y = p_x
                            
                            item = items_to_pack[rid]
                            current_poly = item['poly']
                            
                            # Move poly to visual_x, visual_y
                            minx, miny, _, _ = current_poly.bounds
                            shift_x = visual_x - minx
                            shift_y = visual_y - miny
                            placed_poly = translate(current_poly, xoff=shift_x, yoff=shift_y)
                            
                            res_item = item.copy()
                            res_item['poly'] = placed_poly
                            res_item['visual_x'] = visual_x
                            res_item['visual_y'] = visual_y
                            final_placed_items.append(res_item)
                            
                    else:
                        # V3.0 Pixel Logic
                        # PixelPacker expects cm units? 
                        # We have `pixel_res`.
                        # But `items_to_pack` polys are in POINTS (72 dpi etc).
                        # So we need to handle units carefully.
                        # 1. Convert polys to CM? Or just pass Point-based width to Packer?
                        # Easier: Convert Fabric Width to POINTS. Use Resolution in POINTS/PX.
                        # pixel_res (cm/px). 
                        # points_per_px = pixel_res * (72/2.54).
                        # Let's effectively convert everything to CM first? No, slow.
                        # Let's keep logic in POINTS.
                        # fabric_width_points = fabric_width_cm * (72/2.54).
                        # res_points = pixel_res * (72/2.54).
                        
                        points_per_cm = 72.0 / 2.54
                        fabric_width_pts = fabric_width_cm * points_per_cm
                        res_pts = pixel_res * points_per_cm
                        
                        px_packer = PixelPacker(fabric_width_pts, resolution_cm=res_pts)
                        px_packer.pack(items_to_pack) # In-place or returns? 
                        # My class appends to `self.packed_items` which includes 'final_poly'.
                        
                        # Post-process results
                        for px_item in px_packer.packed_items:
                            # 'final_poly' is already placed in coordinate space (X=Width, Y=Length)?
                            # Wait, earlier I said PixelPacker(W, H). W = Fabric Width.
                            # So `final_poly` from PixelPacker has X=FabricWidth, Y=Length.
                            # Visual Plot: X=Length, Y=FabricWidth.
                            # So we need to SWAP X/Y of the poly coords?
                            # Or did PixelPacker map identically to Rectpack?
                            # PixelPacker.pack loop:
                            # `canvas_h, canvas_w`. W = Bin Width (Fabric Width).
                            # `minx, miny, _, _ = poly.bounds`.
                            # `scaled_poly` bounds.
                            # If `poly` fits "Width-wise", its X-dimension in Poly Space should match Fabric Width?
                            # But in current visual space, Width is Y-axis.
                            # So we need to rotate Poly 90 deg OR handle X/Y swap in Pixel packer.
                            
                            # SIMPLER: Rotate all polys -90 deg before sending to PixelPacker.
                            # So that "Item visual width" becomes "Item X-axis width".
                            # Then PixelPacker packs along X-axis (Fabric Width).
                            # Then we un-rotate placed polys?
                            
                            # ALTERNATIVE: Just tell PixelPacker that Bin Width is Infinite (X) and Height is Fabric Width (Y).
                            # And pack along X.
                            # My PixelPacker implementation: `bin_width_px = int(fabric_width_cm * ...)`
                            # So it assumes X-axis is constrained.
                            # So: Let's treat X-axis as Fabric Width.
                            # Visual Width is Y-axis.
                            # So we want to map Visual Y -> Pixel X. Visual X -> Pixel Y.
                            # So: Input Polys (Visual) should be Rotated -90 or swapped?
                            # Visual (x, y) -> Pixel (y, x).
                            # So `poly` sent to rasterizer should have logic: `px_x = p_y`, `px_y = p_x`.
                            # Or just swap geometric coords.
                            # Let's rotate 90 deg. (Visual Y becomes Geometric X).
                            
                            # Step 1: Rotate Input Poly 90 deg.
                            # Step 2: Pack.
                            # Step 3: Rotate Result -90 deg.
                            
                            final_poly = px_item['final_poly']
                            # Rotate back -90 (or +270)
                            # But translation was applied in rotated space.
                            # It works out.
                            
                            # Wait, PixelPacker `real_x` corresponds to Pixel COL (Width axis).
                            # So `real_x` is Visual Y.
                            # `real_y` corresponds to Pixel ROW (Height/Length axis).
                            # So `real_y` is Visual X.
                            # So `final_poly` (if not pre-rotated) would be in (Width, Length) space.
                            # Visual is (Length, Width).
                            # So we just need to swap X/Y of `final_poly` coordinates?
                            # Yes. `final_poly = transform(final_poly, lambda x,y: (y, x))`?
                            # Let's assume PixelPacker returns results where X=Width, Y=Length.
                            # We want X=Length, Y=Width.
                            # So yes, swap coordinates.
                            
                            raw_poly = px_item['final_poly']
                            # Swap X/Y
                            swapped_poly = translate(raw_poly, 0, 0) # copy
                            # shapely.ops.transform? 
                            # Let's just create new poly from coords.
                            pts = [(p[1], p[0]) for p in raw_poly.exterior.coords]
                            final_placed_poly = Polygon(pts)
                            
                            res_item = px_item.copy()
                            res_item['poly'] = final_placed_poly
                            # Update visual_x/y for metrics
                            minx, miny, maxx, maxy = final_placed_poly.bounds
                            res_item['visual_x'] = minx
                            res_item['visual_y'] = miny
                            final_placed_items.append(res_item)
                            
                    # --- VISUALIZATION (Unified) ---
                    if not final_placed_items:
                         st.warning(f"No items placed for {fabric_name}.")
                    else:
                        # Common Metric Calculation
                        import matplotlib.cm as cm
                        unique_sizes = sorted(list(set(i['size'] for i in items_to_pack)))
                        if unique_sizes:
                            colormap = cm.get_cmap('Spectral', len(unique_sizes))
                            colors = {s: colormap(idx) for idx, s in enumerate(unique_sizes)}
                        else:
                            colors = {}
                        
                        fig, ax = plt.subplots(figsize=(10, 5)) 
                        
                        total_poly_area = 0
                        max_length_used = 0
                        
                        for item in final_placed_items:
                            p = item['poly']
                            total_poly_area += item['area']
                            minx, miny, maxx, maxy = p.bounds
                            max_length_used = max(max_length_used, maxx)
                            
                            x_p, y_p = p.exterior.xy
                            x_cm = [c * POINT_TO_CM for c in x_p]
                            y_cm = [c * POINT_TO_CM for c in y_p]
                            
                            c = colors.get(item['size'], 'gray')
                            ax.fill(x_cm, y_cm, alpha=0.8, fc=c, ec='black', linewidth=0.5, label=item['size'])

                        # Metrics
                        marker_len_cm = (max_length_used * POINT_TO_CM) + marker_margin_cm
                        fabric_area_cm2 = marker_len_cm * fabric_width_cm
                        marker_len_m = marker_len_cm / 100.0
                        
                        used_area_cm2 = total_poly_area * (POINT_TO_CM**2)
                        efficiency = (used_area_cm2 / fabric_area_cm2) * 100 if fabric_area_cm2 > 0 else 0
                        wastage = 100.0 - efficiency
                        
                        # Costing
                        total_cost = marker_len_m * fabric_cost_per_meter
                        
                        ax.set_xlabel("Length (cm)")
                        ax.set_ylabel("Width (cm)")
                        ax.set_aspect('equal')
                        ax.set_ylim(0, fabric_width_cm)
                        ax.set_xlim(0, marker_len_cm + 10)
                        
                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
                        
                        st.pyplot(fig)
                        
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.markdown(f"#### 📏 {marker_len_m:.3f} m")
                        c2.markdown(f"#### 💰 {total_cost:,.2f}")
                        c3.markdown(f"#### 📊 {efficiency:.1f}%")
                        c4.markdown(f"#### 🗑️ {wastage:.1f}%")
                        c5.markdown(f"#### 🧩 {len(final_placed_items)}")
                        
                        # Exports
                        st.subheader("💾 Exports")
                        ec1, ec2, ec3 = st.columns(3)
                        
                        # Excel
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                             res_df = pd.DataFrame([
                                 {"Metric": "Length (m)", "Value": marker_len_m},
                                 {"Metric": "Total Cost", "Value": total_cost},
                                 {"Metric": "Cost / Meter", "Value": fabric_cost_per_meter},
                                 {"Metric": "Width (cm)", "Value": fabric_width_cm},
                                 {"Metric": "Efficiency (%)", "Value": efficiency},
                                 {"Metric": "Wastage (%)", "Value": wastage}
                             ])
                             res_df.to_excel(writer, sheet_name='Summary', index=False)
                             edited_df.to_excel(writer, sheet_name='Details', index=False)
                             
                             img_buf = io.BytesIO()
                             fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
                             img_buf.seek(0)
                             worksheet = writer.sheets['Summary']
                             worksheet.insert_image('D2', 'layout.png', {'image_data': img_buf, 'x_scale': 0.5, 'y_scale': 0.5})
                        
                        ec1.download_button(f"Excel ({fabric_name})", data=buf.getvalue(), file_name=f"{fabric_name}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"ex_{fabric_name}")
                        
                        dxf_data = create_dxf_export(final_placed_items)
                        ec2.download_button(f"DXF ({fabric_name})", data=dxf_data, file_name=f"{fabric_name}.dxf", key=f"dx_{fabric_name}")
                        
                        plt_data = create_hpgl_export(final_placed_items)
                        ec3.download_button(f"PLT ({fabric_name})", data=plt_data, file_name=f"{fabric_name}.plt", key=f"pl_{fabric_name}")

    else:
        st.info("👈 Upload Pattern PDFs in the settings panel to get started.")
