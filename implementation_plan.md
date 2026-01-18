# Fabric Consumption Estimator Implementation Plan

## Goal Description
Build a single-file Streamlit application (`app.py`) that calculates fabric consumption from VStitcher pattern PDFs. The app will extract vector paths from uploaded PDFs, calculate their area, and estimate fabric yield based on a configurable width and waste factor.

## Proposed Changes

### Configuration
#### [NEW] [requirements.txt](file:///Users/jagdishs/.gemini/antigravity/scratch/fabric-estimator/requirements.txt)
- `streamlit`
- `pymupdf` (imported as `fitz`)
- `shapely`
- `pandas`
- `openpyxl` (for Excel export)

### Application Logic
#### [NEW] [app.py](file:///Users/jagdishs/.gemini/antigravity/scratch/fabric-estimator/app.py)
- **Imports**: `streamlit`, `fitz`, `shapely.geometry`, `pandas`, `io`.
- **Sidebar**:
    - `st.sidebar.number_input` for "Fabric Width (cm)".
- **Core Function `extract_pattern_area(uploaded_file)`**:
    - Open PDF from stream using `fitz.open(stream=..., filetype="pdf")`.
    - Iterate through pages.
    - Use `page.get_drawings()` to find paths.
    - Convert paths to Shapely `Polygon` objects.
        - Handle Bezier curves if necessary (simple approximation or bounding box might be needed, but `get_drawings` returns items that can be converted to points).
        - *Self-Correction*: `pymupdf` drawings can be complex. We will approximate area by converting closed paths to polygons. If paths are just lines, we might need a convex hull or simply sum the areas of closed path items. Given "VStitcher pattern PDFs", these are usually vector outlines. We will assume closed paths represent pattern pieces.
    - Sum the area of all polygons.
    - Return total area in $m^2$ (handling unit conversion from PDF points to meters).
        - PDF points are usually 1/72 inch.
        - 1 point = 0.0352778 cm.
        - 1 point = 0.000352778 m.
        - Area in points^2 * (0.000352778)^2 = Area in m^2.
- **Main UI**:
    - `st.file_uploader` (accept_multiple_files=True).
    - Loop through files, call `extract_pattern_area`.
    - Calculate Yield: $(Area * 1.15) / (Width\_cm / 100)$.
    - Store results in a list of dicts.
    - Create DataFrame.
    - Display DataFrame with `st.dataframe`.
    - `st.download_button` for Excel export.

### Version 2.4 Specification (Costing & DXF)
#### Costing Logic
- **UI**: Add `st.number_input("Cost per Meter", ...)` inside the Fabric & Marker expander.
- **Calculation**: `Total Cost = Marker Length (m) * Cost per Meter`.
- **Display**: Add a metric card for "Estimated Cost".
- **Export**: Add "Unit Cost" and "Total Cost" columns/cells to the Excel "Summary" sheet.

#### DXF Import Logic
- **Upload**: Update `st.file_uploader` to type `["pdf", "dxf"]`.
- **Parsing**: Create `extract_polygons_from_dxf(file_stream)`:
    - Use `ezdxf.read(file_stream)`.
    - Iterate `MSP` (Modelspace).
    - Query `LWPOLYLINE` and `POLYLINE` entities.
    - Extract points `(x, y)`.
    - **Scaling**: DXF is often unitless. We will assume millimeters (common in apparel) or check `$INSUNITS`.
    - **UI**: Add a radio button "DXF Import Units" (mm, cm, inch) to let user verify. Default to `mm`.
    - Convert all to internal points (like PDF) or standardize on CM immediately for the packer.
        - *Decision*: Convert everything to **Points** to match existing PDF logic (1 pt = 1/72 inch) or refactor app to work linearly in CM.
        - *Simpler Approach*: Convert incoming DXF coords -> CM -> Points.
        - 1 mm = 2.83465 points.
        - 1 cm = 28.3465 points.
        - 1 inch = 72 points.
    - Return list of Shapely Polygons.

### Version 2.5 Specification (Smart Rotation)
#### Minimum Area Logic
- **Goal**: Minimize waste by finding the optimal rotation for *each* piece that results in the smallest Rectangular Bounding Box.
- **Constraints**: This changes the grainline. It must be an **Optional Setting** (checkbox: "Auto-Align to Grain (Min Bounding Box)").
- **Implementation**:
    1.  Create helper `get_min_area_rotation_angle(poly)`.
    2.  Use `shapely.minimum_rotated_rectangle(poly)` to get the tightest box.
    3.  Calculate the angle of that box's long edge relative to X-axis.
    4.  Rotate the polygon by `-angle`.
    5.  **Integration**: In the packing loop, if checkbox is checked, apply this rotation *before* standard grainline rotation or packing logic.
    - *Note*: Usually, if this is enabled, it *overrides* standard "Length/Cross" file settings because it claims to find the "True" grain alignment (assuming the piece's bounding box aligns with grain).

### Version 2.6 Specification (Cross-Size Grouping)
#### Goal
Reduce user effort by grouping the "same" piece across different sizes (e.g., "Pocket Bag" for Sizes 36, 38, 40) into a single configuration row.

#### Implementation
1.  **Fuzzy Signature**: Update `get_shape_signature`.
    -   Round `norm_area` to 1 decimal place (instead of 2).
    -   Round `aspect_ratio` to 1 decimal place (instead of 2).
    -   *Result*: Small grading variations will now yield the *same signature*.
2.  **Aggregation Logic**:
    -   Group `all_pieces` by `signature` ONLY (remove `size` from the grouping key).
    -   For each group, collect list of sizes involved (e.g., `["36", "38", "40"]`).
    -   Create one row per signature.
    -   **Column 'Size'**: Display string "Mixed (3 sizes)" or list them.
3.  **Override Application**:
    -   The `override_configs` map will key off `signature` alone.
    -   When applying overrides in the packing loop, look up config by `p['signature']`.

### Version 3.0 Specification (True Shape Nesting)
#### Core Concept
Replace the bounding-box packer (`rectpack`) with a custom **Pixel-Based (Raster) Packer**. This allows pieces to "interlock" (e.g., placing a small piece inside the curve of another), significantly improving efficiency for apparel patterns.

#### Technical Architecture
1.  **Dependencies**: Add `numpy`.
    *   We will use `shapely` for vector operations and `rasterio.features` OR custom logic to rasterize polygons into `numpy` boolean grids.
    *   *Decision*: Use `rasterio` if easy, else write a simple scanline rasterizer to keep deps low. `skimage.draw.polygon` is also an option. Let's start with a custom **CV2**-style rasterizer using `PIL.ImageDraw` (which comes with `streamlit` dependencies usually) or `scikit-image`. Actually, `matplotlib.path` contains point-in-polygon logic which is robust.
    *   *Selection*: `numpy` + `cv2` (opencv-python-headless) or `PIL` is best for speed. Let's use `PIL` (Pillow) since `streamlit` already installs it. `ImageDraw.Draw(img).polygon(...)` produces a mask.

2.  **Resolution Strategy**:
    *   Define `PIXELS_PER_CM = 2` (5mm resolution) or configurable.
    *   Higher resolution = Better fit, Slower performance.

3.  **The Packing Algorithm (Heuristic)**:
    *   **Init**: Create a "Fabric Mask" (large numpy array of Zeros). Width = Fabric Width * Resolution. Height = Growable.
    *   **Queue**: Sort pieces by Area (Descending).
    *   **Loop**: For each piece:
        1.  Rasterize it into a small boolean mask (`piece_mask`).
        2.  **Search**: Slide `piece_mask` over the `fabric_mask` from Bottom-Left (y=0, x=0).
        3.  **Check**: `if not np.any(fabric_mask[y:y+h, x:x+w] & piece_mask)`:
            *   **Placed!**
            *   Update `fabric_mask |= (piece_mask at x,y)`.
            *   Save Coordinates.
            *   Break and move to next piece.

4.  **UI Updates**:
    *   Add "Nesting Mode" Toggle: `["Fast (Rectangular)", "Advanced (True Shape)"]`.
    *   Add "Quality/Resolution" Slider (Low/Med/High).

## Verification Plan
### Automated Tests
- I cannot easily run automated UI tests for Streamlit in this environment.
- I will verify the logic by ensuring the code assumes 72 DPI (standard PDF) unless coordinates suggest otherwise.

### Manual Verification
- Code review of the area calculation logic.
- Ensure all imports are present.
