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

## Verification Plan
### Automated Tests
- I cannot easily run automated UI tests for Streamlit in this environment.
- I will verify the logic by ensuring the code assumes 72 DPI (standard PDF) unless coordinates suggest otherwise.

### Manual Verification
- Code review of the area calculation logic.
- Ensure all imports are present.
