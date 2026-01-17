# Fabric Estimator Pro Walkthrough

Version 2.1 introduces Grainline controls to solve orientation mismatches and support Bias cutting.

## New Features
- **Grainline Control**: Specify "Length", "Cross", or "Bias" for each pattern file.
    - **Length (90°)**: Rotates upright PDF pieces 90° to align with Fabric Length (Selvage). Default.
    - **Cross (0°)**: Keeps PDF upright orientation aligned with Fabric Width.
    - **Bias (45°)**: Rotates pieces 45° for bias packing.
- **Improved Engine**: Packing now strictly respects your grainline choices.
- **Pro UI**: Updated title and layout.

## How to Run

1. **Open Terminal** (if not already open).
2. **Navigate to the project directory**:
   ```bash
   cd /Users/jagdishs/.gemini/antigravity/scratch/fabric-estimator
   ```
3. **Run the app**:
   ```bash
   bash run_app.sh
   ```

## Usage
1. **Settings**: Set Fabric Width, Buffer, etc. on the right.
2. **Upload**: Upload PDF patterns.
3. **Configure Order**:
   - **Quantity**: Set number of items per file.
   - **Grainline (Global)**: Set default grainline for the file (Length/Cross/Bias).
   - **Piece Details (Advanced)**:
     - Expand the "Piece Details" section.
     - View thumbnails of grouped piece shapes.
     - Set specific **Grainline Overrides** (e.g., set a Sleeve to "Bias" while keeping the rest "Length").
4. **Calculate**: Click **Calculate / Regenerate Layout**.
5. **Analyze**: View the layout. Note that 45° pieces will appear rotated.
6. **Export**: download reports as needed.
