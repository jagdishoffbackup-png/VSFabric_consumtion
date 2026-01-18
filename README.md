# 🧵 Fabric Consumption Estimator Pro

An advanced marker making and fabric consumption estimation tool for garment professionals. This tool supports PDF and DXF patterns, handles multi-fabric compositions, and features high-efficiency "True Shape" nesting.

## ✨ Key Features
*   **Multi-Format Support**: Upload `.pdf` or `.dxf` patterns.
*   **True Shape Nesting (V3.0)**: Advanced pixel-based engine that allows pieces to interlock, reducing waste by up to 15% compared to bounding-box methods.
*   **Costing**: Real-time cost calculation based on marker length.
*   **Smart Auto-Rotation**: Automatically aligns pieces to their minimum area bounding box for maximum efficiency.
*   **Cross-Size Grouping**: Smartly groups similar shapes across different sizes for rapid configuration.
*   **Professional Exports**: Export markers to PDF, DXF, HPGL, and detailed Excel reports with thumbnails.

## 🚀 Getting Started

### Prerequisites
*   Python 3.9+

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/fabric-estimator.git
    cd fabric-estimator
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App
```bash
streamlit run app.py
```

## 🛠️ Tech Stack
*   **Frontend**: Streamlit
*   **Geometry**: Shapely, NumPy, Pillow
*   **Parsing**: PyMuPDF (fitz), ezdxf
*   **Packing**: rectpack (Standard Mode), Custom Pixel-Engine (Advanced Mode)

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
