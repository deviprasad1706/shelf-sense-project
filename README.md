# 🛒 ShelfSense Edge (PS ID: SIH26179)

### 🏆 Smart India Hackathon Shortlisted Prototype Build
An optimized, privacy-compliant Edge AI retail checkout simulation pipeline built using **YOLOv8 (SKU-110k Retail Pre-trained)**, **ONNX Runtime**, and **Streamlit**.

## 💡 Core Innovation: The Under-Shelf Architecture
Traditional smart retail camera layouts rely on distant, front-facing aisle cameras that suffer from severe **occlusion** (front items blocking the view of empty spaces behind them) and violate data privacy standards by capturing consumer faces.

**ShelfSense Edge** transitions the computing perspective to localized, interior edge nodes:
- **Zero Occlusion:** Micro-camera lenses are mounted flush underneath upper shelf tiers pointing down, maintaining a clear line of sight over the tray floor.
- **Privacy by Design (DPDP Act, 2023 Compliance):** The structural field of view focuses entirely on product shapes and hands; shopper faces are never captured or processed.
- **Cross-Camera Billing Fusion:** Localized grab events are instantly cross-referenced with overhead CCTV tracking coordinates to automate digital customer billing ledgers seamlessly.

## 🛠️ Software Simulation Tech Stack
- **AI Inferencing Engine:** Ultralytics YOLOv8 compiled via **ONNX Runtime Engine (`model.onnx`)**
- **Dataset Focus:** SKU-110k (Dense Retail Object Matrix)
- **Frontend Dashboard Interface:** Streamlit UI Framework
- **Temporal Count Smoothing Layer:** Python Collections Frame Buffer Array Matrix
- **Core Video Pipeline:** OpenCV (`cv2`)
