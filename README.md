# Tile2Net Inspector

A web-based visualization and analysis tool for inspecting Tile2Net network predictions. This tool provides interactive maps, quality metrics, and detailed analysis of predicted pedestrian networks.

## Links

- **Tile2Net Repository**: https://github.com/VIDA-NYU/tile2net
- **Presentation Slides**: [slides/Tile2Net Inspector.pptx](slides/Tile2Net%20Inspector.pptx)

## Features

- **Interactive Map View**: Visualize predicted networks, ground truth, and analysis results on an interactive map
- **Quality Metrics**: Calculate IoU, precision, recall, F1 score, and mean displacement error (MDE)
- **Network Topology Analysis**: Analyze deadends, intersections, and network connectivity
- **Tile-level Analysis**: Detailed metrics and visualization for individual tiles
- **Fault Detection**: Identify true positives, false positives, and false negatives

## Semantic Segmentation Requirements

Same as Tile2Net:

**Hardware requirements:**
- `==1 CUDA-enabled GPU for inference`

**Software requirements:**
- `CUDA==11.7`
- `Python==3.10.9`
- `pytorch==2.0.0`

## Installation

Install dependencies:
```bash
pip install -r requirements-dev.txt
```

## Usage

Run the application:
```bash
python interface.py
```

The web interface will be available at `http://localhost:8000/app/`

