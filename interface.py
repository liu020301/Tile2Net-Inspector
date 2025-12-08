from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from collections import Counter
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from tqdm import tqdm

from tile2net import Raster


DEFAULT_LOCATION = os.getenv("INSPECTOR_LOCATION", "Boston Common, Boston, MA, US")
OUTPUT_DIR = Path(os.getenv("INSPECTOR_OUTPUT_DIR", "./example_dir"))
NUM_TILES = int(os.getenv("INSPECTOR_NUM_TILES", "4"))
BUFFER_SIZE = float(os.getenv("INSPECTOR_BUFFER_SIZE", "2.0"))
DISTANCE_THRESHOLD_METERS = float(os.getenv("INSPECTOR_DISTANCE_THRESHOLD", "5"))
RESULTS_DIRNAME = "analysis_results"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def mean_displacement_error(gdf_gt: gpd.GeoDataFrame, gdf_pred_network: gpd.GeoDataFrame) -> float:
    # compute average distance between gt and predicted lines
    total_distance = 0.0
    num_comparisons = 0
    print("Calculating Mean Displacement Error (MDE)...")
    for _, gt_row in tqdm(gdf_gt.iterrows(), total=len(gdf_gt)):
        gt_line = gt_row["geometry"]
        for _, pred_row in gdf_pred_network.iterrows():
            pred_line = pred_row["geometry"]
            nearest_geom = nearest_points(gt_line, pred_line)[1]
            distance = gt_line.distance(nearest_geom)
            total_distance += distance
            num_comparisons += 1
    return total_distance / max(num_comparisons, 1)


def evaluate_iou(gt_line, pred_line, buffer_size: float = BUFFER_SIZE) -> Tuple[float, float, float]:
    # buffer both lines and compute intersection/union
    gt_buf = gt_line.buffer(buffer_size)
    pred_buf = pred_line.buffer(buffer_size)

    inter = gt_buf.intersection(pred_buf).area
    union = gt_buf.union(pred_buf).area
    gt_area = gt_buf.area
    pred_area = pred_buf.area

    if union == 0 or gt_area == 0 or pred_area == 0:
        return 0.0, 0.0, 0.0

    iou = inter / union
    precision = inter / pred_area
    recall = inter / gt_area
    return iou, precision, recall


def calculate_iou_scores(gdf_gt: gpd.GeoDataFrame, gdf_pred_network: gpd.GeoDataFrame) -> Tuple[float, float, float]:
    # use spatial index to speed up matching
    gt_geoms = gdf_gt.geometry.values
    pred_geoms = gdf_pred_network.geometry.values
    tree = STRtree(pred_geoms)

    iou_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []

    print("Calculating IoU scores with STRtree...")

    for gt_line in gt_geoms:
        # find nearby predicted lines
        candidates = tree.query(gt_line)
        if len(candidates) == 0:
            iou_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue

        # pick the best match
        best_iou = 0.0
        best_precision = 0.0
        best_recall = 0.0

        for pred_line_idx in candidates:
            pred_line = pred_geoms[pred_line_idx]
            iou, precision, recall = evaluate_iou(gt_line, pred_line)
            if iou > best_iou:
                best_iou = iou
                best_precision = precision
                best_recall = recall

        iou_scores.append(best_iou)
        precision_scores.append(best_precision)
        recall_scores.append(best_recall)

    return (np.mean(iou_scores), np.mean(precision_scores), np.mean(recall_scores))


def ensure_prediction_artifacts(raster: Raster, tile_count: int) -> None:
    # make sure we have tiles and predictions, generate if missing
    tiles_dir = Path(raster.project.tiles.static)
    network_dir = Path(raster.project.network)
    polygons_dir = Path(raster.project.polygons)

    tiles_dir.mkdir(parents=True, exist_ok=True)
    network_dir.mkdir(parents=True, exist_ok=True)
    polygons_dir.mkdir(parents=True, exist_ok=True)

    if not tiles_dir.exists() or not any(tiles_dir.iterdir()):
        print("Tiles missing, generating...")
        raster.generate(tile_count)

    if not any(network_dir.iterdir()) or not any(polygons_dir.iterdir()):
        print("Predictions missing, running inference...")
        raster.inference()


def _latest_child_dir(path: Path) -> Path:
    # find the most recently created subdirectory
    candidates = [p for p in path.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No directories found inside {path}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _first_shapefile(path: Path) -> Path:
    for shp in path.glob("*.shp"):
        return shp
    raise FileNotFoundError(f"No shapefile found inside {path}")


def load_prediction_outputs(raster: Raster) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # load the latest network and polygon shapefiles
    network_parent = Path(raster.project.network)
    polygons_parent = Path(raster.project.polygons)

    network_dir = _latest_child_dir(network_parent)
    network_file = _first_shapefile(network_dir)

    # try to match polygon dir by name, fallback to latest
    polygon_dir_name = network_dir.name.replace("Network", "Polygons")
    polygon_dir = polygons_parent / polygon_dir_name
    if not polygon_dir.exists():
        polygon_dir = _latest_child_dir(polygons_parent)
    polygons_file = _first_shapefile(polygon_dir)

    print(f"Predicted network file: {network_file}")
    print(f"Polygons file: {polygons_file}")

    network = gpd.read_file(network_file)
    polygons = gpd.read_file(polygons_file)
    return network, polygons


def analyze_tiles(
    raster: Raster,
    network: gpd.GeoDataFrame,
    dead_ends: gpd.GeoDataFrame,
    gdf_gt: gpd.GeoDataFrame,
) -> List[Dict]:
    """Analyze each tile for deadends and metrics"""
    # Create tile grid GeoDataFrame - this includes all active tiles
    tile_grid = raster.create_grid_gdf()
    tile_grid = tile_grid.to_crs(4326)
    
    # Ensure all GeoDataFrames are in the same CRS
    network_wgs = network.to_crs(4326) if network.crs != 4326 else network
    dead_ends_wgs = dead_ends.to_crs(4326) if dead_ends.crs != 4326 else dead_ends
    gdf_gt_wgs = gdf_gt.to_crs(4326) if gdf_gt.crs != 4326 else gdf_gt
    
    tile_analyses = []
    
    # Create a map of all tiles in the grid (including inactive ones for matrix)
    all_tiles_map = {}
    for idx, tile_row in tile_grid.iterrows():
        tile_geom = tile_row.geometry
        tile_bbox = tile_geom.bounds
        
        # Get position - calculate from xtile and ytile to ensure accuracy
        xtile = int(tile_row.get("xtile", 0))
        ytile = int(tile_row.get("ytile", 0))
        col_idx = xtile - raster.xtile
        row_idx = ytile - raster.ytile
        pos = (col_idx, row_idx)
        pos_key = pos
        
        # Count deadends in this tile
        deadends_in_tile = dead_ends_wgs[dead_ends_wgs.geometry.intersects(tile_geom)]
        deadends_count = len(deadends_in_tile)
        
        # Get network segments in this tile
        network_in_tile = network_wgs[network_wgs.geometry.intersects(tile_geom)]
        
        # Get ground truth in this tile
        gt_in_tile = gdf_gt_wgs[gdf_gt_wgs.geometry.intersects(tile_geom)]
        
        # Calculate metrics for this tile
        tile_iou = 0.0
        tile_precision = 0.0
        tile_recall = 0.0
        tile_f1 = 0.0
        tile_mde = 0.0
        
        if len(network_in_tile) > 0 and len(gt_in_tile) > 0:
            try:
                iou, precision, recall = calculate_iou_scores(gt_in_tile, network_in_tile)
                tile_iou = float(iou)
                tile_precision = float(precision)
                tile_recall = float(recall)
                # Calculate F1 score
                tile_f1 = 2 * (tile_precision * tile_recall) / (tile_precision + tile_recall) if (tile_precision + tile_recall) > 0 else 0.0
                tile_mde = float(mean_displacement_error(gt_in_tile, network_in_tile))
            except Exception as e:
                print(f"Error calculating metrics for tile {idx}: {e}")
        
        tile_data = {
            "tile_id": int(tile_row.get("idd", idx)),
            "xtile": int(tile_row.get("xtile", 0)),
            "ytile": int(tile_row.get("ytile", 0)),
            "position": list(pos_key),
            "deadends_count": deadends_count,
            "network_segments": len(network_in_tile),
            "gt_segments": len(gt_in_tile),
            "metrics": {
                "iou": round(tile_iou, 3),
                "precision": round(tile_precision, 3),
                "recall": round(tile_recall, 3),
                "f1": round(tile_f1, 3),
                "mde": round(tile_mde, 3),
            },
            "geometry": json.loads(tile_grid.iloc[[idx]].to_json())
        }
        
        tile_analyses.append(tile_data)
        all_tiles_map[pos_key] = tile_data
    
    return tile_analyses


def gdf_to_graph(network_gdf: gpd.GeoDataFrame) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """Convert GeoDataFrame of LineStrings to NetworkX graph
    Returns:
        G: NetworkX graph
        node_coords: Dictionary mapping node_id to (lon, lat) coordinates
    """
    G = nx.Graph()
    network_wgs = network_gdf.to_crs(4326) if network_gdf.crs != 4326 else network_gdf
    
    # Tolerance for connecting nodes (in degrees, ~1.5 meters)
    # Use same tolerance as deadend detection for consistency
    # 1.5 meters ≈ 0.000015 degrees
    tolerance = 0.000015
    
    # Collect all nodes (endpoints) with their coordinates
    nodes = {}  # (lon, lat) -> node_id
    node_coords = {}  # node_id -> (lon, lat) rounded coordinates
    node_id_counter = 0
    edges = []
    
    for idx, row in network_wgs.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiLineString":
            line_strings = list(geom.geoms)
        else:
            line_strings = [geom]
        
        for line in line_strings:
            if len(line.coords) < 2:
                continue
            
            coords = list(line.coords)
            # Get start and end points
            start_coord = coords[0]
            end_coord = coords[-1]
            
            # Find or create nodes for start and end points
            def get_or_create_node(coord):
                nonlocal node_id_counter
                # Round coordinates to tolerance to merge nearby points
                rounded = (round(coord[0] / tolerance) * tolerance, 
                          round(coord[1] / tolerance) * tolerance)
                if rounded not in nodes:
                    nodes[rounded] = node_id_counter
                    node_coords[node_id_counter] = rounded
                    node_id_counter += 1
                return nodes[rounded]
            
            start_node = get_or_create_node(start_coord)
            end_node = get_or_create_node(end_coord)
            
            # Calculate edge length
            line_length = line.length * 111000  # Approximate meters (rough conversion)
            
            # Add edge if nodes are different
            if start_node != end_node:
                edges.append((start_node, end_node, line_length))
    
    # Build graph
    for start, end, length in edges:
        if G.has_edge(start, end):
            # If edge exists, update weight (take minimum or average)
            G[start][end]['weight'] = min(G[start][end].get('weight', length), length)
        else:
            G.add_edge(start, end, weight=length)
    
    return G, node_coords


def calculate_degree_distribution(network_gdf: gpd.GeoDataFrame) -> Dict:
    """Calculate node degree distribution for the network"""
    try:
        G, _ = gdf_to_graph(network_gdf)
        
        if len(G.nodes()) == 0:
            return {
                "total_nodes": 0,
                "avg_degree": 0.0,
                "median_degree": 0.0,
                "degree_distribution": {},
                "deadends_ratio": 0.0,
                "intersections_ratio": 0.0
            }
        
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        
        # Calculate statistics
        total_nodes = len(degrees)
        avg_degree = np.mean(degree_values) if degree_values else 0.0
        median_degree = np.median(degree_values) if degree_values else 0.0
        
        # Count nodes by degree
        degree_dist = Counter(degree_values)
        degree_distribution = {int(k): int(v) for k, v in degree_dist.items()}
        
        # Calculate ratios
        deadends_count = degree_dist.get(1, 0)
        intersections_count = sum(v for k, v in degree_dist.items() if k >= 3)
        
        deadends_ratio = deadends_count / total_nodes if total_nodes > 0 else 0.0
        intersections_ratio = intersections_count / total_nodes if total_nodes > 0 else 0.0
        
        return {
            "total_nodes": total_nodes,
            "avg_degree": round(float(avg_degree), 3),
            "median_degree": round(float(median_degree), 2),
            "degree_distribution": degree_distribution,
            "deadends_ratio": round(float(deadends_ratio), 3),
            "intersections_ratio": round(float(intersections_ratio), 3)
        }
    except Exception as e:
        print(f"Error calculating degree distribution: {e}")
        return {
            "total_nodes": 0,
            "avg_degree": 0.0,
            "median_degree": 0.0,
            "degree_distribution": {},
            "deadends_ratio": 0.0,
            "intersections_ratio": 0.0
        }


def calculate_network_connectivity(network_gdf: gpd.GeoDataFrame) -> Dict:
    """Calculate network connectivity metrics"""
    try:
        G, _ = gdf_to_graph(network_gdf)
        
        if len(G.nodes()) == 0:
            return {
                "num_components": 0,
                "largest_component_size": 0,
                "connectivity_ratio": 0.0,
                "isolated_segments": 0
            }
        
        # Find connected components
        connected_components = list(nx.connected_components(G))
        num_components = len(connected_components)
        
        if num_components == 0:
            return {
                "num_components": 0,
                "largest_component_size": 0,
                "connectivity_ratio": 0.0,
                "isolated_segments": 0
            }
        
        # Find largest component
        largest_component = max(connected_components, key=len)
        largest_component_size = len(largest_component)
        total_nodes = len(G.nodes())
        
        connectivity_ratio = largest_component_size / total_nodes if total_nodes > 0 else 0.0
        
        # Count isolated segments (components with < 3 nodes, likely disconnected fragments)
        isolated_segments = sum(1 for comp in connected_components if len(comp) < 3)
        
        # Calculate component size distribution
        component_sizes = [len(comp) for comp in connected_components]
        component_size_dist = Counter(component_sizes)
        component_size_distribution = {int(k): int(v) for k, v in component_size_dist.items()}
        
        return {
            "num_components": num_components,
            "largest_component_size": largest_component_size,
            "connectivity_ratio": round(float(connectivity_ratio), 3),
            "isolated_segments": isolated_segments,
            "component_size_distribution": component_size_distribution
        }
    except Exception as e:
        print(f"Error calculating network connectivity: {e}")
        return {
            "num_components": 0,
            "largest_component_size": 0,
            "connectivity_ratio": 0.0,
            "isolated_segments": 0,
            "network_density": 0.0
        }


def load_or_fetch_ground_truth(bbox: List[float], output_dir: Path, project_name: str) -> gpd.GeoDataFrame:
    # load cached gt or fetch from osm
    gt_path = output_dir / project_name / "gt.geojson"
    if gt_path.exists():
        print(f"Loading cached GT from {gt_path}")
        return gpd.read_file(gt_path)

    print("Fetching GT from OSM via osmnx...")
    G = ox.graph_from_bbox(bbox, network_type="all")
    gdf_gt = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_gt.to_file(gt_path, driver="GeoJSON")
    return gdf_gt


def detect_faults(
    gdf_pred_wgs: gpd.GeoDataFrame,
    gdf_gt_wgs: gpd.GeoDataFrame,
    distance_threshold: float,
) -> Tuple[List, List, List]:
    # convert to web mercator for distance calculations
    gdf_pred = gdf_pred_wgs.to_crs(3857)
    gdf_gt = gdf_gt_wgs.to_crs(3857)

    gt_sindex = gdf_gt.sindex

    false_positive = []
    true_positive = []
    false_negative = []  # Store full geometries instead of points

    # check each predicted line against gt
    for _, row in gdf_pred.iterrows():
        pred_geom = row.geometry
        candidate_idx = list(gt_sindex.intersection(pred_geom.bounds))
        if not candidate_idx:
            false_positive.append(pred_geom)
            continue
        min_dist = np.min([pred_geom.distance(gdf_gt.iloc[i].geometry) for i in candidate_idx])
        if min_dist > distance_threshold:
            false_positive.append(pred_geom)
        else:
            true_positive.append(pred_geom)

    # check each gt line against predictions (for false negatives)
    pred_sindex = gdf_pred.sindex
    for _, row in gdf_gt.iterrows():
        gt_geom = row.geometry
        candidate_idx = list(pred_sindex.intersection(gt_geom.bounds))
        if not candidate_idx:
            false_negative.append(gt_geom)  # Store full geometry
            continue
        min_dist = np.min([gt_geom.distance(gdf_pred.iloc[i].geometry) for i in candidate_idx])
        if min_dist > distance_threshold:
            false_negative.append(gt_geom)  # Store full geometry

    return true_positive, false_positive, false_negative


def geoms_to_gdf(geoms: List, source_crs: str) -> gpd.GeoDataFrame:
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=source_crs)
    return gpd.GeoDataFrame(geometry=geoms, crs=source_crs)


def gdf_to_geojson_dict(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:4326") -> Dict:
    # convert to wgs84 geojson for the frontend
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must define a CRS before serialization.")
    return json.loads(gdf.to_crs(target_crs).to_json())


def persist_analysis(result: Dict, base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = base_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result["metrics"], indent=2))
    for layer_name, geojson in result["layers"].items():
        layer_path = base_dir / f"{layer_name}.geojson"
        layer_path.write_text(json.dumps(geojson))


def slugify_name(location: str) -> str:
    # convert location string to a safe directory name
    slug = "".join(ch if ch.isalnum() else "-" for ch in location.lower())
    slug = "-".join(filter(None, slug.split("-")))
    return slug or "tile2net-project"


def build_analysis_payload(
    location: str,
    progress_cb: Callable[[int, str], None] | None = None,
) -> Dict:
    # main analysis pipeline: generate tiles, run inference, compute metrics
    def _progress(value: int, message: str) -> None:
        if progress_cb:
            progress_cb(value, message)

    project_name = slugify_name(location)
    raster = Raster(location=location, name=project_name, output_dir=str(OUTPUT_DIR))

    _progress(5, "Ensuring tiles and predictions")
    ensure_prediction_artifacts(raster, NUM_TILES)

    _progress(25, "Loading predictions")
    network, polygons = load_prediction_outputs(raster)
    bbox = network.total_bounds.tolist()

    _progress(40, "Loading ground truth")
    gdf_gt = load_or_fetch_ground_truth(bbox, OUTPUT_DIR, raster.name)

    _progress(60, "Calculating metrics")
    avg_iou, avg_precision, avg_recall = calculate_iou_scores(gdf_gt, network)
    # Calculate F1 score
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
    mde = mean_displacement_error(gdf_gt, network)
    
    _progress(65, "Analyzing network topology")
    degree_stats = calculate_degree_distribution(network)
    connectivity_stats = calculate_network_connectivity(network)

    _progress(80, "Detecting faults")
    true_positive, false_positive, false_negative = detect_faults(
        network, gdf_gt, DISTANCE_THRESHOLD_METERS
    )

    _progress(90, "Preparing map layers")
    tp_gdf = geoms_to_gdf(true_positive, "EPSG:3857")
    fp_gdf = geoms_to_gdf(false_positive, "EPSG:3857")
    fn_gdf = geoms_to_gdf(false_negative, "EPSG:3857")
    
    # Detect deadends for tile-level analysis
    from shapely.geometry import Point
    
    # Improved deadend detection: find line endpoints that only connect to one other line
    # This version checks both endpoint proximity and line proximity to avoid false positives
    # at turns where lines have small gaps
    network_wgs = network.to_crs(4326)
    
    # Collect all endpoints with their corresponding line indices
    endpoints = []
    for idx, row in network_wgs.iterrows():
        geom = row.geometry
        if geom.geom_type == "MultiLineString":
            for ls in geom.geoms:
                if len(ls.coords) >= 2:
                    endpoints.append((Point(ls.coords[0]), idx, ls))
                    endpoints.append((Point(ls.coords[-1]), idx, ls))
        else:
            if len(geom.coords) >= 2:
                endpoints.append((Point(geom.coords[0]), idx, geom))
                endpoints.append((Point(geom.coords[-1]), idx, geom))
    
    # Build spatial index for endpoints and lines
    from shapely.strtree import STRtree
    endpoint_tree = STRtree([ep[0] for ep in endpoints])
    line_tree = STRtree(network_wgs.geometry.values)
    
    # Use larger tolerance to account for small gaps at turns (about 1.5 meters)
    endpoint_tolerance = 0.00001  # ~1 meter in degrees for endpoint-to-endpoint
    line_tolerance = 0.000015  # ~1.5 meters in degrees for endpoint-to-line
    
    deadend_points = []  # Store deadend points instead of lines
    deadend_coords_set = set()  # Use set to track unique coordinates
    
    for endpoint, line_idx, line_geom in endpoints:
        # Find nearby endpoints (within small tolerance)
        nearby_endpoints = endpoint_tree.query(endpoint.buffer(endpoint_tolerance))
        
        # Also check if endpoint is close to any line segment (not just endpoints)
        # This handles cases where lines have small gaps at turns
        nearby_lines = line_tree.query(endpoint.buffer(line_tolerance))
        
        # Remove the current line from nearby_lines
        nearby_lines_filtered = [i for i in nearby_lines if i != line_idx]
        
        # Check if endpoint is actually close to any nearby line
        is_connected = False
        if nearby_lines_filtered:
            for nearby_line_idx in nearby_lines_filtered:
                nearby_line = network_wgs.iloc[nearby_line_idx].geometry
                # Check distance from endpoint to the line segment
                dist = endpoint.distance(nearby_line)
                if dist <= line_tolerance:
                    is_connected = True
                    break
        
        # Also check if there are other endpoints nearby (traditional check)
        if len(nearby_endpoints) > 2:  # More than just this endpoint and its pair
            is_connected = True
        
        # If not connected to any other line, it's a deadend
        if not is_connected:
            coord_key = (round(endpoint.x, 8), round(endpoint.y, 8))
            if coord_key not in deadend_coords_set:
                deadend_coords_set.add(coord_key)
                deadend_points.append(endpoint)
    
    # Convert deadend points to GeoDataFrame
    dead_ends_gdf = gpd.GeoDataFrame(geometry=deadend_points, crs=4326).to_crs(3857) if deadend_points else gpd.GeoDataFrame(geometry=[], crs=3857)
    
    # Calculate intersections (nodes with degree >= 3)
    _progress(91, "Calculating intersections")
    G, node_coords = gdf_to_graph(network_wgs)
    intersection_points = []
    if len(G.nodes()) > 0:
        degrees = dict(G.degree())
        # Find nodes with degree >= 3 (intersections)
        for node_id, degree in degrees.items():
            if degree >= 3 and node_id in node_coords:
                lon, lat = node_coords[node_id]
                intersection_points.append(Point(lon, lat))
    
    intersections_gdf = gpd.GeoDataFrame(geometry=intersection_points, crs=4326).to_crs(3857) if intersection_points else gpd.GeoDataFrame(geometry=[], crs=3857)
    
    _progress(92, "Analyzing tiles")
    tile_analyses = analyze_tiles(raster, network, dead_ends_gdf, gdf_gt)

    # Create tile grid GeoDataFrame for frontend
    tile_grid_gdf = raster.create_grid_gdf()
    
    # Get tiles bounding box to clip network layers
    # Ensure tile_grid_gdf is in WGS84 for clipping
    tile_grid_wgs = tile_grid_gdf.to_crs(4326) if tile_grid_gdf.crs != 4326 else tile_grid_gdf
    tiles_bbox = tile_grid_wgs.total_bounds  # [minx, miny, maxx, maxy]
    tiles_boundary = Polygon.from_bounds(tiles_bbox[0], tiles_bbox[1], tiles_bbox[2], tiles_bbox[3])
    tiles_boundary_gdf = gpd.GeoDataFrame(geometry=[tiles_boundary], crs=4326)
    
    # Clip network layers to tiles boundary
    def clip_layer(gdf, layer_name):
        """Clip a GeoDataFrame to tiles boundary"""
        if gdf is None or len(gdf) == 0:
            return gdf
        # Convert to WGS84 if needed
        gdf_wgs = gdf.to_crs(4326) if gdf.crs != 4326 else gdf.copy()
        # Clip to boundary
        clipped = gpd.clip(gdf_wgs, tiles_boundary_gdf)
        # Convert back to original CRS if needed
        if gdf.crs != 4326:
            clipped = clipped.to_crs(gdf.crs)
        return clipped
    
    # Clip all network-related layers
    network_clipped = clip_layer(network, "predicted_network")
    polygons_clipped = clip_layer(polygons, "polygons")
    gdf_gt_clipped = clip_layer(gdf_gt, "ground_truth")
    tp_gdf_clipped = clip_layer(tp_gdf, "true_positive")
    fp_gdf_clipped = clip_layer(fp_gdf, "false_positive")
    fn_gdf_clipped = clip_layer(fn_gdf, "false_negative")
    dead_ends_clipped = clip_layer(dead_ends_gdf, "dead_ends")
    intersections_clipped = clip_layer(intersections_gdf, "intersections")

    layers = {
        "predicted_network": gdf_to_geojson_dict(network_clipped),
        "polygons": gdf_to_geojson_dict(polygons_clipped),
        "ground_truth": gdf_to_geojson_dict(gdf_gt_clipped),
        "true_positive": gdf_to_geojson_dict(tp_gdf_clipped),
        "false_positive": gdf_to_geojson_dict(fp_gdf_clipped),
        "false_negative": gdf_to_geojson_dict(fn_gdf_clipped),
        "dead_ends": gdf_to_geojson_dict(dead_ends_clipped),
        "intersections": gdf_to_geojson_dict(intersections_clipped),
    }
    tile_grid_geojson = gdf_to_geojson_dict(tile_grid_gdf)
    
    # Get grid dimensions
    grid_dimensions = {
        "width": int(raster.width),
        "height": int(raster.height),
        "base_width": int(raster.base_width),
        "base_height": int(raster.base_height)
    }
    
    # Get tiles info for satellite layer
    tiles_dir = Path(raster.project.tiles.static)
    info_file = Path(raster.project.tiles.info)
    
    actual_tiles_path = tiles_dir
    if tiles_dir.exists():
        for subdir in tiles_dir.rglob("*"):
            if subdir.is_dir():
                jpg_files = list(subdir.glob("*.jpg"))
                if jpg_files:
                    actual_tiles_path = subdir
                    break
    
    extension = "jpg"
    zoom = raster.zoom
    
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
            zoom = info.get("zoom", raster.zoom)
            if actual_tiles_path.exists():
                sample_files = list(actual_tiles_path.glob("*.*"))
                if sample_files:
                    extension = sample_files[0].suffix.lstrip(".")
    
    tiles_info = {
        "zoom": zoom,
        "tiles_path": str(actual_tiles_path),
        "project_name": raster.name,
        "extension": extension
    }
    
    payload = {
        "metrics": {
            "mde": round(float(mde), 3),
            "iou": round(float(avg_iou), 3),
            "precision": round(float(avg_precision), 3),
            "recall": round(float(avg_recall), 3),
            "f1": round(float(avg_f1), 3),
            "distance_threshold_m": DISTANCE_THRESHOLD_METERS,
            "buffer_size": BUFFER_SIZE,
        },
        "network_topology": {
            "degree_distribution": degree_stats,
            "connectivity": connectivity_stats,
        },
        "layers": layers,
        "tile_analyses": tile_analyses,
        "tile_grid": tile_grid_geojson,
        "grid_dimensions": grid_dimensions,
        "tiles": tiles_info,
        "metadata": {
            "location": location,
            "project_name": raster.name,
            "bbox": bbox,
        },
    }

    results_dir = OUTPUT_DIR / raster.name / RESULTS_DIRNAME
    persist_analysis(payload, results_dir)
    _progress(100, "Completed")
    return payload


class RunRequest(BaseModel):
    location: str


# global state for the current analysis
analysis_cache: Dict | None = None
analysis_status: Dict = {
    "state": "loading",
    "message": "Initializing analysis...",
    "progress": 0,
    "location": DEFAULT_LOCATION,
}
analysis_lock = threading.Lock()

app = FastAPI(title="Tile2Net Inspector", version="0.2.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _set_status(state: str, message: str, progress: int, location: str | None = None) -> None:
    # update status for frontend polling
    analysis_status.update(
        {
            "state": state,
            "message": message,
            "progress": progress,
            "location": location or analysis_status.get("location", DEFAULT_LOCATION),
        }
    )


def _run_analysis(location: str) -> None:
    # run the full analysis pipeline in a background thread
    global analysis_cache

    def progress_cb(value: int, message: str) -> None:
        _set_status("running", message, value, location)

    try:
        with analysis_lock:
            analysis_cache = build_analysis_payload(location, progress_cb=progress_cb)
        _set_status("idle", "Analysis completed.", 100, location)
    except Exception as exc:
        _set_status("error", f"Analysis failed: {exc}", 0, location)
        raise


@app.on_event("startup")
def load_analysis_cache() -> None:
    # pre-run default location so frontend has data on startup
    print("Building analysis cache...")
    try:
        _run_analysis(DEFAULT_LOCATION)
    except Exception as exc:
        print(f"Startup analysis failed: {exc}")


@app.get("/api/analysis")
def get_analysis():
    """Get full analysis payload including metrics and network topology"""
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    return JSONResponse(analysis_cache)

@app.get("/api/metrics")
def get_metrics():
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    return JSONResponse(analysis_cache["metrics"])


@app.get("/api/layers")
def list_layers():
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    return {"layers": list(analysis_cache["layers"].keys())}


@app.get("/api/layers/{layer_id}")
def get_layer(layer_id: str):
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    layer = analysis_cache["layers"].get(layer_id)
    if not layer:
        raise HTTPException(status_code=404, detail=f"Layer '{layer_id}' not found.")
    return JSONResponse(layer)


@app.get("/api/metadata")
def get_metadata():
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    return JSONResponse(analysis_cache["metadata"])


@app.get("/api/tile-analyses")
def get_tile_analyses():
    """Get tile-level analysis data for matrix visualization"""
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    if "tile_analyses" not in analysis_cache:
        raise HTTPException(status_code=404, detail="Tile analyses not available.")
    return JSONResponse(analysis_cache["tile_analyses"])


@app.get("/api/tile-grid")
def get_tile_grid():
    """Get tile grid GeoDataFrame for map visualization"""
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    if "tile_grid" not in analysis_cache:
        raise HTTPException(status_code=404, detail="Tile grid not available.")
    return JSONResponse(analysis_cache["tile_grid"])


@app.get("/api/grid-dimensions")
def get_grid_dimensions():
    """Get grid dimensions for matrix visualization"""
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    if "grid_dimensions" not in analysis_cache:
        raise HTTPException(status_code=404, detail="Grid dimensions not available.")
    return JSONResponse(analysis_cache["grid_dimensions"])


@app.get("/api/tiles-info")
def get_tiles_info_endpoint():
    """Get tiles information for satellite layer"""
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    if "tiles" not in analysis_cache:
        raise HTTPException(status_code=404, detail="Tiles information not available.")
    return JSONResponse(analysis_cache["tiles"])


@app.get("/api/tiles/{z}/{x}/{y}")
def get_tile(z: int, x: int, y: int):
    """Serve tile2net generated tiles using xyz format"""
    if not analysis_cache:
        raise HTTPException(status_code=503, detail="Analysis is still loading.")
    
    if "tiles" not in analysis_cache:
        raise HTTPException(status_code=404, detail="Tiles information not available.")
    
    tiles_info = analysis_cache["tiles"]
    tiles_path = Path(tiles_info["tiles_path"])
    extension = tiles_info.get("extension", "jpg")
    zoom = tiles_info.get("zoom", z)
    
    if z != zoom:
        raise HTTPException(status_code=404, detail=f"Zoom level mismatch. Expected {zoom}, got {z}.")
    
    tile_file = tiles_path / f"{x}_{y}.{extension}"
    
    if not tile_file.exists():
        for ext in ["jpg", "jpeg", "png"]:
            alt_file = tiles_path / f"{x}_{y}.{ext}"
            if alt_file.exists():
                tile_file = alt_file
                extension = ext
                break
    
    if not tile_file.exists():
        raise HTTPException(status_code=404, detail=f"Tile not found: {x}_{y} at zoom {z}")
    
    from fastapi.responses import FileResponse
    
    media_type_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png"
    }
    media_type = media_type_map.get(extension, "image/jpeg")
    
    return FileResponse(
        tile_file,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000"}
    )


@app.get("/api/status")
def get_status():
    return analysis_status


@app.post("/api/run")
def trigger_run(request: RunRequest, background_tasks: BackgroundTasks):
    # start a new analysis run from the frontend
    if analysis_status["state"] == "running":
        raise HTTPException(status_code=409, detail="Analysis already running.")
    location = request.location.strip()
    if not location:
        raise HTTPException(status_code=400, detail="Location must not be empty.")

    _set_status("running", "Starting analysis...", 1, location)
    background_tasks.add_task(_run_analysis, location)
    return {"message": "Analysis started.", "location": location}


# serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/")
def root():
    if FRONTEND_DIR.exists():
        return RedirectResponse(url="/app/")
    return {"message": "Frontend not found. Visit /api/docs for API usage."}


def run() -> None:
    # entrypoint when running directly
    import uvicorn

    uvicorn.run(
        "interface:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(int(os.getenv("UVICORN_RELOAD", "0"))),
    )


if __name__ == "__main__":
    run()