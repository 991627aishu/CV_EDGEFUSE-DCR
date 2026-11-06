"""
EdgeFuse-DCR: Edge-aware Confidence-Guided Depth Completion and Refinement.
"""

from .data import NYUDepthV2Dataset, load_nyu_sample, load_kitti_val_samples, read_kitti_sample, load_middlebury_scene, generate_controlled_input
from .depth_engine import reconstruct_depth, estimate_confidence_map, detect_rgb_edges, detect_depth_discontinuities
from .model import EdgeFuseNet, load_model, save_checkpoint
from .evaluation import compute_depth_metrics, run_comparative_benchmark, format_evaluation_table_markdown

