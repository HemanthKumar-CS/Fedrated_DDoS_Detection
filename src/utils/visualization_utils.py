"""
Visualization utilities for federated learning analysis
Provides common functions for directory management, plot styling, and file operations
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def create_output_directory(base_path: str = "results/validations", timestamp: bool = True) -> str:
    """
    Create organized output directory structure with timestamp-based organization
    
    Args:
        base_path: Base directory path for outputs
        timestamp: Whether to include timestamp in directory name
        
    Returns:
        str: Path to created directory
    """
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_path, timestamp_str)
    else:
        output_dir = base_path
    
    # Create subdirectories
    subdirs = [
        "confusion_matrices",
        "classification_reports", 
        "roc_curves",
        "precision_recall_curves",
        "training_progress"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    logger.info(f"Created output directory structure at: {output_dir}")
    return output_dir

def apply_consistent_styling() -> None:
    """Apply consistent styling and formatting for all visualizations"""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set consistent parameters
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.dpi': 300
    })

def save_plot_with_metadata(fig, filepath: str, metadata: Dict[str, Any] = None) -> str:
    """
    Save plot with metadata and proper formatting
    
    Args:
        fig: Matplotlib figure object
        filepath: Path to save the plot
        metadata: Optional metadata to include
        
    Returns:
        str: Path where plot was saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the plot
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Save metadata if provided
    if metadata:
        metadata_path = filepath.replace('.png', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Plot saved to: {filepath}")
    return filepath

def format_classification_table(report_dict: Dict) -> str:
    """
    Format classification report dictionary into readable table string
    
    Args:
        report_dict: Classification report dictionary from sklearn
        
    Returns:
        str: Formatted table string
    """
    if not report_dict:
        return "No classification report data available"
    
    # Extract metrics for each class
    classes = [key for key in report_dict.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
    
    table_lines = []
    table_lines.append("=" * 70)
    table_lines.append("CLASSIFICATION REPORT")
    table_lines.append("=" * 70)
    table_lines.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    table_lines.append("-" * 70)
    
    # Add class-specific metrics
    for class_name in classes:
        if class_name in report_dict:
            metrics = report_dict[class_name]
            table_lines.append(
                f"{class_name:<15} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1-score']:<12.4f} "
                f"{int(metrics['support']):<10}"
            )
    
    table_lines.append("-" * 70)
    
    # Add summary metrics
    if 'accuracy' in report_dict:
        table_lines.append(f"{'Accuracy':<15} {'':<12} {'':<12} {report_dict['accuracy']:<12.4f} {'':<10}")
    
    if 'macro avg' in report_dict:
        metrics = report_dict['macro avg']
        table_lines.append(
            f"{'Macro Avg':<15} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1-score']:<12.4f} "
            f"{int(metrics['support']):<10}"
        )
    
    if 'weighted avg' in report_dict:
        metrics = report_dict['weighted avg']
        table_lines.append(
            f"{'Weighted Avg':<15} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1-score']:<12.4f} "
            f"{int(metrics['support']):<10}"
        )
    
    table_lines.append("=" * 70)
    
    return "\n".join(table_lines)

def get_class_labels() -> list:
    """Get consistent class labels for DDoS detection"""
    return ['Benign', 'Attack']

def get_color_palette(n_colors: int = None) -> list:
    """Get consistent color palette for visualizations"""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']
    if n_colors:
        return colors[:n_colors]
    return colors

def create_figure_with_style(figsize: tuple = (10, 6)) -> plt.Figure:
    """Create a matplotlib figure with consistent styling applied"""
    apply_consistent_styling()
    fig = plt.figure(figsize=figsize)
    return fig

def ensure_directory_exists(filepath: str) -> None:
    """Ensure the directory for a filepath exists"""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)