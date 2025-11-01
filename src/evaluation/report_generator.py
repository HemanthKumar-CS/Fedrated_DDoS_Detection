"""
Report generator for organizing and formatting federated learning evaluation outputs
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generate formatted reports and manage file organization"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_client_report(self, client_id: str, metrics: Dict[str, Any], 
                             visualizations: Dict[str, str]) -> str:
        """
        Generate client-specific report with metrics and visualization paths
        
        Args:
            client_id: Client identifier
            metrics: Dictionary of calculated metrics
            visualizations: Dictionary of visualization file paths
            
        Returns:
            str: Formatted report string
        """
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append(f"CLIENT EVALUATION REPORT - {client_id}")
            report_lines.append("=" * 80)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Performance Metrics Section
            report_lines.append("📊 PERFORMANCE METRICS")
            report_lines.append("-" * 40)
            if metrics:
                report_lines.append(f"Accuracy:           {metrics.get('accuracy', 0):.4f}")
                report_lines.append(f"Precision:          {metrics.get('precision', 0):.4f}")
                report_lines.append(f"Recall:             {metrics.get('recall', 0):.4f}")
                report_lines.append(f"F1-Score:           {metrics.get('f1_score', 0):.4f}")
                report_lines.append(f"ROC-AUC:            {metrics.get('roc_auc', 0):.4f}")
                report_lines.append(f"Average Precision:  {metrics.get('average_precision', 0):.4f}")
            else:
                report_lines.append("No metrics available")
            
            report_lines.append("")
            
            # Confusion Matrix Section
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                report_lines.append("🔍 CONFUSION MATRIX")
                report_lines.append("-" * 40)
                report_lines.append("           Predicted")
                report_lines.append("         Benign  Attack")
                report_lines.append(f"Benign   {cm[0][0]:6d}  {cm[0][1]:6d}")
                report_lines.append(f"Attack   {cm[1][0]:6d}  {cm[1][1]:6d}")
                report_lines.append("")
            
            # Generated Files Section
            report_lines.append("📁 GENERATED VISUALIZATIONS")
            report_lines.append("-" * 40)
            if visualizations:
                for viz_type, filepath in visualizations.items():
                    if client_id in viz_type:
                        filename = os.path.basename(filepath)
                        report_lines.append(f"• {viz_type}: {filename}")
            else:
                report_lines.append("No visualizations generated")
            
            report_lines.append("")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating client report for {client_id}: {e}")
            return f"Error generating report for {client_id}: {e}"
    
    def generate_aggregated_report(self, aggregated_metrics: Dict[str, Any], 
                                 visualizations: Dict[str, str],
                                 client_metrics: Dict[str, Dict] = None) -> str:
        """
        Generate aggregated report combining all client results
        
        Args:
            aggregated_metrics: Dictionary of aggregated metrics
            visualizations: Dictionary of visualization file paths
            client_metrics: Optional individual client metrics
            
        Returns:
            str: Formatted aggregated report string
        """
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("FEDERATED LEARNING EVALUATION REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Summary Section
            report_lines.append("📈 FEDERATED LEARNING SUMMARY")
            report_lines.append("-" * 40)
            if aggregated_metrics:
                num_clients = aggregated_metrics.get('num_clients', 0)
                report_lines.append(f"Number of Clients:  {num_clients}")
                report_lines.append(f"Mean Accuracy:      {aggregated_metrics.get('accuracy_mean', 0):.4f} ± {aggregated_metrics.get('accuracy_std', 0):.4f}")
                report_lines.append(f"Mean Precision:     {aggregated_metrics.get('precision_mean', 0):.4f} ± {aggregated_metrics.get('precision_std', 0):.4f}")
                report_lines.append(f"Mean Recall:        {aggregated_metrics.get('recall_mean', 0):.4f} ± {aggregated_metrics.get('recall_std', 0):.4f}")
                report_lines.append(f"Mean F1-Score:      {aggregated_metrics.get('f1_score_mean', 0):.4f} ± {aggregated_metrics.get('f1_score_std', 0):.4f}")
                report_lines.append(f"Mean ROC-AUC:       {aggregated_metrics.get('roc_auc_mean', 0):.4f} ± {aggregated_metrics.get('roc_auc_std', 0):.4f}")
            else:
                report_lines.append("No aggregated metrics available")
            
            report_lines.append("")
            
            # Individual Client Performance
            if client_metrics:
                report_lines.append("👥 INDIVIDUAL CLIENT PERFORMANCE")
                report_lines.append("-" * 40)
                report_lines.append(f"{'Client':<12} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1-Score':<10}")
                report_lines.append("-" * 60)
                
                for client_id, metrics in client_metrics.items():
                    report_lines.append(
                        f"{client_id:<12} "
                        f"{metrics.get('accuracy', 0):<10.4f} "
                        f"{metrics.get('precision', 0):<11.4f} "
                        f"{metrics.get('recall', 0):<10.4f} "
                        f"{metrics.get('f1_score', 0):<10.4f}"
                    )
                
                report_lines.append("")
            
            # Aggregated Confusion Matrix
            if 'aggregated_confusion_matrix' in aggregated_metrics:
                cm = aggregated_metrics['aggregated_confusion_matrix']
                report_lines.append("🔍 AGGREGATED CONFUSION MATRIX")
                report_lines.append("-" * 40)
                report_lines.append("           Predicted")
                report_lines.append("         Benign  Attack")
                report_lines.append(f"Benign   {cm[0][0]:6d}  {cm[0][1]:6d}")
                report_lines.append(f"Attack   {cm[1][0]:6d}  {cm[1][1]:6d}")
                report_lines.append("")
            
            # Performance Analysis
            report_lines.append("📊 PERFORMANCE ANALYSIS")
            report_lines.append("-" * 40)
            
            if aggregated_metrics:
                acc_std = aggregated_metrics.get('accuracy_std', 0)
                acc_mean = aggregated_metrics.get('accuracy_mean', 0)
                
                if acc_std < 0.05:
                    consistency = "HIGH"
                elif acc_std < 0.1:
                    consistency = "MODERATE"
                else:
                    consistency = "LOW"
                
                report_lines.append(f"Client Consistency: {consistency} (std: {acc_std:.4f})")
                
                if acc_mean > 0.9:
                    performance = "EXCELLENT"
                elif acc_mean > 0.8:
                    performance = "GOOD"
                elif acc_mean > 0.7:
                    performance = "FAIR"
                else:
                    performance = "POOR"
                
                report_lines.append(f"Overall Performance: {performance} (mean: {acc_mean:.4f})")
            
            report_lines.append("")
            
            # Generated Files Section
            report_lines.append("📁 GENERATED VISUALIZATIONS")
            report_lines.append("-" * 40)
            if visualizations:
                # Group by type
                confusion_matrices = [k for k in visualizations.keys() if 'confusion_matrix' in k]
                roc_curves = [k for k in visualizations.keys() if 'roc_curve' in k]
                pr_curves = [k for k in visualizations.keys() if 'pr_curve' in k]
                training_plots = [k for k in visualizations.keys() if 'training' in k or 'convergence' in k or 'comparison' in k]
                reports = [k for k in visualizations.keys() if 'report' in k]
                
                if confusion_matrices:
                    report_lines.append("Confusion Matrices:")
                    for viz in confusion_matrices:
                        filename = os.path.basename(visualizations[viz])
                        report_lines.append(f"  • {filename}")
                
                if roc_curves:
                    report_lines.append("ROC Curves:")
                    for viz in roc_curves:
                        filename = os.path.basename(visualizations[viz])
                        report_lines.append(f"  • {filename}")
                
                if pr_curves:
                    report_lines.append("Precision-Recall Curves:")
                    for viz in pr_curves:
                        filename = os.path.basename(visualizations[viz])
                        report_lines.append(f"  • {filename}")
                
                if training_plots:
                    report_lines.append("Training Progress:")
                    for viz in training_plots:
                        filename = os.path.basename(visualizations[viz])
                        report_lines.append(f"  • {filename}")
                
                if reports:
                    report_lines.append("Classification Reports:")
                    for viz in reports:
                        filename = os.path.basename(visualizations[viz])
                        report_lines.append(f"  • {filename}")
            else:
                report_lines.append("No visualizations generated")
            
            report_lines.append("")
            report_lines.append("=" * 80)
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating aggregated report: {e}")
            return f"Error generating aggregated report: {e}"
    
    def organize_output_files(self, base_dir: str, generated_files: Dict[str, str]) -> Dict[str, str]:
        """
        Organize output files with descriptive filenames and timestamps
        
        Args:
            base_dir: Base directory for outputs
            generated_files: Dictionary of generated file paths
            
        Returns:
            Dict mapping organized file types to paths
        """
        try:
            organized = {}
            
            # Create summary report
            summary_path = os.path.join(base_dir, f"summary_report_{self.timestamp}.md")
            
            # Generate summary content
            summary_lines = []
            summary_lines.append(f"# Federated Learning Visualization Summary")
            summary_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append("")
            summary_lines.append("## Generated Files")
            summary_lines.append("")
            
            # Organize by category
            categories = {
                'Confusion Matrices': [k for k in generated_files.keys() if 'confusion_matrix' in k],
                'ROC Curves': [k for k in generated_files.keys() if 'roc_curve' in k],
                'Precision-Recall Curves': [k for k in generated_files.keys() if 'pr_curve' in k],
                'Training Progress': [k for k in generated_files.keys() if any(x in k for x in ['training', 'convergence', 'comparison'])],
                'Classification Reports': [k for k in generated_files.keys() if 'report' in k]
            }
            
            for category, files in categories.items():
                if files:
                    summary_lines.append(f"### {category}")
                    summary_lines.append("")
                    for file_key in files:
                        filepath = generated_files[file_key]
                        filename = os.path.basename(filepath)
                        relative_path = os.path.relpath(filepath, base_dir)
                        summary_lines.append(f"- [{filename}]({relative_path})")
                    summary_lines.append("")
            
            # Save summary
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            
            organized['summary_report'] = summary_path
            
            # Create metadata file
            metadata_path = os.path.join(base_dir, f"metadata_{self.timestamp}.json")
            metadata = {
                'generation_timestamp': self.timestamp,
                'generation_datetime': datetime.now().isoformat(),
                'total_files': len(generated_files),
                'file_categories': {cat: len(files) for cat, files in categories.items()},
                'generated_files': generated_files
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            organized['metadata'] = metadata_path
            
            logger.info(f"Organized {len(generated_files)} files with summary and metadata")
            return organized
            
        except Exception as e:
            logger.error(f"Error organizing output files: {e}")
            return {}
    
    def save_comprehensive_report(self, base_dir: str, aggregated_metrics: Dict[str, Any],
                                client_metrics: Dict[str, Dict], visualizations: Dict[str, str]) -> str:
        """
        Save comprehensive report combining all analysis results
        
        Args:
            base_dir: Base directory for outputs
            aggregated_metrics: Aggregated metrics across all clients
            client_metrics: Individual client metrics
            visualizations: Generated visualization file paths
            
        Returns:
            str: Path to saved comprehensive report
        """
        try:
            report_path = os.path.join(base_dir, f"comprehensive_report_{self.timestamp}.txt")
            
            # Generate comprehensive report
            report_content = self.generate_aggregated_report(
                aggregated_metrics, visualizations, client_metrics)
            
            # Add individual client reports
            report_content += "\n\n" + "=" * 80 + "\n"
            report_content += "INDIVIDUAL CLIENT REPORTS\n"
            report_content += "=" * 80 + "\n\n"
            
            for client_id, metrics in client_metrics.items():
                client_report = self.generate_client_report(client_id, metrics, visualizations)
                report_content += client_report + "\n\n"
            
            # Save comprehensive report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Comprehensive report saved: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error saving comprehensive report: {e}")
            return ""