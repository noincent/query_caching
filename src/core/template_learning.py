"""
Template Learning Module

This module provides functionality for learning and adapting templates based on
usage patterns and feedback to improve accuracy over time.
"""

import logging
import time
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)

class TemplateLearning:
    """Learn and adapt templates based on usage patterns."""
    
    def __init__(self, query_cache):
        self.query_cache = query_cache
        self.feedback_history = []
    
    def record_feedback(self, template_id: int, success: bool, execution_time: float = None, error: str = None):
        """Record feedback for template usage."""
        feedback = {
            'template_id': template_id,
            'success': success,
            'timestamp': time.time(),
            'execution_time': execution_time,
            'error': error
        }
        self.feedback_history.append(feedback)
        
        # Update template immediately
        self._update_template_from_feedback(template_id, success)
    
    def _update_template_from_feedback(self, template_id: int, success: bool):
        """Update template statistics based on feedback."""
        # Find template
        template = None
        for idx, t in enumerate(self.query_cache.template_matcher.templates):
            if t.get('id') == template_id:
                template = t
                template_idx = idx
                break
        
        if not template:
            return
        
        # Update success rate with adaptive learning rate
        usage_count = template.get('usage_count', 0)
        # Higher usage count = lower learning rate (more stable)
        learning_rate = 1.0 / (1.0 + np.log1p(usage_count))
        
        current_success_rate = template.get('success_rate', 0.5)
        new_success_rate = (1 - learning_rate) * current_success_rate + learning_rate * (1.0 if success else 0.0)
        
        template['success_rate'] = new_success_rate
        template['last_feedback'] = time.time()
        
        # Mark template for review if success rate is too low
        if new_success_rate < 0.3 and usage_count > 10:
            template['needs_review'] = True
            logger.warning(f"Template {template_id} marked for review due to low success rate: {new_success_rate:.2f}")
    
    def suggest_template_improvements(self, template_id: int) -> List[Dict[str, Any]]:
        """Suggest improvements for a template based on usage patterns."""
        suggestions = []
        
        # Analyze feedback history for this template
        template_feedback = [f for f in self.feedback_history if f['template_id'] == template_id]
        
        if not template_feedback:
            return suggestions
        
        # Analyze common errors
        errors = [f['error'] for f in template_feedback if f.get('error')]
        if errors:
            # Group similar errors
            error_groups = {}
            for error in errors:
                key = error[:50]  # Group by first 50 chars
                error_groups[key] = error_groups.get(key, 0) + 1
            
            most_common_error = max(error_groups.items(), key=lambda x: x[1])
            suggestions.append({
                'type': 'common_error',
                'description': f"Most common error: {most_common_error[0]}",
                'frequency': most_common_error[1]
            })
        
        # Analyze execution time
        exec_times = [f['execution_time'] for f in template_feedback if f.get('execution_time')]
        if exec_times:
            avg_time = np.mean(exec_times)
            if avg_time > 1000:  # More than 1 second
                suggestions.append({
                    'type': 'performance',
                    'description': f"Average execution time is high: {avg_time:.0f}ms",
                    'recommendation': "Consider adding indexes or optimizing the SQL query"
                })
        
        return suggestions
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about template learning and performance."""
        if not self.feedback_history:
            return {'message': 'No feedback data available'}
        
        total_feedback = len(self.feedback_history)
        successful_uses = sum(1 for f in self.feedback_history if f['success'])
        success_rate = successful_uses / total_feedback if total_feedback > 0 else 0
        
        # Template performance analysis
        template_performance = {}
        for feedback in self.feedback_history:
            template_id = feedback['template_id']
            if template_id not in template_performance:
                template_performance[template_id] = {'successes': 0, 'failures': 0}
            
            if feedback['success']:
                template_performance[template_id]['successes'] += 1
            else:
                template_performance[template_id]['failures'] += 1
        
        # Find best and worst performing templates
        best_templates = []
        worst_templates = []
        
        for template_id, perf in template_performance.items():
            total = perf['successes'] + perf['failures']
            if total >= 5:  # Only consider templates with at least 5 uses
                rate = perf['successes'] / total
                if rate >= 0.8:
                    best_templates.append({'id': template_id, 'success_rate': rate})
                elif rate <= 0.3:
                    worst_templates.append({'id': template_id, 'success_rate': rate})
        
        return {
            'total_feedback_entries': total_feedback,
            'overall_success_rate': success_rate,
            'best_performing_templates': sorted(best_templates, key=lambda x: x['success_rate'], reverse=True)[:5],
            'worst_performing_templates': sorted(worst_templates, key=lambda x: x['success_rate'])[:5],
            'templates_needing_review': len([t for t in self.query_cache.template_matcher.templates if t.get('needs_review', False)])
        }
    
    def auto_tune_thresholds(self) -> Dict[str, float]:
        """Automatically tune similarity thresholds based on performance data."""
        if len(self.feedback_history) < 50:  # Need sufficient data
            return {}
        
        # Analyze success rates by similarity method
        method_performance = {}
        
        for feedback in self.feedback_history[-100:]:  # Last 100 feedback entries
            # This would require storing which method was used for matching
            # For now, we'll use a simplified approach
            pass
        
        # Calculate optimal thresholds (simplified implementation)
        current_thresholds = self.query_cache.config.get('similarity_thresholds', {})
        suggested_thresholds = current_thresholds.copy()
        
        # If success rate is low, lower thresholds slightly
        overall_success = sum(1 for f in self.feedback_history[-50:] if f['success'])
        recent_success_rate = overall_success / min(50, len(self.feedback_history))
        
        if recent_success_rate < 0.6:
            # Lower thresholds by 5%
            for method in suggested_thresholds:
                suggested_thresholds[method] *= 0.95
            logger.info(f"Suggested lowering thresholds due to low success rate: {recent_success_rate:.2f}")
        elif recent_success_rate > 0.9:
            # Raise thresholds by 2% for better precision
            for method in suggested_thresholds:
                suggested_thresholds[method] *= 1.02
            logger.info(f"Suggested raising thresholds due to high success rate: {recent_success_rate:.2f}")
        
        return suggested_thresholds