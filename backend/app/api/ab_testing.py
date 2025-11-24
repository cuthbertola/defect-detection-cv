"""A/B Testing Framework for model versions."""

import random
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class ModelVersion:
    """Represents a model version for A/B testing."""
    name: str
    path: str
    weight: float = 0.5
    metrics: Dict = field(default_factory=dict)
    request_count: int = 0
    total_inference_time: float = 0.0
    defects_detected: int = 0
    quality_passed: int = 0


class ABTestingFramework:
    """A/B Testing framework for comparing model versions."""
    
    def __init__(self):
        self.models: Dict[str, ModelVersion] = {}
        self.active_test: Optional[str] = None
        self.test_results: list = []
        self.start_time: Optional[datetime] = None
        
    def register_model(self, name: str, path: str, weight: float = 0.5) -> None:
        """Register a model version for testing."""
        self.models[name] = ModelVersion(name=name, path=path, weight=weight)
        
    def start_test(self, test_name: str, model_a: str, model_b: str, 
                   weight_a: float = 0.5) -> Dict:
        """Start an A/B test between two models."""
        if model_a not in self.models or model_b not in self.models:
            raise ValueError("Both models must be registered first")
        
        self.active_test = test_name
        self.start_time = datetime.now()
        
        self.models[model_a].weight = weight_a
        self.models[model_b].weight = 1 - weight_a
        
        for model in [model_a, model_b]:
            self.models[model].request_count = 0
            self.models[model].total_inference_time = 0.0
            self.models[model].defects_detected = 0
            self.models[model].quality_passed = 0
        
        return {
            "test_name": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "weight_a": weight_a,
            "weight_b": 1 - weight_a,
            "started_at": self.start_time.isoformat()
        }
    
    def select_model(self) -> str:
        """Select a model based on traffic weights."""
        if not self.models:
            raise ValueError("No models registered")
        
        rand = random.random()
        cumulative = 0.0
        
        for name, model in self.models.items():
            cumulative += model.weight
            if rand <= cumulative:
                return name
        
        return list(self.models.keys())[0]
    
    def record_result(self, model_name: str, inference_time: float, 
                     has_defect: bool, confidence: float) -> None:
        """Record inference result for a model."""
        if model_name not in self.models:
            return
        
        model = self.models[model_name]
        model.request_count += 1
        model.total_inference_time += inference_time
        
        if has_defect:
            model.defects_detected += 1
        else:
            model.quality_passed += 1
        
        self.test_results.append({
            "model": model_name,
            "inference_time": inference_time,
            "has_defect": has_defect,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_test_stats(self) -> Dict:
        """Get current A/B test statistics."""
        stats = {
            "test_name": self.active_test,
            "started_at": self.start_time.isoformat() if self.start_time else None,
            "total_requests": sum(m.request_count for m in self.models.values()),
            "models": {}
        }
        
        for name, model in self.models.items():
            avg_inference = (model.total_inference_time / model.request_count 
                          if model.request_count > 0 else 0)
            
            stats["models"][name] = {
                "request_count": model.request_count,
                "traffic_weight": model.weight,
                "avg_inference_time_ms": round(avg_inference * 1000, 2),
                "defects_detected": model.defects_detected,
                "quality_passed": model.quality_passed,
                "defect_rate": round(
                    model.defects_detected / model.request_count * 100, 2
                ) if model.request_count > 0 else 0
            }
        
        return stats
    
    def determine_winner(self, metric: str = "avg_inference_time_ms") -> Dict:
        """Determine the winning model based on a metric."""
        stats = self.get_test_stats()
        
        if not stats["models"]:
            return {"winner": None, "reason": "No models tested"}
        
        best_model = None
        best_value = None
        
        for name, model_stats in stats["models"].items():
            value = model_stats.get(metric, 0)
            
            if best_value is None:
                best_model = name
                best_value = value
            elif metric in ["avg_inference_time_ms"]:
                if value < best_value:
                    best_model = name
                    best_value = value
            else:
                if value > best_value:
                    best_model = name
                    best_value = value
        
        return {
            "winner": best_model,
            "metric": metric,
            "value": best_value,
            "stats": stats
        }
    
    def save_results(self, filepath: str = "ab_test_results.json") -> None:
        """Save test results to file."""
        results = {
            "test_name": self.active_test,
            "started_at": self.start_time.isoformat() if self.start_time else None,
            "ended_at": datetime.now().isoformat(),
            "stats": self.get_test_stats(),
            "winner": self.determine_winner(),
            "raw_results": self.test_results[-100:]
        }
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)


# Global instance
ab_framework = ABTestingFramework()

# Register default models
ab_framework.register_model("v1_baseline", "models/production/model.onnx", weight=0.5)
ab_framework.register_model("v2_optimized", "models/production/model.onnx", weight=0.5)
