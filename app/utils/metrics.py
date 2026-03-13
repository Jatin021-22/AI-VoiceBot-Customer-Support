import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class RequestMetrics:
    request_id: str
    endpoint: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_latency_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None

    def finish(self, success: bool = True, error: str = None):
        self.end_time = time.time()
        self.total_latency_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error = error


class MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests: List[RequestMetrics] = []
        self._counters = defaultdict(int)

    def record_request(self, metrics: RequestMetrics):
        with self._lock:
            self._requests.append(metrics)
            self._counters["total_requests"] += 1
            if metrics.success:
                self._counters["successful_requests"] += 1
            else:
                self._counters["failed_requests"] += 1

    def get_summary(self) -> dict:
        with self._lock:
            if not self._requests:
                return {"message": "No requests recorded yet"}
            completed = [r for r in self._requests if r.total_latency_ms is not None]
            latencies = [r.total_latency_ms for r in completed]
            return {
                "total_requests": self._counters["total_requests"],
                "successful_requests": self._counters["successful_requests"],
                "failed_requests": self._counters["failed_requests"],
                "success_rate": round(
                    self._counters["successful_requests"] /
                    max(self._counters["total_requests"], 1) * 100, 2
                ),
                "latency_stats": {
                    "avg_total_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
                    "min_total_ms": round(min(latencies), 2) if latencies else 0,
                    "max_total_ms": round(max(latencies), 2) if latencies else 0,
                },
            }


metrics_collector = MetricsCollector()