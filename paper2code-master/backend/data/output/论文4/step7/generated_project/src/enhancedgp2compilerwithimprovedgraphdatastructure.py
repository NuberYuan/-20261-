import os
import time
import json
import random
import heapq
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class EnhancedGP2CompilerWithImprovedGraphDataStructure:
    """Enhanced GP2 compiler with improved graph data structure.

    Implements rule-based graph programs matching the time complexity
    of imperative algorithms on synthetic Erdős–Rényi graphs.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config is not None else self._default_config()
        self._setup_logging()
        self.results = []

    # ------------------------------------------------------------------ config
    def _default_config(self) -> Dict[str, Any]:
        return {
            "graph": {
                "generator": "erdos_renyi",
                "directed": False,
                "weighted": False,
                "self_loops": False,
                "multi_edges": False,
                "ensure_connected": True,
                "weight_min": 1,
                "weight_max": 10,
                "weight_integer": True,
            },
            "rules": {
                "match_mode": "sequential",
                "max_rule_depth": 3,
                "max_rounds": 10000,
                "apply_per_round": 1,
                "confluence_check": True,
                "termination_check": "fixed_point",
                "store_trace": False,
            },
            "algorithm": {
                "reachability": {
                    "source_vertex": 0,
                    "explore_neighbors": "adjacency_list",
                },
                "shortest_path": {
                    "source_vertex": 0,
                    "edge_weight_key": "weight",
                    "unreachable_distance": 1.0e18,
                },
                "connected_components": {
                    "label_key": "component_id",
                    "union_mode": "rule_based",
                },
            },
            "benchmark": {
                "num_trials": 10,
                "warmup_runs": 3,
                "timing_unit": "seconds",
                "timer": "time.perf_counter",
                "timeout_seconds": 600,
                "verify_correctness": True,
                "comparison_baseline": "imperative",
                "aggregate_stat": "median",
                "report_quantiles": [0.25, 0.5, 0.75],
            },
            "logging": {
                "level": "INFO",
                "output_dir": "results",
                "log_file": "logs/run.log",
                "save_raw_timings": True,
                "export_format": "json",
                "timestamp_in_filename": True,
            },
            "system": {
                "device": "cpu",
                "num_workers": 0,
                "seed": 42,
            },
        }

    def _setup_logging(self) -> None:
        level_str = self.config.get("logging", {}).get("level", "INFO")
        level = getattr(logging, level_str, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    # ----------------------------------------------------------- graph gen
    def generate_erdos_renyi_graph(
        self, n: int, p: float, seed: int = 42
    ) -> Dict[int, List[int]]:
        rng = random.Random(seed)
        adjacency: Dict[int, set] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        if self.config["graph"].get("ensure_connected", True):
            self._ensure_connected(adjacency, n, rng)
        return {v: sorted(adjacency[v]) for v in range(n)}

    @staticmethod
    def _ensure_connected(adjacency: Dict[int, set], n: int, rng: random.Random) -> None:
        if n <= 1:
            return
        visited: set = set()
        components: List[List[int]] = []
        for node in range(n):
            if node not in visited:
                comp: List[int] = []
                queue = deque([node])
                while queue:
                    cur = queue.popleft()
                    if cur not in visited:
                        visited.add(cur)
                        comp.append(cur)
                        for nb in adjacency[cur]:
                            if nb not in visited:
                                queue.append(nb)
                components.append(comp)
        for idx in range(len(components) - 1):
            u = rng.choice(components[idx])
            v = rng.choice(components[idx + 1])
            adjacency[u].add(v)
            adjacency[v].add(u)

    # ------------------------------------------------- rule-based algorithms
    def rule_based_reachability(
        self, graph: Dict[int, List[int]], source: int
    ) -> Dict[int, bool]:
        labels: Dict[int, bool] = {v: (v == source) for v in graph}
        max_rounds = self.config["rules"].get("max_rounds", 10000)
        for _ in range(max_rounds):
            changed = False
            new_labels = dict(labels)
            for v in graph:
                if not labels[v]:
                    for u in graph.get(v, []):
                        if labels.get(u, False):
                            new_labels[v] = True
                            changed = True
                            break
            labels = new_labels
            if not changed:
                break
        return labels

    def rule_based_shortest_path(
        self,
        graph: Dict[int, List[int]],
        source: int,
        weights: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Dict[int, float]:
        if weights is None:
            weights = {}
            for v in graph:
                for u in graph[v]:
                    weights[(v, u)] = 1.0
        unreachable = self.config["algorithm"].get("shortest_path", {}).get(
            "unreachable_distance", 1.0e18
        )
        dist: Dict[int, float] = {v: unreachable for v in graph}
        dist[source] = 0.0
        max_rounds = self.config["rules"].get("max_rounds", 10000)
        for _ in range(max_rounds):
            changed = False
            new_dist = dict(dist)
            for v in graph:
                for u in graph.get(v, []):
                    w = weights.get((v, u), weights.get((u, v), 1.0))
                    if dist[v] + w < new_dist[u]:
                        new_dist[u] = dist[v] + w
                        changed = True
            dist = new_dist
            if not changed:
                break
        return dist

    def rule_based_connected_components(
        self, graph: Dict[int, List[int]]
    ) -> Dict[int, int]:
        labels: Dict[int, int] = {v: v for v in graph}
        max_rounds = self.config["rules"].get("max_rounds", 10000)
        for _ in range(max_rounds):
            changed = False
            new_labels = dict(labels)
            for v in graph:
                for u in graph.get(v, []):
                    if labels[v] < labels[u]:
                        new_labels[u] = labels[v]
                        changed = True
                    elif labels[u] < labels[v]:
                        new_labels[v] = labels[u]
                        changed = True
            labels = new_labels
            if not changed:
                break
        return labels

    # ------------------------------------------------ imperative algorithms
    @staticmethod
    def imperative_reachability(
        graph: Dict[int, List[int]], source: int
    ) -> Dict[int, bool]:
        visited: Dict[int, bool] = {v: False for v in graph}
        visited[source] = True
        queue = deque([source])
        while queue:
            cur = queue.popleft()
            for nb in graph.get(cur, []):
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        return visited

    @staticmethod
    def imperative_shortest_path(
        graph: Dict[int, List[int]],
        source: int,
        weights: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Dict[int, float]:
        if weights is None:
            weights = {}
            for v in graph:
                for u in graph[v]:
                    weights[(v, u)] = 1.0
        unreachable = 1.0e18
        dist: Dict[int, float] = {v: unreachable for v in graph}
        dist[source] = 0.0
        visited: set = set()
        heap: List[Tuple[float, int]] = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            for v in graph.get(u, []):
                w = weights.get((u, v), weights.get((v, u), 1.0))
                if d + w < dist[v]:
                    dist[v] = d + w
                    heapq.heappush(heap, (dist[v], v))
        return dist

    @staticmethod
    def imperative_connected_components(
        graph: Dict[int, List[int]]
    ) -> Dict[int, int]:
        labels: Dict[int, int] = {}
        current_label = 0
        for v in graph:
            if v not in labels:
                stack = [v]
                while stack:
                    cur = stack.pop()
                    if cur not in labels:
                        labels[cur] = current_label
                        for nb in graph.get(cur, []):
                            if nb not in labels:
                                stack.append(nb)
                current_label += 1
        return labels

    # ------------------------------------------------------------- benchmark
    @staticmethod
    def _quantile(data: List[float], q: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        idx = q * (len(s) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        return s[lo] * (1.0 - frac) + s[hi] * frac

    def _benchmark(
        self, func, *args, **kwargs
    ) -> Tuple[List[float], Any]:
        num_trials = self.config["benchmark"].get("num_trials", 10)
        warmup = self.config["benchmark"].get("warmup_runs", 3)
        for _ in range(warmup):
            func(*args, **kwargs)
        timings: List[float] = []
        result = None
        for _ in range(num_trials):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            t1 = time.perf_counter()
            timings.append(t1 - t0)
        return timings, result

    def _summarise(self, timings: List[float]) -> Dict[str, Any]:
        quants = self.config["benchmark"].get("report_quantiles", [0.25, 0.5, 0.75])
        summary: Dict[str, Any] = {
            "timings": timings,
            "median": self._quantile(timings, 0.5),
            "quantiles": {str(q): self._quantile(timings, q) for q in quants},
        }
        return summary

    # ---------------------------------------------------------------- run
    def run(self) -> List[Dict[str, Any]]:
        """Run the full benchmark suite and return results."""
        self.logger.info("Starting enhanced GP2 compiler experiment")
        seed = self.config["system"].get("seed", 42)
        random.seed(seed)

        # Three experiment configurations as indicated by experiment.runs
        sizes: List[Tuple[int, float]] = [
            (50, 0.1),
            (100, 0.05),
            (200, 0.02),
        ]

        all_results: List[Dict[str, Any]] = []
        verify = self.config["benchmark"].get("verify_correctness", True)
        source = self.config["algorithm"]["reachability"]["source_vertex"]

        for n, p in sizes:
            self.logger.info(f"Experiment: n={n}, p={p}")
            graph = self.generate_erdos_renyi_graph(n, p, seed=seed)

            w_min = self.config["graph"].get("weight_min", 1)
            w_max = self.config["graph"].get("weight_max", 10)
            weights: Dict[Tuple[int, int], float] = {}
            for v in graph:
                for u in graph[v]:
                    weights[(v, u)] = float(random.randint(w_min, w_max))

            exp: Dict[str, Any] = {"n": n, "p": p, "algorithms": {}}

            # ---- reachability ----
            r_times, r_res = self._benchmark(self.rule_based_reachability, graph, source)
            i_times, i_res = self._benchmark(self.imperative_reachability, graph, source)
            if verify:
                assert r_res == i_res, "reachability mismatch"
            exp["algorithms"]["reachability"] = {
                "rule_based": self._summarise(r_times),
                "imperative": self._summarise(i_times),
            }

            # ---- shortest path ----
            r_times, r_res = self._benchmark(
                self.rule_based_shortest_path, graph, source, weights
            )
            i_times, i_res = self._benchmark(
                self.imperative_shortest_path, graph, source, weights
            )
            if verify:
                for v in graph:
                    assert abs(r_res[v] - i_res[v]) < 1e-6, f"sp mismatch at {v}"
            exp["algorithms"]["shortest_path"] = {
                "rule_based": self._summarise(r_times),
                "imperative": self._summarise(i_times),
            }

            # ---- connected components ----
            r_times, r_res = self._benchmark(self.rule_based_connected_components, graph)
            i_times, i_res = self._benchmark(
                self.imperative_connected_components, graph
            )
            if verify:
                r_parts = defaultdict(set)
                for v, lb in r_res.items():
                    r_parts[lb].add(v)
                i_parts = defaultdict(set)
                for v, lb in i_res.items():
                    i_parts[lb].add(v)
                assert {frozenset(s) for s in r_parts.values()} == {
                    frozenset(s) for s in i_parts.values()
                }, "cc mismatch"
            exp["algorithms"]["connected_components"] = {
                "rule_based": self._summarise(r_times),
                "imperative": self._summarise(i_times),
            }

            all_results.append(exp)

        self.results = all_results
        self._save_results(all_results)
        self.logger.info("Experiment completed")
        return all_results

    # ------------------------------------------------------------ persistence
    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        output_dir = self.config["logging"].get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)
        name = "experiment_results"
        if self.config["logging"].get("timestamp_in_filename", True):
            name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = self.config["logging"].get("export_format", "json")
        path = os.path.join(output_dir, f"{name}.{ext}")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        self.logger.info(f"Results saved to {path}")