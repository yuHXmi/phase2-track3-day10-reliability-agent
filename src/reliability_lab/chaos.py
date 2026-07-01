from __future__ import annotations

import json
import random
from pathlib import Path

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker
from reliability_lab.config import LabConfig, ScenarioConfig
from reliability_lab.gateway import ReliabilityGateway
from reliability_lab.metrics import RunMetrics
from reliability_lab.providers import FakeLLMProvider


def load_queries(path: str | Path = "data/sample_queries.jsonl") -> list[str]:
    queries: list[str] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        queries.append(json.loads(line)["query"])
    return queries


def build_gateway(config: LabConfig, provider_overrides: dict[str, float] | None = None) -> ReliabilityGateway:
    providers = []
    for p in config.providers:
        fail_rate = provider_overrides.get(p.name, p.fail_rate) if provider_overrides else p.fail_rate
        providers.append(FakeLLMProvider(p.name, fail_rate, p.base_latency_ms, p.cost_per_1k_tokens))
    breakers = {
        p.name: CircuitBreaker(
            name=p.name,
            failure_threshold=config.circuit_breaker.failure_threshold,
            reset_timeout_seconds=config.circuit_breaker.reset_timeout_seconds,
            success_threshold=config.circuit_breaker.success_threshold,
        )
        for p in config.providers
    }
    cache: ResponseCache | SharedRedisCache | None = None
    if config.cache.enabled:
        if config.cache.backend == "redis":
            cache = SharedRedisCache(
                config.cache.redis_url,
                config.cache.ttl_seconds,
                config.cache.similarity_threshold,
            )
        else:
            cache = ResponseCache(config.cache.ttl_seconds, config.cache.similarity_threshold)
    return ReliabilityGateway(providers, breakers, cache)


def calculate_recovery_time_ms(gateway: ReliabilityGateway) -> float | None:
    """Derive recovery time from circuit breaker transition logs.

    TODO(student): Implement recovery time calculation:
    1. For each breaker in gateway.breakers.values():
       - Walk breaker.transition_log entries
       - Track when circuit goes to "open" (save ts)
       - Track when circuit goes to "closed" (compute delta from open ts)
       - Recovery time = (close_ts - open_ts) * 1000 (convert to ms)
    2. Return average of all recovery times, or None if no recovery occurred.

    Each transition_log entry is a dict with keys: "from", "to", "reason", "ts"
    where "ts" is time.time() (epoch seconds).
    """
    deltas = []
    for breaker in gateway.breakers.values():
        last_open_ts = None
        for entry in breaker.transition_log:
            state_to = entry["to"]
            ts = entry["ts"]
            if not isinstance(ts, (int, float)):
                continue
            if state_to == "open":
                last_open_ts = ts
            elif state_to == "closed":
                if last_open_ts is not None and isinstance(last_open_ts, (int, float)):
                    deltas.append((ts - last_open_ts) * 1000.0)
                    last_open_ts = None
    if deltas:
        return sum(deltas) / len(deltas)
    return None


def run_scenario(config: LabConfig, queries: list[str], scenario: ScenarioConfig) -> RunMetrics:
    """Run a single named chaos scenario.

    TODO(student): Implement the scenario runner:
    1. Build gateway with build_gateway(config, scenario.provider_overrides or None)
    2. Create empty RunMetrics()
    3. Loop config.load_test.requests times:
       a. Pick random query from queries
       b. Call gateway.complete(prompt)
       c. Update metrics:
          - total_requests += 1
          - estimated_cost += result.estimated_cost
          - If cache_hit: cache_hits += 1, estimated_cost_saved += 0.001
          - If route == "fallback": fallback_successes += 1, successful_requests += 1
          - If route == "static_fallback": static_fallbacks += 1, failed_requests += 1
          - Else: successful_requests += 1
          - If result.latency_ms > 0: append to latencies_ms
    4. Count circuit_open_count from breaker transition logs (entries where to == "open")
    5. Set recovery_time_ms via calculate_recovery_time_ms(gateway)
    6. Return metrics
    """
    gateway = build_gateway(config, scenario.provider_overrides or None)
    metrics = RunMetrics()

    for _ in range(config.load_test.requests):
        query = random.choice(queries)
        result = gateway.complete(query)

        metrics.total_requests += 1
        metrics.estimated_cost += result.estimated_cost

        if result.cache_hit:
            metrics.cache_hits += 1
            metrics.estimated_cost_saved += 0.001

        if result.route == "fallback":
            metrics.fallback_successes += 1
            metrics.successful_requests += 1
        elif result.route == "static_fallback":
            metrics.static_fallbacks += 1
            metrics.failed_requests += 1
        else:
            metrics.successful_requests += 1

        if result.latency_ms > 0:
            metrics.latencies_ms.append(result.latency_ms)

    for breaker in gateway.breakers.values():
        metrics.circuit_open_count += sum(1 for entry in breaker.transition_log if entry["to"] == "open")

    metrics.recovery_time_ms = calculate_recovery_time_ms(gateway)
    return metrics


def run_simulation(config: LabConfig, queries: list[str]) -> RunMetrics:
    """Run all named scenarios from config, or a default run if none defined.

    TODO(student): Add a cache vs no-cache comparison scenario.
    Extend with your own custom scenarios (e.g., cost cap near limit).
    """
    import copy
    import time
    from reliability_lab.circuit_breaker import CircuitState, CircuitBreaker

    if not config.scenarios:
        default_scenario = ScenarioConfig(name="default", description="baseline run")
        metrics = run_scenario(config, queries, default_scenario)
        metrics.scenarios = {"default": "pass" if metrics.successful_requests > 0 else "fail"}
        return metrics

    # 1. Cache vs No-Cache Comparison
    print("\n" + "=" * 40)
    print("  CACHE VS NO-CACHE COMPARISON")
    print("=" * 40)

    # Run with cache
    cache_enabled_config = copy.deepcopy(config)
    cache_enabled_config.cache.enabled = True
    warmup_gw = build_gateway(cache_enabled_config)
    for q in queries[:20]:
        warmup_gw.complete(q)
    with_cache_metrics = run_scenario(cache_enabled_config, queries, ScenarioConfig(name="with_cache", provider_overrides={}))

    # Run without cache
    cache_disabled_config = copy.deepcopy(config)
    cache_disabled_config.cache.enabled = False
    no_cache_metrics = run_scenario(cache_disabled_config, queries, ScenarioConfig(name="no_cache", provider_overrides={}))

    p50_no = no_cache_metrics.percentile(50)
    p50_with = with_cache_metrics.percentile(50)
    p95_no = no_cache_metrics.percentile(95)
    p95_with = with_cache_metrics.percentile(95)
    cost_no = no_cache_metrics.estimated_cost
    cost_with = with_cache_metrics.estimated_cost
    hit_with = with_cache_metrics.cache_hit_rate

    print("| Metric          | Without cache | With cache | Delta |")
    print("|---|---:|---:|---|")
    print(f"| latency_p50_ms  | {p50_no:13.2f} | {p50_with:10.2f} | {p50_with - p50_no:5.2f} |")
    print(f"| latency_p95_ms  | {p95_no:13.2f} | {p95_with:10.2f} | {p95_with - p95_no:5.2f} |")
    print(f"| estimated_cost  | {cost_no:13.6f} | {cost_with:10.6f} | {cost_with - cost_no:5.6f} |")
    print(f"| cache_hit_rate  | {0.0:12.2f}% | {hit_with * 100:9.2f}% | {hit_with * 100:4.2f}% |")
    print("=" * 40 + "\n")

    combined = RunMetrics()

    # Run default scenarios from config
    for scenario in config.scenarios:
        result = run_scenario(config, queries, scenario)

        # Define pass/fail criteria per scenario
        if scenario.name == "primary_timeout_100":
            passed = result.fallback_success_rate > 0.8
        elif scenario.name == "primary_flaky_50":
            passed = result.availability >= 0.8
        elif scenario.name == "all_healthy":
            passed = result.availability >= 0.95
        else:
            passed = result.successful_requests > 0

        combined.scenarios[scenario.name] = "pass" if passed else "fail"

        combined.total_requests += result.total_requests
        combined.successful_requests += result.successful_requests
        combined.failed_requests += result.failed_requests
        combined.fallback_successes += result.fallback_successes
        combined.static_fallbacks += result.static_fallbacks
        combined.cache_hits += result.cache_hits
        combined.circuit_open_count += result.circuit_open_count
        combined.estimated_cost += result.estimated_cost
        combined.estimated_cost_saved += result.estimated_cost_saved
        combined.latencies_ms.extend(result.latencies_ms)
        if result.recovery_time_ms is not None:
            if combined.recovery_time_ms is None:
                combined.recovery_time_ms = result.recovery_time_ms
            else:
                combined.recovery_time_ms = (combined.recovery_time_ms + result.recovery_time_ms) / 2

    # 2. Custom Scenario: Recovery Scenario
    print("Running Custom Scenario: recovery_scenario...")
    rec_config = copy.deepcopy(config)
    rec_providers = []
    for p in rec_config.providers:
        fail_rate = 1.0 if p.name == "primary" else p.fail_rate
        rec_providers.append(FakeLLMProvider(p.name, fail_rate, p.base_latency_ms, p.cost_per_1k_tokens))
    rec_breakers = {
        p.name: CircuitBreaker(
            name=p.name,
            failure_threshold=rec_config.circuit_breaker.failure_threshold,
            reset_timeout_seconds=rec_config.circuit_breaker.reset_timeout_seconds,
            success_threshold=rec_config.circuit_breaker.success_threshold,
        )
        for p in rec_config.providers
    }
    rec_gateway = ReliabilityGateway(rec_providers, rec_breakers, None)

    # Trip breaker
    for _ in range(rec_config.circuit_breaker.failure_threshold + 1):
        rec_gateway.complete(random.choice(queries))

    assert rec_gateway.breakers["primary"].state == CircuitState.OPEN

    # Sleep to let reset timeout elapse
    time.sleep(rec_config.circuit_breaker.reset_timeout_seconds + 0.1)

    # Make primary healthy
    for prov in rec_gateway.providers:
        if prov.name == "primary":
            prov.fail_rate = 0.0

    # Probe & close breaker
    for _ in range(rec_config.circuit_breaker.success_threshold + 1):
        rec_gateway.complete(random.choice(queries))

    assert rec_gateway.breakers["primary"].state == CircuitState.CLOSED  # type: ignore[comparison-overlap]

    rec_time = calculate_recovery_time_ms(rec_gateway)
    if rec_time is not None:
        combined.recovery_time_ms = rec_time
        print(f"Captured recovery time: {rec_time:.2f} ms")
        combined.scenarios["recovery_scenario"] = "pass"
    else:
        combined.scenarios["recovery_scenario"] = "fail"

    # 3. Custom Scenario: Cost Budget Exceeded
    print("Running Custom Scenario: cost_budget_exceeded...")
    cost_config = copy.deepcopy(config)
    cost_config.cache.enabled = False
    cost_gateway = build_gateway(cost_config)
    cost_gateway.cost_budget = 0.002
    for _ in range(15):
        cost_gateway.complete(random.choice(queries))
    print(f"Cost simulation completed. Cumulative cost: {cost_gateway.cumulative_cost:.6f}")
    if cost_gateway.cumulative_cost >= cost_gateway.cost_budget:
        combined.scenarios["cost_budget_exceeded"] = "pass"
    else:
        combined.scenarios["cost_budget_exceeded"] = "fail"

    return combined
