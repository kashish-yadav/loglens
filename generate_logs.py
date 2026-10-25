"""Generate synthetic log data (fixed seed, no real data).  Usage: python generate_logs.py

Writes three CSVs into data/:
  sample_logs.csv    demo input for the dashboard (no labels; includes duplicates + a few bad rows)
  training_logs.csv  labeled training data (column `label`)
  evaluation_logs.csv labeled held-out data (column `label`)

Leakage control: every event has several phrasings (templates). Training and evaluation use
DIFFERENT templates, so the evaluation set tests unseen wording. The sample uses all of them,
so it must not be used to measure accuracy.
"""
import random
import re
import string
from datetime import datetime, timedelta

import pandas as pd

from common import DATA_DIR, EVAL_PATH, INPUT_COLUMNS, KEEP, REVIEW, SAMPLE_PATH, SUPPRESS, TRAIN_PATH
from common import normalize_message

SEED = 42
START = datetime(2026, 10, 12)
ALL = ("payment-service", "auth-service", "api-gateway", "database-service", "notification-service")
EPS = ("/api/v1/orders", "/api/v1/cart", "/api/v1/products")

# label, services, [(level, weight)], endpoints, statuses, latency range, weight, train templates, eval templates
EVENTS = [
    (KEEP, ("payment-service",), [("ERROR", 8), ("WARN", 2)], ("/api/v1/payments/charge",), (402, 500), (150, 2800), 6,
     ["Payment declined for order {order}: issuer response code {code}",
      "Payment processing failed: gateway rejected transaction {txn}",
      "Charge failed for {amount} on order {order}, card authorization error"],
     ["Transaction {txn} declined by issuer (insufficient funds)",
      "Unable to capture payment for order {order}: payment gateway returned failure"]),
    (KEEP, ("auth-service",), [("WARN", 6), ("ERROR", 4)], ("/api/v1/auth/login",), (401, 403), (20, 400), 6,
     ["Authentication failed for user {user}: invalid credentials",
      "Login rejected: account {user} locked after repeated failed attempts",
      "Token validation failed for session {sess}: signature mismatch"],
     ["Access denied for user {user}, JWT expired", "Failed login attempt from {ip} for account {user}"]),
    (KEEP, ("api-gateway",), [("ERROR", 1)], EPS, (504, 408), (5000, 30000), 5,
     ["Upstream timeout after {ms}ms calling {svc}", "Gateway timeout on {ep}: {svc} did not respond",
      "Request to {svc} timed out while handling {ep}"],
     ["Upstream {svc} failed to respond within {ms} ms, returning 504", "Timeout waiting for response from {svc}"]),
    (KEEP, ("database-service",), [("CRITICAL", 6), ("ERROR", 4)], ("",), (None, 503), None, 4,
     ["Database connection refused by host {host}", "Connection pool exhausted: unable to acquire connection to {host}",
      "Failed to connect to database primary {host}: connection timed out"],
     ["Cannot reach database {host}, connection attempt failed", "Too many connections: {host} refusing new client sessions"]),
    (KEEP, ALL, [("CRITICAL", 6), ("ERROR", 4)], ("",), (500, None), None, 3,
     ["Unhandled exception in OrderProcessor.process: NullPointerException",
      "Uncaught TypeError while serializing response payload", "FATAL: out of memory, worker process terminated"],
     ["Critical exception: IndexOutOfBoundsException thrown in InvoiceBuilder.render",
      "Service crashed with segmentation fault, restarting worker"]),
    (KEEP, ("api-gateway", "payment-service", "auth-service"), [("WARN", 1)], EPS, (200,), (1500, 9000), 5,
     ["High latency detected: {ep} responded in {ms}ms (threshold 1000ms)", "Slow response of {ms}ms for {ep}",
      "Latency SLO breach on {ep}: p95 above target"],
     ["Response time of {ms} ms exceeded warning threshold for {ep}", "{ep} is responding slowly ({ms}ms)"]),
    (SUPPRESS, ALL, [("INFO", 1)], ("/health",), (200,), (2, 45), 14,
     ["Health check passed: {service} responded 200 in {ms}ms", "Health check OK", "Liveness probe succeeded for {service}"],
     ["Readiness check ok, all dependencies reachable", "Service {service} reports healthy status"]),
    (SUPPRESS, ALL, [("INFO", 6), ("DEBUG", 4)], ("",), (None,), None, 6,
     ["Heartbeat received from {service}", "Heartbeat sent to cluster coordinator", "Keepalive ping acknowledged"],
     ["Node heartbeat acknowledged by scheduler", "Periodic ping completed, service {service} alive"]),
    (SUPPRESS, ("api-gateway", "payment-service", "auth-service"), [("INFO", 1)], EPS, (200, 200, 201, 204), (15, 280), 12,
     ["Request completed: {method} {ep} returned {status} in {ms}ms", "{method} {ep} handled successfully",
      "Served {method} request for {ep}"],
     ["Successfully processed {method} {ep} (status {status})", "Response sent to client for {ep} with {status}"]),
    (SUPPRESS, ("payment-service", "notification-service", "database-service"), [("INFO", 1)], ("",), (None,), (50, 900), 6,
     ["Scheduled job {job} completed", "Job {job} finished in {ms}ms", "Cron task {job} executed successfully"],
     ["Background task {job} ran to completion", "{job} job completed, next run scheduled"]),
    (SUPPRESS, ("auth-service",), [("INFO", 7), ("DEBUG", 3)], ("/api/v1/auth/session",), (200, 201), (10, 150), 5,
     ["Session created for user {user}", "Token issued for user {user}, expires in 3600s", "User {user} logged in successfully"],
     ["New session {sess} established", "Access token refreshed for {user}"]),
    (REVIEW, ("api-gateway", "payment-service", "auth-service"), [("INFO", 5), ("WARN", 5)], EPS,
     (418, 299, 599, 460, 520), (20, 800), 5,
     ["Non-standard response returned for {ep}", "Unrecognized status code from downstream while serving {ep}",
      "Response code outside expected set for {ep}"],
     ["Downstream replied with an unusual code on {ep}", "Unmapped HTTP status observed for {ep}"]),
    (REVIEW, ALL, [("INFO", 4), ("WARN", 3), ("DEBUG", 3)], ("",), (None,), None, 5,
     ["Webhook received with unrecognized event type {etype}", "Unknown message format on queue {queue}, default handler used",
      "Feature flag {flag} evaluated to unexpected value"],
     ["Unexpected event type {etype} encountered, skipping enrichment",
      "Payload schema mismatch on queue {queue}, applying defaults"]),
    (REVIEW, ALL, [("INFO", 5), ("WARN", 5)], ("",), (None, 200), (20, 900), 5,
     ["Retrying operation after transient condition", "Request routed through fallback path",
      "Deprecated API version v1 invoked by client"],
     ["Client approaching rate limit threshold", "Cache miss ratio above expected range"]),
]


def _slot(name, rng, rec):
    hexs = lambda n: "".join(rng.choices("0123456789abcdef", k=n))  # noqa: E731
    return {
        "order": lambda: f"ORD-{rng.randint(10000, 99999)}", "txn": lambda: f"TXN-{rng.randint(10**7, 10**8 - 1)}",
        "user": lambda: f"user_{rng.randint(1000, 9999)}", "sess": lambda: f"sess_{hexs(8)}",
        "ip": lambda: f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
        "amount": lambda: f"${rng.randint(5, 900)}.{rng.randint(0, 99):02d}",
        "code": lambda: rng.choice(["05", "51", "54", "91"]),
        "svc": lambda: rng.choice(["inventory-service", "shipping-service", "pricing-service"]),
        "host": lambda: rng.choice(["db-primary", "db-replica"]),
        "job": lambda: rng.choice(["cache-refresh", "report-rollup", "token-cleanup"]),
        "etype": lambda: rng.choice(["order.reconcile", "user.merge", "invoice.sync"]),
        "queue": lambda: rng.choice(["billing-events", "audit-stream"]),
        "flag": lambda: rng.choice(["new-checkout-flow", "beta-search"]),
        "method": lambda: rng.choice(["GET", "GET", "POST"]),
        "ms": lambda: str(rec["latency_ms"] or rng.randint(1100, 4000)),
        "status": lambda: str(rec["status_code"] or 200), "ep": lambda: rec["endpoint"] or EPS[0],
        "service": lambda: rec["service"],
    }[name]()


def _timestamp(rng):
    hour = rng.choices(range(24), weights=[1] * 6 + [3, 5, 7, 8, 8, 8, 7, 8, 8, 8, 7, 6, 5, 4, 3, 2, 2, 1], k=1)[0]
    return START + timedelta(days=rng.randrange(7), hours=hour, minutes=rng.randrange(60), seconds=rng.randrange(60))


def _generate(part: str, n: int, seed: int) -> list[dict]:
    """part: 'train' | 'eval' | 'all' - which template set to draw from."""
    rng = random.Random(seed)
    total = sum(e[6] for e in EVENTS)
    rows = []
    for label, services, levels, eps, statuses, lat, weight, train_t, eval_t in EVENTS:
        pool = {"train": train_t, "eval": eval_t, "all": train_t + eval_t}[part]
        for i in range(max(2, round(n * weight / total))):
            template = pool[i % len(pool)] if i < len(pool) else rng.choice(pool)
            status, latency = rng.choice(statuses), (rng.randint(*lat) if lat else None)
            rec = {"timestamp": _timestamp(rng).strftime("%Y-%m-%d %H:%M:%S"),
                   "level": rng.choices([l for l, _ in levels], weights=[w for _, w in levels])[0],
                   "service": rng.choice(services), "request_id": "req-" + "".join(rng.choices(string.hexdigits[:16].lower(), k=8)),
                   "endpoint": rng.choice(eps), "status_code": "" if status is None else status,
                   "latency_ms": "" if latency is None else latency}
            rec["message"] = re.sub(r"\{(\w+)\}", lambda m: _slot(m.group(1), rng, rec), template)
            rec["label"] = label
            for col in ("request_id", "endpoint", "latency_ms"):      # occasional missing optional fields
                if rng.random() < 0.03:
                    rec[col] = ""
            rows.append(rec)
    return sorted(rows, key=lambda r: r["timestamp"])


def build(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    sample = _generate("all", 300, seed)
    sample += [dict(r) for r in rng.sample(sample, 10)]                 # exact duplicate copies
    sample = sorted(sample, key=lambda r: r["timestamp"])
    for i, (col, bad) in zip(rng.sample(range(len(sample)), 5), [
            ("timestamp", "not-a-timestamp"), ("timestamp", "yesterday"), ("message", ""),
            ("status_code", "abc"), ("level", "VERBOSE")]):
        sample[i][col] = bad                                            # a few deliberately bad values
    train, evaluation = _generate("train", 300, seed + 1), _generate("eval", 150, seed + 2)
    assert not ({normalize_message(r["message"]) for r in train}
                & {normalize_message(r["message"]) for r in evaluation}), "train/eval leakage"
    frame = lambda rows, label: pd.DataFrame(rows)[INPUT_COLUMNS + (["label"] if label else [])]  # noqa: E731
    return {"sample": frame(sample, False), "train": frame(train, True), "eval": frame(evaluation, True)}


def main():
    DATA_DIR.mkdir(exist_ok=True)
    sets = build()
    for name, path in (("sample", SAMPLE_PATH), ("train", TRAIN_PATH), ("eval", EVAL_PATH)):
        df = sets[name]
        df.to_csv(path, index=False)
        print(f"wrote {path.name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
