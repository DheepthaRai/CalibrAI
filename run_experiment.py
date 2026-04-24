"""
CalibrAI Experiment Runner
Runs calibration for Banking and Healthcare, 10 queries each,
then prints a results table with block rates, FP/FN rates per level.
"""
import asyncio
import json
import time
import websockets
import httpx

API = "http://localhost:8000"
WS  = "ws://localhost:8000"

INDUSTRIES = ["Banking", "Healthcare"]
WAVE_SIZE  = 10

async def run_industry(industry: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Starting calibration: {industry}  (wave_size={WAVE_SIZE})")
    print('='*60)

    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API}/api/calibration/start", json={
            "industry":          industry,
            "wave_size":         WAVE_SIZE,
            "cost_per_violation": 5000,
            "weekly_volume":      10000,
            "base_url":          "http://localhost:11434",
            "llama_model":       "llama3.1:latest",
            "deepseek_model":    "deepseek-r1:7b",
        }, timeout=10)
        resp.raise_for_status()
        run_id = resp.json()["run_id"]
        print(f"  run_id: {run_id}")

    uri = f"{WS}/ws/calibration/{run_id}"
    result = None
    t0 = time.time()

    async with websockets.connect(uri, ping_interval=None, open_timeout=15) as ws:
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=600)
            except asyncio.TimeoutError:
                print("  [TIMEOUT] No message for 600s")
                break

            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "heartbeat":
                elapsed = time.time() - t0
                print(f"  [heartbeat] {elapsed:.0f}s elapsed")

            elif mtype == "status":
                print(f"  [status] {msg['message']}")

            elif mtype == "queries_ready":
                print(f"  [queries_ready] {msg['message']}")

            elif mtype == "query_result":
                pct = msg.get("pct", 0)
                qid = msg.get("qid")
                attack = "ATTACK" if msg.get("is_attack") else "legit"
                q = msg.get("query", "")[:50]
                # per-level blocked summary
                levels = msg.get("results", {})
                blocked_at = [str(lvl) for lvl, r in levels.items() if r.get("blocked")]
                print(f"  [{pct:5.1f}%] q{qid} ({attack}) blocked@L[{','.join(blocked_at) or 'none'}]  \"{q}\"")

            elif mtype == "complete":
                result = msg
                print(f"\n  [complete] recommended_level={msg.get('recommended_level')}")
                break

            elif mtype == "error":
                print(f"  [ERROR] {msg.get('message')}")
                break

    return {"industry": industry, "run_id": run_id, "result": result}


def print_table(runs: list[dict]):
    LEVEL_NAMES = {1: "Very Strict", 2: "Strict", 3: "Balanced", 4: "Permissive", 5: "Very Permissive"}

    for run in runs:
        industry = run["industry"]
        result   = run.get("result")
        if not result:
            print(f"\n{industry}: NO RESULT (run failed or timed out)")
            continue

        summary = result.get("summary", {})
        rec     = result.get("recommended_level", "?")
        rec_text = result.get("recommendation_text", "")

        print(f"\n{'─'*72}")
        print(f"  {industry.upper()}  (recommended: Level {rec} — {LEVEL_NAMES.get(rec, '?')})")
        print(f"{'─'*72}")
        print(f"  {'Level':<6} {'Name':<16} {'Atk Block%':>10} {'FP Rate%':>9} {'Score':>7} {'Atk#':>5} {'FP#':>4}")
        print(f"  {'─'*6} {'─'*16} {'─'*10} {'─'*9} {'─'*7} {'─'*5} {'─'*4}")

        for lvl in range(1, 6):
            s = summary.get(str(lvl)) or summary.get(lvl)
            if not s:
                print(f"  {lvl:<6} {'(no data)'}")
                continue
            marker = " ◀ optimal" if lvl == rec else ""
            print(
                f"  {lvl:<6} {s['name']:<16}"
                f" {s['attack_block_rate']*100:>9.1f}%"
                f" {s['fp_rate']*100:>8.1f}%"
                f" {s['score']:>7.4f}"
                f" {s['attacks_blocked']:>3}/{s['attack_count']:<2}"
                f" {s['legit_blocked']:>2}/{s['legit_count']:<2}"
                f"{marker}"
            )

        print(f"\n  Recommendation: {rec_text[:200]}")

    print(f"\n{'═'*72}")
    print("  CROSS-INDUSTRY SUMMARY")
    print(f"{'═'*72}")
    print(f"  {'Industry':<14} {'Optimal':>8} {'L1 Block%':>10} {'L3 Block%':>10} {'L5 Block%':>10}")
    print(f"  {'─'*14} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for run in runs:
        result = run.get("result")
        if not result:
            continue
        s = result.get("summary", {})
        rec = result.get("recommended_level", "?")
        def abr(lvl):
            d = s.get(str(lvl)) or s.get(lvl)
            return f"{d['attack_block_rate']*100:.1f}%" if d else "—"
        print(f"  {run['industry']:<14} {'L'+str(rec):>8} {abr(1):>10} {abr(3):>10} {abr(5):>10}")


async def main():
    runs = []
    for ind in INDUSTRIES:
        run = await run_industry(ind)
        runs.append(run)

    print_table(runs)

asyncio.run(main())
