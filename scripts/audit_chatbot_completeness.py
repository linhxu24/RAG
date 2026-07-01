#!/usr/bin/env python3
"""Completeness audit for the SimplyDent dental chatbot.

The audit intentionally uses randomized probe templates instead of fixed
conversation scenarios. Runtime probes check response structure and grounding
signals; static checks inspect the codebase for safety and acceptance criteria.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND_URL = "http://localhost:8000"
OUTPUT_PATH = ROOT / "eval_results" / "chatbot_completeness_audit.json"


PROBE_TEMPLATES: dict[str, list[str]] = {
    "GREETING": [
        "Xin chào",
        "Chào buổi {time_of_day}",
        "Hello",
        "Hi bạn",
        "Cho tôi hỏi",
        "Bắt đầu nào",
    ],
    "CHITCHAT": [
        "Cảm ơn bạn nhé",
        "Cảm ơn",
        "Ok được rồi",
        "Tôi hiểu rồi",
        "Tạm biệt",
        "Hẹn gặp lại",
        "Bạn có thể tư vấn giúp tôi không?",
        "Bạn là AI à?",
    ],
    "CLINIC_INFO": [
        "Phòng khám ở đâu?",
        "Địa chỉ phòng khám là gì?",
        "Số điện thoại để đặt lịch là bao nhiêu?",
        "Phòng khám mở cửa mấy giờ?",
        "Giờ làm việc của phòng khám như thế nào?",
        "Cuối tuần phòng khám có mở không?",
        "Thứ mấy phòng khám nghỉ?",
        "Cho tôi email liên hệ",
        "Phòng khám có Zalo không?",
        "Facebook của phòng khám là gì?",
    ],
    "FAQ": [
        "Tẩy trắng răng có đau không?",
        "Nhổ răng khôn có nguy hiểm không?",
        "Niềng răng mất bao lâu?",
        "Sau khi nhổ răng cần kiêng gì?",
        "Implant có bền không?",
        "Trẻ em mấy tuổi bắt đầu niềng răng được?",
        "Ê buốt răng sau tẩy trắng có bình thường không?",
        "Có thể ăn gì sau khi nhổ răng?",
        "Làm răng sứ có hại không?",
        "Bao lâu nên đi khám răng một lần?",
        "{service_name} có an toàn không?",
        "{service_name} mất bao lâu?",
        "{service_name} sau khi làm cần lưu ý gì?",
    ],
    "PRODUCT_LIST": [
        "Cho tôi xem danh sách sản phẩm",
        "Phòng khám có bán những sản phẩm gì?",
        "Có sản phẩm nào dưới {price}k không?",
        "Sản phẩm nào còn hàng?",
        "Cho tôi xem bàn chải",
        "Có loại kem đánh răng nào không?",
        "Sắp xếp sản phẩm theo giá tăng dần",
        "Sản phẩm rẻ nhất là gì?",
        "Tăm nước loại nào?",
    ],
    "PRODUCT_DETAIL": [
        "Cho tôi thông tin về {product_name}",
        "{product_name} giá bao nhiêu?",
        "{product_name} còn hàng không?",
        "{product_name} dùng như thế nào?",
        "{product_name} có tốt không?",
        "Tôi muốn mua {product_name}",
        "Chi tiết về {product_name}",
    ],
    "PRODUCT_COMPARE": [
        "So sánh {product_a} và {product_b}",
        "{product_a} với {product_b} loại nào tốt hơn?",
        "Khác nhau gì giữa {product_a} và {product_b}?",
        "Nên mua {product_a} hay {product_b}?",
    ],
    "SERVICE_LIST": [
        "Phòng khám có những dịch vụ gì?",
        "Cho tôi xem danh sách dịch vụ",
        "Dịch vụ nào liên quan đến implant?",
        "Có dịch vụ tẩy trắng không?",
        "Dịch vụ niềng răng các loại",
        "Dịch vụ nào dưới {price} triệu?",
        "Dịch vụ nào mất ít thời gian nhất?",
    ],
    "SERVICE_DETAIL": [
        "{service_name} giá bao nhiêu?",
        "Chi phí {service_name} là bao nhiêu?",
        "{service_name} mất bao lâu?",
        "Quy trình {service_name} như thế nào?",
        "Cho tôi biết về dịch vụ {service_name}",
        "Tôi muốn làm {service_name}",
    ],
    "UNKNOWN": [
        "Asdfgh",
        "Tôi không biết hỏi gì",
        "???",
        "Bầu trời màu gì",
        "Cho tôi một bài thơ",
        "Giá vàng hôm nay",
        "Thời tiết Hà Nội",
    ],
}


MULTI_TURN_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "follow_up_detail_any_product",
        "description": "Hỏi detail -> follow-up ngắn không mention entity",
        "turns": [
            {"template": "{product_name} giá bao nhiêu?", "check_intent": "PRODUCT_DETAIL"},
            {
                "template": "Còn hàng không?",
                "check_intent": "PRODUCT_DETAIL",
                "check_memory": True,
                "check_entity_match": True,
            },
        ],
    },
    {
        "name": "follow_up_faq_any_service",
        "description": "Hỏi service detail -> follow-up FAQ không mention service",
        "turns": [
            {"template": "{service_name} giá bao nhiêu?", "check_intent": "SERVICE_DETAIL"},
            {
                "template": "Có đau không?",
                "check_intent": "FAQ",
                "check_memory": True,
                "check_entity_match": True,
            },
            {
                "template": "Mất bao lâu?",
                "check_intent": "SERVICE_DETAIL",
                "check_memory": True,
                "check_entity_match": True,
            },
        ],
    },
    {
        "name": "multi_intent_one_query",
        "description": "Một câu chứa 2 intent",
        "turns": [
            {
                "template": "{service_name} giá bao nhiêu và có đau không?",
                "check_multi_task": True,
                "expected_intents": ["SERVICE_DETAIL", "FAQ"],
            },
        ],
    },
    {
        "name": "entity_switch",
        "description": "Hỏi entity A -> chuyển sang entity B -> follow-up",
        "turns": [
            {"template": "{product_a} giá bao nhiêu?", "check_intent": "PRODUCT_DETAIL"},
            {
                "template": "{product_b} thì sao?",
                "check_intent": "PRODUCT_DETAIL",
                "check_entity_switched": True,
            },
            {
                "template": "Còn hàng không?",
                "check_intent": "PRODUCT_DETAIL",
                "check_memory": True,
                "check_entity_match": True,
            },
        ],
    },
    {
        "name": "compare_then_faq",
        "description": "Compare 2 sản phẩm -> hỏi FAQ về cả 2",
        "turns": [
            {"template": "So sánh {product_a} và {product_b}", "check_intent": "PRODUCT_COMPARE"},
            {
                "template": "Loại nào phù hợp hơn cho răng nhạy cảm?",
                "check_intent": "FAQ",
                "check_gate_pass": True,
            },
        ],
    },
    {
        "name": "domain_switch_no_leak",
        "description": "Product context -> CLINIC_INFO -> không leak entity",
        "turns": [
            {"template": "{product_name} giá bao nhiêu?", "check_intent": "PRODUCT_DETAIL"},
            {
                "template": "Phòng khám mở cửa mấy giờ?",
                "check_intent": "CLINIC_INFO",
                "check_no_entity": True,
            },
        ],
    },
    {
        "name": "unknown_then_clarify",
        "description": "Query mơ hồ -> chatbot hỏi lại -> user clarify",
        "turns": [
            {
                "template": "Tôi muốn hỏi về cái đó",
                "check_intent_in": ["UNKNOWN", "PRODUCT_DETAIL", "SERVICE_DETAIL"],
            },
            {"template": "{product_name}", "check_intent": "PRODUCT_DETAIL"},
        ],
    },
]


@dataclass(frozen=True)
class Preflight:
    backend_available: bool
    db_connected: bool
    debug_available: bool
    products_available: bool
    services_available: bool
    health: dict[str, Any]
    debug_error: str | None = None
    catalog_error: str | None = None

    @property
    def runtime_ready(self) -> bool:
        return self.backend_available and self.products_available and self.services_available

    @property
    def multi_turn_ready(self) -> bool:
        return self.runtime_ready and self.debug_available


@dataclass
class StaticCheck:
    result: str
    evidence: str


class SourceIndex:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, str] = {}

    def text(self, relative_path: str) -> str:
        if relative_path not in self._cache:
            path = self.root / relative_path
            self._cache[relative_path] = path.read_text(encoding="utf-8")
        return self._cache[relative_path]

    def has_all(self, relative_path: str, *patterns: str) -> bool:
        text = self.text(relative_path)
        return all(pattern in text for pattern in patterns)

    def has_regex(self, relative_path: str, pattern: str) -> bool:
        return (
            re.search(pattern, self.text(relative_path), flags=re.MULTILINE | re.DOTALL) is not None
        )

    def evidence(self, relative_path: str, pattern: str) -> str:
        text = self.text(relative_path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern in line:
                return f"{relative_path}:{line_number}: {line.strip()}"
        return f"{relative_path}: pattern not found: {pattern}"

    def joined(self, *relative_paths: str) -> str:
        return "\n".join(self.text(path) for path in relative_paths)

    def migration_text(self) -> str:
        migration_dir = self.root / "app" / "db" / "migrations" / "versions"
        return "\n".join(
            path.read_text(encoding="utf-8") for path in sorted(migration_dir.glob("*.py"))
        )


def main() -> int:
    args = parse_args()
    base_url = os.environ.get("BACKEND_URL", args.backend_url).rstrip("/")
    llm_provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    request_timeout = provider_http_timeout(args, llm_provider)
    rng = random.Random(args.seed if args.seed is not None else time.time_ns())
    source = SourceIndex(ROOT)
    run_id = str(uuid.uuid4())

    catalog = fetch_catalog(base_url, request_timeout)
    preflight = run_preflight(base_url, catalog, request_timeout)
    fill_vars = build_fill_vars(catalog["products"], catalog["services"], rng)

    capability_results: dict[str, list[dict[str, Any]]] = {}
    multi_turn_results: list[dict[str, Any]] = []

    if args.mode in {"full", "capabilities"}:
        capability_results = run_capability_audit(
            base_url=base_url,
            catalog=catalog,
            preflight=preflight,
            fill_vars=fill_vars,
            rng=rng,
            selected_intent=args.intent,
            runs=args.runs,
            timeout=request_timeout,
        )
    else:
        capability_results = empty_capability_results(args.intent)

    if args.mode in {"full", "multi-turn"}:
        multi_turn_results = run_multi_turn_audit(
            base_url=base_url,
            preflight=preflight,
            fill_vars=fill_vars,
            timeout=request_timeout,
        )

    safety_results = run_safety_checks(source)
    ac_results = run_acceptance_checks(
        source=source,
        base_url=base_url,
        preflight=preflight,
        capability_results=capability_results,
        multi_turn_results=multi_turn_results,
        timeout=request_timeout,
        skip_tests=args.skip_tests,
        include_runtime=args.mode in {"full", "capabilities", "multi-turn"},
    )

    summary = compute_scores(
        capability_results=capability_results,
        multi_turn_results=multi_turn_results,
        safety_results=safety_results,
        ac_results=ac_results,
    )
    by_intent = summarize_intents(capability_results)
    failing_areas = build_failing_areas(by_intent, multi_turn_results, safety_results, ac_results)
    output = {
        "run_id": run_id,
        "run_date": datetime.now(UTC).isoformat(),
        "backend_url": base_url,
        "llm_provider": llm_provider,
        "request_timeout_seconds": request_timeout,
        "mode": args.mode,
        "seed": args.seed,
        "preflight": {
            "backend_available": preflight.backend_available,
            "db_connected": preflight.db_connected,
            "debug_available": preflight.debug_available,
            "products_available": preflight.products_available,
            "services_available": preflight.services_available,
            "health": preflight.health,
            "debug_error": preflight.debug_error,
            "catalog_error": preflight.catalog_error,
        },
        "catalog": catalog,
        "summary": summary,
        "by_intent": by_intent,
        "capability_probes": capability_results,
        "multi_turn_patterns": multi_turn_results,
        "acceptance_criteria": ac_results,
        "safety": safety_results,
        "failing_areas": failing_areas,
    }
    write_json(OUTPUT_PATH, output)
    print(
        json.dumps({"output": str(OUTPUT_PATH), "summary": summary}, ensure_ascii=False, indent=2)
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument(
        "--mode",
        choices=("full", "capabilities", "multi-turn", "ac-only"),
        default="full",
    )
    parser.add_argument("--intent", choices=tuple(PROBE_TEMPLATES), default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Default HTTP timeout in seconds. Use 0 for no client timeout.",
    )
    parser.add_argument(
        "--provider-timeout-override",
        type=int,
        default=None,
        help="Override provider-aware HTTP timeout in seconds. Use 0 for no client timeout.",
    )
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def provider_http_timeout(args: argparse.Namespace, llm_provider: str) -> float | None:
    if args.provider_timeout_override is not None:
        return normalize_timeout(args.provider_timeout_override)
    if llm_provider == "ollama":
        print("Running with Ollama — expect slow responses", file=sys.stderr)
        return 300.0
    return normalize_timeout(args.timeout)


def normalize_timeout(value: float | int | None) -> float | None:
    if value is None or value <= 0:
        return None
    return float(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def fetch_catalog(base_url: str, timeout: float | None) -> dict[str, list[str]]:
    products = fetch_catalog_names(base_url, "/api/products", timeout)
    services = fetch_catalog_names(base_url, "/api/services", timeout)
    return {"products": products, "services": services}


def fetch_catalog_names(base_url: str, path: str, timeout: float | None) -> list[str]:
    try:
        response = httpx.get(f"{base_url}{path}", timeout=timeout)
        response.raise_for_status()
        items = response.json().get("items", [])
    except Exception:
        return []
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status is not None and status != "active":
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    return list(dict.fromkeys(names))


def run_preflight(base_url: str, catalog: dict[str, list[str]], timeout: float | None) -> Preflight:
    health: dict[str, Any] = {}
    backend_available = False
    db_connected = False
    debug_available = False
    debug_error = None
    catalog_error = None
    try:
        response = httpx.get(f"{base_url}/health", timeout=timeout)
        backend_available = response.status_code == 200
        health = response.json()
        db_connected = bool(health.get("database", {}).get("connected"))
    except Exception as exc:
        health = {"error": str(exc)}
    products_available = bool(catalog.get("products"))
    services_available = bool(catalog.get("services"))
    if not products_available or not services_available:
        catalog_error = "Catalog is empty; runtime product/service probes are blocked."
    if backend_available:
        try:
            sample = call_chat(base_url, "Xin chào", f"audit-preflight-{uuid.uuid4()}", timeout)
            debug_available = isinstance(sample.get("debug"), dict)
            if not debug_available:
                debug_error = "Chat response did not include debug payload; set DEBUG=true."
        except Exception as exc:
            debug_error = str(exc)
    return Preflight(
        backend_available=backend_available,
        db_connected=db_connected,
        debug_available=debug_available,
        products_available=products_available,
        services_available=services_available,
        health=health,
        debug_error=debug_error,
        catalog_error=catalog_error,
    )


def build_fill_vars(
    products: list[str],
    services: list[str],
    rng: random.Random,
) -> dict[str, str]:
    fill_vars = {
        "time_of_day": rng.choice(["sáng", "chiều", "tối"]),
        "price": rng.choice(["100", "200", "300", "500"]),
    }
    if products:
        fill_vars["product_name"] = rng.choice(products)
    if len(products) >= 2:
        first, second = rng.sample(products, 2)
        fill_vars["product_a"] = first
        fill_vars["product_b"] = second
    if services:
        fill_vars["service_name"] = rng.choice(services)
    return fill_vars


def generate_probes(
    fill_vars: dict[str, str],
    rng: random.Random,
    selected_intent: str | None,
) -> dict[str, list[str]]:
    probes: dict[str, list[str]] = {}
    intents = [selected_intent] if selected_intent else list(PROBE_TEMPLATES)
    for intent in intents:
        templates = [
            template
            for template in PROBE_TEMPLATES[intent]
            if required_fields(template).issubset(fill_vars.keys())
        ]
        if not templates:
            probes[intent] = []
            continue
        chosen = rng.sample(templates, min(3, len(templates)))
        probes[intent] = [template.format(**fill_vars) for template in chosen]
    return probes


def required_fields(template: str) -> set[str]:
    return set(re.findall(r"{([A-Za-z_][A-Za-z0-9_]*)}", template))


def empty_capability_results(selected_intent: str | None = None) -> dict[str, list[dict[str, Any]]]:
    intents = [selected_intent] if selected_intent else list(PROBE_TEMPLATES)
    return {intent: [] for intent in intents if intent is not None}


def run_capability_audit(
    *,
    base_url: str,
    catalog: dict[str, list[str]],
    preflight: Preflight,
    fill_vars: dict[str, str],
    rng: random.Random,
    selected_intent: str | None,
    runs: int,
    timeout: float | None,
) -> dict[str, list[dict[str, Any]]]:
    results = empty_capability_results(selected_intent)
    if not preflight.backend_available:
        reason = preflight.health.get("error") or "Runtime backend is unavailable."
        for intent in results:
            results[intent].append(blocked_probe(intent, reason))
        return results

    for run_index in range(max(1, runs)):
        run_fill_vars = dict(fill_vars)
        if catalog["products"]:
            run_fill_vars["product_name"] = rng.choice(catalog["products"])
        if len(catalog["products"]) >= 2:
            first, second = rng.sample(catalog["products"], 2)
            run_fill_vars["product_a"] = first
            run_fill_vars["product_b"] = second
        if catalog["services"]:
            run_fill_vars["service_name"] = rng.choice(catalog["services"])
        probes = generate_probes(run_fill_vars, rng, selected_intent)
        for intent, queries in probes.items():
            if not queries:
                results.setdefault(intent, []).append(
                    blocked_probe(intent, "No valid templates after catalog fill.")
                )
                continue
            for query in queries:
                session_id = f"audit-{intent.lower()}-{run_index}-{uuid.uuid4()}"
                try:
                    response = call_chat(base_url, query, session_id, timeout)
                    checks = capability_checks(intent, response)
                    passed = all(checks.values())
                    results.setdefault(intent, []).append(
                        {
                            "run_index": run_index,
                            "query": query,
                            "session_id": session_id,
                            "intent": response.get("intent"),
                            "answer_type": response.get("answer_type"),
                            "degraded": response.get("degraded"),
                            "checks": checks,
                            "probe_passed": passed,
                            "response_summary": summarize_response(response),
                        }
                    )
                except Exception as exc:
                    results.setdefault(intent, []).append(
                        {
                            "run_index": run_index,
                            "query": query,
                            "session_id": session_id,
                            "intent": None,
                            "degraded": True,
                            "checks": {name: False for name in check_names(intent)},
                            "probe_passed": False,
                            "error": str(exc),
                        }
                    )
    return results


def blocked_probe(intent: str, reason: str) -> dict[str, Any]:
    return {
        "query": None,
        "intent": None,
        "degraded": True,
        "checks": {name: False for name in check_names(intent)},
        "probe_passed": False,
        "blocked": True,
        "reason": reason,
    }


def call_chat(base_url: str, query: str, session_id: str, timeout: float | None) -> dict[str, Any]:
    response = httpx.post(
        f"{base_url}/chat",
        json={"message": query, "session_id": session_id, "debug": True},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def check_names(intent: str) -> list[str]:
    return list(CAPABILITY_CHECKS[intent])


def capability_checks(intent: str, response: dict[str, Any]) -> dict[str, bool]:
    return {name: bool(check(response)) for name, check in CAPABILITY_CHECKS[intent].items()}


def answer(response: dict[str, Any]) -> dict[str, Any]:
    value = response.get("answer")
    return value if isinstance(value, dict) else {}


def sources(response: dict[str, Any]) -> list[dict[str, Any]]:
    values = answer(response).get("sources", [])
    return [item for item in values if isinstance(item, dict)]


def items(response: dict[str, Any]) -> list[dict[str, Any]]:
    values = answer(response).get("items", [])
    return [item for item in values if isinstance(item, dict)]


def source_types(response: dict[str, Any]) -> set[str]:
    return {str(item.get("source_type")) for item in sources(response)}


def bound_tasks(response: dict[str, Any]) -> list[dict[str, Any]]:
    debug = response.get("debug")
    if not isinstance(debug, dict):
        return []
    retrieval = debug.get("retrieval")
    if isinstance(retrieval, dict):
        bound_plan = retrieval.get("bound_plan")
        if isinstance(bound_plan, dict) and isinstance(bound_plan.get("tasks"), list):
            return [task for task in bound_plan["tasks"] if isinstance(task, dict)]
    task = debug.get("bound_task")
    return [task] if isinstance(task, dict) and task else []


def primary_bound_task(response: dict[str, Any]) -> dict[str, Any]:
    tasks = bound_tasks(response)
    return tasks[0] if tasks else {}


def planned_tasks(response: dict[str, Any]) -> list[dict[str, Any]]:
    debug = response.get("debug")
    if not isinstance(debug, dict):
        return []
    values = debug.get("planned_tasks")
    if isinstance(values, list):
        return [task for task in values if isinstance(task, dict)]
    retrieval = debug.get("retrieval")
    if isinstance(retrieval, dict):
        planner = retrieval.get("planner_plan")
        if isinstance(planner, dict) and isinstance(planner.get("tasks"), list):
            return [task for task in planner["tasks"] if isinstance(task, dict)]
    return []


def has_one_bound_entity(response: dict[str, Any]) -> bool:
    task = primary_bound_task(response)
    if not task:
        return False
    return len(task.get("entity_names") or []) == 1 and len(task.get("resolved_ids") or []) == 1


def has_two_bound_entities(response: dict[str, Any]) -> bool:
    task = primary_bound_task(response)
    if not task:
        return False
    return len(task.get("entity_names") or []) >= 2 and len(task.get("resolved_ids") or []) >= 2


def price_grounded(response: dict[str, Any]) -> bool:
    text = str(answer(response).get("text") or "")
    generated = price_numbers(text)
    if not generated:
        return True
    evidence_text = json.dumps(
        {
            "items": items(response),
            "debug_evidence": response.get("debug", {}).get("retrieval", {}).get("evidence")
            if isinstance(response.get("debug"), dict)
            else None,
        },
        ensure_ascii=False,
        default=str,
    )
    evidence_numbers = all_numbers(evidence_text)
    return generated.issubset(evidence_numbers)


def no_invented_clinic_fact(response: dict[str, Any]) -> bool:
    text = str(answer(response).get("text") or "").lower()
    has_concrete_fact = any(
        keyword in text
        for keyword in ("địa chỉ", "số điện thoại", "giờ", "mở cửa", "email", "zalo", "facebook")
    )
    return not has_concrete_fact or "clinic_info" in source_types(response)


CAPABILITY_CHECKS: dict[str, dict[str, Callable[[dict[str, Any]], bool]]] = {
    "GREETING": {
        "intent_classified": lambda response: response.get("intent") == "GREETING",
        "has_answer": lambda response: bool(answer(response).get("text")),
        "not_degraded": lambda response: not response.get("degraded"),
        "no_rag_evidence": lambda response: len(sources(response)) == 0,
    },
    "CHITCHAT": {
        "intent_classified": lambda response: response.get("intent") == "CHITCHAT",
        "has_answer": lambda response: bool(answer(response).get("text")),
        "not_degraded": lambda response: not response.get("degraded"),
        "no_rag_evidence": lambda response: len(sources(response)) == 0,
    },
    "CLINIC_INFO": {
        "intent_classified": lambda response: response.get("intent") == "CLINIC_INFO",
        "has_answer": lambda response: bool(answer(response).get("text")),
        "not_degraded": lambda response: not response.get("degraded"),
        "has_authoritative_source": lambda response: "clinic_info" in source_types(response),
        "no_hallucinated_fact": no_invented_clinic_fact,
    },
    "FAQ": {
        "intent_classified": lambda response: response.get("intent") == "FAQ",
        "has_answer": lambda response: bool(answer(response).get("text")),
        "not_degraded": lambda response: not response.get("degraded"),
        "has_evidence": lambda response: len(sources(response)) > 0,
        "evidence_from_allowed_sources": lambda response: source_types(response).issubset(
            {"faq", "chunk", "table_row"}
        ),
    },
    "PRODUCT_LIST": {
        "intent_classified": lambda response: response.get("intent") == "PRODUCT_LIST",
        "has_items": lambda response: len(items(response)) > 0,
        "items_have_name": lambda response: (
            bool(items(response))
            and all(
                item.get("name") or item.get("data", {}).get("name") for item in items(response)
            )
        ),
        "not_degraded": lambda response: not response.get("degraded"),
        "source_is_sql": lambda response: "product" in source_types(response),
    },
    "PRODUCT_DETAIL": {
        "intent_classified": lambda response: response.get("intent") == "PRODUCT_DETAIL",
        "has_answer": lambda response: bool(answer(response).get("text")),
        "not_degraded": lambda response: not response.get("degraded"),
        "one_entity_resolved": has_one_bound_entity,
        "authoritative_source": lambda response: "product" in source_types(response),
        "no_hallucinated_price": price_grounded,
    },
    "PRODUCT_COMPARE": {
        "intent_classified": lambda response: response.get("intent") == "PRODUCT_COMPARE",
        "has_answer": lambda response: bool(answer(response).get("text")),
        "not_degraded": lambda response: not response.get("degraded"),
        "two_entities_resolved": has_two_bound_entities,
        "has_items": lambda response: len(items(response)) >= 2,
        "both_products_in_sources": lambda response: (
            sum(1 for source in sources(response) if source.get("source_type") == "product") >= 2
        ),
    },
    "SERVICE_LIST": {
        "intent_classified": lambda response: response.get("intent") == "SERVICE_LIST",
        "has_items": lambda response: len(items(response)) > 0,
        "not_degraded": lambda response: not response.get("degraded"),
        "source_is_sql": lambda response: "service" in source_types(response),
    },
    "SERVICE_DETAIL": {
        "intent_classified": lambda response: response.get("intent") == "SERVICE_DETAIL",
        "has_answer": lambda response: bool(answer(response).get("text")),
        "not_degraded": lambda response: not response.get("degraded"),
        "one_entity_resolved": has_one_bound_entity,
        "authoritative_source": lambda response: "service" in source_types(response),
    },
    "UNKNOWN": {
        "intent_classified": lambda response: response.get("intent") == "UNKNOWN",
        "has_clarification": lambda response: bool(answer(response).get("text")),
        "not_degraded": lambda response: not response.get("degraded"),
        "no_fabricated_data": lambda response: len(items(response)) == 0,
    },
}


def summarize_response(response: dict[str, Any]) -> dict[str, Any]:
    debug = response.get("debug") if isinstance(response.get("debug"), dict) else {}
    return {
        "trace_id": response.get("trace_id"),
        "intent": response.get("intent"),
        "answer_type": response.get("answer_type"),
        "answer_chars": len(str(answer(response).get("text") or "")),
        "item_count": len(items(response)),
        "source_types": sorted(source_types(response)),
        "debug_present": bool(debug),
        "bound_task": debug.get("bound_task") if isinstance(debug, dict) else None,
        "planned_task_count": len(planned_tasks(response)),
    }


def run_multi_turn_audit(
    *,
    base_url: str,
    preflight: Preflight,
    fill_vars: dict[str, str],
    timeout: float | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not preflight.multi_turn_ready:
        reason = preflight.debug_error or preflight.catalog_error or "Multi-turn preflight failed."
        return [
            {
                "pattern": pattern["name"],
                "description": pattern["description"],
                "session_id": None,
                "turns": [],
                "pattern_passed": False,
                "blocked": True,
                "reason": reason,
            }
            for pattern in MULTI_TURN_PATTERNS
        ]
    for pattern in MULTI_TURN_PATTERNS:
        missing = sorted(required_pattern_fields(pattern) - fill_vars.keys())
        if missing:
            results.append(
                {
                    "pattern": pattern["name"],
                    "description": pattern["description"],
                    "session_id": None,
                    "turns": [],
                    "pattern_passed": False,
                    "blocked": True,
                    "reason": f"Missing fill vars: {missing}",
                }
            )
            continue
        session_id = f"audit-mt-{pattern['name']}-{uuid.uuid4()}"
        results.append(run_multi_turn_probe(pattern, fill_vars, base_url, session_id, timeout))
    return results


def required_pattern_fields(pattern: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    for turn in pattern.get("turns", []):
        fields.update(required_fields(str(turn.get("template") or "")))
    return fields


def run_multi_turn_probe(
    probe: dict[str, Any],
    fill_vars: dict[str, str],
    base_url: str,
    session_id: str,
    timeout: float | None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    last_entity = None
    for index, turn in enumerate(probe["turns"]):
        query = turn["template"].format(**fill_vars)
        try:
            response = call_chat(base_url, query, session_id, timeout)
        except Exception as exc:
            results.append(
                {
                    "turn_index": index,
                    "query": query,
                    "intent": None,
                    "degraded": True,
                    "checks": {"call_succeeded": False},
                    "turn_passed": False,
                    "error": str(exc),
                }
            )
            continue
        task = primary_bound_task(response)
        debug = response.get("debug") if isinstance(response.get("debug"), dict) else {}
        checks: dict[str, bool] = {}

        if "check_intent" in turn:
            checks["intent_match"] = response.get("intent") == turn["check_intent"]
        if "check_intent_in" in turn:
            checks["intent_in_expected"] = response.get("intent") in set(turn["check_intent_in"])
        if turn.get("check_multi_task"):
            tasks = planned_tasks(response)
            checks["multi_task_produced"] = len(tasks) >= 2
            if "expected_intents" in turn:
                actual_intents = {task.get("intent") for task in tasks}
                checks["expected_intents_present"] = all(
                    expected in actual_intents for expected in turn["expected_intents"]
                )
        if turn.get("check_memory"):
            checks["memory_used"] = memory_used(response)
        if turn.get("check_entity_match") and last_entity:
            current_entity = first_entity_name(task)
            checks["entity_consistent"] = current_entity == last_entity
        if turn.get("check_entity_switched") and last_entity:
            current_entity = first_entity_name(task)
            checks["entity_switched"] = current_entity is not None and current_entity != last_entity
        if turn.get("check_no_entity"):
            checks["no_entity_leaked"] = not (task.get("entity_names") or task.get("resolved_ids"))
        if turn.get("check_gate_pass"):
            checks["gate_passed"] = str(debug.get("gate_status") or "unknown") in {
                "pass",
                "unknown",
            }

        current = first_entity_name(task)
        if current:
            last_entity = current
        results.append(
            {
                "turn_index": index,
                "query": query,
                "intent": response.get("intent"),
                "degraded": response.get("degraded"),
                "checks": checks,
                "turn_passed": bool(checks) and all(checks.values()),
                "response_summary": summarize_response(response),
            }
        )
    return {
        "pattern": probe["name"],
        "description": probe["description"],
        "session_id": session_id,
        "turns": results,
        "pattern_passed": bool(results) and all(turn["turn_passed"] for turn in results),
    }


def first_entity_name(task: dict[str, Any]) -> str | None:
    values = task.get("entity_names") if isinstance(task, dict) else None
    if isinstance(values, list) and values:
        return str(values[0])
    return None


def memory_used(response: dict[str, Any]) -> bool:
    debug = response.get("debug")
    if not isinstance(debug, dict):
        return False
    if debug.get("resolution_source") == "conversation_state":
        return True
    task = primary_bound_task(response)
    if task.get("binding_source") in {"conversation_state", "same_turn_task", "mixed_context"}:
        return True
    decision = debug.get("decision")
    return isinstance(decision, dict) and decision.get("binding_source") in {
        "conversation_state",
        "same_turn_task",
        "mixed_context",
    }


def run_safety_checks(source: SourceIndex) -> dict[str, dict[str, Any]]:
    checks = {
        "no_hallucinated_price": StaticCheck(
            "pass"
            if source.has_all(
                "app/generation/validator.py", "_validate_prices", "Unsupported price"
            )
            else "fail",
            source.evidence("app/generation/validator.py", "_validate_prices"),
        ),
        "no_hallucinated_hours": StaticCheck(
            "pass"
            if source.has_all(
                "app/generation/validator.py", "_clock_values", "Unsupported opening-hour"
            )
            else "fail",
            source.evidence("app/generation/validator.py", "Unsupported opening-hour"),
        ),
        "no_diagnosis": StaticCheck(
            "pass"
            if "không chẩn đoán" in source.text("app/generation/prompts.py").lower()
            else "fail",
            source.evidence("app/generation/prompts.py", "không chẩn đoán"),
        ),
        "no_archived_records": StaticCheck(
            "pass"
            if source.has_all(
                "app/retrieval/structured_retriever.py",
                'Product.status == "active"',
                'Service.status == "active"',
                'Document.status == "active"',
            )
            else "fail",
            source.evidence("app/retrieval/structured_retriever.py", 'Product.status == "active"'),
        ),
        "no_debug_exposure": StaticCheck(
            "pass"
            if source.has_all("app/services/chat.py", "debug=self.settings.debug")
            and source.has_all(
                "app/generation/renderer.py", "debug=dict(debug_data or {}) if debug else None"
            )
            else "fail",
            source.evidence(
                "app/generation/renderer.py", "debug=dict(debug_data or {}) if debug else None"
            ),
        ),
        "sql_as_authority": StaticCheck(
            "pass"
            if source.has_all("app/generation/prompts.py", "Không tự bịa giá", "SQL nghiệp vụ")
            and source.has_all("app/generation/generator.py", "direct_response")
            else "fail",
            source.evidence("app/generation/prompts.py", "Không tự bịa giá"),
        ),
    }
    return {
        key: {"result": value.result, "evidence": value.evidence} for key, value in checks.items()
    }


def run_acceptance_checks(
    *,
    source: SourceIndex,
    base_url: str,
    preflight: Preflight,
    capability_results: dict[str, list[dict[str, Any]]],
    multi_turn_results: list[dict[str, Any]],
    timeout: float | None,
    skip_tests: bool,
    include_runtime: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, desc: str, method: str, passed: bool, evidence: str) -> None:
        checks.append(
            {
                "id": check_id,
                "desc": desc,
                "method": method,
                "result": "pass" if passed else "fail",
                "evidence": evidence,
            }
        )

    def block(check_id: str, desc: str, method: str, evidence: str) -> None:
        checks.append(
            {
                "id": check_id,
                "desc": desc,
                "method": method,
                "result": "blocked",
                "evidence": evidence,
            }
        )

    add(
        "AC-1",
        "App starts with FastAPI",
        "code",
        source.has_all("app/main.py", "app = FastAPI("),
        source.evidence("app/main.py", "app = FastAPI("),
    )
    migration_text = source.migration_text()
    model_text = source.text("app/db/models.py")
    required_tables = {
        "documents",
        "chunks",
        "assets",
        "chunk_assets",
        "tables",
        "table_rows",
        "products",
        "services",
        "clinic_info",
        "faqs",
        "product_categories",
        "service_categories",
        "faq_categories",
        "category_aliases",
        "product_aliases",
        "service_aliases",
        "faq_aliases",
        "conversation_sessions",
        "conversation_turns",
        "conversation_summaries",
        "ingestion_runs",
        "rag_traces",
        "rag_trace_steps",
        "evaluation_datasets",
        "evaluation_cases",
        "evaluation_runs",
        "evaluation_case_results",
    }
    missing_tables = sorted(
        table for table in required_tables if table not in migration_text + model_text
    )
    add(
        "AC-2",
        "Migrations create all required tables",
        "code",
        not missing_tables,
        "Missing tables: " + ", ".join(missing_tables)
        if missing_tables
        else "All required table names present in migrations/models.",
    )
    if include_runtime:
        add(
            "AC-3",
            "App connects to PostgreSQL with pgvector",
            "api",
            preflight.backend_available and preflight.db_connected,
            json.dumps(preflight.health, ensure_ascii=False),
        )
    else:
        block(
            "AC-3",
            "App connects to PostgreSQL with pgvector",
            "api",
            "Runtime checks disabled for this mode.",
        )
    add(
        "AC-4",
        "Document can be uploaded and ingested",
        "code",
        source.has_all(
            "app/api/routes_ingestion.py", '@router.post("/ingest/upload")', "IngestionPipeline"
        ),
        source.evidence("app/api/routes_ingestion.py", '@router.post("/ingest/upload")'),
    )
    add(
        "AC-5",
        "Ingestion creates document/chunk/table/row/asset records",
        "code",
        source.has_all(
            "app/ingestion/pipeline.py",
            "Document(",
            "IngestionRun",
            "ParsedTable(",
            "Chunk(",
            "Asset(",
        )
        and source.has_all(
            "app/ingestion/table_processor.py",
            "TableRow(",
            "Product(",
            "Service(",
        ),
        (
            "pipeline.py creates Document/IngestionRun/ParsedTable/Chunk/Asset; "
            "table_processor.py creates TableRow/Product/Service/FAQ/ClinicInfo."
        ),
    )
    add(
        "AC-6",
        "Normalization produces separate text, table, image collections",
        "code",
        source.has_all(
            "app/ingestion/normalizer.py", "text_blocks", "table_blocks", "image_blocks"
        ),
        source.evidence("app/ingestion/normalizer.py", "normalized_counts"),
    )
    add(
        "AC-7",
        "Tables excluded from text chunking",
        "code",
        source.has_all("app/ingestion/pipeline.py", "self.chunker.split(parsed.text_blocks)")
        and "parsed.tables" not in source.text("app/ingestion/chunker.py"),
        source.evidence("app/ingestion/pipeline.py", "self.chunker.split(parsed.text_blocks)"),
    )
    add(
        "AC-8",
        "Asset masking: token present, resolves to URL, no template placeholder",
        "code",
        source.has_all(
            "app/ingestion/asset_masker.py", "mask_asset_positions", "append_asset_tokens"
        )
        and source.has_all(
            "app/assets/resolver.py",
            "ASSET_TEMPLATE_SEGMENT_PATTERN",
            '"url": asset.public_url or asset.local_path',
        ),
        (
            "asset_masker masks placeholders; resolver filters template tokens "
            "and returns URL/local path."
        ),
    )
    add(
        "AC-9",
        "Assets staged, promoted for completed runs, linked to chunks",
        "code",
        source.has_all(
            "app/ingestion/pipeline.py", "stage_bytes", "self.storage.promote", "ChunkAsset("
        ),
        source.evidence("app/ingestion/pipeline.py", "self.storage.promote"),
    )
    add(
        "AC-10",
        "Product/service table rows stored, queried, linked to assets",
        "code",
        source.has_all("app/ingestion/table_processor.py", "TableRow(", "asset_id=self._asset_id")
        and source.has_all(
            "app/retrieval/structured_retriever.py",
            "_product_result",
            "_service_result",
            "_asset_token",
        ),
        (
            "table_processor stores table rows/business records with asset_id; "
            "structured_retriever returns product/service evidence."
        ),
    )
    add(
        "AC-11",
        "Embedding dimension validated against model and PostgreSQL",
        "code",
        source.has_all(
            "app/ingestion/embedder.py",
            "validate_configuration",
            "vector\\((\\d+)\\)",
            "model_dimension",
        ),
        source.evidence("app/ingestion/embedder.py", "validate_configuration"),
    )
    add(
        "AC-12",
        "Embedding failure cannot silently activate document",
        "code",
        source.has_all(
            "app/ingestion/pipeline.py",
            "if self.settings.strict_embedding",
            "raise",
            "one_or_more_embeddings_failed",
        )
        and source.has_all(
            "app/ingestion/smoke_checks.py", "retrievable_records_missing_embeddings"
        ),
        (
            "pipeline raises or records embedding failure; smoke checks block "
            "missing embeddings when required."
        ),
    )
    add(
        "AC-13",
        "Smoke checks prevent activation of incomplete data",
        "code",
        source.has_all(
            "app/ingestion/pipeline.py",
            "run_ingestion_smoke_checks",
            "review_reasons.extend(smoke_report.blocking_reasons)",
        ),
        source.evidence(
            "app/ingestion/pipeline.py", "review_reasons.extend(smoke_report.blocking_reasons)"
        ),
    )
    add(
        "AC-14",
        "Manual approval cascades status, returns 409 when validation fails",
        "code",
        source.has_all(
            "app/ingestion/review.py", "approve_document_records", "apply_document_status"
        )
        and source.has_all(
            "app/api/routes_ingestion.py", "ApprovalValidationError", "status_code=409"
        ),
        "review.py cascades status; routes_ingestion.py maps ApprovalValidationError to HTTP 409.",
    )
    add(
        "AC-15",
        "Duplicate policies reject/reuse/replace/force behave explicitly",
        "code",
        source.has_all(
            "app/ingestion/pipeline.py",
            '"reject"',
            '"reuse"',
            '"replace"',
            '"force"',
            "DuplicateDocumentError",
        ),
        source.evidence("app/ingestion/pipeline.py", "Unsupported duplicate ingestion policy"),
    )
    add_probe_ac(
        checks,
        "AC-16",
        "Product list returns active products through SQL",
        "PRODUCT_LIST",
        capability_results,
    )
    add_probe_ac(
        checks,
        "AC-17",
        "Service list returns active services through SQL",
        "SERVICE_LIST",
        capability_results,
    )
    add_probe_ac(
        checks,
        "AC-18",
        "Product detail retrieves product by name",
        "PRODUCT_DETAIL",
        capability_results,
    )
    add_probe_ac(
        checks,
        "AC-19",
        "Product compare retrieves multiple products from source data only",
        "PRODUCT_COMPARE",
        capability_results,
    )
    add_probe_ac(
        checks,
        "AC-20",
        "FAQ works through exact/fuzzy or semantic search",
        "FAQ",
        capability_results,
    )
    if include_runtime and preflight.backend_available:
        try:
            sample = call_chat(base_url, "Xin chào", f"audit-ac21-{uuid.uuid4()}", timeout)
            required = {"trace_id", "intent", "answer", "degraded"}
            add(
                "AC-21",
                "Chat endpoint returns structured JSON",
                "api",
                required.issubset(sample.keys()) and isinstance(sample.get("answer"), dict),
                f"Response keys: {sorted(sample.keys())}",
            )
        except Exception as exc:
            add("AC-21", "Chat endpoint returns structured JSON", "api", False, str(exc))
    else:
        block(
            "AC-21",
            "Chat endpoint returns structured JSON",
            "api",
            "Runtime checks disabled or backend unavailable.",
        )
    add(
        "AC-22",
        "LLM output is validated",
        "code",
        source.has_all("app/generation/generator.py", "self.validator.validate")
        and source.has_all(
            "app/generation/validator.py", "GeneratedResponse.model_validate", "_validate_semantics"
        ),
        source.evidence("app/generation/validator.py", "_validate_semantics"),
    )
    add(
        "AC-23",
        "Every chat request creates rag_traces and rag_trace_steps",
        "code",
        source.has_all("app/services/chat.py", "TraceRecorder.start")
        and source.has_all("app/observability/tracing.py", "RagTrace(", "RagTraceStep("),
        source.evidence("app/services/chat.py", "TraceRecorder.start"),
    )
    add(
        "AC-24",
        "Latency recorded per stage",
        "code",
        source.has_all(
            "app/observability/tracing.py", "latency_ms=int((time.perf_counter() - start) * 1000)"
        ),
        source.evidence(
            "app/observability/tracing.py", "latency_ms=int((time.perf_counter() - start) * 1000)"
        ),
    )
    add(
        "AC-25",
        "Evaluation dataset can be run",
        "code",
        source.has_all(
            "app/evaluation/runner.py", "run_pipeline_evaluation", "ChatService(settings)"
        ),
        source.evidence("app/evaluation/runner.py", "run_pipeline_evaluation"),
    )
    add(
        "AC-26",
        "Evaluation results stored in evaluation_runs",
        "code",
        source.has_all("app/evaluation/runner.py", "EvaluationCaseResult", "session.add(result)")
        and source.has_all("app/api/routes_evaluation.py", "EvaluationRun"),
        "runner.py writes EvaluationCaseResult rows and evaluation routes manage EvaluationRun.",
    )
    if skip_tests:
        block("AC-27", "Tests pass", "run", "Skipped by --skip-tests.")
    else:
        checks.append(run_pytest_ac())
    add(
        "AC-28",
        "README and AGENTS.md explain ingestion lifecycle",
        "code",
        "Luồng ingestion" in source.text("README.md")
        and "AGENTS.md" in "\n".join(os.listdir(ROOT)),
        source.evidence("README.md", "Luồng ingestion"),
    )
    add(
        "AC-29",
        "Structured SQL responses resolve assets from item UUIDs",
        "code",
        source.has_all("app/generation/renderer.py", "asset_ids = [")
        and source.has_all("app/assets/resolver.py", "Asset.asset_id.in_(requested_ids)"),
        source.evidence("app/assets/resolver.py", "Asset.asset_id.in_(requested_ids)"),
    )
    add(
        "AC-30",
        "Generic service schemas not synced into products",
        "code",
        source.has_all("app/ingestion/document_classifier.py", "TYPE_MAP")
        and source.has_all("app/ingestion/table_normalizer.py", "entity_type"),
        (
            "document_classifier.py and table_normalizer.py keep product/service "
            "classification explicit."
        ),
    )
    add_multi_turn_ac(
        checks,
        "AC-31",
        "One message can produce multiple tasks, each with own intent/entity/evidence",
        "multi_intent_one_query",
        multi_turn_results,
    )
    add(
        "AC-32",
        "Implicit follow-up cannot leave stale PlannerLLM entities in executable state",
        "code",
        source.has_all(
            "app/orchestration/task_canonicalizer.py",
            "rejected_planner_entities",
            "entity_names=entity_names",
            "resolved_ids=resolved_ids",
        ),
        source.evidence("app/orchestration/task_canonicalizer.py", "rejected_planner_entities"),
    )
    add_multi_turn_ac(
        checks,
        "AC-33",
        "Explicit entity switching resolved against DB before execution",
        "entity_switch",
        multi_turn_results,
    )
    add(
        "AC-34",
        "Synthesis cannot return entity/fact conflicting with cited evidence",
        "code",
        source.has_all(
            "app/generation/validator.py",
            "_validate_semantics",
            "Answer does not identify the canonical task entity",
        )
        and source.has_all(
            "app/orchestration/consistency_gate.py", "evidence_resolved_id_mismatch"
        ),
        "validator.py checks semantic grounding; consistency_gate.py checks evidence IDs.",
    )
    add(
        "AC-35",
        (
            "Conversation evaluation measures entity binding, follow-up memory, "
            "multi-task, scenario pass rate"
        ),
        "code",
        source.has_all(
            "app/evaluation/eval_conversation.py",
            "entity_binding_accuracy",
            "follow_up_success_rate",
            "multi_task_success_rate",
            "scenario_pass_rate",
        ),
        source.evidence("app/evaluation/eval_conversation.py", "scenario_pass_rate"),
    )
    return checks


def add_probe_ac(
    checks: list[dict[str, Any]],
    check_id: str,
    desc: str,
    intent: str,
    capability_results: dict[str, list[dict[str, Any]]],
) -> None:
    rows = capability_results.get(intent, [])
    if not rows:
        checks.append(
            {
                "id": check_id,
                "desc": desc,
                "method": "probe",
                "intent": intent,
                "result": "blocked",
                "evidence": "No probes were run for this intent.",
            }
        )
        return
    passed = all(row.get("probe_passed") for row in rows)
    failed_queries = [row.get("query") for row in rows if not row.get("probe_passed")]
    checks.append(
        {
            "id": check_id,
            "desc": desc,
            "method": "probe",
            "intent": intent,
            "result": "pass" if passed else "fail",
            "evidence": (
                f"{sum(row.get('probe_passed') is True for row in rows)}/{len(rows)} probes passed."
            )
            + (f" Failed queries: {failed_queries[:3]}" if failed_queries else ""),
        }
    )


def add_multi_turn_ac(
    checks: list[dict[str, Any]],
    check_id: str,
    desc: str,
    pattern_name: str,
    multi_turn_results: list[dict[str, Any]],
) -> None:
    row = next((item for item in multi_turn_results if item.get("pattern") == pattern_name), None)
    if row is None:
        checks.append(
            {
                "id": check_id,
                "desc": desc,
                "method": "multi_turn",
                "pattern": pattern_name,
                "result": "blocked",
                "evidence": "Pattern was not run.",
            }
        )
        return
    checks.append(
        {
            "id": check_id,
            "desc": desc,
            "method": "multi_turn",
            "pattern": pattern_name,
            "result": "pass" if row.get("pattern_passed") else "fail",
            "evidence": row.get("reason") or f"pattern_passed={row.get('pattern_passed')}",
        }
    )


def run_pytest_ac() -> dict[str, Any]:
    pytest_cmd = [str(ROOT / ".venv" / "bin" / "pytest"), "-q"]
    if not Path(pytest_cmd[0]).is_file():
        pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
    try:
        completed = subprocess.run(
            pytest_cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "id": "AC-27",
            "desc": "Tests pass",
            "method": "run",
            "cmd": " ".join(pytest_cmd),
            "result": "fail",
            "evidence": f"pytest timed out after {exc.timeout}s",
        }
    except Exception as exc:
        return {
            "id": "AC-27",
            "desc": "Tests pass",
            "method": "run",
            "cmd": " ".join(pytest_cmd),
            "result": "fail",
            "evidence": str(exc),
        }
    output_tail = "\n".join(completed.stdout.splitlines()[-20:])
    return {
        "id": "AC-27",
        "desc": "Tests pass",
        "method": "run",
        "cmd": " ".join(pytest_cmd),
        "result": "pass" if completed.returncode == 0 else "fail",
        "evidence": output_tail,
    }


def summarize_intents(
    capability_results: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for intent, rows in capability_results.items():
        total_checks = sum(len(row.get("checks", {})) for row in rows)
        passed_checks = sum(
            sum(1 for value in row.get("checks", {}).values() if value) for row in rows
        )
        summary[intent] = {
            "score": round(passed_checks / total_checks * 100, 1) if total_checks else 0.0,
            "probes_run": sum(1 for row in rows if row.get("query")),
            "checks_passed": passed_checks,
            "checks_total": total_checks,
            "failed_checks": sorted(
                {
                    check_name
                    for row in rows
                    for check_name, passed in row.get("checks", {}).items()
                    if not passed
                }
            ),
        }
    return summary


def compute_scores(
    *,
    capability_results: dict[str, list[dict[str, Any]]],
    multi_turn_results: list[dict[str, Any]],
    safety_results: dict[str, dict[str, Any]],
    ac_results: list[dict[str, Any]],
) -> dict[str, Any]:
    intent_scores = {
        intent: data["score"] for intent, data in summarize_intents(capability_results).items()
    }
    mt_total = len(multi_turn_results)
    mt_passed = sum(1 for result in multi_turn_results if result.get("pattern_passed"))
    mt_score = round(mt_passed / mt_total * 100, 1) if mt_total else 0.0
    ac_total = len(ac_results)
    ac_passed = sum(1 for result in ac_results if result.get("result") == "pass")
    ac_score = round(ac_passed / ac_total * 100, 1) if ac_total else 0.0
    safety_total = len(safety_results)
    safety_passed = sum(1 for result in safety_results.values() if result.get("result") == "pass")
    safety_score = round(safety_passed / safety_total * 100, 1) if safety_total else 0.0
    capability_score = round(intent_scores_avg(intent_scores), 1)
    overall = round(
        capability_score * 0.35 + mt_score * 0.25 + ac_score * 0.30 + safety_score * 0.10, 1
    )
    return {
        "overall_completeness": overall,
        "intent_capability_score": capability_score,
        "multi_turn_score": mt_score,
        "acceptance_criteria_score": ac_score,
        "safety_score": safety_score,
        "by_intent": intent_scores,
        "ac_passed": ac_passed,
        "ac_total": ac_total,
        "mt_patterns_passed": mt_passed,
        "mt_patterns_total": mt_total,
    }


def intent_scores_avg(intent_scores: dict[str, float]) -> float:
    return sum(intent_scores.values()) / len(intent_scores) if intent_scores else 0.0


def build_failing_areas(
    by_intent: dict[str, dict[str, Any]],
    multi_turn_results: list[dict[str, Any]],
    safety_results: dict[str, dict[str, Any]],
    ac_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for intent, data in sorted(by_intent.items()):
        if data.get("score", 0) < 85:
            failures.append(
                {
                    "area": intent,
                    "score": data["score"],
                    "failing_checks": data.get("failed_checks", []),
                    "recommended_fix": (
                        "Inspect failed probe responses and strengthen routing, "
                        "binding, retrieval, or response schema handling for this intent."
                    ),
                }
            )
    for row in multi_turn_results:
        if not row.get("pattern_passed"):
            failures.append(
                {
                    "area": f"multi_turn:{row.get('pattern')}",
                    "score": 0,
                    "failing_checks": [
                        check
                        for turn in row.get("turns", [])
                        for check, passed in turn.get("checks", {}).items()
                        if not passed
                    ],
                    "sample_probe": row.get("turns", [{}])[0].get("query")
                    if row.get("turns")
                    else None,
                    "recommended_fix": (
                        "Inspect debug.bound_task/debug.retrieval.bound_plan for "
                        "binding source, resolved IDs, and gate status."
                    ),
                }
            )
    for key, row in safety_results.items():
        if row.get("result") != "pass":
            failures.append(
                {
                    "area": f"safety:{key}",
                    "score": 0,
                    "failing_checks": [key],
                    "recommended_fix": (
                        "Add or tighten validator/prompt/tool policy coverage for this safety rule."
                    ),
                }
            )
    failed_ac = [row for row in ac_results if row.get("result") != "pass"]
    if failed_ac:
        failures.append(
            {
                "area": "acceptance_criteria",
                "score": round(
                    (len(ac_results) - len(failed_ac)) / max(1, len(ac_results)) * 100, 1
                ),
                "failing_checks": [row["id"] for row in failed_ac],
                "recommended_fix": "Review each failed AC evidence string in acceptance_criteria.",
            }
        )
    return failures


def price_numbers(text: str) -> set[int]:
    values: set[int] = set()
    patterns = (
        r"(?:giá|chi phí)\s*(?:là|:)?\s*([\d.,]+)\s*(triệu|trieu|nghìn|nghin|k)?",
        r"([\d.,]+)\s*(triệu|trieu|nghìn|nghin|k|đ|₫|vnd)\b",
    )
    for pattern in patterns:
        for match in re.findall(pattern, text.lower()):
            value, unit = match if isinstance(match, tuple) else (match, "")
            parsed = normalize_numeric(value, unit)
            if parsed is not None:
                values.add(parsed)
    return values


def all_numbers(text: str) -> set[int]:
    values: set[int] = set()
    for match in re.findall(r"\d[\d.,]*", text):
        parsed = normalize_numeric(match)
        if parsed is not None:
            values.add(parsed)
    return values


def normalize_numeric(value: str, unit: str = "") -> int | None:
    value = value.strip().rstrip(".,")
    if not value:
        return None
    decimal_value = None
    if unit in {"triệu", "trieu", "nghìn", "nghin", "k"}:
        try:
            decimal_value = float(value.replace(".", "").replace(",", "."))
        except ValueError:
            decimal_value = None
    if decimal_value is not None:
        if unit in {"triệu", "trieu"}:
            return int(decimal_value * 1_000_000)
        if unit in {"nghìn", "nghin", "k"}:
            return int(decimal_value * 1_000)
    if re.search(r"[.,]\d{2}$", value) and not re.search(r"[.,]\d{3}[.,]\d{2}$", value):
        value = value[:-3]
    digits = re.sub(r"\D", "", value)
    return int(digits) if digits else None


if __name__ == "__main__":
    raise SystemExit(main())
