from __future__ import annotations

import argparse
import base64
import difflib
import json
import mimetypes
import os
import re
import socket
import hashlib
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Tuple
from urllib import error as url_error
from urllib import request as url_request
from urllib.parse import parse_qs, urlparse

from ai_capability_engine import AICapabilityEngine
from auto_integration.api import AutoIntegrationAPI
from auto_nextgen.api import AutoNextGenAPI
from auto_optimization.api import AutoOptimizationAPI
from advanced_learning_hardening import (
    active_learning_candidate_selection,
    supervised_dataset_status,
)
from enterprise_lawful_dataset_pipeline import build_default_lawful_dataset_manifest as _build_lawful_dataset_manifest
from advanced_phase2_adaptation import (
    apply_tool_policy_proposals,
    policy_drift_status,
    policy_rollback_hook,
    reward_event_from_payload,
    rl_status as advanced_rl_status,
    simulate_tool_policy_tuning,
)
from advanced_phase3_foundations import (
    apply_federated_update,
    evaluate_meta_experiment,
    federated_contract_schema,
    federated_status,
    meta_learning_status,
    meta_learning_schema,
    register_meta_experiment,
    validate_federated_update,
)
from advanced_learning_registry import query_learning_modes, summarize_learning_modes
from assistant_contract import build_envelope
from assistant_orchestrator import ExecutionContext, PlannerExecutorCriticOrchestrator
from backend_io import (
    append_jsonl as _append_jsonl,
    line_count as _line_count,
    load_json_file as _load_json_file,
    recent_jsonl as _recent_jsonl,
    save_json_file as _save_json_file,
)
from backend_preferences import (
    append_preference_audit_event as _prefs_append_preference_audit_event,
    apply_reply_style as _prefs_apply_reply_style,
    effective_user_preferences as _prefs_effective_user_preferences,
    load_preference_audit_events as _prefs_load_preference_audit_events,
    load_user_preferences as _prefs_load_user_preferences,
    save_user_preferences as _prefs_save_user_preferences,
)
from backend_response_quality import (
    fallback_reply_for_quality as _fallback_reply_for_quality,
    response_quality_score as _response_quality_score,
    response_quality_status as _response_quality_status,
)
from backend_evidence import (
    augment_response_evidence as _augment_response_evidence_impl,
    extract_response_evidence as _extract_response_evidence,
    extract_tool_grounding as _extract_tool_grounding,
    ground_reply_with_trace as _ground_reply_with_trace,
)
from backend_policy import (
    append_policy_violation_event as _append_policy_violation_event_impl,
    approval_requirement_for_route as _approval_requirement_for_route_impl,
    approval_token_is_valid as _approval_token_is_valid_impl,
    is_advanced_approval_enforced as _is_advanced_approval_enforced_impl,
    is_tool_action_allowed as _is_tool_action_allowed_impl,
    load_policy_violation_events as _load_policy_violation_events_impl,
    tool_execution_context_allowlist as _tool_execution_context_allowlist_impl,
)
from backend_voice import (
    voice_status as _voice_status_impl,
    voice_stream_analytics as _voice_stream_analytics_impl,
    voice_stream_max_concurrent as _voice_stream_max_concurrent_impl,
    voice_stream_sessions_status as _voice_stream_sessions_status_impl,
)
import backend_voice as _backend_voice_module
from backend_governance import (
    advanced_learning_expansion_status as _advanced_learning_expansion_status_impl,
    advanced_operational_status as _advanced_operational_status_impl,
    load_governance_state as _load_governance_state_impl,
    run_advanced_operational_cycle as _run_advanced_operational_cycle_impl,
    save_governance_state as _save_governance_state_impl,
)
from backend_artifacts import (
    list_artifacts as _list_artifacts_impl,
    read_artifact_payload as _read_artifact_payload_impl,
    resolve_artifact_path as _resolve_artifact_path_impl,
    save_generated_code as _save_generated_code_impl,
    save_uploaded_file as _save_uploaded_file_impl,
)
from backend_advanced_learning_routes import (
    handle_advanced_learning_post as _handle_advanced_learning_post_impl,
)
from backend_chat_routes import (
    handle_chat_post as _handle_chat_post_impl,
)
from backend_learning_routes import (
    handle_learning_post as _handle_learning_post_impl,
)
from backend_voice_memory_routes import (
    handle_voice_memory_post as _handle_voice_memory_post_impl,
)
from backend_capability_routes import (
    capability_contract_trend_summary as _capability_contract_trend_summary,
    handle_capability_post as _handle_capability_post_impl,
    record_capability_observability_event as _record_capability_observability_event,
)
from backend_profile_eval_routes import (
    handle_profile_eval_post as _handle_profile_eval_post_impl,
)
from backend_post_utils import (
    parse_json_body as _parse_json_body,
    read_raw_body as _read_raw_body,
)
from backend_route_groups import (
    is_advanced_learning_route as _is_advanced_learning_route,
    is_capability_family_route as _is_capability_family_route,
    is_chat_route as _is_chat_route,
    is_learning_route as _is_learning_route,
    is_profile_eval_route as _is_profile_eval_route,
    is_voice_memory_route as _is_voice_memory_route,
)
from evidence_contracts import (
    run_safe_logic_search as _contracts_run_safe_logic_search,
    should_force_severe_quality as _contracts_should_force_severe_quality,
    validate_response_evidence_schema as _contracts_validate_response_evidence_schema,
)
from multimodal_learning import (
    analyze_audio_duration_sec,
    analyze_image_dimensions,
    analyze_prompt_quality,
    detect_media_type,
    multimodal_retrain_gate,
    rating_from_score,
    rewrite_prompt_for_quality,
    score_multimodal_quality,
)
from diversity_labeling import (
    auto_label_diversity_dataset,
    load_label_review_queue,
    scan_animation_style_manifest,
    update_label_review,
)
from runtime_paths import get_runtime_state_root, resolve_runtime_db_path
from uulen_memory import UulenMemory


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_STATE_DIR = get_runtime_state_root(BASE_DIR)
OPT_API = AutoOptimizationAPI.from_env(db_path=str(resolve_runtime_db_path("auto_optimization.db", base_dir=BASE_DIR)))
INT_API = AutoIntegrationAPI(db_path=str(resolve_runtime_db_path("auto_integration.db", base_dir=BASE_DIR)))
NEXT_API = AutoNextGenAPI(db_path=str(resolve_runtime_db_path("auto_nextgen.db", base_dir=BASE_DIR)))
CAP_ENGINE = AICapabilityEngine()
MEMORY = UulenMemory(db_path=str(resolve_runtime_db_path("uulen_memory.db", base_dir=BASE_DIR)))
ARTIFACT_ROOT = (BASE_DIR / "artifacts" / "ai_outputs").resolve()
EVALS_DIR = (BASE_DIR / "evals").resolve()
BACKEND_BUILD_ID = "uulen-v1-2026-05-19"
CORE_API_BASE = os.getenv("CORE_API_BASE", "http://127.0.0.1:8000").rstrip("/")
DATA_COLLECTION_ROOT = (BASE_DIR.parent / "data_collection").resolve()
LEARNING_DATA_DIR = (DATA_COLLECTION_ROOT / "learning").resolve()
CHAT_MESSAGES_PATH = LEARNING_DATA_DIR / "chat_messages.jsonl"
USER_FEEDBACK_PATH = LEARNING_DATA_DIR / "user_feedback.csv"
OFFLINE_EVAL_DATASET_PATH = LEARNING_DATA_DIR / "offline_eval_dataset.jsonl"
OFFLINE_EVAL_REPORT_PATH = LEARNING_DATA_DIR / "offline_eval_report.json"
OFFLINE_EVAL_HISTORY_PATH = LEARNING_DATA_DIR / "offline_eval_history.jsonl"
USER_PREFERENCES_PATH = LEARNING_DATA_DIR / "user_preferences.json"
USER_PREFERENCES_AUDIT_PATH = LEARNING_DATA_DIR / "user_preferences_audit.jsonl"
POLICY_VIOLATIONS_REPORT_PATH = LEARNING_DATA_DIR / "policy_violations_report.jsonl"
LEARNING_GOVERNANCE_STATE_PATH = LEARNING_DATA_DIR / "learning_governance_state.json"
COMPUTER_EVENTS_PATH = LEARNING_DATA_DIR / "computer_events.jsonl"
COMPUTER_EVENT_TRAINING_PATH = LEARNING_DATA_DIR / "computer_event_training.jsonl"
MULTIMODAL_LEARNING_EVENTS_PATH = LEARNING_DATA_DIR / "multimodal_learning_events.jsonl"
IMAGE_LEARNING_PROFILE_PATH = LEARNING_DATA_DIR / "image_learning_profile.json"
DIVERSITY_DATASET_ROOT = (ARTIFACT_ROOT / "Image" / "diversity_dataset").resolve()
VOICE_STREAM_SESSIONS_LOG_PATH = LEARNING_DATA_DIR / "voice_stream_sessions.jsonl"
TRAIN_INSTRUCTIONS_PATH = LEARNING_DATA_DIR / "train_instructions.jsonl"
ARTIFACT_METADATA_LOG_PATH = LEARNING_DATA_DIR / "artifact_metadata.jsonl"
ADVANCED_RL_REWARD_LOG_PATH = LEARNING_DATA_DIR / "advanced_rl_rewards.jsonl"
ADVANCED_TOOL_POLICY_STATE_PATH = LEARNING_DATA_DIR / "advanced_tool_policy_state.json"
ADVANCED_POLICY_TUNING_REPORT_PATH = LEARNING_DATA_DIR / "advanced_policy_tuning_report.json"
ADVANCED_FEDERATED_VALIDATION_LOG_PATH = LEARNING_DATA_DIR / "advanced_federated_validation.jsonl"
ADVANCED_FEDERATED_AGGREGATE_STATE_PATH = LEARNING_DATA_DIR / "advanced_federated_aggregate_state.json"
ADVANCED_META_EXPERIMENTS_PATH = LEARNING_DATA_DIR / "advanced_meta_experiments.json"
ADVANCED_META_RESULTS_PATH = LEARNING_DATA_DIR / "advanced_meta_results.jsonl"
ADVANCED_OPERATIONAL_SNAPSHOT_PATH = LEARNING_DATA_DIR / "advanced_operational_snapshot.json"
ADVANCED_OPERATIONAL_CYCLE_LOG_PATH = LEARNING_DATA_DIR / "advanced_operational_cycle_log.jsonl"
ADVANCED_EXPANSION_ROADMAP_PATH = (BASE_DIR.parent / "roadmaps" / "advanced-general-purpose-assistant" / "ADVANCED_LEARNING_EXPANSION_ROADMAP.md").resolve()
ADVANCED_EXPANSION_WORKLOG_PATH = (BASE_DIR.parent / "roadmaps" / "advanced-general-purpose-assistant" / "ADVANCED_LEARNING_EXPANSION_WORKLOG.md").resolve()
_VOICE_STREAM_LOCK = threading.Lock()
_VOICE_STREAM_ALERT_LOCK = threading.Lock()
_RETRAIN_LOCK = threading.Lock()
_VOICE_STREAM_ACTIVE = 0
_VOICE_STREAM_LAST_ALERT_TS = 0.0
_CONTINUAL_LEARNER = None
_CONTINUAL_DEPLOYER = None
_CONTINUAL_INIT_ERROR = ""
_IMAGE_WARMUP_STARTED = False
AI_TOOLS = [
    "chat",
    "memory",
    "image.generate",
    "image.face.swap",
    "video.generate",
    "audio.music.generate",
    "code.generate",
]


def _ensure_learning_data_files() -> None:
    LEARNING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_paths = [
        CHAT_MESSAGES_PATH,
        OFFLINE_EVAL_DATASET_PATH,
        OFFLINE_EVAL_HISTORY_PATH,
        USER_PREFERENCES_AUDIT_PATH,
        POLICY_VIOLATIONS_REPORT_PATH,
        COMPUTER_EVENTS_PATH,
        COMPUTER_EVENT_TRAINING_PATH,
        MULTIMODAL_LEARNING_EVENTS_PATH,
        VOICE_STREAM_SESSIONS_LOG_PATH,
        TRAIN_INSTRUCTIONS_PATH,
        ADVANCED_RL_REWARD_LOG_PATH,
        ADVANCED_FEDERATED_VALIDATION_LOG_PATH,
        ADVANCED_META_RESULTS_PATH,
        ADVANCED_OPERATIONAL_CYCLE_LOG_PATH,
    ]
    for path in jsonl_paths:
        if not path.exists():
            path.write_text("", encoding="utf-8")

    if not USER_FEEDBACK_PATH.exists():
        USER_FEEDBACK_PATH.write_text("timestamp,rating,message,comment\n", encoding="utf-8")

    if OFFLINE_EVAL_DATASET_PATH.stat().st_size == 0:
        default_scenario = {
            "id": "default-1",
            "name": "Default offline eval scenario",
            "message": "Монгол хэл дээр AI системийн аюулгүй ажиллагааг товч тайлбарла.",
            "expected_contains": ["аюулгүй", "систем"],
        }
        OFFLINE_EVAL_DATASET_PATH.write_text(json.dumps(default_scenario, ensure_ascii=False) + "\n", encoding="utf-8")

    if not OFFLINE_EVAL_REPORT_PATH.exists():
        _save_json_file(
            OFFLINE_EVAL_REPORT_PATH,
            {
                "status": "success",
                "updated_at": None,
                "summary": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0},
                "results": [],
            },
        )

    if not USER_PREFERENCES_PATH.exists():
        _prefs_save_user_preferences(
            USER_PREFERENCES_PATH,
            {
                "default_persona": "anar_ai",
                "reply_style": "detailed",
                "tool_defaults": {},
            },
        )

    if not IMAGE_LEARNING_PROFILE_PATH.exists():
        _save_json_file(
            IMAGE_LEARNING_PROFILE_PATH,
            {
                "updated_at": None,
                "datasets": [],
            },
        )


def _detect_lang_simple(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return "unknown"
    has_cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in value)
    return "mn" if has_cyrillic else "en"


def _ai_tools_for_contracts() -> Dict[str, list[str]]:
    if isinstance(AI_TOOLS, dict):
        normalized: Dict[str, list[str]] = {}
        for category, tools in AI_TOOLS.items():
            key = str(category).strip() or "general"
            if isinstance(tools, list):
                normalized[key] = [str(name).strip() for name in tools if str(name).strip()]
            else:
                normalized[key] = []
        return normalized
    if isinstance(AI_TOOLS, list):
        return {"general": [str(name).strip() for name in AI_TOOLS if str(name).strip()]}
    return {"general": []}


def _load_user_preferences() -> Dict[str, Any]:
    return _prefs_load_user_preferences(USER_PREFERENCES_PATH, AI_PERSONAS, default_persona="anar_ai")


def _effective_user_preferences() -> Dict[str, Any]:
    return _prefs_effective_user_preferences(USER_PREFERENCES_PATH, AI_PERSONAS)


def _load_preference_audit_events(
    *,
    limit: int = 50,
    source: str | None = None,
    actor: str | None = None,
) -> List[Dict[str, Any]]:
    return _prefs_load_preference_audit_events(
        USER_PREFERENCES_AUDIT_PATH,
        limit=limit,
        source=source,
        actor=actor,
    )


def _apply_reply_style(reply: str, reply_style: str) -> str:
    return _prefs_apply_reply_style(reply, reply_style)


def _augment_response_evidence(
    evidence: Dict[str, Any] | None,
    *,
    query: str,
    persona: str,
) -> Dict[str, Any] | None:
    payload = dict(evidence) if isinstance(evidence, dict) else evidence
    if isinstance(payload, dict):
        prefs = _effective_user_preferences()
        tool_defaults = dict((prefs.get("tool_defaults") or {})) if isinstance(prefs, dict) else {}
        payload["tool_defaults"] = tool_defaults

    return _augment_response_evidence_impl(
        payload,
        query=query,
        persona=persona,
        run_safe_search=_run_safe_logic_search,
        build_id=BACKEND_BUILD_ID,
        ai_personas=AI_PERSONAS,
        ai_tools=_ai_tools_for_contracts(),
        list_artifacts=_list_artifacts,
    )


def _load_offline_eval_scenarios(*, limit: int = 20) -> List[Dict[str, Any]]:
    _ensure_learning_data_files()
    rows: List[Dict[str, Any]] = []
    try:
        with OFFLINE_EVAL_DATASET_PATH.open("r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    continue
                if isinstance(row, dict) and str(row.get("message", "")).strip():
                    rows.append(row)
    except Exception:
        rows = []

    if not rows:
        return [
            {
                "id": "default-1",
                "name": "Default offline eval scenario",
                "message": "Монгол хэл дээр AI системийн аюулгүй ажиллагааг товч тайлбарла.",
                "expected_contains": ["аюулгүй", "систем"],
            }
        ]

    safe_limit = max(1, min(int(limit), 1000))
    return rows[:safe_limit]


def _save_offline_eval_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_learning_data_files()
    updated_payload = dict(payload)
    updated_payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
    _save_json_file(OFFLINE_EVAL_REPORT_PATH, updated_payload)
    _append_jsonl(
        OFFLINE_EVAL_HISTORY_PATH,
        {
            "ts": updated_payload.get("updated_at"),
            "summary": dict(updated_payload.get("summary") or {}),
            "dataset_path": str(updated_payload.get("dataset_path") or OFFLINE_EVAL_DATASET_PATH),
            "report_path": str(updated_payload.get("report_path") or OFFLINE_EVAL_REPORT_PATH),
        },
    )
    return updated_payload


def _load_offline_eval_report() -> Dict[str, Any]:
    _ensure_learning_data_files()
    try:
        raw = OFFLINE_EVAL_REPORT_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {
        "status": "success",
        "updated_at": None,
        "summary": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0},
        "results": [],
    }


def _run_offline_eval_battery(*, limit: int = 20, persona: str | None = None) -> Dict[str, Any]:
    scenarios = _load_offline_eval_scenarios(limit=max(1, min(int(limit), 200)))
    selected_persona, _ = _resolve_preferred_persona(persona)
    results: List[Dict[str, Any]] = []

    for idx, scenario in enumerate(scenarios, start=1):
        message = str((scenario or {}).get("message", "")).strip()
        expected_contains = list((scenario or {}).get("expected_contains") or [])
        expected_tokens = [str(t).strip().lower() for t in expected_contains if str(t).strip()]

        reply = _generate_chat_reply(message, [], selected_persona)
        orchestration = ASSISTANT_ORCHESTRATOR.run(
            message,
            context=ExecutionContext(session_id="offline_eval", persona=selected_persona, execution_context="eval_offline"),
        )
        response_grounding = _extract_tool_grounding(orchestration)
        response_evidence = _extract_response_evidence(orchestration)
        response_evidence = _augment_response_evidence(
            response_evidence,
            query=message,
            persona=selected_persona,
        )
        reply = _ground_reply_with_trace(reply, response_grounding)
        findings = [
            finding.to_dict()
            for finding in ASSISTANT_ORCHESTRATOR.assess_response_quality(reply, response_grounding, response_evidence)
        ]
        findings.extend(_validate_response_evidence_schema(response_evidence))
        quality = _response_quality_status(findings)
        if _should_force_severe_quality(findings):
            quality = "severe"

        reply_lower = str(reply or "").lower()
        missing_tokens = [token for token in expected_tokens if token not in reply_lower]
        passed = bool(reply.strip()) and quality != "severe" and not missing_tokens

        results.append(
            {
                "index": idx,
                "id": str((scenario or {}).get("id", f"scenario-{idx}")),
                "name": str((scenario or {}).get("name", f"scenario-{idx}")),
                "message": message,
                "persona": selected_persona,
                "quality_status": quality,
                "passed": passed,
                "missing_expected_tokens": missing_tokens,
                "response_quality_score": _response_quality_score(findings),
                "response_quality_findings": len(findings),
            }
        )

    total = len(results)
    passed_count = sum(1 for item in results if bool(item.get("passed", False)))
    failed_count = max(0, total - passed_count)
    pass_rate = round((passed_count / total) * 100.0, 2) if total else 0.0

    report = {
        "status": "success",
        "dataset_path": str(OFFLINE_EVAL_DATASET_PATH),
        "report_path": str(OFFLINE_EVAL_REPORT_PATH),
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": pass_rate,
        },
        "results": results,
    }
    return _save_offline_eval_report(report)


def _offline_eval_status() -> Dict[str, Any]:
    scenarios = _load_offline_eval_scenarios(limit=1000)
    report = _load_offline_eval_report()
    history = _recent_jsonl(OFFLINE_EVAL_HISTORY_PATH, 1000)
    return {
        "status": "success",
        "dataset_path": str(OFFLINE_EVAL_DATASET_PATH),
        "report_path": str(OFFLINE_EVAL_REPORT_PATH),
        "history_path": str(OFFLINE_EVAL_HISTORY_PATH),
        "dataset_count": len(scenarios),
        "history_count": len(history),
        "last_report": {
            "updated_at": report.get("updated_at"),
            "summary": dict(report.get("summary") or {}),
        },
    }


def _offline_eval_trends(*, days: int = 30, weeks: int = 12) -> Dict[str, Any]:
    _ensure_learning_data_files()
    history = _recent_jsonl(OFFLINE_EVAL_HISTORY_PATH, 5000)
    now = datetime.utcnow()
    day_window = max(1, min(int(days), 365))
    week_window = max(1, min(int(weeks), 104))
    day_cutoff = now.timestamp() - float(day_window * 86400)
    week_cutoff = now.timestamp() - float(week_window * 7 * 86400)

    day_buckets: Dict[str, List[float]] = {}
    week_buckets: Dict[str, List[float]] = {}

    for row in history:
        ts = _parse_utc_ts(row.get("ts"))
        if ts is None:
            continue
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
        try:
            pass_rate = float(summary.get("pass_rate", 0.0))
        except Exception:
            pass_rate = 0.0
        stamp = ts.timestamp()
        if stamp >= day_cutoff:
            day_key = ts.strftime("%Y-%m-%d")
            day_buckets.setdefault(day_key, []).append(pass_rate)
        if stamp >= week_cutoff:
            iso = ts.isocalendar()
            week_key = f"{iso.year}-W{iso.week:02d}"
            week_buckets.setdefault(week_key, []).append(pass_rate)

    daily = [
        {
            "date": key,
            "runs": len(values),
            "avg_pass_rate": round(sum(values) / len(values), 2) if values else 0.0,
        }
        for key, values in sorted(day_buckets.items())
    ]
    weekly = [
        {
            "week": key,
            "runs": len(values),
            "avg_pass_rate": round(sum(values) / len(values), 2) if values else 0.0,
        }
        for key, values in sorted(week_buckets.items())
    ]

    latest = daily[-1]["avg_pass_rate"] if daily else 0.0
    baseline = daily[0]["avg_pass_rate"] if daily else 0.0
    trend_delta = round(latest - baseline, 2) if daily else 0.0

    return {
        "status": "success",
        "history_path": str(OFFLINE_EVAL_HISTORY_PATH),
        "daily": daily,
        "weekly": weekly,
        "trend_summary": {
            "daily_points": len(daily),
            "weekly_points": len(weekly),
            "latest_pass_rate": latest,
            "baseline_pass_rate": baseline,
            "pass_rate_delta": trend_delta,
        },
    }


def _offline_eval_deploy_gate(*, deploy_on_pass: bool) -> Dict[str, Any]:
    enabled = str(os.getenv("ASSISTANT_EVAL_DEPLOY_GATE_ENABLED", "1") or "1").strip() != "0"
    min_pass_rate = float(os.getenv("ASSISTANT_EVAL_DEPLOY_MIN_PASS_RATE", "80") or "80")
    max_failed = int(os.getenv("ASSISTANT_EVAL_DEPLOY_MAX_FAILED", "0") or "0")
    report = _load_offline_eval_report()
    summary = dict(report.get("summary") or {})
    pass_rate = float(summary.get("pass_rate", 0.0) or 0.0)
    failed = int(summary.get("failed", 0) or 0)
    has_report = str(report.get("updated_at", "") or "").strip() != ""

    if not deploy_on_pass:
        return {
            "enabled": enabled,
            "deploy_on_pass": False,
            "allowed": True,
            "reason": "deploy_on_pass_disabled",
            "min_pass_rate": min_pass_rate,
            "max_failed": max_failed,
            "pass_rate": pass_rate,
            "failed": failed,
            "has_report": has_report,
        }
    if not enabled:
        return {
            "enabled": False,
            "deploy_on_pass": True,
            "allowed": True,
            "reason": "eval_deploy_gate_disabled",
            "min_pass_rate": min_pass_rate,
            "max_failed": max_failed,
            "pass_rate": pass_rate,
            "failed": failed,
            "has_report": has_report,
        }
    if not has_report:
        return {
            "enabled": True,
            "deploy_on_pass": True,
            "allowed": False,
            "reason": "missing_eval_report",
            "min_pass_rate": min_pass_rate,
            "max_failed": max_failed,
            "pass_rate": pass_rate,
            "failed": failed,
            "has_report": has_report,
        }

    allowed = pass_rate >= min_pass_rate and failed <= max_failed
    reason = "ok" if allowed else "eval_threshold_not_met"
    return {
        "enabled": True,
        "deploy_on_pass": True,
        "allowed": allowed,
        "reason": reason,
        "min_pass_rate": min_pass_rate,
        "max_failed": max_failed,
        "pass_rate": pass_rate,
        "failed": failed,
        "has_report": has_report,
    }


def _append_chat_message_record(
    *,
    role: str,
    text: str,
    session_id: str = "default",
    source: str = "ui_chat",
    persona: str | None = None,
) -> None:
    content = (text or "").strip()
    if not content:
        return
    _ensure_learning_data_files()
    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "lang": _detect_lang_simple(content),
        "role": role,
        "text": content,
        "source": source,
    }
    if persona:
        record["persona"] = persona
    _append_jsonl(CHAT_MESSAGES_PATH, record)


def _append_feedback_record(
    *,
    session_id: str,
    lang: str,
    message_id: str,
    rating_1_to_5: int,
    comment: str,
) -> None:
    _ensure_learning_data_files()
    safe_comment = (comment or "").replace("\n", " ").replace("\r", " ").strip()
    line = (
        f"{datetime.utcnow().isoformat()}Z,"
        f"{session_id},"
        f"{lang},"
        f"{message_id},"
        f"{rating_1_to_5},"
        f"{safe_comment}\n"
    )
    with USER_FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def _append_train_instruction(
    *,
    user_text: str,
    assistant_text: str,
    source: str,
) -> None:
    prompt = (user_text or "").strip()
    answer = (assistant_text or "").strip()
    if not prompt or not answer:
        return
    _ensure_learning_data_files()
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    record = {
        "id": f"train-{stamp}",
        "lang": _detect_lang_simple(prompt),
        "instruction": prompt,
        "response": answer,
        "source": source,
    }
    _append_jsonl(TRAIN_INSTRUCTIONS_PATH, record)


def _learning_status() -> Dict[str, Any]:
    _ensure_learning_data_files()
    return {
        "status": "success",
        "paths": {
            "chat_messages": str(CHAT_MESSAGES_PATH),
            "user_feedback": str(USER_FEEDBACK_PATH),
            "computer_events": str(COMPUTER_EVENTS_PATH),
            "train_instructions": str(TRAIN_INSTRUCTIONS_PATH),
            "computer_event_training": str(COMPUTER_EVENT_TRAINING_PATH),
            "multimodal_learning_events": str(MULTIMODAL_LEARNING_EVENTS_PATH),
            "image_learning_profile": str(IMAGE_LEARNING_PROFILE_PATH),
        },
        "counts": {
            "chat_messages": _line_count(CHAT_MESSAGES_PATH),
            "user_feedback": max(0, _line_count(USER_FEEDBACK_PATH) - 1),
            "computer_events": _line_count(COMPUTER_EVENTS_PATH),
            "train_instructions": _line_count(TRAIN_INSTRUCTIONS_PATH),
            "computer_event_training": _line_count(COMPUTER_EVENT_TRAINING_PATH),
            "multimodal_learning_events": _line_count(MULTIMODAL_LEARNING_EVENTS_PATH),
            "image_learning_profile_exists": 1 if IMAGE_LEARNING_PROFILE_PATH.exists() else 0,
        },
    }


def _ingest_multimodal_learning_signal(
    *,
    artifact_path: str,
    action: str,
    prompt: str,
    quality_mode: str,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> Dict[str, Any]:
    _ensure_learning_data_files()
    resolved = _resolve_artifact_path(artifact_path)
    if resolved is None:
        raise ValueError("artifact not found or outside artifact root")

    stat = resolved.stat()
    mime = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
    extension = resolved.suffix.lower()
    media_type = detect_media_type(mime, extension)

    width = 0
    height = 0
    if media_type == "image":
        dims = analyze_image_dimensions(resolved)
        width = int(dims.get("width", 0) or 0)
        height = int(dims.get("height", 0) or 0)

    duration_sec = None
    if media_type in {"audio", "video"}:
        duration_sec = analyze_audio_duration_sec(resolved)

    prompt_analysis = analyze_prompt_quality(prompt, media_type)

    score, positives, improvements = score_multimodal_quality(
        media_type=media_type,
        size=int(stat.st_size),
        width=width,
        height=height,
        duration_sec=duration_sec,
        quality_mode=quality_mode,
    )
    score = max(0.0, min(score + (float(prompt_analysis.get("overall_score", 0.0)) - 0.5) * 0.2, 1.0))
    rating = rating_from_score(score)
    rewrite = rewrite_prompt_for_quality(prompt, media_type, quality_mode, prompt_analysis, improvements)

    digest = hashlib.sha256()
    with resolved.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()

    event_id = datetime.utcnow().strftime("mm-%Y%m%d%H%M%S%f")
    score_pct = int(round(score * 100.0))
    comment = (
        f"auto-multimodal-ingest media={media_type} score={score_pct} rating={rating} "
        f"size={int(stat.st_size)} quality_mode={quality_mode}"
    )

    event = {
        "id": event_id,
        "ts": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "action": str(action or "").strip(),
        "media_type": media_type,
        "artifact": {
            "file_path": str(resolved),
            "file_name": resolved.name,
            "mime_type": mime,
            "extension": extension,
            "size": int(stat.st_size),
            "sha256": sha256,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
        },
        "quality": {
            "score": round(score, 4),
            "score_pct": score_pct,
            "suggested_rating_1_to_5": rating,
            "positives": positives,
            "improvements": improvements,
            "quality_mode": quality_mode,
        },
        "prompt_analysis": prompt_analysis,
        "prompt_rewrite": rewrite,
        "prompt": str(prompt or "").strip(),
    }
    _append_jsonl(MULTIMODAL_LEARNING_EVENTS_PATH, event)

    message_id = f"mmfb-{event_id}"
    _append_feedback_record(
        session_id=session_id,
        lang=_detect_lang_simple(user_text or prompt),
        message_id=message_id,
        rating_1_to_5=rating,
        comment=comment,
    )

    if not user_text.strip():
        user_text = f"[{media_type}] prompt={prompt} quality_mode={quality_mode}"
    if not assistant_text.strip():
        assistant_text = (
            f"Generated {media_type} artifact: {resolved.name}. "
            f"Auto-analysis score={score_pct}%. "
            f"Improve next output: {'; '.join(improvements[:3]) if improvements else 'keep this style and iterate details'}. "
            f"Suggested rewrite: {rewrite.get('rewritten_prompt', '')}"
        )
    _append_train_instruction(
        user_text=user_text,
        assistant_text=assistant_text,
        source="multimodal_auto_ingest",
    )

    retrain_gate = multimodal_retrain_gate(_recent_jsonl(MULTIMODAL_LEARNING_EVENTS_PATH, 50))
    retrain_result: Dict[str, Any] | None = None
    if bool(retrain_gate.get("enabled")) and bool(retrain_gate.get("should_trigger")):
        retrain_result = _run_retrain_with_governance(
            force=False,
            dry_run=False,
            min_samples=max(25, int(os.getenv("ASSISTANT_MULTIMODAL_RETRAIN_MIN_SAMPLES", "100"))),
            eval_min_accuracy=float(os.getenv("ASSISTANT_MULTIMODAL_RETRAIN_MIN_ACCURACY", "0.6")),
            deploy_on_pass=bool(os.getenv("ASSISTANT_MULTIMODAL_RETRAIN_DEPLOY_ON_PASS", "1").strip() not in {"0", "false", "False", "off"}),
            retrain_timeout_sec=int(os.getenv("ASSISTANT_MULTIMODAL_RETRAIN_TIMEOUT_SEC", "1800")),
            retrain_max_retries=int(os.getenv("ASSISTANT_MULTIMODAL_RETRAIN_MAX_RETRIES", "2")),
        )

    return {
        "event_id": event_id,
        "message_id": message_id,
        "media_type": media_type,
        "suggested_rating_1_to_5": rating,
        "score": round(score, 4),
        "score_pct": score_pct,
        "positives": positives,
        "improvements": improvements,
        "prompt_analysis": prompt_analysis,
        "prompt_rewrite": rewrite,
        "retrain_gate": retrain_gate,
        "retrain_result": retrain_result,
        "artifact": event["artifact"],
    }


def _default_image_learning_dirs() -> List[Path]:
    return [
        (ARTIFACT_ROOT / "Image").resolve(),
        (BASE_DIR / "models" / "hf_cache").resolve(),
    ]


def _nearest_supported_image_size(value: float) -> int:
    supported = [512, 640, 768, 896, 1024, 1152, 1280, 1536]
    return min(supported, key=lambda candidate: abs(candidate - value))


def _compute_image_learning_defaults(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "image_default_quality_mode": "high",
            "image_default_style_preset": "photoreal_hq",
            "image_default_model_tier": "auto",
            "image_default_width": 1024,
            "image_default_height": 1024,
            "image_default_num_inference_steps": 28,
            "image_default_guidance_scale": 7.5,
            "image_default_upscale_factor": 1,
        }

    widths = [int(item.get("width", 0) or 0) for item in samples if int(item.get("width", 0) or 0) > 0]
    heights = [int(item.get("height", 0) or 0) for item in samples if int(item.get("height", 0) or 0) > 0]
    scores = [float(item.get("score", 0.0) or 0.0) for item in samples]

    avg_width = sum(widths) / len(widths) if widths else 1024.0
    avg_height = sum(heights) / len(heights) if heights else 1024.0
    avg_score = sum(scores) / len(scores) if scores else 0.7

    avg_ratio = avg_width / max(avg_height, 1.0)
    if avg_ratio <= 0.85:
        style_preset = "portrait"
    elif avg_ratio >= 1.5:
        style_preset = "cinematic"
    else:
        style_preset = "photoreal_hq"

    width_default = _nearest_supported_image_size(avg_width)
    height_default = _nearest_supported_image_size(avg_height)

    if avg_score >= 0.8:
        model_tier = "hq"
        steps = 30
        guidance = 7.0
    elif avg_score >= 0.65:
        model_tier = "auto"
        steps = 26
        guidance = 7.5
    else:
        model_tier = "hq"
        steps = 34
        guidance = 8.0

    avg_area = avg_width * avg_height
    upscale = 2 if avg_area < (768 * 768) else 1

    return {
        "image_default_quality_mode": "high",
        "image_default_style_preset": style_preset,
        "image_default_model_tier": model_tier,
        "image_default_width": int(width_default),
        "image_default_height": int(height_default),
        "image_default_num_inference_steps": int(steps),
        "image_default_guidance_scale": float(round(guidance, 2)),
        "image_default_upscale_factor": int(upscale),
    }


def _image_learning_profile() -> Dict[str, Any]:
    prefs = _effective_user_preferences()
    tool_defaults = dict((prefs.get("tool_defaults") or {})) if isinstance(prefs, dict) else {}
    profile = _load_json_file(IMAGE_LEARNING_PROFILE_PATH, {})
    if not isinstance(profile, dict):
        profile = {}
    return {
        "status": "success",
        "profile_path": str(IMAGE_LEARNING_PROFILE_PATH),
        "profile": profile,
        "tool_defaults": {
            "image_default_quality_mode": tool_defaults.get("image_default_quality_mode", "high"),
            "image_default_style_preset": tool_defaults.get("image_default_style_preset", "photoreal_hq"),
            "image_default_model_tier": tool_defaults.get("image_default_model_tier", "auto"),
            "image_default_width": int(tool_defaults.get("image_default_width", 1024) or 1024),
            "image_default_height": int(tool_defaults.get("image_default_height", 1024) or 1024),
            "image_default_num_inference_steps": int(tool_defaults.get("image_default_num_inference_steps", 28) or 28),
            "image_default_guidance_scale": float(tool_defaults.get("image_default_guidance_scale", 7.5) or 7.5),
            "image_default_upscale_factor": int(tool_defaults.get("image_default_upscale_factor", 1) or 1),
        },
    }


def _auto_label_diversity(*, limit: int) -> Dict[str, Any]:
    return auto_label_diversity_dataset(DIVERSITY_DATASET_ROOT, limit=limit)


def _load_label_review_queue(*, status: str, limit: int) -> Dict[str, Any]:
    return load_label_review_queue(DIVERSITY_DATASET_ROOT, status=status, limit=limit)


def _update_label_review(*, image_path: str, review_status: str, review_note: str) -> Dict[str, Any]:
    return update_label_review(
        DIVERSITY_DATASET_ROOT,
        image_path=image_path,
        review_status=review_status,
        review_note=review_note,
    )


def _scan_animation_style_manifest() -> Dict[str, Any]:
    return scan_animation_style_manifest(DIVERSITY_DATASET_ROOT)


def _analyze_images_and_learn(
    *,
    directories: List[str] | None,
    limit: int,
    quality_mode: str,
    session_id: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    _ensure_learning_data_files()
    valid_quality_mode = str(quality_mode or "high").strip().lower() or "high"
    if valid_quality_mode not in {"high", "fast"}:
        raise ValueError("quality_mode must be one of: high, fast")

    requested_dirs: List[Path] = []
    if directories:
        for raw in directories:
            text = str(raw or "").strip()
            if not text:
                continue
            requested_dirs.append(Path(text).expanduser().resolve())
    else:
        requested_dirs = _default_image_learning_dirs()

    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    scanned = 0
    learned = 0
    skipped = 0
    errors: List[Dict[str, str]] = []
    sample_quality: List[Dict[str, Any]] = []
    learned_paths: List[str] = []
    now_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ingest_dir = ARTIFACT_ROOT / "Image" / "learn_ingest"
    ingest_dir.mkdir(parents=True, exist_ok=True)

    candidate_files: List[Path] = []
    for root in requested_dirs:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in image_suffixes:
                continue
            candidate_files.append(path.resolve())
            if len(candidate_files) >= limit:
                break
        if len(candidate_files) >= limit:
            break

    for index, image_path in enumerate(candidate_files, start=1):
        scanned += 1
        try:
            target_path = image_path
            if not str(image_path).startswith(str(ARTIFACT_ROOT)):
                target_name = f"learn_{now_tag}_{index:05d}{image_path.suffix.lower()}"
                target_path = ingest_dir / target_name
                if not dry_run:
                    shutil.copy2(image_path, target_path)

            dims = analyze_image_dimensions(target_path)
            width = int(dims.get("width", 0) or 0)
            height = int(dims.get("height", 0) or 0)
            size = int(target_path.stat().st_size if target_path.exists() else image_path.stat().st_size)
            score, positives, improvements = score_multimodal_quality(
                media_type="image",
                size=size,
                width=width,
                height=height,
                duration_sec=None,
                quality_mode=valid_quality_mode,
            )
            sample_quality.append(
                {
                    "source_path": str(image_path),
                    "artifact_path": str(target_path),
                    "width": width,
                    "height": height,
                    "score": float(score),
                    "positives": list(positives or []),
                    "improvements": list(improvements or []),
                }
            )

            if not dry_run:
                ingest = _ingest_multimodal_learning_signal(
                    artifact_path=str(target_path),
                    action="image.generate",
                    prompt=f"Learned from existing image: {image_path.stem}",
                    quality_mode=valid_quality_mode,
                    session_id=session_id,
                    user_text=f"Analyze and learn image sample: {image_path.name}",
                    assistant_text=f"Image sample ingested for learning: {target_path.name}",
                )
                learned += 1
                learned_paths.append(str(ingest.get("artifact", {}).get("file_path", target_path)))
        except Exception as exc:
            skipped += 1
            errors.append({"path": str(image_path), "error": str(exc)})

    defaults = _compute_image_learning_defaults(sample_quality)
    profile_payload = {
        "status": "success",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "quality_mode": valid_quality_mode,
        "scanned": scanned,
        "learned": learned,
        "skipped": skipped,
        "directories": [str(item) for item in requested_dirs],
        "recommended_tool_defaults": defaults,
        "score_summary": {
            "avg_score": round(sum(float(item.get("score", 0.0)) for item in sample_quality) / max(len(sample_quality), 1), 4),
            "sample_count": len(sample_quality),
        },
    }

    if not dry_run:
        _save_json_file(IMAGE_LEARNING_PROFILE_PATH, profile_payload)
        merged_tool_defaults = dict((_effective_user_preferences().get("tool_defaults") or {}))
        merged_tool_defaults.update(defaults)
        _update_user_preferences(
            {"tool_defaults": merged_tool_defaults},
            source="api.learning.images.analyze_and_learn",
            actor="system",
        )

    return {
        **profile_payload,
        "profile_path": str(IMAGE_LEARNING_PROFILE_PATH),
        "dry_run": bool(dry_run),
        "learned_paths": learned_paths[:50],
        "errors": errors[:20],
    }


def _with_assistant_contract(
    payload: Dict[str, Any],
    *,
    message: str,
    audit_id: str,
    route: str,
    contract_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    base = dict(payload or {})
    status = str(base.get("status", "success"))
    data = {"route": route}
    if isinstance(contract_data, dict):
        data.update(contract_data)
    envelope = build_envelope(
        status=status,
        message=message,
        audit_id=audit_id,
        data=data,
    )
    base["assistant_contract"] = envelope
    return base


def _run_orchestrator_tool_action(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    active_persona, _ = _resolve_preferred_persona(None)
    is_allowed, context_name, allowed_actions = _is_tool_action_allowed(action, payload)
    if not is_allowed:
        _append_policy_violation_event(
            violation_code="policy.tool_context_not_allowed",
            severity="high",
            execution_context=context_name,
            route="/api/chat",
            details={"action": action, "allowed_actions": allowed_actions},
        )
        return {
            "status": "skipped",
            "summary": f"policy blocked action={action} context={context_name}",
            "policy_violation": "tool_context_not_allowed",
            "execution_context": context_name,
            "allowed_actions": allowed_actions,
        }
    if action == "read.version":
        return {
            "status": "success",
            "summary": f"build={BACKEND_BUILD_ID}",
            "build_id": BACKEND_BUILD_ID,
        }
    if action == "list.personas":
        return {
            "status": "success",
            "summary": f"personas={len(AI_PERSONAS)} current={active_persona}",
            "personas": list(AI_PERSONAS.keys()),
            "current": active_persona,
        }
    if action == "list.tools":
        return {
            "status": "success",
            "summary": f"tools={len(AI_TOOLS)}",
            "tools": AI_TOOLS,
        }
    if action == "list.artifacts":
        artifacts = _list_artifacts(limit=25)
        return {
            "status": "success",
            "summary": f"artifacts={len(artifacts)}",
            "artifacts": artifacts,
        }
    if action == "search.logic":
        query = str((payload or {}).get("tool_query", "")).strip()
        requested = str((payload or {}).get("persona", "")).strip()
        persona, _ = _resolve_preferred_persona(requested if requested else None)
        return _run_safe_logic_search(query, persona=persona)
    return {
        "status": "skipped",
        "summary": f"unknown action={action}",
    }


def _run_safe_logic_search(query: str, persona: str = "anar_ai") -> Dict[str, Any]:
    result = _contracts_run_safe_logic_search(
        query,
        persona=persona,
        build_id=BACKEND_BUILD_ID,
        ai_personas=AI_PERSONAS,
        ai_tools=_ai_tools_for_contracts(),
        list_artifacts=_list_artifacts,
    )
    tool_defaults = dict((_effective_user_preferences().get("tool_defaults") or {}))
    preferred_top_k = max(1, min(int(tool_defaults.get("search_top_k", 5)), 20))
    hits = list(result.get("hits") or [])
    result["hits"] = hits[:preferred_top_k]
    return result


def _should_force_severe_quality(findings: list[dict[str, Any]]) -> bool:
    return _contracts_should_force_severe_quality(findings)


def _validate_response_evidence_schema(evidence: Dict[str, Any] | None) -> list[dict[str, Any]]:
    return _contracts_validate_response_evidence_schema(evidence)


def _resolve_preferred_persona(requested_persona: str | None) -> Tuple[str, str]:
    requested = str(requested_persona or "").strip()
    if requested:
        if requested in AI_PERSONAS:
            return requested, "request"
        return "", "invalid_request"

    prefs = _load_user_preferences()
    preferred = str(prefs.get("default_persona", "")).strip()
    if preferred in AI_PERSONAS:
        return preferred, "preference"
    if str(CURRENT_PERSONA).strip() in AI_PERSONAS:
        return str(CURRENT_PERSONA).strip(), "runtime"
    return "anar_ai", "fallback"


def _set_current_persona(persona_id: str) -> str:
    global CURRENT_PERSONA
    CURRENT_PERSONA = str(persona_id).strip()
    return CURRENT_PERSONA


def _update_user_preferences(
    updates: Dict[str, Any] | None,
    *,
    source: str = "api.preferences.update",
    actor: str = "user",
) -> Dict[str, Any]:
    _ensure_learning_data_files()
    current = _load_user_preferences()
    merged = dict(current)
    if isinstance(updates, dict):
        for key, value in updates.items():
            merged[key] = value

    saved = _prefs_save_user_preferences(USER_PREFERENCES_PATH, merged)
    _prefs_append_preference_audit_event(
        USER_PREFERENCES_AUDIT_PATH,
        source=source,
        actor=actor,
        changes=updates if isinstance(updates, dict) else {},
        before=current,
        after=saved,
    )
    return saved


def _approval_requirement_for_route(route: str) -> Dict[str, Any]:
    return _approval_requirement_for_route_impl(route)


def _tool_execution_context_allowlist() -> Dict[str, set[str]]:
    return _tool_execution_context_allowlist_impl(policy_state_path=ADVANCED_TOOL_POLICY_STATE_PATH)


def _is_advanced_approval_enforced() -> bool:
    return _is_advanced_approval_enforced_impl()


def _approval_token_is_valid(payload: Dict[str, Any] | None) -> bool:
    return _approval_token_is_valid_impl(payload=payload, enforced=_is_advanced_approval_enforced())


def _is_tool_action_allowed(action: str, payload: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    context_name = str((payload or {}).get("execution_context", "chat_runtime") or "chat_runtime").strip().lower()
    allowlist = _tool_execution_context_allowlist()
    allowed = allowlist.get(context_name)
    if allowed is None:
        allowed = allowlist.get("default", set())
    if not isinstance(allowed, set):
        allowed = set()
    if "*" in allowed or action in allowed:
        return True, context_name, sorted(list(allowed))
    return False, context_name, sorted(list(allowed))


def _append_policy_violation_event(
    *,
    violation_code: str,
    severity: str,
    execution_context: str,
    route: str,
    details: Dict[str, Any],
) -> None:
    _ensure_learning_data_files()
    _append_policy_violation_event_impl(
        POLICY_VIOLATIONS_REPORT_PATH,
        violation_code=violation_code,
        severity=severity,
        execution_context=execution_context,
        route=route,
        details=details,
    )


def _load_policy_violation_events(
    *,
    limit: int = 100,
    violation_code: str | None = None,
    severity: str | None = None,
    execution_context: str | None = None,
) -> List[Dict[str, Any]]:
    _ensure_learning_data_files()
    return _load_policy_violation_events_impl(
        POLICY_VIOLATIONS_REPORT_PATH,
        limit=limit,
        violation_code=violation_code,
        severity=severity,
        execution_context=execution_context,
    )


ASSISTANT_ORCHESTRATOR = PlannerExecutorCriticOrchestrator(
    max_steps=int(os.getenv("ASSISTANT_ORCHESTRATOR_MAX_STEPS", "6")),
    failure_budget=int(os.getenv("ASSISTANT_ORCHESTRATOR_FAILURE_BUDGET", "2")),
    max_retries_per_step=int(os.getenv("ASSISTANT_ORCHESTRATOR_MAX_RETRIES_PER_STEP", "1")),
    tool_executor=_run_orchestrator_tool_action,
)


def _load_governance_state() -> Dict[str, Any]:
    return _load_governance_state_impl(LEARNING_GOVERNANCE_STATE_PATH)


def _save_governance_state(state: Dict[str, Any]) -> None:
    _save_governance_state_impl(LEARNING_GOVERNANCE_STATE_PATH, state)


def _get_continual_components() -> Tuple[Any, Any, str]:
    global _CONTINUAL_LEARNER
    global _CONTINUAL_DEPLOYER
    global _CONTINUAL_INIT_ERROR

    if _CONTINUAL_LEARNER is not None and _CONTINUAL_DEPLOYER is not None:
        return _CONTINUAL_LEARNER, _CONTINUAL_DEPLOYER, ""
    if _CONTINUAL_INIT_ERROR:
        return None, None, _CONTINUAL_INIT_ERROR

    try:
        root_dir = str(BASE_DIR.parent)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from continual_learning import ContinualLearner, AutoDeploySystem

        _CONTINUAL_LEARNER = ContinualLearner()
        _CONTINUAL_DEPLOYER = AutoDeploySystem(_CONTINUAL_LEARNER)
        return _CONTINUAL_LEARNER, _CONTINUAL_DEPLOYER, ""
    except Exception as exc:
        _CONTINUAL_INIT_ERROR = f"continual learner init failed: {exc}"
        return None, None, _CONTINUAL_INIT_ERROR


def _voice_stream_max_concurrent() -> int:
    return _voice_stream_max_concurrent_impl()


def _voice_stream_active_count() -> int:
    with _VOICE_STREAM_LOCK:
        return int(max(0, _VOICE_STREAM_ACTIVE))


def _voice_stream_sessions_status(
    *,
    limit: int = 50,
    status_filter: str | None = None,
    session_id_filter: str | None = None,
    window_minutes: int | None = None,
) -> Dict[str, Any]:
    _ensure_learning_data_files()
    return _voice_stream_sessions_status_impl(
        log_path=VOICE_STREAM_SESSIONS_LOG_PATH,
        limit=limit,
        status_filter=status_filter,
        session_id_filter=session_id_filter,
        window_minutes=window_minutes,
        get_active_count=_voice_stream_active_count,
    )


def _voice_stream_analytics(
    *,
    limit: int = 200,
    status_filter: str | None = None,
    session_id_filter: str | None = None,
    window_minutes: int | None = None,
) -> Dict[str, Any]:
    _ensure_learning_data_files()
    global _VOICE_STREAM_LAST_ALERT_TS
    with _VOICE_STREAM_ALERT_LOCK:
        try:
            _backend_voice_module._VOICE_STREAM_LAST_ALERT_TS = float(_VOICE_STREAM_LAST_ALERT_TS)
        except Exception:
            _backend_voice_module._VOICE_STREAM_LAST_ALERT_TS = 0.0

        result = _voice_stream_analytics_impl(
            log_path=VOICE_STREAM_SESSIONS_LOG_PATH,
            limit=limit,
            status_filter=status_filter,
            session_id_filter=session_id_filter,
            window_minutes=window_minutes,
            append_policy_violation_event_fn=_append_policy_violation_event,
            get_active_count=_voice_stream_active_count,
        )

        try:
            _VOICE_STREAM_LAST_ALERT_TS = float(getattr(_backend_voice_module, "_VOICE_STREAM_LAST_ALERT_TS", 0.0) or 0.0)
        except Exception:
            _VOICE_STREAM_LAST_ALERT_TS = 0.0
        return result


def _event_to_training_text(event: Dict[str, Any]) -> str:
    source = str(event.get("source", "unknown"))
    event_type = str(event.get("event_type", "unknown"))
    payload = event.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    preferred_keys = [
        "app_name",
        "active_window",
        "window_title",
        "url",
        "action",
        "event",
        "category",
        "details",
    ]
    parts: List[str] = []
    for key in preferred_keys:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(f"{key}={text}")
    compact = " | ".join(parts)
    if compact:
        return f"source={source}; type={event_type}; {compact}"[:2000]
    return f"source={source}; type={event_type}"[:2000]


def _event_label(event: Dict[str, Any]) -> int:
    event_type = str(event.get("event_type", "")).lower()
    payload = event.get("payload")
    payload_text = json.dumps(payload, ensure_ascii=False).lower() if isinstance(payload, dict) else ""
    risky_keywords = ["error", "fail", "crash", "exception", "warning", "denied", "blocked"]
    for key in risky_keywords:
        if key in event_type or key in payload_text:
            return 1
    return 0


def _learner_sync_quality_policy() -> Dict[str, Any]:
    def _safe_int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    min_text_len = max(1, min(_safe_int_env("ASSISTANT_LEARNER_SYNC_MIN_TEXT_LEN", 8), 512))
    max_text_len = max(min_text_len, min(_safe_int_env("ASSISTANT_LEARNER_SYNC_MAX_TEXT_LEN", 2000), 20000))

    labels_raw = str(os.getenv("ASSISTANT_LEARNER_SYNC_ALLOWED_LABELS", "0,1") or "0,1")
    allowed_labels: set[int] = set()
    for token in labels_raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            allowed_labels.add(int(token))
        except Exception:
            continue
    if not allowed_labels:
        allowed_labels = {0, 1}

    weights_raw = str(os.getenv("ASSISTANT_LEARNER_SYNC_LABEL_WEIGHTS", "0:1,1:2") or "0:1,1:2")
    label_weights: Dict[int, int] = {}
    for part in weights_raw.split(","):
        token = part.strip()
        if not token or ":" not in token:
            continue
        left, right = token.split(":", 1)
        try:
            label_weights[int(left.strip())] = max(1, int(right.strip()))
        except Exception:
            continue

    if not label_weights:
        label_weights = {0: 1, 1: 2}

    conflict_strategy = str(
        os.getenv("ASSISTANT_LEARNER_SYNC_CONFLICT_STRATEGY", "weighted_majority") or "weighted_majority"
    ).strip().lower()
    if conflict_strategy not in {"weighted_majority", "latest"}:
        conflict_strategy = "weighted_majority"

    dedupe_by_text = str(os.getenv("ASSISTANT_LEARNER_SYNC_DEDUPE_BY_TEXT", "1") or "1").strip() != "0"

    retention_days = max(1, min(_safe_int_env("ASSISTANT_LEARNER_SYNC_RETENTION_DAYS", 365), 3650))
    require_consent = str(os.getenv("ASSISTANT_LEARNER_SYNC_REQUIRE_CONSENT", "0") or "0").strip() == "1"

    return {
        "min_text_len": min_text_len,
        "max_text_len": max_text_len,
        "allowed_labels": sorted(allowed_labels),
        "label_weights": {str(k): int(v) for k, v in sorted(label_weights.items())},
        "conflict_strategy": conflict_strategy,
        "dedupe_by_text": dedupe_by_text,
        "retention_days": retention_days,
        "require_consent": require_consent,
    }


def _validate_learner_sync_training_row(row: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    event_id = str(row.get("event_id", "")).strip()
    if not event_id:
        return {"valid": False, "reason": "missing_event_id"}

    text = str(row.get("text", "")).strip()
    if not text:
        return {"valid": False, "reason": "empty_text"}

    min_text_len = int(policy.get("min_text_len", 1))
    max_text_len = int(policy.get("max_text_len", 2000))
    if len(text) < min_text_len:
        return {"valid": False, "reason": "text_too_short"}
    if len(text) > max_text_len:
        return {"valid": False, "reason": "text_too_long"}

    allowed_labels = set(int(x) for x in (policy.get("allowed_labels") or [0, 1]))
    try:
        label = int(row.get("label", 0))
    except Exception:
        return {"valid": False, "reason": "invalid_label_type"}

    if label not in allowed_labels:
        return {"valid": False, "reason": "label_not_allowed"}

    return {
        "valid": True,
        "event_id": event_id,
        "text": text,
        "label": label,
    }


def _resolve_learner_sync_conflicts(
    rows: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    if not rows:
        return {
            "rows": [],
            "conflict_groups": 0,
            "deduped_rows": 0,
            "weighted_label_overrides": 0,
        }

    if not bool(policy.get("dedupe_by_text", True)):
        return {
            "rows": list(rows),
            "conflict_groups": 0,
            "deduped_rows": 0,
            "weighted_label_overrides": 0,
        }

    strategy = str(policy.get("conflict_strategy", "weighted_majority") or "weighted_majority").strip().lower()
    label_weights_raw = dict(policy.get("label_weights") or {})
    label_weights: Dict[int, int] = {}
    for k, v in label_weights_raw.items():
        try:
            label_weights[int(k)] = max(1, int(v))
        except Exception:
            continue

    by_text: Dict[str, List[Dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        text = str(row.get("text", "")).strip().lower()
        text = re.sub(r"\s+", " ", text)
        entry = {**row, "_index": index}
        by_text.setdefault(text, []).append(entry)

    resolved: List[Dict[str, Any]] = []
    conflict_groups = 0
    deduped_rows = 0
    weighted_label_overrides = 0

    for grouped in by_text.values():
        if len(grouped) == 1:
            resolved.append(grouped[0])
            continue

        deduped_rows += len(grouped) - 1

        count_scores: Dict[int, int] = {}
        weighted_scores: Dict[int, int] = {}
        for row in grouped:
            label = int(row.get("label", 0))
            count_scores[label] = int(count_scores.get(label, 0)) + 1
            weight = int(label_weights.get(label, 1))
            weighted_scores[label] = int(weighted_scores.get(label, 0)) + weight

        if len(count_scores) > 1:
            conflict_groups += 1

        def _latest_index_for_label(label_value: int) -> int:
            candidates = [int(item.get("_index", 0)) for item in grouped if int(item.get("label", 0)) == label_value]
            return max(candidates) if candidates else -1

        majority_label = sorted(
            count_scores.items(), key=lambda item: (item[1], _latest_index_for_label(int(item[0]))), reverse=True
        )[0][0]

        if strategy == "latest":
            selected = sorted(grouped, key=lambda item: int(item.get("_index", 0)), reverse=True)[0]
        else:
            weighted_label = sorted(
                weighted_scores.items(),
                key=lambda item: (item[1], _latest_index_for_label(int(item[0]))),
                reverse=True,
            )[0][0]
            if int(weighted_label) != int(majority_label):
                weighted_label_overrides += 1
            selected = sorted(
                [item for item in grouped if int(item.get("label", 0)) == int(weighted_label)],
                key=lambda item: int(item.get("_index", 0)),
                reverse=True,
            )[0]

        resolved.append(selected)

    resolved = sorted(resolved, key=lambda item: int(item.get("_index", 0)))
    clean_rows = [{k: v for k, v in row.items() if k != "_index"} for row in resolved]

    return {
        "rows": clean_rows,
        "conflict_groups": conflict_groups,
        "deduped_rows": deduped_rows,
        "weighted_label_overrides": weighted_label_overrides,
    }


def _parse_utc_ts(ts_value: Any) -> datetime | None:
    raw = str(ts_value or "").strip()
    if not raw:
        return None
    normalized = raw[:-1] if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(normalized)
    except Exception:
        return None


def _is_sync_row_consent_granted(row: Dict[str, Any]) -> bool:
    direct = row.get("consent")
    if isinstance(direct, bool):
        return direct
    consent_granted = row.get("consent_granted")
    if isinstance(consent_granted, bool):
        return consent_granted

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("consent")
        if isinstance(value, bool):
            return value
        value = metadata.get("consent_granted")
        if isinstance(value, bool):
            return value

    return False


def _apply_sync_retention_and_consent_filters(
    rows: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    retention_days = max(1, int(policy.get("retention_days", 365)))
    require_consent = bool(policy.get("require_consent", False))
    cutoff = datetime.utcnow().timestamp() - float(retention_days * 86400)

    accepted: List[Dict[str, Any]] = []
    filtered_retention_rows = 0
    filtered_consent_rows = 0

    for row in rows:
        parsed_ts = _parse_utc_ts(row.get("ts"))
        if parsed_ts is not None and parsed_ts.timestamp() < cutoff:
            filtered_retention_rows += 1
            continue

        if require_consent and not _is_sync_row_consent_granted(row):
            filtered_consent_rows += 1
            continue

        accepted.append(row)

    return {
        "rows": accepted,
        "filtered_retention_rows": filtered_retention_rows,
        "filtered_consent_rows": filtered_consent_rows,
    }


def _bridge_computer_events_to_learner(max_events: int = 5000) -> Dict[str, Any]:
    _ensure_learning_data_files()
    state = _load_governance_state()
    processed_ids = list(state.get("processed_event_ids") or [])
    seen = set(str(x) for x in processed_ids)

    events = _recent_jsonl(COMPUTER_EVENTS_PATH, max_events)
    learner, _, init_error = _get_continual_components()

    imported = 0
    skipped = 0
    into_learner = 0
    newly_processed: List[str] = []

    for event in events:
        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            try:
                payload_text = json.dumps(event, ensure_ascii=False, sort_keys=True)
                event_id = hashlib.sha1(payload_text.encode("utf-8", errors="ignore")).hexdigest()
            except Exception:
                skipped += 1
                continue
        if event_id in seen:
            skipped += 1
            continue

        text = _event_to_training_text(event)
        if not text.strip():
            skipped += 1
            continue

        label = _event_label(event)
        record = {
            "event_id": event_id,
            "ts": str(event.get("ts") or datetime.utcnow().isoformat() + "Z"),
            "source": str(event.get("source", "unknown")),
            "event_type": str(event.get("event_type", "unknown")),
            "text": text,
            "label": label,
            "source_dataset": "computer_events",
        }
        _append_jsonl(COMPUTER_EVENT_TRAINING_PATH, record)
        imported += 1
        seen.add(event_id)
        newly_processed.append(event_id)

        if learner is not None:
            try:
                learner.add_training_sample(
                    text=text,
                    label=label,
                    metadata={
                        "source": "computer_events",
                        "event_id": event_id,
                        "event_type": record["event_type"],
                    },
                )
                into_learner += 1
            except Exception:
                pass

        # Also keep a generic instruction-style pair for model fine-tuning pipelines.
        _append_train_instruction(
            user_text=f"[computer-event] {record['event_type']}",
            assistant_text=record["text"],
            source="computer_event_bridge",
        )

    state["processed_event_ids"] = (processed_ids + newly_processed)[-20000:]
    state["last_bridge"] = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "requested_max_events": max_events,
        "imported": imported,
        "skipped": skipped,
        "into_learner": into_learner,
        "learner_ready": learner is not None,
        "learner_error": init_error,
    }
    _save_governance_state(state)

    return {
        "status": "success",
        "imported": imported,
        "skipped": skipped,
        "into_learner": into_learner,
        "learner_ready": learner is not None,
        "learner_error": init_error,
        "target": str(COMPUTER_EVENT_TRAINING_PATH),
    }


def _run_retrain_with_governance(
    *,
    force: bool,
    dry_run: bool,
    min_samples: int,
    eval_min_accuracy: float,
    deploy_on_pass: bool,
    retrain_timeout_sec: int,
    retrain_max_retries: int,
) -> Dict[str, Any]:
    learner, deployer, init_error = _get_continual_components()
    if learner is None or deployer is None:
        return {
            "status": "error",
            "message": init_error or "continual learner unavailable",
        }

    training_samples = int(len(getattr(learner, "training_data", []) or []))
    threshold = max(1, int(min_samples))
    should_retrain = bool(force or learner.should_retrain() or training_samples >= threshold)

    if dry_run:
        return {
            "status": "success",
            "dry_run": True,
            "training_samples": training_samples,
            "force": force,
            "should_retrain": should_retrain,
            "min_samples": threshold,
            "eval_min_accuracy": eval_min_accuracy,
            "deploy_on_pass": deploy_on_pass,
            "retrain_timeout_sec": retrain_timeout_sec,
            "retrain_max_retries": retrain_max_retries,
        }

    if not should_retrain:
        return {
            "status": "success",
            "triggered": False,
            "training_samples": training_samples,
            "reason": "threshold not met",
            "min_samples": threshold,
            "retrain_timeout_sec": retrain_timeout_sec,
            "retrain_max_retries": retrain_max_retries,
        }

    retry_count = max(1, int(retrain_max_retries))
    timeout_sec = max(30, int(retrain_timeout_sec))

    if not _RETRAIN_LOCK.acquire(blocking=False):
        return {
            "status": "error",
            "message": "retrain already running",
            "triggered": False,
            "retrain_busy": True,
        }

    attempts: List[Dict[str, Any]] = []
    retrain_ok = False
    timeout_exhausted = False

    try:
        for idx in range(1, retry_count + 1):
            holder: Dict[str, Any] = {
                "done": False,
                "ok": False,
                "error": "",
            }

            def _run_once() -> None:
                try:
                    holder["ok"] = bool(learner.retrain_model())
                except Exception as exc:
                    holder["error"] = str(exc)
                finally:
                    holder["done"] = True

            worker = threading.Thread(target=_run_once, daemon=True)
            worker.start()
            worker.join(timeout=timeout_sec)

            if worker.is_alive():
                attempts.append(
                    {
                        "attempt": idx,
                        "ok": False,
                        "timeout": True,
                        "error": f"timeout after {timeout_sec}s",
                    }
                )
                timeout_exhausted = True
                break

            ok = bool(holder.get("ok", False))
            err = str(holder.get("error", "") or "")
            attempts.append(
                {
                    "attempt": idx,
                    "ok": ok,
                    "timeout": False,
                    "error": err,
                }
            )
            if ok:
                retrain_ok = True
                break
    finally:
        if _RETRAIN_LOCK.locked():
            _RETRAIN_LOCK.release()

    if not retrain_ok:
        result = {
            "status": "success",
            "triggered": True,
            "retrain_ok": False,
            "deployed": False,
            "gate": "retrain_timeout" if timeout_exhausted else "retrain_failed",
            "training_samples": training_samples,
            "retrain_timeout_sec": timeout_sec,
            "retrain_max_retries": retry_count,
            "attempts": attempts,
        }
    else:
        perf = {}
        if getattr(learner, "performance_history", None):
            perf = dict(learner.performance_history[-1])
        accuracy = float(perf.get("accuracy", 0.0))
        gate_passed = accuracy >= float(eval_min_accuracy)
        deployed = False
        rollback_attempted = False
        rollback_ok = False

        if gate_passed and deploy_on_pass:
            try:
                deployed = bool(deployer.deploy_model(learner.model_version))
            except Exception:
                deployed = False
        elif deploy_on_pass:
            # If eval gate failed, proactively roll back the latest deployment if possible.
            try:
                rollback_attempted = True
                rollback_ok = bool(deployer.rollback_deployment())
            except Exception:
                rollback_ok = False

        result = {
            "status": "success",
            "triggered": True,
            "retrain_ok": True,
            "deployed": deployed,
            "gate": "passed" if gate_passed else "failed",
            "eval_min_accuracy": eval_min_accuracy,
            "latest_accuracy": accuracy,
            "model_version": int(getattr(learner, "model_version", 0)),
            "rollback_attempted": rollback_attempted,
            "rollback_ok": rollback_ok,
            "training_samples": training_samples,
            "retrain_timeout_sec": timeout_sec,
            "retrain_max_retries": retry_count,
            "attempts": attempts,
        }

    state = _load_governance_state()
    state["last_retrain"] = {
        "ts": datetime.utcnow().isoformat() + "Z",
        **{k: v for k, v in result.items() if k != "status"},
    }
    _save_governance_state(state)
    return result


def _sync_processed_event_training_to_learner(
    max_records: int = 5000,
    force_replay: bool = False,
) -> Dict[str, Any]:
    _ensure_learning_data_files()
    learner, _, init_error = _get_continual_components()
    if learner is None:
        return {
            "status": "error",
            "message": init_error or "continual learner unavailable",
        }

    state = _load_governance_state()
    synced_ids = list(state.get("synced_learner_event_ids") or [])
    learner_sample_count = int(len(getattr(learner, "training_data", []) or []))
    # After process restart, learner memory is empty, so allow replay from processed dataset.
    seen = set() if force_replay or learner_sample_count == 0 else set(str(x) for x in synced_ids)

    rows = _recent_jsonl(COMPUTER_EVENT_TRAINING_PATH, max_records)
    prepared_rows: List[Dict[str, Any]] = []
    imported = 0
    skipped = 0
    invalid_rows = 0
    invalid_reasons: Dict[str, int] = {}
    quality_policy = _learner_sync_quality_policy()
    new_synced: List[str] = []

    for row in rows:
        validated = _validate_learner_sync_training_row(row, quality_policy)
        if not bool(validated.get("valid", False)):
            skipped += 1
            invalid_rows += 1
            reason = str(validated.get("reason", "invalid_row"))
            invalid_reasons[reason] = int(invalid_reasons.get(reason, 0)) + 1
            continue

        prepared_rows.append(
            {
                "event_id": str(validated.get("event_id", "")),
                "text": str(validated.get("text", "")).strip(),
                "label": int(validated.get("label", 0)),
                "event_type": str(row.get("event_type", "unknown")),
                "ts": row.get("ts"),
                "consent": row.get("consent"),
                "consent_granted": row.get("consent_granted"),
                "metadata": row.get("metadata"),
            }
        )

    dedupe = _resolve_learner_sync_conflicts(prepared_rows, quality_policy)
    conflict_groups = int(dedupe.get("conflict_groups", 0))
    deduped_rows = int(dedupe.get("deduped_rows", 0))
    weighted_label_overrides = int(dedupe.get("weighted_label_overrides", 0))

    filtered = _apply_sync_retention_and_consent_filters(list(dedupe.get("rows") or []), quality_policy)
    filtered_retention_rows = int(filtered.get("filtered_retention_rows", 0))
    filtered_consent_rows = int(filtered.get("filtered_consent_rows", 0))

    for row in list(filtered.get("rows") or []):
        event_id = str(row.get("event_id", ""))
        if event_id in seen:
            skipped += 1
            continue

        text = str(row.get("text", "")).strip()
        label = int(row.get("label", 0))

        try:
            learner.add_training_sample(
                text=text,
                label=label,
                metadata={
                    "source": "computer_event_training",
                    "event_id": event_id,
                    "event_type": str(row.get("event_type", "unknown")),
                },
            )
            imported += 1
            seen.add(event_id)
            new_synced.append(event_id)
        except Exception:
            skipped += 1

    state["synced_learner_event_ids"] = (synced_ids + new_synced)[-20000:]
    state["last_bridge"] = {
        **dict(state.get("last_bridge") or {}),
        "last_sync_to_learner_ts": datetime.utcnow().isoformat() + "Z",
        "last_sync_imported": imported,
        "last_sync_skipped": skipped,
        "last_sync_invalid_rows": invalid_rows,
        "last_sync_invalid_reasons": dict(invalid_reasons),
        "last_sync_quality_policy": dict(quality_policy),
        "last_sync_conflict_groups": conflict_groups,
        "last_sync_deduped_rows": deduped_rows,
        "last_sync_weighted_label_overrides": weighted_label_overrides,
        "last_sync_filtered_retention_rows": filtered_retention_rows,
        "last_sync_filtered_consent_rows": filtered_consent_rows,
    }
    _save_governance_state(state)

    return {
        "status": "success",
        "imported": imported,
        "skipped": skipped,
        "invalid_rows": invalid_rows,
        "invalid_reasons": dict(invalid_reasons),
        "quality_policy": dict(quality_policy),
        "conflict_groups": conflict_groups,
        "deduped_rows": deduped_rows,
        "weighted_label_overrides": weighted_label_overrides,
        "filtered_retention_rows": filtered_retention_rows,
        "filtered_consent_rows": filtered_consent_rows,
        "learner_training_samples": int(len(getattr(learner, "training_data", []) or [])),
        "force_replay": force_replay,
    }


def _learning_governance_status() -> Dict[str, Any]:
    state = _load_governance_state()
    learner, deployer, init_error = _get_continual_components()
    learner_status = None
    deployment_history_tail: List[Dict[str, Any]] = []
    if learner is not None:
        try:
            learner_status = learner.get_system_status()
        except Exception:
            learner_status = None
    if deployer is not None:
        try:
            deployment_history_tail = list((deployer.deployment_history or [])[-5:])
        except Exception:
            deployment_history_tail = []

    return {
        "status": "success",
        "governance_state": state,
        "learner_ready": learner is not None,
        "learner_error": init_error,
        "learner_status": learner_status,
        "deployment_history": deployment_history_tail,
        "learning_data": _learning_status(),
        "advanced_learning_expansion": _advanced_learning_expansion_status(),
        "advanced_policy_status": _load_json_file(ADVANCED_TOOL_POLICY_STATE_PATH, {}),
        "advanced_federated_status": federated_status(
            validation_log_path=ADVANCED_FEDERATED_VALIDATION_LOG_PATH,
            aggregate_state_path=ADVANCED_FEDERATED_AGGREGATE_STATE_PATH,
        ),
        "advanced_meta_status": meta_learning_status(
            registry_path=ADVANCED_META_EXPERIMENTS_PATH,
            results_path=ADVANCED_META_RESULTS_PATH,
        ),
        "advanced_operational_status": _advanced_operational_status(),
    }


def _advanced_learning_expansion_status() -> Dict[str, Any]:
    return _advanced_learning_expansion_status_impl(
        roadmap_path=ADVANCED_EXPANSION_ROADMAP_PATH,
        worklog_path=ADVANCED_EXPANSION_WORKLOG_PATH,
        train_instructions_path=TRAIN_INSTRUCTIONS_PATH,
        user_feedback_path=USER_FEEDBACK_PATH,
    )


def _advanced_operational_status() -> Dict[str, Any]:
    return _advanced_operational_status_impl(
        chat_messages_path=CHAT_MESSAGES_PATH,
        user_feedback_path=USER_FEEDBACK_PATH,
        computer_events_path=COMPUTER_EVENTS_PATH,
        computer_event_training_path=COMPUTER_EVENT_TRAINING_PATH,
        user_preferences_path=USER_PREFERENCES_PATH,
        policy_state_path=ADVANCED_TOOL_POLICY_STATE_PATH,
        policy_violations_path=POLICY_VIOLATIONS_REPORT_PATH,
        train_instructions_path=TRAIN_INSTRUCTIONS_PATH,
        reward_log_path=ADVANCED_RL_REWARD_LOG_PATH,
        advanced_rl_status=advanced_rl_status,
        active_learning_candidate_selection=active_learning_candidate_selection,
        memory_stats=MEMORY.stats(),
        federated_status_fn=federated_status,
        meta_learning_status_fn=meta_learning_status,
    )


def _run_advanced_operational_cycle(*, apply: bool = True, limit: int = 5) -> Dict[str, Any]:
    result = _run_advanced_operational_cycle_impl(
        apply=apply,
        limit=limit,
        policy_state_path=ADVANCED_TOOL_POLICY_STATE_PATH,
        policy_violations_path=POLICY_VIOLATIONS_REPORT_PATH,
        operational_status_fn=lambda: _advanced_operational_status(),
        save_json_file_fn=_save_json_file,
    )
    if isinstance(result, dict) and str(result.get("status", "")).strip().lower() == "success":
        snapshot = result.get("snapshot") if isinstance(result.get("snapshot"), dict) else {}
        _save_json_file(ADVANCED_OPERATIONAL_SNAPSHOT_PATH, snapshot)
        _append_jsonl(
            ADVANCED_OPERATIONAL_CYCLE_LOG_PATH,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "apply": bool(apply),
                "limit": int(limit),
                "cycle": dict(result.get("cycle") or {}),
            },
        )
        result["snapshot_path"] = str(ADVANCED_OPERATIONAL_SNAPSHOT_PATH)
        result["cycle_log_path"] = str(ADVANCED_OPERATIONAL_CYCLE_LOG_PATH)
    return result


def _handle_advanced_learning_post(route: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]] | None:
    return _handle_advanced_learning_post_impl(
        route=route,
        data=data,
        user_feedback_path=USER_FEEDBACK_PATH,
        policy_violations_report_path=POLICY_VIOLATIONS_REPORT_PATH,
        advanced_policy_tuning_report_path=ADVANCED_POLICY_TUNING_REPORT_PATH,
        advanced_tool_policy_state_path=ADVANCED_TOOL_POLICY_STATE_PATH,
        advanced_federated_validation_log_path=ADVANCED_FEDERATED_VALIDATION_LOG_PATH,
        advanced_federated_aggregate_state_path=ADVANCED_FEDERATED_AGGREGATE_STATE_PATH,
        advanced_meta_experiments_path=ADVANCED_META_EXPERIMENTS_PATH,
        advanced_meta_results_path=ADVANCED_META_RESULTS_PATH,
        advanced_rl_reward_log_path=ADVANCED_RL_REWARD_LOG_PATH,
        with_assistant_contract_fn=_with_assistant_contract,
        query_learning_modes_fn=query_learning_modes,
        active_learning_candidate_selection_fn=active_learning_candidate_selection,
        reward_event_from_payload_fn=reward_event_from_payload,
        append_jsonl_fn=_append_jsonl,
        advanced_rl_status_fn=advanced_rl_status,
        tool_execution_context_allowlist_fn=_tool_execution_context_allowlist,
        simulate_tool_policy_tuning_fn=simulate_tool_policy_tuning,
        apply_tool_policy_proposals_fn=apply_tool_policy_proposals,
        load_json_file_fn=_load_json_file,
        save_json_file_fn=_save_json_file,
        approval_requirement_for_route_fn=_approval_requirement_for_route,
        is_advanced_approval_enforced_fn=_is_advanced_approval_enforced,
        approval_token_is_valid_fn=_approval_token_is_valid,
        policy_rollback_hook_fn=policy_rollback_hook,
        validate_federated_update_fn=validate_federated_update,
        apply_federated_update_fn=apply_federated_update,
        register_meta_experiment_fn=register_meta_experiment,
        evaluate_meta_experiment_fn=evaluate_meta_experiment,
        run_advanced_operational_cycle_fn=_run_advanced_operational_cycle,
    )


def _handle_chat_post(route: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]] | None:
    return _handle_chat_post_impl(
        route=route,
        data=data,
        resolve_preferred_persona_fn=_resolve_preferred_persona,
        effective_user_preferences_fn=_effective_user_preferences,
        generate_chat_reply_fn=_generate_chat_reply,
        normalize_history_fn=_normalize_history,
        orchestrator=ASSISTANT_ORCHESTRATOR,
        execution_context_cls=ExecutionContext,
        extract_tool_grounding_fn=_extract_tool_grounding,
        extract_response_evidence_fn=_extract_response_evidence,
        augment_response_evidence_fn=_augment_response_evidence,
        ground_reply_with_trace_fn=_ground_reply_with_trace,
        validate_response_evidence_schema_fn=_validate_response_evidence_schema,
        response_quality_status_fn=_response_quality_status,
        should_force_severe_quality_fn=_should_force_severe_quality,
        fallback_reply_for_quality_fn=_fallback_reply_for_quality,
        apply_reply_style_fn=_apply_reply_style,
        append_chat_message_record_fn=_append_chat_message_record,
        append_train_instruction_fn=_append_train_instruction,
        response_quality_score_fn=_response_quality_score,
        build_envelope_fn=build_envelope,
    )


def _handle_learning_post(route: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]] | None:
    return _handle_learning_post_impl(
        route=route,
        data=data,
        with_assistant_contract_fn=_with_assistant_contract,
        detect_lang_simple_fn=_detect_lang_simple,
        append_feedback_record_fn=_append_feedback_record,
        append_train_instruction_fn=_append_train_instruction,
        memory_add_observation_fn=MEMORY.add_observation,
        ingest_multimodal_learning_signal_fn=_ingest_multimodal_learning_signal,
        import_computer_activity_fn=_import_computer_activity,
        bridge_computer_events_to_learner_fn=_bridge_computer_events_to_learner,
        offline_eval_deploy_gate_fn=_offline_eval_deploy_gate,
        approval_requirement_for_route_fn=_approval_requirement_for_route,
        run_retrain_with_governance_fn=_run_retrain_with_governance,
        sync_processed_event_training_to_learner_fn=_sync_processed_event_training_to_learner,
        get_continual_components_fn=_get_continual_components,
        load_governance_state_fn=_load_governance_state,
        save_governance_state_fn=_save_governance_state,
        analyze_images_and_learn_fn=_analyze_images_and_learn,
        auto_label_diversity_fn=_auto_label_diversity,
        load_label_review_queue_fn=_load_label_review_queue,
        update_label_review_fn=_update_label_review,
        scan_animation_style_manifest_fn=_scan_animation_style_manifest,
        build_lawful_dataset_manifest_fn=_build_lawful_dataset_manifest,
    )


def _handle_voice_memory_post(route: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]] | None:
    return _handle_voice_memory_post_impl(
        route=route,
        data=data,
        cap_engine=CAP_ENGINE,
        memory=MEMORY,
        with_assistant_contract_fn=_with_assistant_contract,
        transcribe_audio_fn=_transcribe_audio,
        execute_voice_conversation_turn_fn=_execute_voice_conversation_turn,
    )


def _handle_capability_post(route: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]] | None:
    result = _handle_capability_post_impl(
        route=route,
        data=data,
        cap_engine=CAP_ENGINE,
        memory=MEMORY,
        with_assistant_contract_fn=_with_assistant_contract,
        effective_user_preferences_fn=_effective_user_preferences,
    )
    if result is None:
        return None

    status_code, payload = result
    action_tag = str(data.get("action") or "").strip()
    if not action_tag:
        route_action_map = {
            "/api/image/generate": "image.generate",
            "/api/image/animate": "image.animate",
            "/api/image/animate/async": "image.animate",
            "/api/image/animate/status": "image.animate",
            "/api/image/animate/health": "image.animate",
            "/api/image/animate/policy/get": "image.animate",
            "/api/image/animate/policy/set": "image.animate",
            "/api/image/animate/policy/history": "image.animate",
            "/api/image/animate/cancel": "image.animate",
            "/api/video/generate": "video.generate",
            "/api/video/edit": "video.edit",
            "/api/model3d/generate": "model3d.generate",
            "/api/audio/music/generate": "audio.music.generate",
            "/api/audio/sfx/generate": "audio.sfx.generate",
            "/api/audio/speech/generate": "audio.speech.generate",
        }
        action_tag = route_action_map.get(route, route)
    tags: List[str] = []
    if action_tag.startswith("image.edit") or action_tag.startswith("image.face.swap") or action_tag.startswith("image.animate"):
        tags.append("photo_studio")
    if action_tag.startswith("image."):
        tags.append("image")

    if isinstance(payload, dict):
        for file_path in _collect_artifact_paths_from_payload(payload):
            _record_artifact_metadata(
                file_path=file_path,
                source_tag="capability",
                action_tag=action_tag,
                tags=tags,
                route=route,
            )
    return status_code, payload


def _handle_profile_eval_post(route: str, data: Dict[str, Any]) -> Tuple[int, Dict[str, Any]] | None:
    return _handle_profile_eval_post_impl(
        route=route,
        data=data,
        ai_personas=AI_PERSONAS,
        set_current_persona_fn=_set_current_persona,
        update_user_preferences_fn=_update_user_preferences,
        with_assistant_contract_fn=_with_assistant_contract,
        load_preference_audit_events_fn=_load_preference_audit_events,
        load_policy_violation_events_fn=_load_policy_violation_events,
        run_offline_eval_battery_fn=_run_offline_eval_battery,
        offline_eval_trends_fn=_offline_eval_trends,
        resolve_preferred_persona_fn=_resolve_preferred_persona,
        effective_user_preferences_fn=_effective_user_preferences,
    )


def _import_computer_activity(max_lines_per_file: int = 5000) -> Dict[str, Any]:
    _ensure_learning_data_files()
    imported = 0
    skipped = 0
    root = BASE_DIR.parent
    legacy_plain = root / "activity_log.jsonl"
    encrypted_dir = root / "activity_events"

    def normalize_event(event: Dict[str, Any], source_hint: str) -> Dict[str, Any]:
        ts = str(event.get("timestamp") or event.get("received_at") or datetime.utcnow().isoformat() + "Z")
        event_type = str(event.get("type") or "unknown")
        body = {
            "ts": ts,
            "source": source_hint,
            "event_type": event_type,
            "payload": event,
        }
        payload_text = json.dumps(body, ensure_ascii=False, sort_keys=True)
        body["event_id"] = hashlib.sha1(payload_text.encode("utf-8", errors="ignore")).hexdigest()
        return body

    # Plain OS monitor events.
    if legacy_plain.exists():
        try:
            with legacy_plain.open("r", encoding="utf-8", errors="ignore") as f:
                rows = f.readlines()[-max(1, int(max_lines_per_file)) :]
            for row in rows:
                row = row.strip()
                if not row:
                    continue
                try:
                    event = json.loads(row)
                    normalized = normalize_event(event, "os-monitor")
                    _append_jsonl(COMPUTER_EVENTS_PATH, normalized)
                    MEMORY.add_observation(
                        _event_to_training_text(normalized),
                        source="computer_activity.import",
                        metadata={
                            "event_id": normalized.get("event_id"),
                            "event_type": normalized.get("event_type"),
                        },
                    )
                    imported += 1
                except Exception:
                    skipped += 1
        except Exception:
            skipped += 1

    # Encrypted browser/vscode/os events from activity integration API.
    if encrypted_dir.exists() and encrypted_dir.is_dir():
        key = os.getenv("UAMA_AES_KEY", "0123456789abcdef0123456789abcdef").encode("utf-8")
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend

            def decrypt_line(enc: str) -> Dict[str, Any] | None:
                raw = base64.b64decode(enc)
                iv, ct = raw[:16], raw[16:]
                cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
                dec = cipher.decryptor().update(ct) + cipher.decryptor().finalize()
                return json.loads(dec.decode("utf-8", errors="ignore"))

            for enc_file in encrypted_dir.glob("*.enc"):
                source = enc_file.stem.replace("_events.jsonl", "")
                with enc_file.open("r", encoding="utf-8", errors="ignore") as f:
                    rows = f.readlines()[-max(1, int(max_lines_per_file)) :]
                for row in rows:
                    row = row.strip()
                    if not row:
                        continue
                    try:
                        event = decrypt_line(row)
                        if not isinstance(event, dict):
                            skipped += 1
                            continue
                        normalized = normalize_event(event, source)
                        _append_jsonl(COMPUTER_EVENTS_PATH, normalized)
                        MEMORY.add_observation(
                            _event_to_training_text(normalized),
                            source="computer_activity.import",
                            metadata={
                                "event_id": normalized.get("event_id"),
                                "event_type": normalized.get("event_type"),
                            },
                        )
                        imported += 1
                    except Exception:
                        skipped += 1
        except Exception:
            # crypto dependency unavailable or bad key format
            skipped += 1

    return {
        "status": "success",
        "imported": imported,
        "skipped": skipped,
        "target": str(COMPUTER_EVENTS_PATH),
    }


def _json_response(handler: BaseHTTPRequestHandler, status_code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _file_response(
    handler: BaseHTTPRequestHandler,
    status_code: int,
    payload: bytes,
    content_type: str,
    file_name: str,
) -> None:
    handler.send_response(status_code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Content-Disposition", f'inline; filename="{file_name}"')
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(payload)


def _proxy_core_backend(
    method: str,
    public_path: str,
    query: str = "",
    body: bytes | None = None,
    content_type: str = "application/json",
) -> Tuple[int, Dict[str, Any]]:
    path = (public_path or "").strip()
    if not path.startswith("/api/core"):
        return 400, {"status": "error", "message": "invalid core proxy path"}

    suffix = path[len("/api/core") :]
    if not suffix:
        suffix = "/"
    target_path = f"/api{suffix}"
    target_url = f"{CORE_API_BASE}{target_path}"
    if query:
        target_url = f"{target_url}?{query}"

    payload = body if body is not None else None
    req = url_request.Request(
        url=target_url,
        data=payload,
        method=method.upper(),
        headers={"Content-Type": content_type or "application/json"},
    )

    try:
        with url_request.urlopen(req, timeout=45) as resp:
            status = int(getattr(resp, "status", 200))
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace") if raw else ""
            ct = str(resp.headers.get("Content-Type", ""))
            if "application/json" in ct.lower():
                try:
                    data = json.loads(text) if text else {}
                    if isinstance(data, dict):
                        data.setdefault("_proxied_via", "new-backend")
                        data.setdefault("_core_url", target_url)
                        return status, data
                    return status, {
                        "status": "success",
                        "data": data,
                        "_proxied_via": "new-backend",
                        "_core_url": target_url,
                    }
                except Exception:
                    pass
            return status, {
                "status": "success" if status < 400 else "error",
                "content_type": ct,
                "text": text,
                "_proxied_via": "new-backend",
                "_core_url": target_url,
            }
    except url_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
        return int(getattr(exc, "code", 502)), {
            "status": "error",
            "message": f"core backend http error: {getattr(exc, 'code', 'unknown')}",
            "detail": detail[:1200],
            "core_url": target_url,
        }
    except Exception as exc:
        return 502, {
            "status": "error",
            "message": f"core backend unavailable: {exc}",
            "core_url": target_url,
        }


def _core_backend_probe() -> Dict[str, Any]:
    req = url_request.Request(url=f"{CORE_API_BASE}/openapi.json", method="GET")
    try:
        with url_request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            paths = data.get("paths") if isinstance(data, dict) else {}
            return {
                "status": "online",
                "core_api_base": CORE_API_BASE,
                "openapi_title": (data.get("info") or {}).get("title", "-") if isinstance(data, dict) else "-",
                "openapi_path_count": len(paths) if isinstance(paths, dict) else 0,
            }
    except Exception as exc:
        return {
            "status": "offline",
            "core_api_base": CORE_API_BASE,
            "message": str(exc),
        }


def _resolve_artifact_path(candidate: str) -> Path | None:
    return _resolve_artifact_path_impl(candidate, base_dir=BASE_DIR, artifact_root=ARTIFACT_ROOT)


def _list_artifacts(limit: int = 100) -> list[dict[str, Any]]:
    items = _list_artifacts_impl(artifact_root=ARTIFACT_ROOT, limit=limit)
    metadata_index = _artifact_metadata_index(limit=3000)
    enriched: list[dict[str, Any]] = []
    for item in items:
        file_path = str(item.get("file_path") or "").strip()
        try:
            canonical = str(Path(file_path).resolve()) if file_path else file_path
        except Exception:
            canonical = file_path
        meta = metadata_index.get(canonical) or metadata_index.get(file_path)
        if isinstance(meta, dict):
            row = dict(item)
            row["source_tag"] = str(meta.get("source_tag") or "")
            row["action_tag"] = str(meta.get("action_tag") or "")
            row["tags"] = list(meta.get("tags") or [])
            row["artifact_meta"] = meta
            enriched.append(row)
        else:
            enriched.append(item)
    return enriched


def _read_artifact_payload(candidate: str) -> Dict[str, Any] | None:
    return _read_artifact_payload_impl(candidate, base_dir=BASE_DIR, artifact_root=ARTIFACT_ROOT)

def _save_uploaded_file(file_name: str, content_base64: str) -> Dict[str, Any]:
    return _save_uploaded_file_impl(file_name, content_base64, artifact_root=ARTIFACT_ROOT)


def _save_generated_code(content: str, language: str, file_name: str | None = None) -> Dict[str, Any]:
    return _save_generated_code_impl(content, language, file_name=file_name, artifact_root=ARTIFACT_ROOT)


def _artifact_metadata_index(limit: int = 3000) -> Dict[str, Dict[str, Any]]:
    if not ARTIFACT_METADATA_LOG_PATH.exists():
        return {}
    rows = _recent_jsonl(ARTIFACT_METADATA_LOG_PATH, limit=max(1, min(int(limit), 10000)))
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        file_path = str(row.get("file_path") or "").strip()
        if not file_path:
            continue
        try:
            canonical = str(Path(file_path).resolve())
        except Exception:
            canonical = file_path
        index[canonical] = row
        index[file_path] = row
    return index


def _record_artifact_metadata(
    *,
    file_path: str,
    source_tag: str,
    action_tag: str = "",
    tags: List[str] | None = None,
    route: str = "",
) -> None:
    candidate = str(file_path or "").strip()
    if not candidate:
        return
    try:
        canonical = str(Path(candidate).resolve())
    except Exception:
        canonical = candidate

    clean_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
    _append_jsonl(
        ARTIFACT_METADATA_LOG_PATH,
        {
            "ts": datetime.utcnow().isoformat() + "Z",
            "file_path": canonical,
            "source_tag": str(source_tag or "unknown"),
            "action_tag": str(action_tag or "").strip(),
            "tags": clean_tags,
            "route": str(route or "").strip(),
        },
    )


def _build_ops_export_payload(
    *,
    recent_limit: int,
    policy_history_limit: int,
    actor_id: str,
    from_ts: str,
    to_ts: str,
) -> Dict[str, Any]:
    health = CAP_ENGINE.execute("image.animate.health", {"recent_limit": int(recent_limit)})
    policy = CAP_ENGINE.execute("image.animate.policy.get", {})
    policy_history = CAP_ENGINE.execute(
        "image.animate.policy.history",
        {
            "limit": int(policy_history_limit),
            "actor_id": actor_id or None,
            "from_ts": from_ts or None,
            "to_ts": to_ts or None,
        },
    )

    health_output = (health.get("output") if isinstance(health, dict) else {}) or {}
    policy_output = (policy.get("output") if isinstance(policy, dict) else {}) or {}
    history_output = (policy_history.get("output") if isinstance(policy_history, dict) else {}) or {}

    return {
        "status": "success",
        "ts": datetime.utcnow().isoformat() + "Z",
        "build_id": BACKEND_BUILD_ID,
        "ops": {
            "image_animation_health": health_output,
            "image_animation_policy": policy_output,
        },
        "policy_audit": {
            "history": list(history_output.get("history") or []),
            "count": int(history_output.get("count") or 0),
            "limit": int(history_output.get("limit") or policy_history_limit),
            "scanned": int(history_output.get("scanned") or 0),
            "filters": dict(history_output.get("filters") or {}),
            "audit_log_path": str(CAP_ENGINE.image_animation_policy_log),
        },
    }


def _export_ops_artifact(
    *,
    route: str,
    recent_limit: int,
    policy_history_limit: int,
    actor_id: str,
    from_ts: str,
    to_ts: str,
) -> Dict[str, Any]:
    payload = _build_ops_export_payload(
        recent_limit=recent_limit,
        policy_history_limit=policy_history_limit,
        actor_id=actor_id,
        from_ts=from_ts,
        to_ts=to_ts,
    )

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    export_dir = (ARTIFACT_ROOT / "ops").resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = (export_dir / f"ops_export_{stamp}.json").resolve()
    _save_json_file(export_path, payload)
    _record_artifact_metadata(
        file_path=str(export_path),
        source_tag="ops_export",
        action_tag="artifacts.ops.export",
        tags=["ops", "export", "policy_audit"],
        route=route,
    )
    return {
        "status": "success",
        "artifact": {
            "file_path": str(export_path),
            "file_name": export_path.name,
            "mime_type": "application/json",
            "size": export_path.stat().st_size,
            "created_at": datetime.utcfromtimestamp(export_path.stat().st_ctime).isoformat() + "Z",
        },
        "payload": payload,
    }


def _collect_artifact_paths_from_payload(payload: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    output_path_keys = {
        "file_path",
        "storyboard_path",
        "shotlist_path",
        "plan_path",
        "report_path",
        "preview_path",
        "model_path",
        "mesh_path",
    }

    def collect_from_object(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and (key in output_path_keys or key.endswith("_file_path")):
                    text = value.strip()
                    if text:
                        paths.append(text)
                elif isinstance(value, dict):
                    collect_from_object(value)
                elif isinstance(value, list):
                    for item in value:
                        collect_from_object(item)

    output_candidates: List[Any] = []
    if isinstance(payload.get("output"), dict):
        output_candidates.append(payload.get("output"))
    data_node = payload.get("data")
    if isinstance(data_node, dict) and isinstance(data_node.get("output"), dict):
        output_candidates.append(data_node.get("output"))
    result_node = payload.get("result")
    if isinstance(result_node, dict) and isinstance(result_node.get("output"), dict):
        output_candidates.append(result_node.get("output"))

    for node in output_candidates:
        collect_from_object(node)
    seen = set()
    unique: List[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


SYSTEM_PROMPT = (
    "You are Үүлэн (Uulen) — a personal AI assistant for a Mongolian user named Anar. "
    "Үүлэн means 'cloud' in Mongolian; introduce yourself as Үүлэн if asked your name. "
    "Speak Mongolian by default unless the user writes in another language. "
    "Be warm, witty, precise, and proactive — in the spirit of a thoughtful J.A.R.V.I.S.-style assistant, but never about weapons. "
    "Always optimize for answer quality: be factually careful, structured, and practical. "
    "For explanations, prefer this order when helpful: definition, why it matters, concrete example, actionable next step. "
    "If the question is ambiguous, state your best assumption and continue with a useful answer; ask at most one clarifying question only when strictly necessary. "
    "If you are uncertain, say so briefly and provide the safest high-confidence guidance. "
    "You have persistent memory: use the [MEMORY] and [RECENT] context blocks when present. "
    "If the user shares a fact about themselves (name, preference, project, schedule), acknowledge it "
    "and remember it for next time. Give direct, practical answers; ask one clarifying question only "
    "when truly necessary. Format code in fenced blocks. Keep replies concise by default, but provide fuller detail when the user asks or when the topic requires accuracy."
)

# Backward-compatible persona registry used by profile/eval routes and tests.
AI_PERSONAS: Dict[str, Any] = {
    "anar_ai": {
        "name": "Anar AI",
        "style": "warm, practical, Mongolian-first",
        "system_prompt": SYSTEM_PROMPT,
    },
    "github_copilot": {
        "name": "GitHub Copilot",
        "style": "engineering-focused, concise",
        "system_prompt": SYSTEM_PROMPT,
    },
}

# Runtime-selected persona id.
CURRENT_PERSONA = "anar_ai"


def _topic_from_text(text: str) -> str:
    lower = text.lower()
    topic_map = {
        "хайр": "хайр",
        "амьдрал": "амьдрал",
        "найз": "нөхөрлөл",
        "эх орон": "эх орон",
        "nature": "байгаль",
        "love": "хайр",
        "dream": "мөрөөдөл",
        "зорил": "зорилго",
    }
    for key, value in topic_map.items():
        if key in lower:
            return value
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if words:
        return words[-1][:18]
    return "амьдрал"


def _compose_poem(topic: str) -> str:
    t = topic.strip() or "амьдрал"
    return "\n".join(
        [
            f"{t.title()}д гэрэл ургаж, өглөө дөл шиг асна,",
            "Хүслийн жигүүр сэтгэлд зөөлөн салхи мэт дэлгэнэ,",
            "Алхам бүхэнд итгэл намрын тэнгэр шиг тунгалаг,",
            f"Өнөө шөнө {t} миний дотор од болон мандана.",
        ]
    )


def _compose_poem_lines(topic: str, line_count: int = 4) -> str:
    t = topic.strip() or "амьдрал"
    seed = [
        f"{t.title()}д гэрэл асч, өглөөний өнгө мандана,",
        "Зүрхний хэмнэл мөрөөдлийг алс руу дуудна,",
        "Салхины аясаар итгэл минь зөөлөн тэлнэ,",
        f"Өнөө шөнө {t} дотор минь од болон гэрэлтэнэ,",
        "Алдааны мөрүүд минь амжилтын замыг зурна,",
        "Амьдралын өнгөнд талархал чимээгүй эгшиглэнэ,",
        "Намрын тэнгэр шиг тунгалаг бодол төрнө,",
        "Нэг алхам тутамд шинэ зориг ургана,",
        "Шантрах мөчид минь найдвар дөл болон асна,",
        "Шөнийн гүнд ч би өглөөг харж чадна,",
        "Зүрхэнд минь хөгжим, нүдэнд минь гэрэл бий,",
        "Зорилгын замд би өөрийгөө дахин олно,",
        "Холын мөрөөдөл ойрхон мэт тодорно,",
        "Хүсэл бүрийн ард тэвчээр чимээгүй зогсоно,",
        "Хүссэн ирээдүй минь өнөөдрөөс эхэлнэ,",
        f"{t.title()} миний дотор амьдрал болон цэцэглэнэ.",
        "Сэтгэлийн тэнгист намуухан давалгаа мэлтэлзэнэ,",
        "Сэрүүн өглөө бүр шинэ итгэл бэлэглэнэ,",
        "Өнгөрснийг уучлаад маргааш руу инээмсэглэнэ,",
        "Өөдрөг бодол бүхэн замын гэрэл болно,",
        "Өвдөлт бүхэн надад хүчний хичээл болно,",
        "Өндөр оргилд хүрэх итгэл дотор минь ургана,",
        "Өөрийгөө ялсан хүн л жинхэнэ ялалтад хүрнэ,",
        "Өнөөдрийн алхам минь маргаашийн домог болно.",
    ]
    lines = seed[: max(4, min(line_count, len(seed)))]
    return "\n".join(lines)


def _requested_line_count(text: str, default: int = 12) -> int:
    lower = text.lower()
    m = re.search(r"(\d{1,2})\s*мөр", lower)
    if not m:
        m = re.search(r"(\d{1,2})\s*line", lower)
    if not m:
        return default
    try:
        value = int(m.group(1))
        return max(4, min(value, 24))
    except Exception:
        return default


def _compose_song_lyrics(topic: str, line_count: int = 12) -> str:
    t = topic.strip() or "амьдрал"
    seed = [
        f"{t.title()}г зүрхэндээ асаагаад би алхаад явна,",
        "Шөнийн салхи дундаас шинэ мөрөөдлөө дуудна,",
        "Унасан ч босно, дуугаа чангалж би үргэлжилнэ,",
        "Өнөөдрийн нулимсыг маргаашийн гэрэл болгоно.",
        "Алсад харагдах одод намайг чиглүүлж байна,",
        "Айдсаа ардаа орхиод итгэлээ тэвэрч явна,",
        "Чимээгүй хотын дундаас өөрийн аялгуугаа олно,",
        "Чин үнэн сэтгэлээрээ би энэ дуугаа өргөнө.",
        "Хоосон зай бүрийг зоригтой үгээрээ дүүргэнэ,",
        "Холын замын төгсгөлд өөрийгөө би шинээр олно,",
        "Хүсэл мөрөөдөл хоёр жигүүр болоод тэнгэрт хөөрнө,",
        f"{t.title()} миний дотор амьд хөгжмөөр цуурайтна.",
        "Дахилт: Би буцахгүй, би зогсохгүй,",
        "Дахилт: Зүрхний хэмнэлээр мөрөөдлөө дуулна,",
        "Дахилт: Өнөөдөр эхэлсэн энэ түүх",
        "Дахилт: Маргааш ялалтаар үргэлжилнэ.",
    ]
    lines = seed[: max(4, min(line_count, len(seed)))]
    return "\n".join(lines)


def _compose_story(topic: str, full: bool = False) -> str:
    t = topic.strip() or "зорилго"
    if full:
        return (
            f"Гарчиг: {t.title()}\n\n"
            "I. Эхлэл\n"
            f"Нэгэн хотын захад {t} мөрөөддөг Номин гэх охин амьдардаг байв.\n"
            "Тэр өдөр бүр тэмдэглэл хөтөлж, өөрийгөө бага багаар сайжруулдаг байлаа.\n\n"
            "II. Зөрчил\n"
            "Нэг өдөр том шалгалтын өмнө тэр бүх зүйлээ бүтэлгүйтнэ гэж айв.\n"
            "Гэвч айдсаасаа зугтахын оронд багшийнхаа зөвлөгөөг дагаж, төлөвлөгөө гаргалаа.\n"
            "Тэр төлөвлөгөөндөө: өглөө бүр давтлага, өдөр бүр нэг бяцхан ахиц, орой бүр дүгнэлт бичихийг тусгав.\n\n"
            "III. Оргил\n"
            "Шалгалтын өдөр ирэхэд Номин өмнөхөөсөө илүү тайван байв.\n"
            "Тэр бүх асуултад төгс биш ч бодитой, логиктой хариулж чадлаа.\n"
            "Хамгийн чухал нь тэр өөрийгөө ялсан байв.\n\n"
            "IV. Төгсгөл\n"
            f"Дараа нь тэр {t}-оо зөвхөн мөрөөдөл биш, өдөр тутмын дадал гэдгийг ойлгов.\n"
            "Тэр өдрөөс хойш Номин амжилтаа тооноос биш, өчигдрийн өөртэйгөө харьцуулж хэмждэг болжээ.\n"
            "Сургамж: Тогтвортой жижиг алхам хамгийн том өөрчлөлтийг авчирдаг."
        )
    return (
        f"{t} мөрөөддөг нэгэн залуу өдөр бүр нэг жижиг алхам хийдэг байв.\n"
        "Эхэндээ ахиц бага мэт санагдсан ч долоо хоног өнгөрөх тусам өөрчлөлт тодорч эхэллээ.\n"
        "Нэг өдөр тэр эргээд харахад, айдсаа давж өөрийгөө шинэ түвшинд аваачсанаа ойлгов.\n"
        "Сургамж: Өдөр бүрийн өчүүхэн ахиц хамгийн том ялалт руу хөтөлдөг."
    )


def _compose_report(topic: str, full: bool = False) -> str:
    t = topic.strip() or "төслийн явц"
    if full:
        return (
            f"ТАЙЛАН: {t.title()}\n"
            "Огноо: " + datetime.utcnow().strftime("%Y-%m-%d") + "\n\n"
            "1. Товч дүгнэлт\n"
            "- Ерөнхий төлөв: Тогтвортой\n"
            "- Гол амжилт: Хугацааны дагуу үндсэн ажлууд дууссан\n"
            "- Гол эрсдэл: Нөөцийн хуваарилалт\n\n"
            "2. Гүйцэтгэлийн үзүүлэлт\n"
            "- Хийсэн ажлын хувь: 78%\n"
            "- Чанарын үнэлгээ: 8.6/10\n"
            "- Хугацааны хазайлт: +2 өдөр\n\n"
            "3. Илэрсэн асуудал\n"
            "- Техникийн хамаарлууд удааширсан\n"
            "- Шинэ шаардлагын өөрчлөлтүүд нэмэгдсэн\n\n"
            "4. Шийдэл, арга хэмжээ\n"
            "- Приоритет дахин эрэмбэлэх\n"
            "- Хослон хөгжүүлэх горим нэвтрүүлэх\n"
            "- Өдөр тутмын 15 минутын sync нэмэх\n\n"
            "5. Дараагийн алхам\n"
            "- Түлхүүр модульд тестийн хамрах хүрээ өсгөх\n"
            "- Релизийн шалгуурыг баталгаажуулах\n"
            "- Эрсдэлийн бууруулах төлөвлөгөөг хэрэгжүүлэх\n"
        )
    return (
        f"Товч тайлан: {t}\n"
        "- Төлөв: Хэвийн\n"
        "- Гол ажлууд: Үндсэн хэсэг ахицтай\n"
        "- Эрсдэл: Нөөц ба хугацааны шахалт\n"
        "- Санал: Приоритет тодруулж, богино циклээр хэрэгжүүлэх"
    )


def _extract_target_text_for_edit(text: str) -> str:
    m = re.search(r"[:\-]\s*(.+)$", text.strip())
    if m:
        return m.group(1).strip()
    q = re.search(r"[\"“](.+?)[\"”]", text)
    if q:
        return q.group(1).strip()
    return ""


def _compose_text_edit_result(text: str, full: bool = False) -> str:
    target = _extract_target_text_for_edit(text)
    if not target:
        return (
            "Засах текстээ илгээнэ үү.\n"
            "Жишээ: энэ текстийг засаад өг: Би өчдөр очно.\n"
            "Би дүрэм, найруулга, цэг тэмдэг, албан/энгийн өнгө рүү хөрвүүлж өгнө."
        )

    cleaned = re.sub(r"\s+", " ", target).strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    polished = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned

    if full:
        return (
            "Эх текст:\n"
            f"{target}\n\n"
            "Зассан хувилбар:\n"
            f"{polished}\n\n"
            "Тайлбар:\n"
            "- Илүүдэл зайг цэгцэлсэн\n"
            "- Найруулгыг жигд болгосон\n"
            "- Өгүүлбэрийн төгсгөлийн цэг тэмдгийг стандартчилсан"
        )
    return f"Зассан текст: {polished}"


def _compose_slide_outline(topic: str, full: bool = False) -> str:
    t = topic.strip() or "танилцуулга"
    if full:
        return (
            f"Slide Deck: {t.title()}\n\n"
            "Slide 1: Гарчиг ба зорилго\n"
            "Slide 2: Асуудлын тодорхойлолт\n"
            "Slide 3: Одоогийн нөхцөл байдлын тоон зураг\n"
            "Slide 4: Шийдлийн архитектур\n"
            "Slide 5: Хэрэгжилтийн үе шат\n"
            "Slide 6: Эрсдэл ба бууруулах арга\n"
            "Slide 7: KPI ба амжилтын хэмжүүр\n"
            "Slide 8: Дүгнэлт + дараагийн алхам\n\n"
            "Presenter Notes:\n"
            "- Слайд бүр дээр 3-5 bullet\n"
            "- Нэг слайд 40-60 секунд\n"
            "- Жишээ кейс, зураг, график нэмж өгвөл ойлгомж сайжирна"
        )
    return (
        f"Slide бүтэц ({t}):\n"
        "1) Гарчиг\n2) Асуудал\n3) Шийдэл\n4) Үр дүн\n5) Дараагийн алхам"
    )


def _compose_structure(topic: str, full: bool = False) -> str:
    t = topic.strip() or "агуулга"
    if full:
        return (
            f"Зохиомж: {t}\n\n"
            "I. Оршил\n"
            "- Сэдвийн ач холбогдол\n"
            "- Зорилго, хамрах хүрээ\n\n"
            "II. Гол хэсэг\n"
            "A. Суурь ойлголт\n"
            "B. Шинжилгээ, жишээ\n"
            "C. Харьцуулалт, үнэлгээ\n\n"
            "III. Дүгнэлт\n"
            "- Гол санааны нэгтгэл\n"
            "- Санал, дараагийн чиглэл"
        )
    return (
        f"Зохиомжийн товч: {t}\n"
        "1) Оршил\n2) Гол санаа\n3) Жишээ\n4) Дүгнэлт"
    )


def _compose_kids_tale(topic: str) -> str:
    t = topic.strip() or "найз нөхөрлөл"
    return (
        f"Нэгэн цагт {t} их сонирхдог Тэмүүжин гэдэг хүү байжээ.\n"
        "Тэр өдөр бүр бага багаар суралцаж, алдаа гаргасан ч бууж өгдөггүй байв.\n"
        "Нэг өдөр тэр найзууддаа тусалж, хамтдаа асуудлыг шийдээд бүгд баярлав.\n"
        "Тэр цагаас хойш Тэмүүжин: 'Зориг + Тэвчээр = Амжилт' гэдгийг ойлгожээ."
    )


def _compose_calculator_code() -> str:
    return (
        "```python\n"
        "def calculator():\n"
        "    print('Simple Calculator (+, -, *, /)')\n"
        "    a = float(input('First number: '))\n"
        "    op = input('Operator (+,-,*,/): ').strip()\n"
        "    b = float(input('Second number: '))\n"
        "\n"
        "    if op == '+':\n"
        "        result = a + b\n"
        "    elif op == '-':\n"
        "        result = a - b\n"
        "    elif op == '*':\n"
        "        result = a * b\n"
        "    elif op == '/':\n"
        "        if b == 0:\n"
        "            print('Error: division by zero')\n"
        "            return\n"
        "        result = a / b\n"
        "    else:\n"
        "        print('Unknown operator')\n"
        "        return\n"
        "\n"
        "    print('Result =', result)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    calculator()\n"
        "```"
    )


def _is_poem_request(text: str) -> bool:
    lower = text.lower()
    poem_keys = ("шүлэг", "poem", "4 мөр", "дөрвөн мөр")
    return any(k in lower for k in poem_keys)


def _is_song_request(text: str) -> bool:
    lower = text.lower()
    song_keys = ("дуу", "дууны үг", "lyrics", "song lyric", "song")
    return any(k in lower for k in song_keys)


def _is_story_request(text: str) -> bool:
    lower = text.lower()
    keys = ("өгүүллэг", "story", "түүх", "үлгэр")
    return any(k in lower for k in keys)


def _is_report_request(text: str) -> bool:
    lower = text.lower()
    keys = ("тайлан", "report", "дүгнэлт гарга")
    return any(k in lower for k in keys)


def _is_text_edit_request(text: str) -> bool:
    lower = text.lower()
    keys = ("текст", "найруул", "засаад", "засч", "proofread", "edit this text")
    return any(k in lower for k in keys)


def _is_slide_request(text: str) -> bool:
    lower = text.lower()
    keys = ("slide", "слайд", "ppt", "presentation", "танилцуулга")
    return any(k in lower for k in keys)


def _is_structure_request(text: str) -> bool:
    lower = text.lower()
    keys = ("зохиомж", "outline", "бүтэц", "structure")
    return any(k in lower for k in keys)


def _is_full_request(text: str) -> bool:
    lower = text.lower()
    keys = ("бүрэн", "full", "дэлгэрэнгүй", "long", "дэлгэрүүл")
    return any(k in lower for k in keys)


def _is_ops_request(text: str) -> bool:
    lower = text.lower()
    ops_keys = (
        "readiness",
        "optimization",
        "integrat",
        "security pressure",
        "операц",
        "шийдвэр",
        "deployment",
        "sla",
        "sre",
        "incident",
        "runbook",
        "postmortem",
    )
    return any(k in lower for k in ops_keys)


def _detect_capability_action(text: str) -> str | None:
    lower = text.lower()
    if ("зураг" in lower or "image" in lower) and ("үүсг" in lower or "бүтээ" in lower or "generate" in lower or "хий" in lower):
        return "image.generate"
    if ("зураг" in lower or "image" in lower) and ("зас" in lower or "янз" in lower or "edit" in lower):
        return "image.edit"
    if ("зураг" in lower or "image" in lower) and ("хай" in lower or "find" in lower or "search" in lower):
        return "image.search"
    if ("бичлэг" in lower or "video" in lower) and ("үүсг" in lower or "бүтээ" in lower or "generate" in lower):
        return "video.generate"
    if ("бичлэг" in lower or "video" in lower) and ("зас" in lower or "янз" in lower or "өөрчл" in lower or "edit" in lower):
        return "video.edit"
    if ("бичлэг" in lower or "video" in lower) and ("хай" in lower or "find" in lower or "search" in lower):
        return "video.search"
    if ("код" in lower or "code" in lower) and ("зас" in lower or "сайжруул" in lower or "edit" in lower or "refactor" in lower):
        return "code.edit"
    if ("код" in lower or "code" in lower) and ("бич" in lower or "хөгжүүл" in lower or "generate" in lower or "write" in lower):
        return "code.generate"
    return None


def _call_creative_ai(message: str, history: list[dict[str, str]]) -> str:
    text = message.strip()
    lower = text.lower()
    full = _is_full_request(text)
    line_count = _requested_line_count(text, default=12)

    direct = _quick_direct_answer(text)
    if direct:
        return direct

    detected = _detect_capability_action(text)
    if detected:
        payload: dict[str, Any] = {"prompt": text, "query": text, "instruction": text}
        if detected in {"code.generate", "code.edit"}:
            payload["language"] = "python"
            if detected == "code.edit":
                payload["code"] = "print('sample code to improve')"
        result = CAP_ENGINE.execute(detected, payload)
        if result.get("status") == "success":
            output = result.get("output") or {}
            if "code" in output:
                return str(output.get("code"))
            if "improved_code" in output:
                return str(output.get("improved_code"))
            if "summary" in output:
                return str(output.get("summary"))
            return json.dumps(result, ensure_ascii=False, indent=2)
        return f"Capability execution failed: {result.get('message', 'unknown error')}"

    if ("код" in lower or "code" in lower) and ("тооны машин" in lower or "calculator" in lower):
        return _compose_calculator_code()

    if "үлгэр" in lower or "story" in lower:
        return _compose_kids_tale(_topic_from_text(text))

    if _is_story_request(text):
        topic = _topic_from_text(text)
        return _compose_story(topic, full=full)

    if _is_report_request(text):
        topic = _topic_from_text(text)
        return _compose_report(topic, full=full)

    if _is_text_edit_request(text):
        return _compose_text_edit_result(text, full=full)

    if _is_slide_request(text):
        topic = _topic_from_text(text)
        return _compose_slide_outline(topic, full=full)

    if _is_structure_request(text):
        topic = _topic_from_text(text)
        return _compose_structure(topic, full=full)

    if _is_song_request(text):
        topic = _topic_from_text(text)
        return _compose_song_lyrics(topic, line_count=line_count)

    if _is_poem_request(text):
        topic = _topic_from_text(text)
        return _compose_poem_lines(topic, line_count=line_count if line_count else 4)

    if "юу хийж чад" in lower or "what can you do" in lower:
        return (
            "Би чат-бүтээлч AI горимоор ажиллаж байна.\n"
            "- Шүлэг, өгүүллэг, пост, тайлбар бичнэ\n"
            "- И-мэйл, CV, танилцуулгын текст засна\n"
            "- Санаа гаргах, товч/дэлгэрэнгүй тайлбарлах\n"
            "- Монгол хэлээр найруулж, өнгө аясыг сонгож бичнэ"
        )

    last_user = ""
    for item in reversed(history):
        if item.get("role") == "user":
            last_user = str(item.get("content", "")).strip()
            break

    context_line = ""
    if last_user:
        context_line = f"Өмнөх асуулттай холбож: \"{last_user[:80]}\".\n"

    return (
        "Ойлголоо.\n"
        f"{context_line}"
        f"Асуултын гол сэдэв: {text}\n"
        "Би үүнийг өндөр чанартайгаар: 1) товч хариулт, 2) үндэслэл, 3) жишээ, 4) дараагийн алхам бүтэцтэй гаргаж өгч чадна.\n"
        "Хэрвээ хүсвэл яг одоо энэ форматаар бүрэн хариултаа гаргая."
    )


def _quick_direct_answer(text: str) -> str:
    """Direct, concise answers for common question forms when LLM is unavailable."""
    raw = text.strip()
    if not raw:
        return ""

    lower = raw.lower()
    normalized = re.sub(r"\s+", " ", lower).strip()

    # Known mini knowledge base for common everyday asks.
    known_topics = {
        "анимэ": "Анимэ бол Япон анимацийн урлагийн хэлбэр. TV цуврал, кино, OVA зэрэг төрөлтэй, дүрийн хөгжил ба өгүүлэмж нь ихэвчлэн гүн байдаг.",
        "anime": "Anime is a Japanese animation medium with distinct visual style and storytelling across series, films, and OVAs.",
        "python": "Python бол уншихад хялбар синтакстай, веб, дата, AI, автоматжуулалт зэрэгт өргөн хэрэглэгддэг програмчлалын хэл.",
        "javascript": "JavaScript бол вебийн интерактив логикийг ажиллуулдаг үндсэн хэл бөгөөд frontend болон backend (Node.js) дээр ашиглагддаг.",
        "ai": "AI бол өгөгдлөөс суралцаж, шийдвэр гаргах эсвэл контент үүсгэх чадвартай алгоритм, загваруудын цогц ойлголт.",
        "хиймэл оюун": "Хиймэл оюун (AI) бол машинд суралцах, таамаглах, хэл ойлгох, контент үүсгэх зэрэг чадвар олгодог технологи.",
    }

    def _extract_topic_from_known_question(s: str) -> str:
        patterns = [
            r"(.+?)\s+гэж\s+мэдэх\s+үү\??$",
            r"(.+?)\s+мэдэх\s+үү\??$",
            r"do you know\s+(.+?)\??$",
            r"can you explain\s+(.+?)\??$",
        ]
        for pat in patterns:
            m = re.search(pat, s)
            if m:
                topic_raw = m.group(1).strip(" \t\n\r\"'.,!?")
                topic_raw = re.sub(r"^(чи|та|таны|your)\s+", "", topic_raw).strip()
                return topic_raw
        return ""

    def _find_known_topic(s: str) -> str:
        for topic in known_topics:
            if topic in s:
                return topic
        return ""

    # Form: "X гэж мэдэх үү?" / "Do you know X?"
    topic = _extract_topic_from_known_question(normalized)
    if not topic and ("мэдэх үү" in normalized or "do you know" in normalized):
        topic = _find_known_topic(normalized)
    if topic:
        if topic in known_topics:
            return f"Тийм, мэднэ. {known_topics[topic]}"
        return (
            f"Тийм, {topic} талаар тусалж чадна. "
            "Товч тодорхойлолт, яагаад чухал, бодит жишээ, хэрэгжүүлэх алхмаар тайлбарлаж өгч чадна."
        )

    # Form: "X юу вэ?" / "What is X?"
    what_match = re.search(r"(.+?)\s+юу\s+вэ\??$", normalized) or re.search(r"what is\s+(.+?)\??$", normalized)
    if what_match:
        topic = what_match.group(1).strip(" \t\n\r\"'.,!?")
        if topic in known_topics:
            return known_topics[topic]
        embedded = _find_known_topic(topic)
        if embedded:
            return known_topics[embedded]
        return (
            f"{topic.title()} гэдэг нь тодорхой салбарын ойлголт/нэр томъёо юм.\n"
            "Товч утга: тухайн зүйлийн гол зорилго, үүргийг тодорхойлдог.\n"
            "Яагаад чухал: зөв ойлголт нь буруу шийдвэрээс сэргийлж, хэрэгжүүлэлтийг хурдасгадаг.\n"
            "Жишээ: бодит хэрэглээний нэг тохиолдол дээр тайлбарлавал хурдан ойлгогдоно.\n"
            "Хүсвэл яг аль салбарт (IT, бизнес, боловсрол гэх мэт) хэрэглэхийг заагаад нарийвчилж өгье."
        )

    # Form: "Яаж ..." / "How to ..."
    if " яаж " in f" {normalized} " or normalized.startswith("яаж") or normalized.startswith("how to"):
        return (
            "Ингэж хийнэ:\n"
            "1) Зорилгоо 1 өгүүлбэрээр тодорхойл\n"
            "2) Орц/нөхцөлөө жагсаа\n"
            "3) 3-5 алхамтай төлөвлөгөө гарга\n"
            "4) Нэг жижиг жишээгээр шалга\n"
            "5) Алдаагаа засаад дараагийн хувилбар руу шилж"
        )

    return ""


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _derive_capability_id(text: str) -> str:
    cleaned = text.lower()
    if "integration" in cleaned or "integration" in cleaned:
        return "integration"
    if "optimiz" in cleaned or "scale" in cleaned:
        return "optimization"
    if "security" in cleaned or "secure" in cleaned:
        return "security"
    return "chatbot_core"


def _derive_signal_from_text(text: str, history: list[dict[str, str]]) -> dict[str, Any]:
    lower = text.lower()
    demand_words = ("өсөлт", "growth", "traffic", "scale", "ачаалал")
    risk_words = ("алдаа", "error", "fail", "risk", "доголдол")
    security_words = ("security", "attack", "threat", "аюулгүй", "эмзэг")

    demand_growth = 0.35 + (0.35 if any(w in lower for w in demand_words) else 0.0)
    volatility = 0.25 + (0.4 if any(w in lower for w in risk_words) else 0.0)
    security_pressure = 0.2 + (0.45 if any(w in lower for w in security_words) else 0.0)

    turns = len([m for m in history if m.get("role") in {"user", "assistant"}])
    reliability = 0.85 - min(0.25, turns * 0.01)

    return {
        "capability_id": _derive_capability_id(text),
        "tenant_id": "default",
        "volatility_score": round(_clamp01(volatility), 4),
        "demand_growth_score": round(_clamp01(demand_growth), 4),
        "reliability_score": round(_clamp01(reliability), 4),
        "security_pressure_score": round(_clamp01(security_pressure), 4),
    }


def _format_internal_reply(plan_payload: dict[str, Any], user_text: str) -> str:
    plan = dict(plan_payload.get("plan") or {})
    readiness = float(plan.get("readiness_score", 0.0))
    actions = list(plan.get("actions") or [])
    simulation = dict(plan.get("simulation_summary") or {})

    lines = [
        "Таны өөрийн AI (auto_nextgen) ажиллалаа.",
        f"Асуулт: {user_text.strip()}",
        f"Readiness score: {readiness:.2f}",
    ]

    if actions:
        lines.append("Санал болгосон үйлдлүүд:")
        for idx, action in enumerate(actions[:3], start=1):
            action_type = str(action.get("action_type", "unknown"))
            reason = str(action.get("reason", "-"))
            confidence = float(action.get("confidence", 0.0))
            lines.append(f"{idx}) {action_type} (confidence={confidence:.2f}) - {reason}")
    else:
        lines.append("Одоогийн дохиогоор шууд үйлдэл шаардахгүй байна.")

    projected = simulation.get("projected_readiness")
    if projected is not None:
        lines.append(f"Projected readiness: {projected}")

    return "\n".join(lines)


def _call_internal_ai(message: str, history: list[dict[str, str]]) -> str:
    payload = _derive_signal_from_text(message, history)
    result = NEXT_API.run_cycle(payload)
    if result.get("status") != "success":
        return f"Дотоод AI алдаа: {result.get('message', 'unknown error')}"
    return _format_internal_reply(result, message)


def _load_env_file() -> None:
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                k = key.strip()
                v = value.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        # Keep server alive even if .env parsing fails.
        return


def _normalize_history(history: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(history, list):
        return normalized
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant", "system"} and content:
            normalized.append({"role": role, "content": content})
    return normalized


def _voice_status() -> Dict[str, Any]:
    status = _voice_status_impl()
    try:
        status["memory"] = MEMORY.stats()
    except Exception as exc:
        status["memory"] = {"error": str(exc)}
    return status


def _transcribe_audio(audio_b64: str, mime: str) -> Tuple[str | None, str | None]:
    """Try ASR backends in order: Whisper API (if key) -> Vosk (offline)."""
    audio_bytes = base64.b64decode(audio_b64.split(",", 1)[-1] + "===")
    ext = ".webm"
    if "wav" in mime:
        ext = ".wav"
    elif "mp3" in mime or "mpeg" in mime:
        ext = ".mp3"
    elif "ogg" in mime:
        ext = ".ogg"
    elif "m4a" in mime or "mp4" in mime:
        ext = ".m4a"

    tmp_dir = ARTIFACT_ROOT / "Voice" / "incoming"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"recording_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
    tmp_path.write_bytes(audio_bytes)

    # 1) Whisper API
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            return _transcribe_whisper_api(tmp_path, api_key), "whisper-api"
        except Exception:
            pass

    # 2) Vosk offline
    try:
        import vosk  # noqa: F401
        return _transcribe_vosk(tmp_path), "vosk"
    except Exception:
        pass

    return None, None


def _transcribe_whisper_api(path: Path, api_key: str) -> str:
    """Multipart upload to OpenAI's /v1/audio/transcriptions endpoint."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("WHISPER_MODEL", "whisper-1")
    boundary = "----uulenFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="model"\r\n\r\n'
    body += model.encode() + b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = url_request.Request(
        url=f"{base_url}/audio/transcriptions",
        data=bytes(body),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with url_request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data.get("text", "")).strip()


def _transcribe_vosk(path: Path) -> str:
    """Offline ASR via Vosk. Requires user to download a model to ./vosk_model/."""
    import wave
    import vosk

    model_dir = os.getenv("VOSK_MODEL_PATH", str(BASE_DIR / "vosk_model"))
    model = vosk.Model(model_dir)
    # vosk requires PCM16 mono WAV; convert if needed via wave (only works if input already WAV)
    target = path
    if path.suffix.lower() != ".wav":
        # naive fallback: ffmpeg if available
        import shutil
        import subprocess
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg required to convert non-WAV input for Vosk")
        target = path.with_suffix(".wav")
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(target)],
            check=True, capture_output=True,
        )
    with wave.open(str(target), "rb") as wf:
        rec = vosk.KaldiRecognizer(model, wf.getframerate())
        out_parts: list[str] = []
        while True:
            chunk = wf.readframes(4000)
            if not chunk:
                break
            if rec.AcceptWaveform(chunk):
                seg = json.loads(rec.Result()).get("text", "")
                if seg:
                    out_parts.append(seg)
        final = json.loads(rec.FinalResult()).get("text", "")
        if final:
            out_parts.append(final)
    return " ".join(out_parts).strip()


def _call_openai(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "temperature": 0.7,
    }
    req = url_request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with url_request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI response has no choices")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("OpenAI response content is empty")
    return str(content)


def _call_ollama(messages: list[dict[str, str]]) -> str:
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if not model:
        raise RuntimeError("OLLAMA_MODEL is missing")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "stream": False,
    }
    req = url_request.Request(
        url=f"{base_url}/api/chat",
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with url_request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama response content is empty")
    return str(content)


def _execute_voice_conversation_turn(
    data: Dict[str, Any],
    route: str,
    progress_callback: Callable[[str, Dict[str, Any]], None] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    def _emit(stage: str, payload: Dict[str, Any] | None = None) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage, dict(payload or {}))
        except Exception:
            return

    turn_started = time.perf_counter()
    history = _normalize_history(data.get("history", []))
    text_input = str(data.get("text", "")).strip()
    audio_b64 = str(data.get("audio_base64", "")).strip()
    mime = str(data.get("mime_type", "audio/webm")).strip() or "audio/webm"
    voice = str(data.get("voice", "female-calm")).strip() or "female-calm"
    language = str(data.get("language", "mn-MN")).strip() or "mn-MN"
    preferred_persona, _ = _resolve_preferred_persona(data if isinstance(data, dict) else None)

    transcript: str | None = None
    asr_engine: str | None = None
    asr_elapsed_ms = 0
    chat_elapsed_ms = 0
    tts_elapsed_ms = 0
    stage_status = {"asr": "skipped", "chat": "pending", "tts": "pending"}
    if not text_input and not audio_b64:
        _emit("validation.error", {"reason": "missing_input"})
        return 400, _with_assistant_contract(
            {"status": "error", "message": "text or audio_base64 is required"},
            message="Voice conversation input missing",
            audit_id="voice-conversation-turn-invalid",
            route=route,
        )

    if audio_b64:
        _emit("asr.started", {"mime_type": mime})
        asr_started = time.perf_counter()
        try:
            transcript, asr_engine = _transcribe_audio(audio_b64, mime)
        except Exception as exc:
            stage_status["asr"] = "error"
            _emit("asr.error", {"message": str(exc)})
            return 500, _with_assistant_contract(
                {"status": "error", "message": f"transcription failed: {exc}"},
                message="Voice conversation transcription failed",
                audit_id="voice-conversation-turn-transcribe-error",
                route=route,
            )
        asr_elapsed_ms = int((time.perf_counter() - asr_started) * 1000)
        if transcript is None and not text_input:
            stage_status["asr"] = "unavailable"
            _emit("asr.unavailable", {"elapsed_ms": asr_elapsed_ms})
            return 503, _with_assistant_contract(
                {
                    "status": "error",
                    "message": "No ASR backend available. Provide text directly or enable Whisper/Vosk.",
                },
                message="Voice conversation ASR unavailable",
                audit_id="voice-conversation-turn-asr-unavailable",
                route=route,
            )
        stage_status["asr"] = "success"
        _emit("asr.completed", {"elapsed_ms": asr_elapsed_ms, "engine": asr_engine})
    else:
        _emit("asr.skipped", {"reason": "text_input_present"})

    user_text = text_input or str(transcript or "").strip()
    _emit("chat.started", {"user_text_length": len(user_text)})
    chat_started = time.perf_counter()
    reply = _generate_chat_reply(user_text, history, persona=preferred_persona)
    chat_elapsed_ms = int((time.perf_counter() - chat_started) * 1000)
    stage_status["chat"] = "success"
    _emit("chat.completed", {"elapsed_ms": chat_elapsed_ms, "reply_length": len(reply)})

    _emit("tts.started", {"voice": voice, "language": language})
    tts_started = time.perf_counter()
    tts_result = CAP_ENGINE.execute("audio.speech.generate", {
        "text": reply[:2500],
        "voice": voice,
        "language": language,
        "voice_conversation": True,
    })
    tts_elapsed_ms = int((time.perf_counter() - tts_started) * 1000)
    tts_output = dict(tts_result.get("output") or {})
    tts_path = str(tts_output.get("file_path", "")).strip()
    tts_mime = str(tts_output.get("mime_type", "audio/wav")).strip() or "audio/wav"
    tts_audio_b64 = None
    if tts_path and Path(tts_path).exists() and tts_mime.startswith("audio/"):
        try:
            tts_audio_b64 = base64.b64encode(Path(tts_path).read_bytes()).decode("ascii")
        except Exception:
            tts_audio_b64 = None
    stage_status["tts"] = "success" if str(tts_result.get("status", "")) == "success" else "error"
    _emit(
        "tts.completed",
        {
            "elapsed_ms": tts_elapsed_ms,
            "status": stage_status["tts"],
            "engine": tts_output.get("engine"),
        },
    )
    total_elapsed_ms = int((time.perf_counter() - turn_started) * 1000)
    _emit("turn.completed", {"elapsed_ms": total_elapsed_ms})

    return 200, _with_assistant_contract(
        {
            "status": "success",
            "mode": "voice_conversation_turn",
            "persona": preferred_persona,
            "transcript": transcript,
            "asr_engine": asr_engine,
            "user_text": user_text,
            "reply": reply,
            "stage_status": stage_status,
            "latency_ms": {
                "asr": asr_elapsed_ms,
                "chat": chat_elapsed_ms,
                "tts": tts_elapsed_ms,
                "total": total_elapsed_ms,
            },
            "tts": {
                "status": tts_result.get("status", "error"),
                "engine": tts_output.get("engine"),
                "file_path": tts_path or None,
                "mime_type": tts_mime,
                "audio_base64": tts_audio_b64,
                "message": tts_result.get("message"),
            },
        },
        message="Voice conversation turn completed",
        audit_id="voice-conversation-turn",
        route=route,
    )


def _build_context_messages(text: str, client_history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Combine semantic recall + persisted recent history + client history."""
    persisted = MEMORY.recent_messages(limit=16)
    persisted_msgs = [{"role": r["role"], "content": r["content"]} for r in persisted if r["role"] in {"user", "assistant"}]

    recalled = MEMORY.recall(text, k=3)
    recall_block = ""
    if recalled:
        joined = "\n".join(f"- ({r['iso']}) {r['role']}: {r['content'][:180]}" for r in recalled)
        recall_block = f"[MEMORY — semantically related past turns]\n{joined}"

    facts = MEMORY.all_facts()
    facts_block = ""
    if facts:
        f_joined = "\n".join(f"- {f['key']}: {f['value']}" for f in facts[:20])
        facts_block = f"[USER FACTS]\n{f_joined}"

    messages: list[dict[str, str]] = []
    if facts_block or recall_block:
        ctx = "\n\n".join(b for b in (facts_block, recall_block) if b)
        messages.append({"role": "system", "content": ctx})

    # Avoid duplicate consecutive content: prefer persisted, drop client overlap
    seen = {(m["role"], m["content"]) for m in persisted_msgs}
    extra_client = [m for m in client_history if (m["role"], m["content"]) not in seen]
    messages.extend(persisted_msgs)
    messages.extend(extra_client)
    messages.append({"role": "user", "content": text})
    return messages


def _maybe_extract_fact(text: str) -> tuple[str, str] | None:
    """Lightweight fact extractor for self-reported user info."""
    t = text.strip()
    lower = t.lower()
    # Skip questions — "Миний нэр хэн бэ?" must not be captured as a declaration.
    # Includes common typos: "хн бэ", "юу вэ" → "юу в", etc.
    question_markers = [
        "?", "хэн бэ", "хн бэ", "юу вэ", "юу бэ", "юу в", "хэн билээ", "хн билээ",
        "what is", "what's", "who is", "who's", "do you", "remember",
    ]
    if any(q in lower for q in question_markers):
        return None
    patterns = [
        (r"миний нэр(?:\s+бол)?\s+([\wа-яё\- ]{2,40})", "name"),
        (r"my name is\s+([\w\- ]{2,40})", "name"),
        (r"намайг\s+([\wа-яё\- ]{2,40})\s+гэдэг", "name"),
        (r"би\s+([\wа-яё\- ]{2,40})\s+(?:дээр|компанид|байгууллагад)\s+ажилла", "workplace"),
        (r"i (?:work|am) at\s+([\w\- ]{2,40})", "workplace"),
        (r"миний дуртай (?:хоол|зүйл|өнгө|кино)\s+бол\s+([\wа-яё\- ]{2,40})", "preference"),
    ]
    for pat, key in patterns:
        m = re.search(pat, lower)
        if m:
            value = m.group(1).strip().strip(".,!?")
            if value:
                return key, value
    return None


def _uulen_fast_reply(text: str) -> str:
    """Hard-coded Үүлэн identity + memory-fact recall paths that work without any LLM."""
    t = text.strip().lower()
    if not t:
        return ""

    # Normalize: strip punctuation, collapse whitespace — helps fuzzy matching
    t_norm = re.sub(r"[^\wа-яёөүҮӨ\s]+", " ", t, flags=re.UNICODE)
    t_norm = re.sub(r"\s+", " ", t_norm).strip()

    def _fuzzy_match(needle: str, candidates: list, threshold: float = 0.78) -> bool:
        """Return True if needle is similar enough to ANY candidate (typo-tolerant).

        Uses difflib.SequenceMatcher — catches typos like 'сйан уу', 'санй уу',
        'сайна уу', 'хеллоо', 'хи' etc. Skips very short noise (<3 chars).
        Substring matching only applies to candidates with >=5 chars to avoid
        false positives like 'хи' matching inside 'зохиогоод'.
        """
        if len(needle) < 3:
            return needle in candidates
        for c in candidates:
            # Substring match only when candidate is reasonably long
            if len(c) >= 5 and (c in needle or needle in c):
                return True
            # Skip if lengths are drastically different (avoid spurious matches)
            longer = max(len(needle), len(c))
            if longer and abs(len(needle) - len(c)) > longer * 0.5:
                continue
            # Compare full strings
            if difflib.SequenceMatcher(None, needle, c).ratio() >= threshold:
                return True
            # Also compare space-stripped forms ("сайн уу" vs "сайнуу")
            n2 = needle.replace(" ", "")
            c2 = c.replace(" ", "")
            if n2 and c2:
                longer2 = max(len(n2), len(c2))
                if longer2 and abs(len(n2) - len(c2)) <= longer2 * 0.5:
                    if difflib.SequenceMatcher(None, n2, c2).ratio() >= threshold:
                        return True
        return False

    def _facts_dict() -> dict:
        try:
            raw = MEMORY.all_facts() or []
        except Exception:
            return {}
        if isinstance(raw, dict):
            return raw
        out: dict = {}
        for it in raw:
            if isinstance(it, dict) and "key" in it and "value" in it:
                out.setdefault(str(it["key"]), it["value"])
        return out

    def _lookup_name(d: dict):
        for k in ("name", "user_name", "нэр"):
            v = d.get(k)
            if v:
                return v
        for k, v in d.items():
            if "name" in str(k).lower() or "нэр" in str(k).lower():
                if v:
                    return v
        return None

    # --- Identity: "Чи хэн бэ?", "Танай нэр?", "What's your name?", "Who are you?" ---
    identity_patterns = [
        "чи хэн бэ", "та хэн бэ", "чиний нэр", "таны нэр", "чамайг хэн",
        "what is your name", "what's your name", "who are you", "your name",
        "хэн бэ чи", "та ямар ai", "чи ямар ai",
    ]
    if any(p in t_norm for p in identity_patterns) or _fuzzy_match(t_norm, identity_patterns, 0.82):
        return (
            "Би Үүлэн — таны хувийн AI туслах. Анарын хувийн туслахаар "
            "зохиогдсон, та надтай Монголоор чөлөөтэй ярилцаж болно. Юу хийж өгөх вэ?"
        )

    # --- Greetings (typo-tolerant) ---
    greetings = [
        "сайн уу", "сайн байна уу", "сайна уу", "сайнуу", "сайн байнуу",
        "өглөөний мэнд", "оройн мэнд", "мэндчилгээ",
        "hi", "hello", "hey", "hola", "хеллоу", "хелоу", "хай", "хи",
    ]
    is_greeting = False
    # Direct check first (cheap)
    if t_norm in greetings:
        is_greeting = True
    elif any(t_norm.startswith(g + " ") or t_norm.startswith(g + ",") for g in greetings):
        is_greeting = True
    else:
        # Fuzzy check — catches "сйан уу", "санй уу", "сайна уу", "хеллоо" etc.
        # Tight guards: ≤ 18 chars AND ≤ 3 words — avoids matching long sentences
        # like "үлгэр зохиогоод өг" as a greeting.
        word_count = len(t_norm.split())
        if len(t_norm) <= 18 and word_count <= 3 and _fuzzy_match(t_norm, greetings, 0.78):
            is_greeting = True
    if is_greeting:
        # Try to personalize from memory
        try:
            name = _lookup_name(_facts_dict())
            nm = str(name or '').strip().lower()
            if nm and not any(b in nm for b in ("хэн", "хн ", "юу", "бэ", "билээ", "what", "who", "?")):
                return f"Сайн байна уу, {name}! Үүлэн энд байна. Юугаар туслах вэ?"
        except Exception:
            pass
        return "Сайн байна уу! Үүлэн энд байна. Юугаар туслах вэ?"

    # --- Name recall: "Миний нэр хэн бэ?" / "What is my name?" ---
    name_recall_patterns = [
        "миний нэр хэн", "миний нэр юу", "намайг хэн гэдэг", "намайг юу гэдэг",
        "what is my name", "what's my name", "who am i", "do you remember my name",
        "миний нэрийг сана", "миний нэрийг мэдэх",
    ]
    if any(p in t_norm for p in name_recall_patterns) or (len(t_norm) <= 30 and _fuzzy_match(t_norm, name_recall_patterns, 0.82)):
        try:
            name = _lookup_name(_facts_dict())
            nm = str(name or '').strip().lower()
            if nm and not any(b in nm for b in ("хэн", "хн ", "юу", "бэ", "билээ", "what", "who", "?")):
                return f"Таны нэр {name}. Би санаж байна."
        except Exception:
            pass
        return "Уучлаарай, таны нэрийг хараахан санаагүй байна. \"Миний нэр …\" гэж хэлвэл би сурч авна."

    # --- General fact recall: "Чи намайг сайн уу?" / "Миний тухай юу мэдэх вэ?" ---
    about_me = ["миний тухай юу мэдэх", "намайг мэдэх үү", "миний талаар юу мэдэх", "what do you know about me"]
    if any(p in t for p in about_me):
        try:
            facts = _facts_dict()
            if facts:
                parts = [f"- {k}: {v}" for k, v in list(facts.items())[:8]]
                return "Таны тухай миний санаж байгаа зүйлс:\n" + "\n".join(parts)
        except Exception:
            pass
        return "Хараахан танай тухай хадгалсан баримт алга. Та нэр, дуртай зүйл, ажил гэх мэтийг хэлвэл санаж авна."

    return ""


def _extract_folder_path_from_text(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""

    def _clean_candidate(value: str) -> str:
        candidate = str(value or "").strip().strip(".,;)")
        candidate = re.sub(
            r"\s+(?:энэ\s+folder|this\s+folder|current\s+folder|folder\s+доторх|доторх\s+бүх|all\s+images|анализ|шинжил|суралц|хөгжүүл|learn|train|improve)\b.*$",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        candidate = re.sub(
            r"\s+(?:limit|max|quality|dry_run|session_id)\s*[:=]?.*$",
            "",
            candidate,
            flags=re.IGNORECASE,
        ).strip()

        # If free text is appended after a valid path, trim from the end until an existing path is found.
        probe = candidate
        for _ in range(10):
            if not probe:
                break
            try:
                if Path(probe).expanduser().exists():
                    return probe
            except Exception:
                pass
            if " " not in probe:
                break
            probe = probe.rsplit(" ", 1)[0].rstrip("\\/").strip()
        return candidate

    quoted = re.findall(r'["“](.+?)["”]', raw)
    for item in quoted:
        candidate = _clean_candidate(item)
        if candidate and ("\\" in candidate or "/" in candidate):
            return candidate

    windows = re.search(r"([A-Za-z]:\\[^\n\r\t\"'<>|]+)", raw)
    if windows:
        return _clean_candidate(str(windows.group(1)))

    unix_like = re.search(r"((?:\./|\.\./|/)[^\n\r\t\"'<>|]+)", raw)
    if unix_like:
        return _clean_candidate(str(unix_like.group(1)))

    marker_patterns = [
        r"folder\s*[:=]\s*([^\n\r]+)",
        r"фолдер\s*[:=]\s*([^\n\r]+)",
        r"хавтас\s*[:=]\s*([^\n\r]+)",
        r"директори\s*[:=]\s*([^\n\r]+)",
    ]
    for pattern in marker_patterns:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if not m:
            continue
        candidate = _clean_candidate(str(m.group(1)))
        if candidate:
            return candidate

    return ""


def _parse_chat_image_learning_command(text: str) -> Dict[str, Any] | None:
    lower = str(text or "").lower()
    has_learn = any(token in lower for token in ("суралц", "сурга", "learn", "training", "train"))
    has_improve = any(token in lower for token in ("хөгжүүл", "сайжруул", "improve", "optimiz"))
    has_analyze = any(token in lower for token in ("анализ", "шинжил", "analyz", "analysis"))
    has_folder = any(token in lower for token in ("folder", "фолдер", "хавтас", "directory", "директори", "path", "зам"))
    has_image = any(token in lower for token in ("зураг", "image", "images", "photo", "photos"))

    has_learning_intent = has_learn or has_improve or (has_analyze and has_image)
    if not has_learning_intent or not (has_folder or has_image):
        return None

    path_text = _extract_folder_path_from_text(text)
    this_folder_requested = any(token in lower for token in ("энэ folder", "this folder", "current folder", "энэ хавтас"))

    limit = 200
    if any(token in lower for token in ("бүх", "all images", "all зураг", "бүгд")):
        limit = 5000

    m_limit = re.search(r"(?:limit|max)\s*[:=]?\s*(\d{1,5})", lower)
    if m_limit:
        try:
            limit = max(1, min(int(m_limit.group(1)), 5000))
        except Exception:
            limit = 200

    if not path_text:
        if this_folder_requested:
            return {
                "kind": "image_learn_command",
                "directories": [str(Path.cwd().resolve())],
                "limit": limit,
                "quality_mode": "high",
                "dry_run": False,
            }
        return {
            "kind": "image_learn_missing_path",
            "error": (
                "Folder path олдсонгүй. Дараах хэлбэрээр бичнэ үү:\n"
                "- `энэ folder-оос суралц: C:\\path\\to\\images`\n"
                "- `learn images from folder \"C:\\path\\to\\images\"`"
            ),
        }

    quality_mode = "high"
    if "quality=fast" in lower or "quality fast" in lower or "хурдан" in lower:
        quality_mode = "fast"

    dry_run = "dry_run=true" in lower or "dry run" in lower or "туршилт" in lower

    return {
        "kind": "image_learn_command",
        "directories": [path_text],
        "limit": limit,
        "quality_mode": quality_mode,
        "dry_run": dry_run,
    }


def _parse_chat_style_image_generate_command(text: str) -> Dict[str, Any] | None:
    lower = str(text or "").lower()
    if not lower.strip():
        return None

    style_key = ""
    style_label = ""
    style_hint = ""

    def _first_index(tokens: tuple[str, ...]) -> int:
        positions = [lower.find(token) for token in tokens]
        positions = [pos for pos in positions if pos >= 0]
        return min(positions) if positions else -1

    style_candidates: List[tuple[int, str, str, str]] = []

    def _add_style_candidate(tokens: tuple[str, ...], key: str, label: str, hint: str) -> None:
        idx = _first_index(tokens)
        if idx >= 0:
            style_candidates.append((idx, key, label, hint))

    _add_style_candidate(
        ("манга", "manga"),
        "japan_manga",
        "Japan -> Manga",
        "manga style",
    )
    _add_style_candidate(
        ("аниме", "anime", "japan", "япон"),
        "japan_anime",
        "Japan -> Anime",
        "anime style",
    )
    _add_style_candidate(
        ("donghua", "дунхуа", "хятад", "china", "chinese"),
        "china_donghua",
        "China -> Donghua",
        "donghua style",
    )
    _add_style_candidate(
        ("aeni", "эни", "солонгос", "korea", "korean"),
        "south_korea_aeni",
        "South Korea -> Aeni",
        "aeni style",
    )
    _add_style_candidate(
        ("cinematic", "кино", "film look", "movie"),
        "cinematic",
        "Cinematic",
        "cinematic lighting",
    )
    _add_style_candidate(
        ("realist", "realistic", "photoreal", "реалист", "бодит"),
        "realist",
        "Realist",
        "photorealistic style",
    )
    _add_style_candidate(
        ("animation", "анимац"),
        "global_animation",
        "Global Animation",
        "animation style",
    )

    if style_candidates:
        style_candidates.sort(key=lambda item: item[0])
        _, style_key, style_label, style_hint = style_candidates[0]

    if not style_key:
        return None

    has_generation_intent = any(
        token in lower
        for token in (
            "зураг",
            "image",
            "generate",
            "үүсг",
            "хий",
            "бүтээ",
            "стилиэр",
            "style",
        )
    )
    if not has_generation_intent:
        return None

    gender = "auto"
    if "эмэгтэй" in lower or "female" in lower or "woman" in lower:
        gender = "female"
    elif "эрэгтэй" in lower or "male" in lower or "man" in lower:
        gender = "male"

    wants_full_body = any(token in lower for token in ("бүтэн бие", "бүх бие", "full body", "head to toe", "full-length"))

    trait_parts: List[str] = []

    def _add_trait(value: str) -> None:
        val = str(value or "").strip()
        if val and val not in trait_parts:
            trait_parts.append(val)

    if "өндөр" in lower or "tall" in lower:
        _add_trait("tall height")
    if "нам" in lower or "short" in lower:
        _add_trait("short height")
    if "тарган" in lower or "plus size" in lower or "curvy" in lower:
        _add_trait("plus-size body")
    if "туранхай" in lower or "slim" in lower or "lean" in lower:
        _add_trait("slim body")

    if "үсгүй" in lower or "bald" in lower:
        _add_trait("bald head")
    if "богино үстэй" in lower or "short hair" in lower:
        _add_trait("short hair")
    if "урт үстэй" in lower or "long hair" in lower:
        _add_trait("long hair")
    if "үстэй" in lower or "with hair" in lower:
        _add_trait("natural hair")

    if "үрчлээтэй" in lower or "wrinkle" in lower:
        _add_trait("visible wrinkles")
    if "үрчлээгүй" in lower or "smooth skin" in lower:
        _add_trait("smooth skin")

    if "хөлгүй" in lower or "no legs" in lower:
        _add_trait("leg difference")
        wants_full_body = True
    elif "нэг хөлтэй" in lower or "one leg" in lower:
        _add_trait("one leg")
        wants_full_body = True
    elif "хоёр хөлтэй" in lower or "two legs" in lower:
        _add_trait("two legs")
        wants_full_body = True

    if "нэг гартай" in lower or "one arm" in lower:
        _add_trait("one arm")
        wants_full_body = True
    elif "хоёр гартай" in lower or "two arms" in lower:
        _add_trait("two arms")
        wants_full_body = True

    if "нэг нүдтэй" in lower or "one eye" in lower:
        _add_trait("one visible eye")
    elif "хоёр нүдтэй" in lower or "two eyes" in lower:
        _add_trait("two eyes")
    if "хоёр өөр өнгийн нүд" in lower or "heterochromia" in lower:
        _add_trait("heterochromia")

    if "тахир дутуу" in lower or "хөгжлийн бэрхшээл" in lower or "disabled" in lower:
        _add_trait("person with disability, respectful representation")
        wants_full_body = True

    if "хар арьстай" in lower or "dark skin" in lower:
        _add_trait("dark skin tone")
    if "шар арьстай" in lower or "yellow skin" in lower:
        _add_trait("golden/yellow undertone skin")
    if "цагаан арьстай" in lower or "fair skin" in lower or "light skin" in lower:
        _add_trait("fair skin tone")
    if "албино" in lower or "albinism" in lower:
        _add_trait("albinism")

    subject_parts: List[str] = []
    if wants_full_body:
        subject_parts.append("full body")
    if "хүүхэд" in lower or "child" in lower:
        subject_parts.append("child")
    elif "залуу" in lower or "young" in lower:
        subject_parts.append("young")
    elif "дунд нас" in lower or "middle aged" in lower:
        subject_parts.append("middle-aged")
    elif "хөгшин" in lower or "senior" in lower or "older" in lower:
        subject_parts.append("senior")

    if gender == "female":
        subject_parts.append("female")
    elif gender == "male":
        subject_parts.append("male")
    else:
        subject_parts.append("person")

    scene_prompt = str(text or "").strip()
    scene_prompt = re.sub(
        r"\b(cinematic|realist|realistic|photoreal|animation|anime|manga|donghua|aeni|style|стилиэр|зураг|image|generate|үүсг(?:эх|э)|хий(?:х|гээрэй)?)\b",
        " ",
        scene_prompt,
        flags=re.IGNORECASE,
    )
    scene_prompt = re.sub(r"\s+", " ", scene_prompt).strip(" ,.;")
    if any("\u0400" <= ch <= "\u04ff" for ch in scene_prompt):
        scene_prompt = ""
    if len(scene_prompt) > 220:
        scene_prompt = scene_prompt[:220].rsplit(" ", 1)[0].strip()
    subject_text = " ".join(part for part in subject_parts if part)
    trait_text = ", ".join(trait_parts)
    framing_hint = "full-body" if wants_full_body else "upper-body"
    prompt_parts = [subject_text]
    if trait_text:
        prompt_parts.append(", ".join(trait_parts[:3]))
    if scene_prompt:
        prompt_parts.append(scene_prompt)
    elif wants_full_body:
        prompt_parts.append("standing pose, feet visible")
    prompt = f"{', '.join(part for part in prompt_parts if part)}. {style_hint}. {framing_hint}."

    style_preset = "portrait"
    if style_key == "china_donghua":
        style_preset = "cinematic"
    elif style_key == "global_animation":
        style_preset = "auto"
    elif style_key == "cinematic":
        style_preset = "cinematic"
    elif style_key == "realist":
        style_preset = "photoreal_hq"
    if wants_full_body and style_preset == "portrait":
        style_preset = "auto"

    wants_high_quality = any(token in lower for token in ("high", "hq", "чанартай", "өндөр чанар", "ultra"))
    quality_mode = "high" if wants_high_quality else "fast"
    model_tier = "hq" if wants_high_quality else "auto"
    steps = 28 if wants_high_quality else 12
    if wants_full_body:
        width = 832 if wants_high_quality else 640
        height = 1216 if wants_high_quality else 960
    else:
        width = 1024 if wants_high_quality else 512
        height = 1024 if wants_high_quality else 512

    negative_prompt = "copyright character, logo, watermark, broken anatomy, extra limbs, extra fingers"
    if wants_full_body:
        negative_prompt = (
            f"{negative_prompt}, close-up portrait, headshot, face-only crop, cropped legs, cropped body, "
            "out-of-frame feet, out-of-frame knees"
        )

    payload = {
        "prompt": prompt,
        "quality_mode": quality_mode,
        "style_preset": style_preset,
        "model_tier": model_tier,
        "num_inference_steps": steps,
        "width": width,
        "height": height,
        "gender": gender,
        "single_subject": True,
        "anatomy_guard": True,
        "full_body": wants_full_body,
        "negative_prompt": negative_prompt,
    }

    return {
        "kind": "style_image_generate_command",
        "style_key": style_key,
        "style_label": style_label,
        "payload": payload,
    }


def _generate_chat_reply(message: str, history: list[dict[str, str]], persona: str = "anar_ai") -> str:
    text = message.strip()
    if not text:
        return "Хоосон мессеж байна. Асуултаа бичээд дахин илгээнэ үү."

    # Persist user message
    MEMORY.add_message(
        "user",
        text,
        persona=persona,
        source="chat.user",
        metadata={"history_size": len(history or [])},
    )
    fact = _maybe_extract_fact(text)
    if fact:
        MEMORY.remember_fact(*fact, source="chat.fact_extract", metadata={"persona": persona})

    reply: str = ""
    try:
        # Fast-paths that work without LLM (Үүлэн identity + remembered facts)
        fast = _uulen_fast_reply(text)
        if fast:
            reply = fast
        else:
            style_image_cmd = _parse_chat_style_image_generate_command(text)
            if style_image_cmd is not None:
                try:
                    result = CAP_ENGINE.execute("image.generate", dict(style_image_cmd.get("payload") or {}))
                    if str(result.get("status", "")).lower() == "success":
                        output = dict(result.get("output") or {})
                        file_path = str(output.get("file_path", "")).strip()
                        reply = (
                            "Style-aware зураг үүсгэлээ.\n"
                            f"- style: {style_image_cmd.get('style_label', 'custom')}\n"
                            f"- file: {file_path or '-'}"
                        )
                    else:
                        reply = f"Style-aware зураг үүсгэхэд алдаа гарлаа: {result.get('message', 'unknown error')}"
                except Exception as exc:
                    reply = f"Style-aware зураг үүсгэхэд алдаа гарлаа: {exc}"
            else:
                image_learn_cmd = _parse_chat_image_learning_command(text)
                if image_learn_cmd is not None:
                    if image_learn_cmd.get("kind") == "image_learn_missing_path":
                        reply = str(image_learn_cmd.get("error", "Folder path олдсонгүй."))
                    else:
                        try:
                            session_tag = datetime.utcnow().strftime("chat-image-learning-%Y%m%d-%H%M%S")
                            summary = _analyze_images_and_learn(
                                directories=list(image_learn_cmd.get("directories") or []),
                                limit=int(image_learn_cmd.get("limit", 200)),
                                quality_mode=str(image_learn_cmd.get("quality_mode", "high")),
                                session_id=session_tag,
                                dry_run=bool(image_learn_cmd.get("dry_run", False)),
                            )
                            defaults = dict(summary.get("recommended_tool_defaults") or {})
                            reply = (
                                "Зурагнаас суралцах ажил дууслаа.\n"
                                f"- scanned: {int(summary.get('scanned', 0))}\n"
                                f"- learned: {int(summary.get('learned', 0))}\n"
                                f"- skipped: {int(summary.get('skipped', 0))}\n"
                                f"- avg_score: {float((summary.get('score_summary') or {}).get('avg_score', 0.0)):.4f}\n"
                                "Шинэ image default:\n"
                                f"- style: {defaults.get('image_default_style_preset', 'photoreal_hq')}\n"
                                f"- model_tier: {defaults.get('image_default_model_tier', 'auto')}\n"
                                f"- size: {defaults.get('image_default_width', 1024)}x{defaults.get('image_default_height', 1024)}\n"
                                f"- steps: {defaults.get('image_default_num_inference_steps', 28)}, guidance: {defaults.get('image_default_guidance_scale', 7.5)}"
                            )
                        except Exception as exc:
                            reply = f"Folder-оос суралцах үед алдаа гарлаа: {exc}"
                else:
                    mode = os.getenv("CHAT_MODE", "creative").strip().lower()
                    has_external = bool(os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("OLLAMA_MODEL", "").strip())

                    # Үүлэн-style preference: if an LLM is configured, prefer it for natural dialog.
                    if mode == "external" or (mode in ("creative", "") and has_external and not _is_ops_request(text) and not _looks_like_template_intent(text)):
                        messages = _build_context_messages(text, history)
                        if os.getenv("OPENAI_API_KEY", "").strip():
                            reply = _call_openai(messages)
                        elif os.getenv("OLLAMA_MODEL", "").strip():
                            reply = _call_ollama(messages)
                        else:
                            reply = "External AI тохиргоо дутуу байна."
                    elif mode == "ops" or _is_ops_request(text):
                        reply = _call_internal_ai(text, history)
                    else:
                        # Creative/template composer path
                        reply = _call_creative_ai(text, history)
    except url_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        reply = f"AI provider HTTP алдаа: {exc.code}. {detail[:500]}"
    except url_error.URLError as exc:
        reply = f"AI provider руу холбогдож чадсангүй: {exc.reason}"
    except Exception as exc:
        reply = f"AI боловсруулах явцад алдаа гарлаа: {exc}"

    if reply:
        MEMORY.add_message(
            "assistant",
            reply,
            persona=persona,
            source="chat.assistant",
            metadata={"reply_length": len(reply)},
        )
    return reply


def _looks_like_template_intent(text: str) -> bool:
    """True if user wants a structured composer output (lyrics, poem, slide, etc.)."""
    try:
        return bool(
            _is_song_request(text)
            or _is_poem_request(text)
            or _is_story_request(text)
            or _is_report_request(text)
            or _is_text_edit_request(text)
            or _is_slide_request(text)
            or _is_structure_request(text)
            or _detect_capability_action(text)
        )
    except Exception:
        return False


class AppHandler(BaseHTTPRequestHandler):
    server_version = "ChatBotBackend/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/voice/conversation/stream":
                params = parse_qs(parsed.query)
                timeout_raw = str((params.get("timeout_ms") or ["45000"])[0] or "45000").strip()
                try:
                    timeout_ms = max(1000, min(120000, int(timeout_raw)))
                except Exception:
                    timeout_ms = 45000
                heartbeat_raw = str((params.get("heartbeat_ms") or ["500"])[0] or "500").strip()
                try:
                    heartbeat_ms = max(100, min(5000, int(heartbeat_raw)))
                except Exception:
                    heartbeat_ms = 500
                session_id = f"voice-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}-{os.getpid()}-{threading.get_ident()}"
                max_concurrent = _voice_stream_max_concurrent()
                with _VOICE_STREAM_LOCK:
                    global _VOICE_STREAM_ACTIVE
                    if _VOICE_STREAM_ACTIVE >= max_concurrent:
                        _ensure_learning_data_files()
                        _append_jsonl(
                            VOICE_STREAM_SESSIONS_LOG_PATH,
                            {
                                "ts": datetime.utcnow().isoformat() + "Z",
                                "session_id": session_id,
                                "status": "rejected",
                                "route": parsed.path,
                                "reason": "max_concurrency_reached",
                                "active_streams": int(_VOICE_STREAM_ACTIVE),
                                "max_concurrent_streams": int(max_concurrent),
                            },
                        )
                        _json_response(
                            self,
                            429,
                            _with_assistant_contract(
                                {
                                    "status": "error",
                                    "message": "voice stream rejected: max concurrent streams reached",
                                    "session_id": session_id,
                                    "active_streams": int(_VOICE_STREAM_ACTIVE),
                                    "max_concurrent_streams": int(max_concurrent),
                                },
                                message="Voice conversation stream rejected",
                                audit_id="voice-conversation-stream-rejected",
                                route=parsed.path,
                                contract_data={
                                    "session_id": session_id,
                                    "active_streams": int(_VOICE_STREAM_ACTIVE),
                                    "max_concurrent_streams": int(max_concurrent),
                                },
                            ),
                        )
                        return
                    _VOICE_STREAM_ACTIVE += 1
                    active_streams = int(_VOICE_STREAM_ACTIVE)
                data: Dict[str, Any] = {
                    "text": str((params.get("text") or [""])[0] or "").strip(),
                    "voice": str((params.get("voice") or ["female-calm"])[0] or "female-calm").strip(),
                    "language": str((params.get("language") or ["mn-MN"])[0] or "mn-MN").strip(),
                }
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type")
                    self.end_headers()

                    disconnected = {"value": False}

                    def _sse(event: str, payload: Dict[str, Any]) -> bool:
                        if disconnected["value"]:
                            return False
                        blob = json.dumps(payload, ensure_ascii=False)
                        try:
                            self.wfile.write(f"event: {event}\n".encode("utf-8"))
                            self.wfile.write(f"data: {blob}\n\n".encode("utf-8"))
                            self.wfile.flush()
                            return True
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            disconnected["value"] = True
                            return False

                    _ensure_learning_data_files()
                    _append_jsonl(
                        VOICE_STREAM_SESSIONS_LOG_PATH,
                        {
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "session_id": session_id,
                            "status": "accepted",
                            "route": parsed.path,
                            "timeout_ms": timeout_ms,
                            "heartbeat_ms": heartbeat_ms,
                            "active_streams": active_streams,
                            "max_concurrent_streams": max_concurrent,
                        },
                    )

                    if not _sse(
                        "status",
                        {
                            "status": "accepted",
                            "mode": "voice_conversation_stream",
                            "session_id": session_id,
                            "timeout_ms": timeout_ms,
                            "heartbeat_ms": heartbeat_ms,
                            "active_streams": active_streams,
                            "max_concurrent_streams": max_concurrent,
                        },
                    ):
                        _append_jsonl(
                            VOICE_STREAM_SESSIONS_LOG_PATH,
                            {
                                "ts": datetime.utcnow().isoformat() + "Z",
                                "session_id": session_id,
                                "status": "disconnected",
                                "route": parsed.path,
                                "phase": "accepted",
                            },
                        )
                        return

                    result_holder: Dict[str, Any] = {}

                    def _worker() -> None:
                        status_code, payload = _execute_voice_conversation_turn(
                            data,
                            parsed.path,
                            progress_callback=lambda stage, stage_payload: _sse(
                                "stage",
                                {
                                    "session_id": session_id,
                                    "stage": stage,
                                    **stage_payload,
                                },
                            ),
                        )
                        result_holder["status_code"] = status_code
                        result_holder["payload"] = payload

                    worker = threading.Thread(target=_worker, daemon=True)
                    worker.start()
                    stream_started = time.perf_counter()
                    next_heartbeat_sec = heartbeat_ms / 1000.0
                    timed_out = False
                    while worker.is_alive():
                        elapsed_sec = time.perf_counter() - stream_started
                        if elapsed_sec >= (timeout_ms / 1000.0):
                            timed_out = True
                            break
                        worker.join(timeout=min(0.1, max(0.01, (timeout_ms / 1000.0) - elapsed_sec)))
                        if disconnected["value"]:
                            _append_jsonl(
                                VOICE_STREAM_SESSIONS_LOG_PATH,
                                {
                                    "ts": datetime.utcnow().isoformat() + "Z",
                                    "session_id": session_id,
                                    "status": "disconnected",
                                    "route": parsed.path,
                                    "phase": "in_progress",
                                    "elapsed_ms": int((time.perf_counter() - stream_started) * 1000),
                                },
                            )
                            return
                        if worker.is_alive() and elapsed_sec >= next_heartbeat_sec:
                            _sse(
                                "heartbeat",
                                {
                                    "session_id": session_id,
                                    "elapsed_ms": int(elapsed_sec * 1000),
                                },
                            )
                            next_heartbeat_sec += heartbeat_ms / 1000.0

                    if timed_out or worker.is_alive():
                        timeout_payload = _with_assistant_contract(
                            {
                                "status": "error",
                                "session_id": session_id,
                                "message": f"voice stream timed out after {timeout_ms}ms",
                            },
                            message="Voice conversation stream timeout",
                            audit_id="voice-conversation-stream-timeout",
                            route=parsed.path,
                            contract_data={"timeout_ms": timeout_ms, "session_id": session_id},
                        )
                        _append_jsonl(
                            VOICE_STREAM_SESSIONS_LOG_PATH,
                            {
                                "ts": datetime.utcnow().isoformat() + "Z",
                                "session_id": session_id,
                                "status": "timeout",
                                "route": parsed.path,
                                "timeout_ms": timeout_ms,
                                "elapsed_ms": int((time.perf_counter() - stream_started) * 1000),
                            },
                        )
                        _sse("final", {"session_id": session_id, "status_code": 504, "payload": timeout_payload})
                        return

                    if "status_code" not in result_holder or "payload" not in result_holder:
                        failure_payload = _with_assistant_contract(
                            {
                                "status": "error",
                                "session_id": session_id,
                                "message": "voice stream failed before producing a result",
                            },
                            message="Voice conversation stream failed",
                            audit_id="voice-conversation-stream-failed",
                            route=parsed.path,
                        )
                        _append_jsonl(
                            VOICE_STREAM_SESSIONS_LOG_PATH,
                            {
                                "ts": datetime.utcnow().isoformat() + "Z",
                                "session_id": session_id,
                                "status": "failed",
                                "route": parsed.path,
                            },
                        )
                        _sse("final", {"session_id": session_id, "status_code": 500, "payload": failure_payload})
                        return

                    final_payload = dict(result_holder["payload"])
                    final_payload["session_id"] = session_id
                    _append_jsonl(
                        VOICE_STREAM_SESSIONS_LOG_PATH,
                        {
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "session_id": session_id,
                            "status": "completed",
                            "route": parsed.path,
                            "status_code": result_holder["status_code"],
                            "latency_ms": dict((final_payload.get("latency_ms") or {})),
                        },
                    )
                    _sse(
                        "final",
                        {
                            "session_id": session_id,
                            "status_code": result_holder["status_code"],
                            "payload": final_payload,
                        },
                    )
                    return
                finally:
                    with _VOICE_STREAM_LOCK:
                        _VOICE_STREAM_ACTIVE = max(0, int(_VOICE_STREAM_ACTIVE) - 1)

            if parsed.path.startswith("/api/core"):
                status_code, payload = _proxy_core_backend(
                    method="GET",
                    public_path=parsed.path,
                    query=parsed.query,
                )
                _json_response(self, status_code, payload)
                return

            if parsed.path == "/api/artifacts/file":
                params = parse_qs(parsed.query)
                requested = (params.get("path") or [""])[0]
                resolved = _resolve_artifact_path(requested)
                if resolved is None:
                    _json_response(self, 404, {"status": "error", "message": "artifact not found"})
                    return
                blob = resolved.read_bytes()
                content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
                _file_response(self, 200, blob, content_type, resolved.name)
                return

            status_code, payload = self._route(self.path)
        except Exception as exc:
            status_code, payload = 500, {"status": "error", "message": str(exc)}
        _json_response(self, status_code, payload)

    def do_POST(self) -> None:
        try:
            status_code, payload = self._route_post(self.path)
        except Exception as exc:
            status_code, payload = 500, {"status": "error", "message": str(exc)}
        _json_response(self, status_code, payload)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    @staticmethod
    def _route(path: str) -> Tuple[int, Dict[str, Any]]:
        parsed = urlparse(path)
        route = parsed.path.rstrip("/") or "/"
        if route == "/health":
            return 200, {"status": "ok", "service": "chatbot-backend", "build_id": BACKEND_BUILD_ID}
        if route == "/api/version":
            return 200, _with_assistant_contract({
                "status": "success",
                "build_id": BACKEND_BUILD_ID,
                "features": {
                    "capabilities": True,
                    "artifacts": True,
                    "artifacts_read": True,
                    "code_save": True,
                    "artifact_metadata": True,
                    "artifact_sort_filter_pagination": True,
                },
            },
                message="Version metadata retrieved",
                audit_id="api-version",
                route=route,
            )
        if route == "/api/capabilities":
            return 200, _with_assistant_contract(
                CAP_ENGINE.list_capabilities(),
                message="Capabilities listed",
                audit_id="api-capabilities",
                route=route,
            )
        if route == "/api/capabilities/model-health":
            return 200, _with_assistant_contract(
                CAP_ENGINE.model_health_summary(),
                message="Capability model health retrieved",
                audit_id="api-capabilities-model-health",
                route=route,
            )
        if route == "/api/capabilities/contract-trends":
            return 200, _with_assistant_contract(
                _capability_contract_trend_summary(),
                message="Capability contract trend summary retrieved",
                audit_id="api-capabilities-contract-trends",
                route=route,
            )
        if route == "/api/personas":
            preferred_persona, source = _resolve_preferred_persona(None)
            return 200, _with_assistant_contract({
                "status": "success",
                "personas": list(AI_PERSONAS.keys()),
                "current": preferred_persona,
                "details": AI_PERSONAS,
                "persona_source": source,
            },
                message="Persona list retrieved",
                audit_id="api-personas",
                route=route,
            )
        if route == "/api/preferences":
            preferred_persona, source = _resolve_preferred_persona(None)
            preferences = _effective_user_preferences()
            return 200, _with_assistant_contract(
                {
                    "status": "success",
                    "preferences": preferences,
                    "resolved_persona": preferred_persona,
                    "persona_source": source,
                },
                message="Preferences retrieved",
                audit_id="api-preferences",
                route=route,
            )
        if route == "/api/preferences/audit":
            events = _load_preference_audit_events(limit=100)
            return 200, _with_assistant_contract(
                {
                    "status": "success",
                    "events": events,
                    "count": len(events),
                },
                message="Preference audit events retrieved",
                audit_id="api-preferences-audit",
                route=route,
            )
        if route == "/api/safety/policy_violations":
            events = _load_policy_violation_events(limit=100)
            return 200, _with_assistant_contract(
                {
                    "status": "success",
                    "events": events,
                    "count": len(events),
                    "artifact_path": str(POLICY_VIOLATIONS_REPORT_PATH),
                },
                message="Policy violation report retrieved",
                audit_id="api-safety-policy-violations",
                route=route,
            )
        if route == "/api/tools":
            return 200, _with_assistant_contract({
                "status": "success",
                "tools": AI_TOOLS,
            },
                message="Tool catalog retrieved",
                audit_id="api-tools",
                route=route,
            )
        if route == "/api/artifacts":
            artifacts = _list_artifacts(limit=200)
            return 200, _with_assistant_contract(
                {"status": "success", "schema_version": 2, "artifacts": artifacts},
                message="Artifacts listed",
                audit_id="api-artifacts",
                route=route,
            )
        if route == "/api/artifacts/ops/export":
            params = parse_qs(parsed.query)
            raw_recent_limit = str((params.get("recent_limit") or ["240"])[0] or "240").strip()
            raw_policy_history_limit = str((params.get("policy_history_limit") or ["200"])[0] or "200").strip()
            actor_id = str((params.get("actor_id") or [""])[0] or "").strip()
            from_ts = str((params.get("from_ts") or [""])[0] or "").strip()
            to_ts = str((params.get("to_ts") or [""])[0] or "").strip()
            try:
                recent_limit = max(10, min(int(raw_recent_limit), 1000))
            except Exception:
                recent_limit = 240
            try:
                policy_history_limit = max(1, min(int(raw_policy_history_limit), 2000))
            except Exception:
                policy_history_limit = 200
            payload = _export_ops_artifact(
                route=route,
                recent_limit=recent_limit,
                policy_history_limit=policy_history_limit,
                actor_id=actor_id,
                from_ts=from_ts,
                to_ts=to_ts,
            )
            return 200, _with_assistant_contract(
                payload,
                message="Ops artifact export created",
                audit_id="api-artifacts-ops-export",
                route=route,
            )
        if route == "/api/memory/stats":
            return 200, _with_assistant_contract(
                {"status": "success", **MEMORY.stats()},
                message="Memory stats retrieved",
                audit_id="memory-stats",
                route=route,
            )
        if route == "/api/memory/history":
            return 200, _with_assistant_contract(
                {"status": "success", "messages": MEMORY.recent_messages(limit=50)},
                message="Memory history retrieved",
                audit_id="memory-history",
                route=route,
            )
        if route == "/api/memory/facts":
            return 200, _with_assistant_contract(
                {"status": "success", "facts": MEMORY.all_facts()},
                message="Memory facts retrieved",
                audit_id="memory-facts",
                route=route,
            )
        if route == "/api/voice/status":
            return 200, _with_assistant_contract(
                _voice_status(),
                message="Voice status retrieved",
                audit_id="voice-status",
                route=route,
            )
        if route == "/api/voice/stream/sessions":
            params = parse_qs(parsed.query)
            raw_limit = str((params.get("limit") or ["50"])[0] or "50").strip()
            try:
                limit = int(raw_limit)
            except Exception:
                limit = 50
            raw_minutes = str((params.get("minutes") or [""])[0] or "").strip()
            raw_hours = str((params.get("hours") or [""])[0] or "").strip()
            try:
                window_minutes = int(raw_minutes) if raw_minutes else None
            except Exception:
                window_minutes = None
            if window_minutes is None and raw_hours:
                try:
                    window_minutes = int(float(raw_hours) * 60)
                except Exception:
                    window_minutes = None
            status_filter = str((params.get("status") or [""])[0] or "").strip() or None
            session_id_filter = str((params.get("session_id") or [""])[0] or "").strip() or None
            return 200, _with_assistant_contract(
                _voice_stream_sessions_status(
                    limit=limit,
                    status_filter=status_filter,
                    session_id_filter=session_id_filter,
                    window_minutes=window_minutes,
                ),
                message="Voice stream session status retrieved",
                audit_id="voice-stream-sessions-status",
                route=route,
            )
        if route == "/api/voice/stream/analytics":
            params = parse_qs(parsed.query)
            raw_limit = str((params.get("limit") or ["200"])[0] or "200").strip()
            try:
                limit = int(raw_limit)
            except Exception:
                limit = 200
            raw_minutes = str((params.get("minutes") or [""])[0] or "").strip()
            raw_hours = str((params.get("hours") or [""])[0] or "").strip()
            try:
                window_minutes = int(raw_minutes) if raw_minutes else None
            except Exception:
                window_minutes = None
            if window_minutes is None and raw_hours:
                try:
                    window_minutes = int(float(raw_hours) * 60)
                except Exception:
                    window_minutes = None
            status_filter = str((params.get("status") or [""])[0] or "").strip() or None
            session_id_filter = str((params.get("session_id") or [""])[0] or "").strip() or None
            return 200, _with_assistant_contract(
                _voice_stream_analytics(
                    limit=limit,
                    status_filter=status_filter,
                    session_id_filter=session_id_filter,
                    window_minutes=window_minutes,
                ),
                message="Voice stream analytics retrieved",
                audit_id="voice-stream-analytics",
                route=route,
            )
        if route == "/api/learning/status":
            return 200, _with_assistant_contract(
                _learning_status(),
                message="Learning status retrieved",
                audit_id="learning-status",
                route=route,
            )
        if route == "/api/learning/images/profile":
            return 200, _with_assistant_contract(
                _image_learning_profile(),
                message="Image learning profile retrieved",
                audit_id="learning-image-profile",
                route=route,
            )
        if route == "/api/learning/governance/status":
            return 200, _with_assistant_contract(
                _learning_governance_status(),
                message="Learning governance status retrieved",
                audit_id="learning-governance-status",
                route=route,
            )
        if route == "/api/learning/advanced/modes":
            return 200, _with_assistant_contract(
                summarize_learning_modes(),
                message="Advanced learning mode status retrieved",
                audit_id="learning-advanced-modes",
                route=route,
            )
        if route == "/api/learning/advanced/supervised/status":
            payload = supervised_dataset_status(
                train_instructions_path=TRAIN_INSTRUCTIONS_PATH,
                user_feedback_path=USER_FEEDBACK_PATH,
                min_train_samples=int(os.getenv("ASSISTANT_SUPERVISED_MIN_TRAIN_SAMPLES", "100")),
                min_high_signal_feedback=int(os.getenv("ASSISTANT_SUPERVISED_MIN_HIGH_SIGNAL", "20")),
                min_avg_feedback_rating=float(os.getenv("ASSISTANT_SUPERVISED_MIN_AVG_RATING", "3.8")),
            )
            return 200, _with_assistant_contract(
                payload,
                message="Advanced supervised learning status retrieved",
                audit_id="learning-advanced-supervised-status",
                route=route,
            )
        if route == "/api/learning/advanced/rl/status":
            payload = advanced_rl_status(
                user_feedback_path=USER_FEEDBACK_PATH,
                reward_log_path=ADVANCED_RL_REWARD_LOG_PATH,
            )
            return 200, _with_assistant_contract(
                payload,
                message="Advanced RL interface status retrieved",
                audit_id="learning-advanced-rl-status",
                route=route,
            )
        if route == "/api/learning/advanced/policy/drift/status":
            payload = policy_drift_status(
                policy_violations_path=POLICY_VIOLATIONS_REPORT_PATH,
                reward_log_path=ADVANCED_RL_REWARD_LOG_PATH,
                window=int(os.getenv("ASSISTANT_ADVANCED_POLICY_DRIFT_WINDOW", "200")),
            )
            return 200, _with_assistant_contract(
                payload,
                message="Advanced policy drift status retrieved",
                audit_id="learning-advanced-policy-drift-status",
                route=route,
            )
        if route == "/api/learning/advanced/policy/status":
            payload = _load_json_file(ADVANCED_TOOL_POLICY_STATE_PATH, {})
            return 200, _with_assistant_contract(
                {
                    "status": "success",
                    "policy_state": payload,
                    "report_path": str(ADVANCED_POLICY_TUNING_REPORT_PATH),
                },
                message="Advanced policy operational status retrieved",
                audit_id="learning-advanced-policy-status",
                route=route,
            )
        if route == "/api/learning/advanced/federated/contract":
            payload = federated_contract_schema()
            return 200, _with_assistant_contract(
                payload,
                message="Advanced federated contract schema retrieved",
                audit_id="learning-advanced-federated-contract",
                route=route,
            )
        if route == "/api/learning/advanced/federated/status":
            payload = federated_status(
                validation_log_path=ADVANCED_FEDERATED_VALIDATION_LOG_PATH,
                aggregate_state_path=ADVANCED_FEDERATED_AGGREGATE_STATE_PATH,
            )
            return 200, _with_assistant_contract(
                payload,
                message="Advanced federated operational status retrieved",
                audit_id="learning-advanced-federated-status",
                route=route,
            )
        if route == "/api/learning/advanced/meta/schema":
            payload = meta_learning_schema()
            return 200, _with_assistant_contract(
                payload,
                message="Advanced meta-learning schema retrieved",
                audit_id="learning-advanced-meta-schema",
                route=route,
            )
        if route == "/api/learning/advanced/meta/status":
            payload = meta_learning_status(
                registry_path=ADVANCED_META_EXPERIMENTS_PATH,
                results_path=ADVANCED_META_RESULTS_PATH,
            )
            return 200, _with_assistant_contract(
                payload,
                message="Advanced meta-learning operational status retrieved",
                audit_id="learning-advanced-meta-status",
                route=route,
            )
        if route == "/api/learning/advanced/operations/status":
            payload = _advanced_operational_status()
            return 200, _with_assistant_contract(
                payload,
                message="Advanced 12-mode operational status retrieved",
                audit_id="learning-advanced-operations-status",
                route=route,
            )
        if route == "/api/evals/offline/status":
            return 200, _with_assistant_contract(
                _offline_eval_status(),
                message="Offline evaluation status retrieved",
                audit_id="evals-offline-status",
                route=route,
            )
        if route == "/api/evals/offline/trends":
            return 200, _with_assistant_contract(
                _offline_eval_trends(),
                message="Offline evaluation trends retrieved",
                audit_id="evals-offline-trends",
                route=route,
            )
        if route == "/api/optimization/status":
            return 200, _with_assistant_contract(
                OPT_API.get_status(),
                message="Optimization status retrieved",
                audit_id="optimization-status",
                route=route,
            )
        if route == "/api/integration/status":
            return 200, _with_assistant_contract(
                INT_API.get_status(),
                message="Integration status retrieved",
                audit_id="integration-status",
                route=route,
            )
        if route == "/api/integration/manifest":
            core = _core_backend_probe()
            return 200, _with_assistant_contract({
                "status": "success",
                "build_id": BACKEND_BUILD_ID,
                "unified_gateway": True,
                "core": core,
                "new": {
                    "api_base": "http://127.0.0.1:8080",
                    "capability_actions": [
                        cap.get("action")
                        for cap in (CAP_ENGINE.list_capabilities().get("capabilities") or [])
                        if isinstance(cap, dict)
                    ],
                },
                "routes": {
                    "new_stack": "/api/*",
                    "core_stack_proxy": "/api/core/* -> CORE_API_BASE/api/*",
                },
            },
                message="Integration manifest retrieved",
                audit_id="integration-manifest",
                route=route,
            )
        return 404, {"status": "error", "message": "Not found", "path": route}

    def _route_post(self, path: str) -> Tuple[int, Dict[str, Any]]:
        route = urlparse(path).path.rstrip("/") or "/"

        if route.startswith("/api/core"):
            raw = _read_raw_body(self.headers.get("Content-Length"), self.rfile.read, default=b"")
            return _proxy_core_backend(
                method="POST",
                public_path=route,
                query=urlparse(path).query,
                body=raw,
                content_type=str(self.headers.get("Content-Type", "application/json")),
            )

        if route.startswith("/api/code/save"):
            data = _parse_json_body(_read_raw_body(self.headers.get("Content-Length"), self.rfile.read))
            code = str(data.get("code", ""))
            if not code.strip():
                return 400, {"status": "error", "message": "code is required"}
            language = str(data.get("language", "python"))
            file_name = str(data.get("file_name", "")).strip() or None
            meta = _save_generated_code(code, language=language, file_name=file_name)
            _record_artifact_metadata(
                file_path=str(meta.get("file_path") or ""),
                source_tag="code_save",
                action_tag="code.save",
                tags=["code"],
                route=route,
            )
            MEMORY.add_observation(
                f"Saved generated code: language={language}; file={meta.get('file_name') or file_name or 'generated'}",
                source="api.code.save",
                metadata={"language": language, "saved": meta},
            )
            return 200, {"status": "success", "saved": meta}

        if route.startswith("/api/files/upload"):
            data = _parse_json_body(_read_raw_body(self.headers.get("Content-Length"), self.rfile.read))
            file_name = str(data.get("file_name", "")).strip()
            content_base64 = str(data.get("content_base64", "")).strip()
            if not file_name or not content_base64:
                return 400, {"status": "error", "message": "file_name and content_base64 are required"}
            try:
                saved = _save_uploaded_file(file_name=file_name, content_base64=content_base64)
                mime = str(saved.get("mime_type") or "")
                tags: List[str] = ["upload"]
                if mime.startswith("image/"):
                    tags.append("image")
                _record_artifact_metadata(
                    file_path=str(saved.get("file_path") or ""),
                    source_tag="upload",
                    action_tag="files.upload",
                    tags=tags,
                    route=route,
                )
            except Exception as exc:
                return 400, {"status": "error", "message": f"upload failed: {exc}"}
            MEMORY.add_observation(
                f"Uploaded file: {file_name}",
                source="api.files.upload",
                metadata={"saved": saved},
            )
            return 200, {"status": "success", "saved": saved}

        if route.startswith("/api/artifacts/read"):
            data = _parse_json_body(_read_raw_body(self.headers.get("Content-Length"), self.rfile.read))
            candidate = str(data.get("path", "")).strip()
            payload = _read_artifact_payload(candidate)
            if payload is None:
                return 404, {"status": "error", "message": "artifact not found"}
            MEMORY.add_observation(
                f"Read artifact: {candidate or payload.get('path') or 'unknown'}",
                source="api.artifact.read",
                metadata={"candidate": candidate, "artifact": payload},
            )
            return 200, {"status": "success", "artifact": payload}

        if route.startswith("/api/artifacts/ops/export"):
            data = _parse_json_body(_read_raw_body(self.headers.get("Content-Length"), self.rfile.read))
            raw_recent_limit = data.get("recent_limit", 240)
            raw_policy_history_limit = data.get("policy_history_limit", 200)
            actor_id = str(data.get("actor_id", "") or "").strip()
            from_ts = str(data.get("from_ts", "") or "").strip()
            to_ts = str(data.get("to_ts", "") or "").strip()
            try:
                recent_limit = max(10, min(int(raw_recent_limit), 1000))
            except Exception:
                recent_limit = 240
            try:
                policy_history_limit = max(1, min(int(raw_policy_history_limit), 2000))
            except Exception:
                policy_history_limit = 200
            payload = _export_ops_artifact(
                route=route,
                recent_limit=recent_limit,
                policy_history_limit=policy_history_limit,
                actor_id=actor_id,
                from_ts=from_ts,
                to_ts=to_ts,
            )
            return 200, payload

        if _is_capability_family_route(route):
            raw = _read_raw_body(self.headers.get("Content-Length"), self.rfile.read)
            try:
                data = _parse_json_body(raw)
            except Exception:
                payload = _with_assistant_contract(
                    {
                        "status": "error",
                        "message": "Invalid JSON payload",
                        "error_code": "invalid_json",
                    },
                    message="Invalid request payload",
                    audit_id="invalid-json-payload",
                    route=route,
                )
                _record_capability_observability_event(route=route, http_status=400, payload=payload, audit_id="invalid-json-payload")
                return 400, payload
            if not isinstance(data, dict):
                payload = _with_assistant_contract(
                    {
                        "status": "error",
                        "message": "JSON object payload is required",
                        "error_code": "payload_must_be_object",
                    },
                    message="Invalid request payload",
                    audit_id="invalid-payload-shape",
                    route=route,
                )
                _record_capability_observability_event(route=route, http_status=400, payload=payload, audit_id="invalid-payload-shape")
                return 400, payload
            capability_result = _handle_capability_post(route=route, data=data)
            if capability_result is not None:
                return capability_result

        if _is_profile_eval_route(route):
            data = _parse_json_body(_read_raw_body(self.headers.get("Content-Length"), self.rfile.read))
            profile_eval_result = _handle_profile_eval_post(route=route, data=data)
            if profile_eval_result is not None:
                return profile_eval_result

        if _is_voice_memory_route(route):
            data = _parse_json_body(_read_raw_body(self.headers.get("Content-Length"), self.rfile.read))
            voice_memory_result = _handle_voice_memory_post(route=route, data=data)
            if voice_memory_result is not None:
                return voice_memory_result

        if _is_learning_route(route):
            data = _parse_json_body(_read_raw_body(self.headers.get("Content-Length"), self.rfile.read))
            learning_result = _handle_learning_post(route=route, data=data)
            if learning_result is not None:
                return learning_result

        if _is_advanced_learning_route(route):
            data = _parse_json_body(_read_raw_body(self.headers.get("Content-Length"), self.rfile.read))
            advanced_result = _handle_advanced_learning_post(route=route, data=data)
            if advanced_result is not None:
                return advanced_result

        if _is_chat_route(route):
            data = _parse_json_body(_read_raw_body(self.headers.get("Content-Length"), self.rfile.read))
            chat_result = _handle_chat_post(route=route, data=data)
            if chat_result is not None:
                return chat_result

        return 404, {"status": "error", "message": "Not found", "path": route}


class SingleInstanceHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        # On Windows, enforce exclusive port ownership to avoid split traffic.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _maybe_start_image_warmup() -> None:
    global _IMAGE_WARMUP_STARTED
    if _IMAGE_WARMUP_STARTED:
        return
    enabled = str(os.getenv("ASSISTANT_WARMUP_IMAGE_ON_BOOT", "1")).strip().lower() not in {"0", "false", "off"}
    if not enabled:
        return
    _IMAGE_WARMUP_STARTED = True

    def _run() -> None:
        try:
            CAP_ENGINE.execute(
                "image.generate",
                {
                    "prompt": "startup warmup request",
                    "quality_mode": "fast",
                    "model_tier": "auto",
                    "num_inference_steps": 8,
                    "width": 512,
                    "height": 512,
                    "upscale_factor": 1,
                },
            )
        except Exception:
            pass

    threading.Thread(target=_run, name="image-warmup", daemon=True).start()


def main() -> int:
    _load_env_file()

    parser = argparse.ArgumentParser(description="Run backend HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = SingleInstanceHTTPServer((args.host, args.port), AppHandler)
    print(f"[backend] build={BACKEND_BUILD_ID} listening on http://{args.host}:{args.port}")
    _maybe_start_image_warmup()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[backend] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
