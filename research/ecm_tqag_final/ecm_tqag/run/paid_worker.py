"""Paid phase worker with one fail-closed dispatch point.

The worker builds provider-neutral requests, routes them through the frozen role
roster, and parses every response through the phase's closed schema.  It never
performs transport outside the injected ``transport`` object.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
import base64
import hashlib
import io
import json

from PIL import Image

from ..interfaces import (
    CAPTION_PROMPT,
    caption_request,
    interface_from_caption,
    interface_from_graph,
    ocr_assisted_graph_prompt,
    parse_caption,
    parse_graph_interface,
)
from ..manifest import Corpus, image_part
from ..direct import direct_request, parse_item
from ..guards import (
    assert_realizer_blind,
    excerpt_word_budget,
    guard_plan_v3,
    guard_realized_v3,
    seal_construction,
)
from ..item_gates import build_idf, evaluate_item
from ..prompts import DECODING, planner_prompt, planner_program, realizer_prompt
from ..structure_reader import PROMPT
from ..controls import (
    control_image_data_url,
    control_item,
    controls_commitment,
    validate_controls,
)
from ..evaluation import (
    answerer_request,
    image_audit_request,
    judge_request,
    parse_answer,
    parse_image_audit,
    parse_judgement,
)
from .envelope import response_content
from .transport import TransportConfig, load_response_sidecar


class PaidWorkerBlocked(RuntimeError):
    """Stable fail-closed error raised before a bad result can be checkpointed."""


def _blocked(reason: str) -> PaidWorkerBlocked:
    return PaidWorkerBlocked(f"BLOCKED_PAID_WORKER:{reason}")


class _Transport(Protocol):
    def call(self, config: TransportConfig, payload: dict[str, Any], *,
             metadata: dict[str, Any]) -> dict[str, Any]: ...


ResponseLoader = Callable[[Path, dict[str, Any]], dict[str, Any]]
ArtifactLoader = Callable[[str], Mapping[str, Any]]


def _graph_request(prompt: str, image_data_url: str) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt:
        raise _blocked("graph_prompt_invalid")
    if not isinstance(image_data_url, str) or not image_data_url.startswith("data:image/"):
        raise _blocked("image_payload_invalid")
    return {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}],
        "temperature": DECODING["temperature"],
        "max_tokens": DECODING["max_tokens"],
    }


def parse_extraction_body(body: Mapping[str, Any], *, kind: str,
                          image_index: int, image_sha256: str) -> dict[str, Any]:
    """Uniformly parse one paid extraction response without repair or clipping.

    This pure helper is shared by live execution and cross-freeze import.  A response
    that violates the frozen 0--1000 graph grid becomes an explicit paid
    ``SCHEMA_REJECTED`` result; every other malformed schema fails closed.
    """
    if kind not in {"graph", "caption", "ocr_graph"}:
        raise ValueError("extraction_kind_invalid")
    response_sha256 = hashlib.sha256(
        json.dumps(dict(body), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    try:
        raw = response_content(dict(body))
        if kind == "caption":
            interface = interface_from_caption(parse_caption(raw))
        else:
            graph = parse_graph_interface(raw)
            for node in graph["nodes"]:
                bbox = node["bbox"]
                if not all(0.0 <= float(value) <= 1000.0 for value in bbox):
                    raise ValueError("graph_interface:grid_bbox_out_of_range")
                x0, y0, x1, y1 = (float(value) for value in bbox)
                if not (x0 < x1 and y0 < y1):
                    raise ValueError("graph_interface:grid_bbox_order_invalid")
            interface = {
                "graph_type": graph["graph_type"],
                "nodes": [
                    {"id": node["id"], "label": node["label"],
                     "level": node["level"],
                     "bbox": [round(float(value) / 1000.0, 6)
                              for value in node["bbox"]]}
                    for node in graph["nodes"]
                ],
                "edges": [dict(edge) for edge in graph["edges"]],
            }
    except ValueError as exc:
        reason = str(exc)
        if reason in {
            "graph_interface:grid_bbox_out_of_range",
            "graph_interface:grid_bbox_order_invalid",
        }:
            return {
                "status": "SCHEMA_REJECTED",
                "reason": reason.removeprefix("graph_interface:"),
                "calls_used": 1,
                "extraction_kind": kind,
                "image_index": image_index,
                "image_sha256": image_sha256,
                "response_sha256": response_sha256,
            }
        if kind == "caption" and reason.startswith("caption_schema:"):
            return {
                "status": "SCHEMA_REJECTED",
                "reason": reason.removeprefix("caption_schema:"),
                "calls_used": 1,
                "extraction_kind": kind,
                "image_index": image_index,
                "image_sha256": image_sha256,
                "response_sha256": response_sha256,
            }
        raise
    return {
        "status": "OK",
        "calls_used": 1,
        "extraction_kind": kind,
        "image_index": image_index,
        "image_sha256": image_sha256,
        "interface": interface,
    }


class PaidPhaseWorker:
    """Callable phase dispatcher bound to one corpus and one frozen roster."""

    def __init__(self, *, corpus: Corpus, freeze: Mapping[str, Any],
                 transport: _Transport,
                 role_configs: Mapping[str, TransportConfig], response_root: Path,
                 response_loader: ResponseLoader = load_response_sidecar,
                 extraction_loader: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
                 artifact_loader: ArtifactLoader | None = None,
                 controls: Mapping[str, Any] | None = None,
                 judging_frame: Mapping[str, Any] | None = None):
        if not isinstance(corpus, Corpus):
            raise _blocked("corpus_invalid")
        roles = freeze.get("roles")
        if not isinstance(roles, Mapping):
            raise _blocked("frozen_roster_missing")
        self.corpus = corpus
        self.freeze = dict(freeze)
        self.transport = transport
        self.role_configs = dict(role_configs)
        self.response_root = Path(response_root)
        self.response_loader = response_loader
        self.extraction_loader = extraction_loader
        self.artifact_loader = artifact_loader
        self.controls = dict(controls) if isinstance(controls, Mapping) else None
        self._controls_by_id = {
            str(row.get("control_id")): dict(row)
            for row in (self.controls or {}).get("controls", [])
            if isinstance(row, Mapping) and isinstance(row.get("control_id"), str)
        }
        self.judging_frame = dict(judging_frame) if isinstance(judging_frame, Mapping) else None

        for name, config in self.role_configs.items():
            frozen = roles.get(name)
            if (
                not isinstance(frozen, Mapping)
                or config.provider != frozen.get("provider")
                or config.model != frozen.get("model")
                or config.allow_fallbacks is not False
            ):
                raise _blocked(f"role_config_mismatch:{name}")

        # One deterministic global image index, preserving frozen chunk and image
        # order. Paths remain local and never enter request metadata or results.
        self._images: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for package in corpus.tlv:
            images = package["evidence"]["images"]
            for image in sorted(images, key=lambda row: row["declared_order"]):
                self._images.append((package, image))
        if len(self._images) != 18:
            raise _blocked(f"image_census_mismatch:{len(self._images)}")
        corpus_texts = [package["evidence"]["text"] for package in corpus.tlv]
        self._idf = build_idf(corpus_texts)
        self._n_docs = len(corpus_texts)

    def _config(self, role: str) -> TransportConfig:
        config = self.role_configs.get(role)
        if config is None:
            raise _blocked(f"role_config_missing:{role}")
        frozen = self.freeze["roles"].get(role)
        if (
            not isinstance(frozen, Mapping)
            or config.provider != frozen.get("provider")
            or config.model != frozen.get("model")
            or config.allow_fallbacks is not False
        ):
            raise _blocked(f"role_config_mismatch:{role}")
        return config

    def _call(self, role: str, payload: dict[str, Any], *,
              metadata: dict[str, Any]) -> dict[str, Any]:
        terminal = self.transport.call(self._config(role), payload, metadata=metadata)
        if not isinstance(terminal, dict) or terminal.get("outcome") != "OK":
            outcome = terminal.get("outcome") if isinstance(terminal, dict) else "not_object"
            raise _blocked(f"transport_failed:{outcome}")
        try:
            body = self.response_loader(self.response_root, terminal)
        except Exception as exc:
            raise _blocked(f"response_sidecar_failed:{type(exc).__name__}") from exc
        if not isinstance(body, dict):
            raise _blocked("response_body_not_object")
        return body

    def __call__(self, task: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(task, dict):
            raise _blocked("task_not_object")
        if task.get("calls") == 0:
            if (task.get("phase") != "construction"
                    or task.get("arm") != "gates_off"
                    or task.get("deterministic_rescore") is not True
                    or task.get("construction_stage") != "rescore"):
                raise _blocked("zero_call_task_invalid")
            source_task_id = task.get("parent_task_id")
            if not isinstance(source_task_id, str) or not source_task_id:
                raise _blocked("source_task_id_invalid")
            if self.artifact_loader is None:
                raise _blocked("source_artifact_loader_missing")
            try:
                source = self.artifact_loader(source_task_id)
            except (KeyError, FileNotFoundError):
                raise _blocked(f"source_artifact_missing:{source_task_id}") from None
            except Exception as exc:
                raise _blocked(f"source_artifact_load_failed:{type(exc).__name__}") from exc
            if not isinstance(source, Mapping):
                raise _blocked("source_artifact_not_object")
            if (((source.get("status") == "NOT_APPLICABLE"
                    and source.get("reason") == "upstream_unavailable")
                    or source.get("status") in {
                        "SCHEMA_REJECTED", "GUARD_REJECTED", "ABSTAINED"
                    })
                    and source.get("arm") == "full"
                    and source.get("chunk_id") == task.get("chunk_id")):
                return {
                    "status": "NOT_APPLICABLE",
                    "reason": "upstream_unavailable",
                    "calls_used": 0,
                    "arm": "gates_off",
                    "chunk_id": task.get("chunk_id"),
                    "input_fingerprint": task.get("input_fingerprint"),
                    "source_task_id": source_task_id,
                    "source_status": source.get("status"),
                }
            if (source.get("status") != "PARSED"
                    or source.get("arm") != "full"
                    or source.get("chunk_id") != task.get("chunk_id")):
                raise _blocked("source_artifact_not_parsed_full")
            item = source.get("item")
            gates = source.get("gates")
            if not isinstance(item, Mapping) or not isinstance(gates, Mapping):
                raise _blocked("source_artifact_incomplete")
            return {
                "status": "DETERMINISTIC_RESCORE",
                "calls_used": 0,
                "arm": "gates_off",
                "chunk_id": task.get("chunk_id"),
                "input_fingerprint": task.get("input_fingerprint"),
                "source_task_id": source_task_id,
                "item": dict(item),
                "source_gates": dict(gates),
            }
        phase = task.get("phase")
        if phase == "extraction":
            return self._extract(task)
        if phase == "construction":
            return self._construct(task)
        if phase == "sensitivity_floor":
            return self._sensitivity(task)
        if phase == "secondary_probes":
            return self._probe(task)
        if phase == "image_audit":
            return self._image_audit(task)
        if phase == "judging":
            return self._judge(task)
        raise _blocked(f"phase_not_implemented:{phase}")

    def _package(self, chunk_id: str) -> dict[str, Any]:
        for package in self.corpus.tlv:
            if package.get("chunk_id") == chunk_id:
                return package
        raise _blocked(f"chunk_not_found:{chunk_id}")

    def _images_for(self, package: Mapping[str, Any]) -> list[dict[str, Any]]:
        return sorted(list(package["evidence"].get("images", [])),
                      key=lambda row: row["declared_order"])

    def _answer_call(self, *, role: str, item: Mapping[str, Any], text: str | None,
                     image_urls: list[str], task: Mapping[str, Any], replicate: int) -> dict[str, Any]:
        try:
            payload = answerer_request(dict(item), text=text, image_data_urls=image_urls)
            body = self._call(role, payload, metadata={
                "phase": str(task["phase"]), "role": role,
                "task_id": str(task["task_id"]), "replicate": replicate,
                "chunk_id": task.get("chunk_id"),
            })
            return parse_answer(response_content(body))
        except PaidWorkerBlocked:
            raise
        except Exception as exc:
            raise _blocked(f"answer_parse_failed:{type(exc).__name__}") from exc

    def _sensitivity(self, task: dict[str, Any]) -> dict[str, Any]:
        role = task.get("answerer_role")
        control_id = task.get("control_id")
        if role not in {"answerer_a", "answerer_b"}:
            raise _blocked("answerer_role_invalid")
        if self.controls is None or not self._controls_by_id:
            raise _blocked("controls_missing")
        row = self._controls_by_id.get(str(control_id))
        if row is None:
            raise _blocked("control_id_invalid")
        if task.get("replicates") != 2 or task.get("calls") != 2:
            raise _blocked("control_replicate_metadata_invalid")
        try:
            validate_controls(self.controls)
            commitment = controls_commitment(self.controls)
        except Exception as exc:
            raise _blocked(f"controls_invalid:{type(exc).__name__}") from exc
        expected = self.freeze.get("sensitivity_controls", {}).get("commitment_sha256")
        if commitment != expected:
            raise _blocked("control_commitment_mismatch")
        image_url = control_image_data_url(row) if row["type"] == "positive_visual" else None
        image_urls = [image_url] if image_url is not None else []
        item = control_item(row)
        answers = [self._answer_call(role=role, item=item, text=row["text"],
                                     image_urls=image_urls, task=task, replicate=i)
                   for i in (1, 2)]
        correct = [not a["abstain"] and a["answer_index"] == row["answer_index"]
                   for a in answers]
        outcomes = [(a["answer_index"], a["abstain"]) for a in answers]
        return {"status": "SCORED", "calls_used": 2, "answerer_role": role,
                "control_id": control_id, "control_type": row["type"],
                "correct": correct,
                "replicate_agreement": outcomes[0] == outcomes[1],
                "control_commitment_sha256": commitment}

    def _source_item(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        parent = task.get("parent_task_id")
        if not isinstance(parent, str) or self.artifact_loader is None:
            raise _blocked("full_artifact_missing")
        try:
            source = self.artifact_loader(parent)
        except Exception as exc:
            raise _blocked(f"full_artifact_load_failed:{type(exc).__name__}") from exc
        if not isinstance(source, Mapping):
            raise _blocked("full_artifact_not_eligible")
        if (source.get("status") == "NOT_APPLICABLE"
                and source.get("reason") == "upstream_unavailable"
                and source.get("arm") == "full"):
            raise _blocked("full_item_not_eligible")
        if source.get("status") != "PARSED" or source.get("arm") != "full":
            raise _blocked("full_artifact_not_eligible")
        if source.get("passes_confirmatory_gates") is not True:
            raise _blocked("full_item_not_eligible")
        item = source.get("item")
        if not isinstance(item, Mapping):
            raise _blocked("full_item_missing")
        return item

    def _probe(self, task: dict[str, Any]) -> dict[str, Any]:
        role = task.get("answerer_role")
        condition = task.get("probe_condition")
        chunk_id = task.get("chunk_id")
        if role not in {"answerer_a", "answerer_b"} or condition not in {
            "control", "control_replicate", "label_permutation", "block_shuffle", "text_anchor_removal"
        }:
            raise _blocked("probe_metadata_invalid")
        try:
            item = self._source_item(task)
        except PaidWorkerBlocked as exc:
            if str(exc).endswith(("full_item_not_eligible", "full_artifact_not_eligible")):
                return {"status": "NOT_APPLICABLE", "reason": "full_item_not_eligible",
                        "calls_used": 0, "chunk_id": chunk_id,
                        "probe_condition": condition, "answerer_role": role}
            raise
        package = self._package(str(chunk_id))
        text = package["evidence"]["text"]
        images = self._images_for(package)
        if not images:
            raise _blocked("probe_images_missing")
        try:
            image_payload, _ = image_part(images[0])
            image_url = image_payload["image_url"]["url"]
        except Exception as exc:
            raise _blocked(f"probe_image_load_failed:{type(exc).__name__}") from exc
        # The perturbation constructors are deterministic and remain diagnostic;
        # they never alter the generated item or enter the primary endpoint.
        if condition in {"label_permutation", "block_shuffle"}:
            from ..counterfactual_images import build_arms
            try:
                family = "P1_LABEL_PERMUTE" if condition == "label_permutation" else "P2_BLOCK_SHUFFLE"
                arms = build_arms(Path(images[0]["path"]), seed=0)
                image_url = "data:image/png;base64," + base64.b64encode(arms[family]["png"]).decode("ascii")
            except Exception as exc:
                raise _blocked(f"probe_perturbation_failed:{type(exc).__name__}") from exc
        if condition == "text_anchor_removal":
            text = ""
        answer = self._answer_call(role=role, item=item, text=text,
                                   image_urls=[image_url], task=task, replicate=1)
        return {"status": "SCORED", "calls_used": 1, "answerer_role": role,
                "chunk_id": chunk_id, "probe_condition": condition, "answer": answer}

    def _image_audit(self, task: dict[str, Any]) -> dict[str, Any]:
        index = task.get("image_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(self._images):
            raise _blocked("audit_image_index_invalid")
        if task.get("image_sha256") != self._images[index - 1][1].get("sha256"):
            raise _blocked("audit_image_fingerprint_mismatch")
        parent = task.get("parent_task_id")
        if self.artifact_loader is None or not isinstance(parent, str):
            raise _blocked("audit_parent_missing")
        try:
            source = self.artifact_loader(parent)
            if (isinstance(source, Mapping)
                    and source.get("status") == "NOT_APPLICABLE"
                    and source.get("reason") == "upstream_unavailable"):
                return {"status": "NOT_APPLICABLE",
                        "reason": "upstream_unavailable",
                        "calls_used": 0,
                        "image_index": index,
                        "source_task_id": parent}
            interface = source["interface"]
            image_url = image_part(self._images[index - 1][1])[0]["image_url"]["url"]
            body = self._call("image_auditor", image_audit_request(dict(interface), image_url), metadata={
                "phase": "image_audit", "role": "image_auditor", "task_id": task["task_id"],
                "image_index": index, "image_sha256": task["image_sha256"]})
            audit = parse_image_audit(response_content(body))
        except PaidWorkerBlocked:
            raise
        except Exception as exc:
            raise _blocked(f"image_audit_failed:{type(exc).__name__}") from exc
        return {"status": "AUDITED", "calls_used": 1, "image_index": index, "audit": audit}

    def _judge(self, task: dict[str, Any]) -> dict[str, Any]:
        role = task.get("judge_role")
        frame_index = task.get("frame_index")
        if role not in {"model_judge_a", "model_judge_b"} or self.judging_frame is None:
            raise _blocked("judging_frame_or_role_missing")
        frame = self.judging_frame.get("private_frame")
        if not isinstance(frame, list) or len(frame) != 40 or not isinstance(frame_index, int) or not 1 <= frame_index <= 40:
            raise _blocked("judging_frame_invalid")
        row = frame[frame_index - 1]
        if not isinstance(row, Mapping) or not isinstance(row.get("item"), Mapping):
            raise _blocked("judging_item_missing")
        try:
            body = self._call(role, judge_request(dict(row["item"])), metadata={
                "phase": "judging", "role": role, "task_id": task["task_id"],
                "frame_index": frame_index, "judge_item_id": row.get("judge_item_id")})
            judgement = parse_judgement(response_content(body))
        except PaidWorkerBlocked:
            raise
        except Exception as exc:
            raise _blocked(f"judgement_failed:{type(exc).__name__}") from exc
        return {"status": "JUDGED", "calls_used": 1,
                "judge_role": role, "frame_index": frame_index,
                "judge_item_id": row["judge_item_id"], "judgement": judgement}

    def _item_gates(self, *, item: Mapping[str, Any], arm: str, text: str,
                    sealed: Mapping[str, Any] | None,
                    contract_gate: Mapping[str, Any],
                    seal_gate: Mapping[str, Any]) -> dict[str, Any]:
        """Apply the frozen five-gate endpoint to every completed item."""
        if sealed is None:
            anchors = [text]
            observations: list[str] = []
            atom_values: list[str] = []
        else:
            anchors = [str(row.get("excerpt") or "")
                       for row in sealed.get("anchors", [])
                       if isinstance(row, Mapping)]
            observations = [str(row.get("observation") or "")
                            for row in sealed.get("visual_observations", [])
                            if isinstance(row, Mapping)]
            atom_values = [str(row.get("value") or "")
                           for row in sealed.get("atoms", [])
                           if isinstance(row, Mapping)]
        view = {
            "chunk_id": "private",
            "condition": arm,
            "status": "PARSED",
            "question": item.get("question"),
            "choices": item.get("choices"),
            "answer_index": item.get("answer_index"),
            "anchors": anchors,
            "observations": observations,
            "atom_values": atom_values,
            "contract_gate": dict(contract_gate),
            "seal_gate": dict(seal_gate),
        }
        try:
            return evaluate_item(view, text, self._idf, self._n_docs, 0.95)
        except Exception as exc:
            raise _blocked(f"item_gate_failed:{type(exc).__name__}:{exc}") from exc

    def _construct(self, task: dict[str, Any]) -> dict[str, Any]:
        stage = task.get("construction_stage")
        arm = task.get("arm")
        if stage not in {"planner", "realizer", "direct"}:
            raise _blocked("construction_stage_invalid")
        if arm not in {"full", "caption_mediated", "text_only",
                       "text_assisted_reader", "direct"}:
            raise _blocked("construction_arm_invalid")
        chunk_id = task.get("chunk_id")
        if not isinstance(chunk_id, str):
            raise _blocked("construction_chunk_invalid")
        package = self._package(chunk_id)
        evidence = package["evidence"]
        text = evidence["text"]
        images = self._images_for(package)

        if stage == "planner":
            if arm == "full":
                kind = "graph"
                interface_kind = "closed_graph"
            elif arm == "caption_mediated":
                kind = "caption"
                interface_kind = "caption"
            elif arm == "text_assisted_reader":
                kind = "ocr_graph"
                interface_kind = "closed_graph"
            else:
                raise _blocked("planner_arm_invalid")
            if self.extraction_loader is None:
                raise _blocked("extraction_loader_missing")
            request_task = dict(task)
            request_task.update({"extraction_kind": kind})
            try:
                extracted = self.extraction_loader(request_task)
            except Exception as exc:
                raise _blocked(f"extraction_artifact_load_failed:{type(exc).__name__}") from exc
            if not isinstance(extracted, Mapping):
                raise _blocked("extraction_artifact_invalid")
            if (extracted.get("status") == "NOT_APPLICABLE"
                    and extracted.get("reason") == "upstream_unavailable"):
                return {"status": "NOT_APPLICABLE", "reason": "upstream_unavailable",
                        "calls_used": 0, "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            if not isinstance(extracted.get("interface"), Mapping):
                raise _blocked("extraction_artifact_invalid")
            payload = {"messages": [{"role": "user", "content": planner_prompt(
                text, evidence.get("document_structure") or {},
                dict(extracted["interface"]), interface_kind=interface_kind)}],
                       "temperature": DECODING["temperature"],
                       "max_tokens": DECODING["max_tokens"]}
            body = self._call("generator", payload, metadata={
                "phase": "construction", "role": "generator", "task_id": task["task_id"],
                "construction_stage": "planner", "arm": arm, "chunk_id": chunk_id})
            try:
                program_raw = response_content(body)
                parsed = planner_program(program_raw)
            except Exception as exc:
                # The HTTP call completed, but the model did not satisfy the
                # frozen planner schema.  This is a terminal one-call ITT
                # outcome: preserve it, never repair it and never retry it.
                return {"status": "SCHEMA_REJECTED", "calls_used": 1,
                        "reason": f"planner_schema_rejected:{type(exc).__name__}:{exc}",
                        "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            if "abstain" in parsed:
                return {"status": "ABSTAINED", "calls_used": 1,
                        "reason": "no_cross_modal_dependency", "arm": arm,
                        "chunk_id": chunk_id, "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            try:
                guarded = guard_plan_v3(parsed, text, "TLV",
                                         declared_images=images, require_cross_modal=True)
            except Exception as exc:
                return {"status": "GUARD_REJECTED", "calls_used": 1,
                        "reason": f"planner_guard_error:{type(exc).__name__}:{exc}",
                        "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            if guarded.get("status") != "PLAN_OK":
                return {"status": "GUARD_REJECTED", "calls_used": 1,
                        "reason": str(guarded.get("reason") or "planner_guard_rejected"),
                        "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            try:
                sealed = seal_construction(
                    guarded["program"], guarded["trace"], "TLV",
                    image_nodes=[f"image-{i + 1}" for i in range(len(images))],
                    evidence_text=text,
                )
                blind = assert_realizer_blind(
                    sealed, text,
                    excerpt_word_budget(text, len(sealed.get("anchors", []))),
                )
            except Exception as exc:
                return {"status": "GUARD_REJECTED", "calls_used": 1,
                        "reason": f"planner_seal_error:{type(exc).__name__}:{exc}",
                        "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            if not blind.get("ok"):
                return {"status": "GUARD_REJECTED", "calls_used": 1,
                        "reason": "planner_seal_rejected:"
                                  + ",".join(blind.get("problems", [])),
                        "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            return {"status": "PLAN_OK", "calls_used": 1, "arm": arm,
                    "chunk_id": chunk_id, "construction_stage": stage,
                    "input_fingerprint": task.get("input_fingerprint"),
                    "program": guarded["program"], "execution": guarded["execution"],
                    "trace": guarded["trace"], "visual": guarded.get("visual"),
                    "sealed": sealed, "seal_blindness": blind}

        if stage == "realizer":
            parent = task.get("parent_task_id")
            if not isinstance(parent, str) or not parent or self.artifact_loader is None:
                raise _blocked("planner_artifact_missing")
            try:
                source = self.artifact_loader(parent)
            except (KeyError, FileNotFoundError):
                raise _blocked(f"planner_artifact_missing:{parent}") from None
            except Exception as exc:
                raise _blocked(f"planner_artifact_load_failed:{type(exc).__name__}") from exc
            if not isinstance(source, Mapping):
                raise _blocked("planner_artifact_invalid")
            if ((source.get("status") == "NOT_APPLICABLE"
                    and source.get("reason") == "upstream_unavailable")
                    or source.get("status") in {
                        "SCHEMA_REJECTED", "GUARD_REJECTED", "ABSTAINED"
                    }):
                return {"status": "NOT_APPLICABLE", "reason": "upstream_unavailable",
                        "calls_used": 0, "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "source_status": source.get("status"),
                        "input_fingerprint": task.get("input_fingerprint")}
            if source.get("status") != "PLAN_OK":
                raise _blocked("planner_artifact_not_plan_ok")
            sealed = source.get("sealed")
            if not isinstance(sealed, Mapping):
                raise _blocked("sealed_artifact_missing")
            sealed = dict(sealed)
            try:
                blind = assert_realizer_blind(sealed, text,
                                              excerpt_word_budget(text, len(sealed.get("anchors", []))))
            except Exception as exc:
                raise _blocked(f"seal_blindness_failed:{type(exc).__name__}") from exc
            if not blind.get("ok"):
                raise _blocked("seal_blindness_failed")
            body = self._call("generator", {"messages": [{"role": "user",
                "content": realizer_prompt(sealed)}], "temperature": DECODING["temperature"],
                "max_tokens": DECODING["max_tokens"]}, metadata={
                "phase": "construction", "role": "generator", "task_id": task["task_id"],
                "construction_stage": "realizer", "arm": arm, "chunk_id": chunk_id})
            try:
                item = parse_item(response_content(body))
            except Exception as exc:
                return {"status": "SCHEMA_REJECTED", "calls_used": 1,
                        "reason": f"realizer_schema_rejected:{type(exc).__name__}:{exc}",
                        "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            try:
                realized = guard_realized_v3(item, sealed, text)
            except Exception as exc:
                return {"status": "GUARD_REJECTED", "calls_used": 1,
                        "reason": f"realizer_guard_error:{type(exc).__name__}:{exc}",
                        "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            if realized.get("status") != "PARSED":
                return {"status": "GUARD_REJECTED", "calls_used": 1,
                        "reason": str(realized.get("reason") or "realizer_guard_rejected"),
                        "arm": arm, "chunk_id": chunk_id,
                        "construction_stage": stage,
                        "input_fingerprint": task.get("input_fingerprint")}
            contract_gate = {"ok": True, "reason": None}
            seal_gate = {"ok": True, "reason": None}
            gate_result = self._item_gates(
                item=item, arm=str(arm), text=text, sealed=sealed,
                contract_gate=contract_gate, seal_gate=seal_gate,
            )
            return {"status": "PARSED", "calls_used": 1, "arm": arm,
                    "chunk_id": chunk_id, "construction_stage": stage, "item": item,
                    "sealed": sealed, "contract_gate": contract_gate,
                    "seal_gate": seal_gate, "gates": gate_result["gates"],
                    "passes_confirmatory_gates": gate_result["passes_confirmatory_gates"],
                    "input_fingerprint": task.get("input_fingerprint")}

        if arm not in {"text_only", "direct"}:
            raise _blocked("direct_stage_arm_invalid")
        image_urls = []
        if arm == "direct":
            for image in images:
                try:
                    image_urls.append(image_part(image)[0]["image_url"]["url"])
                except Exception as exc:
                    raise _blocked(f"image_load_failed:{type(exc).__name__}") from exc
        try:
            payload = direct_request(
                text,
                input_mode="ocr_only" if arm == "text_only" else "ocr_layout_pixels",
                image_data_urls=image_urls,
            )
        except Exception as exc:
            raise _blocked(f"direct_request_failed:{type(exc).__name__}:{exc}") from exc
        body = self._call("generator", payload, metadata={
            "phase": "construction", "role": "generator", "task_id": task["task_id"],
            "construction_stage": "direct", "arm": arm, "chunk_id": chunk_id})
        try:
            item = parse_item(response_content(body))
        except Exception as exc:
            return {"status": "SCHEMA_REJECTED", "calls_used": 1,
                    "reason": f"direct_schema_rejected:{type(exc).__name__}:{exc}",
                    "arm": arm, "chunk_id": chunk_id,
                    "construction_stage": stage,
                    "input_fingerprint": task.get("input_fingerprint")}
        contract_gate = {"ok": True, "reason": None}
        seal_gate = {"ok": True, "reason": "not_applicable_direct_path"}
        gate_result = self._item_gates(
            item=item, arm=str(arm), text=text, sealed=None,
            contract_gate=contract_gate, seal_gate=seal_gate,
        )
        return {"status": "PARSED", "calls_used": 1, "arm": arm,
                "chunk_id": chunk_id, "construction_stage": stage, "item": item,
                "contract_gate": contract_gate, "seal_gate": seal_gate,
                "gates": gate_result["gates"],
                "passes_confirmatory_gates": gate_result["passes_confirmatory_gates"],
                "input_fingerprint": task.get("input_fingerprint")}

    def _extract(self, task: dict[str, Any]) -> dict[str, Any]:
        kind = task.get("extraction_kind")
        index = task.get("image_index")
        if kind not in {"graph", "caption", "ocr_graph"}:
            raise _blocked("extraction_kind_invalid")
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(self._images):
            raise _blocked("image_index_invalid")
        package, image = self._images[index - 1]
        if task.get("image_sha256") != image.get("sha256"):
            raise _blocked("image_fingerprint_mismatch")

        try:
            image_payload, _audit = image_part(image)
            data_url = image_payload["image_url"]["url"]
            with Image.open(Path(image["path"])) as opened:
                width, height = opened.size
        except Exception as exc:
            raise _blocked(f"image_load_failed:{type(exc).__name__}") from exc

        if kind == "caption":
            payload = caption_request(data_url)
        elif kind == "graph":
            payload = _graph_request(PROMPT, data_url)
        else:
            payload = _graph_request(
                ocr_assisted_graph_prompt(package["evidence"]["text"]), data_url
            )

        metadata = {
            "phase": "extraction",
            "role": "generator",
            "task_id": task.get("task_id"),
            "extraction_kind": kind,
            "image_index": index,
            "image_sha256": image["sha256"],
        }
        body = self._call("generator", payload, metadata=metadata)
        try:
            return parse_extraction_body(
                body, kind=str(kind), image_index=index,
                image_sha256=str(image["sha256"]),
            )
        except Exception as exc:
            # Only parse_extraction_body may classify a response as an observed
            # SCHEMA_REJECTED outcome.  Any other malformed envelope/JSON/schema
            # remains a fail-closed worker error rather than being broadened into
            # an admissible rejection class here.
            raise _blocked(f"extraction_parse_failed:{type(exc).__name__}:{exc}") from exc


__all__ = ["PaidPhaseWorker", "PaidWorkerBlocked", "parse_extraction_body"]
