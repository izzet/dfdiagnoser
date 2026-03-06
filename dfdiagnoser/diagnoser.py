import glob
import io
import json
import os
import time

import pandas as pd
import structlog

from .scoring import score_metrics
from .types import DiagnosisResult
from .utils.log_utils import console_block

logger = structlog.get_logger()


class Diagnoser:
    def __init__(self):
        from .state import DiagnosisStateStore

        self.state = DiagnosisStateStore()

    def diagnose_checkpoint(self, checkpoint_dir: str, metric_boundaries: dict = {}):
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(
                f"Checkpoint directory {checkpoint_dir} does not exist"
            )
        if not os.path.isdir(checkpoint_dir):
            raise NotADirectoryError(
                f"Checkpoint directory {checkpoint_dir} is not a directory"
            )
        if not os.listdir(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} is empty")

        with console_block("Load raw stats"):
            raw_stats_paths = glob.glob(
                os.path.join(checkpoint_dir, "_raw_stats_*.json")
            )
            if not raw_stats_paths:
                raise ValueError(
                    f"Checkpoint directory {checkpoint_dir} does not contain any raw stats files"
                )
            with open(raw_stats_paths[0], "r") as f:
                raw_stats = json.load(f)
        flat_view_paths = glob.glob(
            os.path.join(checkpoint_dir, "_flat_view_*.parquet")
        )
        if not flat_view_paths:
            raise ValueError(
                f"Checkpoint directory {checkpoint_dir} does not contain any flat view files"
            )

        with console_block("Score flat views"):
            scored_flat_views = []
            for flat_view_path in flat_view_paths:
                flat_view = pd.read_parquet(flat_view_path)
                scored_flat_view = score_metrics(flat_view, metric_boundaries)
                scored_flat_views.append(scored_flat_view)

        return DiagnosisResult(
            flat_view_paths=flat_view_paths,
            scored_flat_views=scored_flat_views,
        )

    def diagnose_mofka(
        self,
        group_file: str,
        topic_name: str,
        metric_boundaries: dict = {},
        stop_name: str = "end",
        output_handler=None,
        consumer_name: str = "",
        idle_timeout_sec: int = 30,
        pull_timeout_ms: int = 1000,
        output_topic: str = "",
    ):
        from .streaming.mofka_io import open_consumer, open_producer

        output_handler = output_handler or (lambda result: None)

        driver, consumer = open_consumer(
            group_file, topic_name, consumer_name=consumer_name or None
        )

        # Open producer for publishing findings to optimizer
        findings_producer = None
        if output_topic:
            try:
                _, findings_producer = open_producer(group_file, output_topic)
                logger.info("diagnoser.findings_producer.open", topic=output_topic)
            except Exception:
                logger.warning("diagnoser.findings_producer.failed", exc_info=True)

        event_count = 0
        flat_view_count = 0
        facts_count = 0
        error_count = 0
        last_event_time = None  # None until first event received

        logger.info(
            "diagnoser.stream.start",
            topic=topic_name,
            idle_timeout_sec=idle_timeout_sec,
            pull_timeout_ms=pull_timeout_ms,
        )

        try:
            timeout_count = 0
            wait_ms = pull_timeout_ms if pull_timeout_ms > 0 else 1000

            future = consumer.pull()
            while True:
                # Check idle timeout (only after first event received)
                now = time.monotonic()
                if (
                    last_event_time is not None
                    and idle_timeout_sec > 0
                    and (now - last_event_time) >= idle_timeout_sec
                ):
                    logger.info(
                        "diagnoser.stream.idle_timeout",
                        idle_sec=round(now - last_event_time, 1),
                        threshold_sec=idle_timeout_sec,
                        timeout_count=timeout_count,
                    )
                    break

                # Wait on current future; timeout is raised as exception
                try:
                    event = future.wait(timeout_ms=wait_ms)
                except Exception as ex:
                    ex_msg = str(ex).lower()
                    if "timeout" in ex_msg:
                        timeout_count += 1
                        continue
                    raise

                if event is None:
                    timeout_count += 1
                    continue

                last_event_time = time.monotonic()
                event_count += 1
                raw_metadata = event.metadata if hasattr(event, "metadata") else None
                if isinstance(raw_metadata, dict):
                    metadata = raw_metadata
                elif isinstance(raw_metadata, str):
                    try:
                        metadata = json.loads(raw_metadata)
                    except (ValueError, TypeError):
                        metadata = {"raw": raw_metadata}
                else:
                    metadata = {}

                artifact_type = metadata.get("artifact_type", "flat_view")
                payload = event.data
                payload_size = 0
                if payload is not None:
                    if isinstance(payload, list):
                        payload_size = sum(len(p) for p in payload)
                    elif isinstance(payload, (bytes, bytearray)):
                        payload_size = len(payload)

                logger.info(
                    "diagnoser.event.received",
                    event_index=event_count,
                    artifact_type=artifact_type,
                    metadata_keys=list(metadata.keys()),
                    payload_size=payload_size,
                    timeouts_before=timeout_count,
                )
                timeout_count = 0

                # Check for stop sentinel
                if metadata.get("name") == stop_name:
                    logger.info("diagnoser.stream.stop_sentinel", event_count=event_count)
                    event.acknowledge()
                    break

                try:
                    if artifact_type == "analysis_facts":
                        self._handle_analysis_facts(event, metadata)
                        facts_count += 1
                        # Emit incremental findings after each facts event
                        # so the optimizer can act before the run ends
                        if findings_producer is not None:
                            incremental = self._build_longitudinal_summary()
                            if incremental:
                                self._publish_findings(findings_producer, incremental)
                                logger.info(
                                    "diagnoser.findings.incremental",
                                    count=len(incremental),
                                    window=self.state.current_window,
                                )
                    else:
                        self._handle_flat_view(
                            event, metadata, metric_boundaries, output_handler
                        )
                        flat_view_count += 1
                except Exception:
                    error_count += 1
                    logger.exception(
                        "diagnoser.event.error",
                        artifact_type=artifact_type,
                        event_index=event_count,
                    )

                self.state.advance_window()
                event.acknowledge()
                future = consumer.pull()

        finally:
            logger.info(
                "diagnoser.stream.done",
                event_count=event_count,
                flat_view_count=flat_view_count,
                facts_count=facts_count,
                error_count=error_count,
            )

            # Build longitudinal summary
            findings = self._build_longitudinal_summary()
            if findings:
                for finding in findings:
                    logger.info(
                        "diagnoser.finding",
                        finding_type=finding.finding_type,
                        motif=finding.motif,
                        severity=finding.severity,
                        confidence=round(finding.confidence, 4),
                        prevalence=round(finding.trend.prevalence, 4),
                        persistence=finding.trend.persistence,
                        trend_direction=finding.trend.trend_direction,
                        opportunity_tags=finding.opportunity_tags,
                        contributing_facts=finding.contributing_facts,
                        summary=finding.summary,
                    )

                # Publish findings to Mofka for optimizer consumption
                if findings_producer is not None:
                    self._publish_findings(findings_producer, findings)

            del consumer
            del driver

    def _handle_flat_view(self, event, metadata, metric_boundaries, output_handler):
        payload = event.data
        if payload is None:
            logger.warning("diagnoser.flat_view.no_data")
            return
        if isinstance(payload, list):
            if not payload:
                logger.warning("diagnoser.flat_view.empty_payload")
                return
            payload = b"".join(payload)

        flat_view = pd.read_parquet(io.BytesIO(payload))
        scored_flat_view = score_metrics(flat_view, metric_boundaries)

        # Record score summaries into state
        self.state.record_scored_summary(scored_flat_view)

        result = DiagnosisResult(
            flat_view_paths=[],
            scored_flat_views=[scored_flat_view],
        )
        output_handler(result)

        logger.info(
            "diagnoser.flat_view.scored",
            rows=len(flat_view),
            view_type=metadata.get("view_type", "unknown"),
        )

    def _handle_analysis_facts(self, event, metadata):
        from .state import FactObservation

        payload = event.data
        if payload is None:
            logger.warning("diagnoser.analysis_facts.no_data")
            return
        if isinstance(payload, list):
            if not payload:
                logger.warning("diagnoser.analysis_facts.empty_payload")
                return
            payload = b"".join(payload)

        envelope = json.loads(payload.decode("utf-8"))
        facts = envelope.get("facts", [])

        logger.info(
            "analysis_facts.received",
            fact_count=len(facts),
            view_type=envelope.get("view_type", "unknown"),
        )

        for fact in facts:
            # severity is a nested dict: {"score": float, "label": str, ...}
            severity = fact.get("severity", {})
            if isinstance(severity, dict):
                severity_score = severity.get("score", 0)
                severity_label = severity.get("label", "unknown")
            else:
                severity_score = float(severity) if severity else 0
                severity_label = "unknown"

            # scope is a nested dict: {"entity": str, "layer": str|null, ...}
            scope = fact.get("scope", "global")
            if isinstance(scope, dict):
                scope_key = scope.get("entity", "global")
            else:
                scope_key = str(scope)

            # window may have epoch info
            window = fact.get("window", {})
            epoch = window.get("epoch") if isinstance(window, dict) else None

            obs = FactObservation(
                window_index=self.state.current_window,
                epoch=epoch,
                severity_score=severity_score,
                severity_label=severity_label,
                evidence=fact.get("evidence", {}),
                opportunity_tags=fact.get("opportunity_tags", []),
            )
            key = (fact.get("fact_type", "unknown"), scope_key)
            self.state.record_fact(key, obs)

            logger.info(
                "diagnoser.fact.recorded",
                window_index=self.state.current_window,
                fact_type=fact.get("fact_type"),
                scope=scope_key,
                severity_score=round(severity_score, 3),
                severity_label=severity_label,
                opportunity_tags=fact.get("opportunity_tags", []),
                epoch=epoch,
            )

    def _build_longitudinal_summary(self):
        from .types import DiagnosisFinding, TrendEvidence

        findings = []

        for key, tracker in self.state.all_trackers():
            fact_type, scope = key
            prevalence = tracker.prevalence()
            persistence = tracker.persistence()

            if not tracker.observations:
                continue

            onset_window = tracker.observations[0].window_index
            peak_obs = max(tracker.observations, key=lambda o: o.severity_score)
            peak_window = peak_obs.window_index

            # Determine trend direction
            if len(tracker.observations) >= 2:
                first_half = tracker.observations[: len(tracker.observations) // 2]
                second_half = tracker.observations[len(tracker.observations) // 2 :]
                avg_first = sum(o.severity_score for o in first_half) / len(first_half)
                avg_second = sum(o.severity_score for o in second_half) / len(second_half)
                if avg_second > avg_first * 1.2:
                    trend_direction = "worsening"
                elif avg_second < avg_first * 0.8:
                    trend_direction = "improving"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "insufficient_data"

            trend = TrendEvidence(
                prevalence=prevalence,
                persistence=persistence,
                onset_window=onset_window,
                peak_severity_window=peak_window,
                trend_direction=trend_direction,
            )

            # Motif classification
            motif, recommendation, confidence = self._classify_motif(
                fact_type, tracker, prevalence, persistence, onset_window, trend_direction
            )

            # Collect all opportunity_tags from observations (deduplicated, ordered)
            all_tags = []
            seen_tags = set()
            for obs in tracker.observations:
                for tag in obs.opportunity_tags:
                    if tag not in seen_tags and tag != "none":
                        all_tags.append(tag)
                        seen_tags.add(tag)

            summary = (
                f"{fact_type}({scope}): motif={motif}, "
                f"prevalence={prevalence:.2f}, persistence={persistence}, "
                f"trend={trend_direction}"
            )

            finding = DiagnosisFinding(
                finding_type=fact_type,
                motif=motif,
                severity=peak_obs.severity_label,
                confidence=confidence,
                trend=trend,
                contributing_facts=[(fact_type, scope)],
                recommendation_bundle=recommendation,
                summary=summary,
                opportunity_tags=all_tags,
            )
            findings.append(finding)

        return findings

    def _classify_motif(
        self, fact_type, tracker, prevalence, persistence, onset_window, trend_direction
    ):
        total_windows = self.state.current_window
        if total_windows == 0:
            total_windows = 1

        # warmup_transient: high severity in first 1-2 windows, declining after
        if onset_window <= 1 and trend_direction == "improving" and prevalence < 0.4:
            return "warmup_transient", "none", 0.7

        # rank_skew_induced: co-occurrence of fetch_imbalance + straggler
        all_fact_types = {k[0] for k, _ in self.state.all_trackers()}
        if (
            "fetch_imbalance" in all_fact_types
            and "straggler" in all_fact_types
            and fact_type in ("fetch_imbalance", "straggler")
        ):
            return "rank_skew_induced", "rank_balance_repartition", 0.75

        # checkpoint_tail_risk
        if "checkpoint" in fact_type and prevalence > 0.3:
            return "checkpoint_tail_risk", "checkpoint_io_batching", 0.65

        # persistent_pressure: prevalence > 0.5, persistence > 3
        if prevalence > 0.5 and persistence > 3:
            return "persistent_pressure", "input_pipeline_tuning", 0.8

        return "unclassified", "investigate", 0.5

    def _publish_findings(self, producer, findings):
        """Publish DiagnosisFindings to Mofka for optimizer consumption."""
        import dataclasses

        for finding in findings:
            payload_dict = {
                "finding_type": finding.finding_type,
                "motif": finding.motif,
                "severity": finding.severity,
                "confidence": finding.confidence,
                "prevalence": finding.trend.prevalence,
                "persistence": finding.trend.persistence,
                "trend_direction": finding.trend.trend_direction,
                "contributing_facts": finding.contributing_facts,
                "recommendation_bundle": finding.recommendation_bundle,
                "opportunity_tags": finding.opportunity_tags,
                "summary": finding.summary,
                "window_index": finding.trend.peak_severity_window,
            }
            payload = json.dumps(payload_dict).encode("utf-8")
            metadata = {
                "type": "diagnosis_finding",
                "finding_type": finding.finding_type,
                "motif": finding.motif,
            }
            try:
                producer.push(metadata=metadata, data=payload)
                logger.info(
                    "diagnoser.finding.published",
                    finding_type=finding.finding_type,
                    motif=finding.motif,
                    tags=finding.opportunity_tags,
                )
            except Exception:
                logger.exception("diagnoser.finding.publish_failed")

        try:
            producer.flush()
            logger.info("diagnoser.findings.flushed", count=len(findings))
        except Exception:
            logger.exception("diagnoser.findings.flush_failed")

    def _diagnose(self, data: dict):
        pass
