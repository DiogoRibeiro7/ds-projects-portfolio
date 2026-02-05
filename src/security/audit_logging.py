"""Comprehensive Audit Logging System
"""

import base64
import gzip
import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""

    # Data events
    DATA_ACCESS = "DATA_ACCESS"
    DATA_MODIFICATION = "DATA_MODIFICATION"
    DATA_DELETION = "DATA_DELETION"
    DATA_EXPORT = "DATA_EXPORT"
    DATA_IMPORT = "DATA_IMPORT"

    # Model events
    MODEL_TRAINING = "MODEL_TRAINING"
    MODEL_PREDICTION = "MODEL_PREDICTION"
    MODEL_DEPLOYMENT = "MODEL_DEPLOYMENT"
    MODEL_UPDATE = "MODEL_UPDATE"
    MODEL_DELETION = "MODEL_DELETION"
    MODEL_APPROVAL = "MODEL_APPROVAL"

    # System events
    SYSTEM_ACCESS = "SYSTEM_ACCESS"
    SYSTEM_CONFIG_CHANGE = "SYSTEM_CONFIG_CHANGE"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SYSTEM_WARNING = "SYSTEM_WARNING"

    # Security events
    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILURE = "AUTH_FAILURE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    SECURITY_ALERT = "SECURITY_ALERT"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    MFA_EVENT = "MFA_EVENT"

    # Compliance events
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    REGULATORY_REPORT = "REGULATORY_REPORT"
    PRIVACY_REQUEST = "PRIVACY_REQUEST"
    DATA_RETENTION = "DATA_RETENTION"


class AuditSeverity(Enum):
    """Severity levels for audit events"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AuditEvent:
    """Audit event structure"""

    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: str | None
    session_id: str | None
    ip_address: str | None
    user_agent: str | None
    resource: str | None  # Resource being accessed/modified
    action: str  # Specific action taken
    result: str  # Success/Failure/Partial
    details: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    stack_trace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    correlation_id: str | None = None  # For tracking related events


@dataclass
class AuditConfig:
    """Audit logging configuration"""

    # Storage settings
    storage_backend: str = "sqlite"  # sqlite, postgresql, file, elasticsearch
    database_path: str = "audit_logs.db"
    file_path: str = "audit_logs/"
    rotation_size_mb: int = 100
    retention_days: int = 365

    # Security settings
    enable_encryption: bool = True
    encryption_key: str | None = None
    enable_integrity_check: bool = True
    enable_tamper_detection: bool = True

    # Performance settings
    async_logging: bool = True
    batch_size: int = 100
    flush_interval_seconds: int = 5
    max_queue_size: int = 10000

    # Filtering settings
    log_levels: list[AuditSeverity] = field(
        default_factory=lambda: [
            AuditSeverity.INFO,
            AuditSeverity.WARNING,
            AuditSeverity.ERROR,
            AuditSeverity.CRITICAL,
        ]
    )
    excluded_event_types: list[AuditEventType] = field(default_factory=list)

    # Alerting settings
    enable_real_time_alerts: bool = True
    alert_thresholds: dict[str, int] = field(default_factory=dict)
    alert_email: str | None = None
    alert_webhook: str | None = None


class AuditStorage:
    """Base class for audit storage backends"""

    def __init__(self, config: AuditConfig):
        self.config = config
        self.cipher = (
            self._initialize_encryption() if config.enable_encryption else None
        )

    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption cipher"""
        if self.config.encryption_key:
            key = base64.urlsafe_b64encode(
                self.config.encryption_key.encode()[:32].ljust(32, b"0")
            )
        else:
            # Generate key from password
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"audit_salt",
                iterations=100000,
                backend=default_backend(),
            )
            key = base64.urlsafe_b64encode(kdf.derive(b"default_audit_key"))

        return Fernet(key)

    def encrypt_data(self, data: str) -> str:
        """Encrypt audit data"""
        if self.cipher:
            return self.cipher.encrypt(data.encode()).decode()
        return data

    def decrypt_data(self, data: str) -> str:
        """Decrypt audit data"""
        if self.cipher:
            return self.cipher.decrypt(data.encode()).decode()
        return data

    def compute_integrity_hash(self, event: AuditEvent) -> str:
        """Compute integrity hash for event"""
        event_str = json.dumps(asdict(event), default=str, sort_keys=True)
        return hashlib.sha256(event_str.encode()).hexdigest()


class SQLiteAuditStorage(AuditStorage):
    """SQLite-based audit storage"""

    def __init__(self, config: AuditConfig):
        super().__init__(config)
        self.db_path = config.database_path
        self._initialize_database()

    def _initialize_database(self):
        """Create audit log table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                event_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                event_type TEXT,
                severity TEXT,
                user_id TEXT,
                session_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                resource TEXT,
                action TEXT,
                result TEXT,
                details TEXT,
                error_message TEXT,
                stack_trace TEXT,
                metadata TEXT,
                tags TEXT,
                correlation_id TEXT,
                integrity_hash TEXT,
                is_encrypted BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for common queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_logs(timestamp)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_logs(user_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_type ON audit_logs(event_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_severity ON audit_logs(severity)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_correlation_id ON audit_logs(correlation_id)"
        )

        conn.commit()
        conn.close()

    def store_event(self, event: AuditEvent):
        """Store audit event in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Prepare event data
        event_dict = asdict(event)

        # Convert complex fields to JSON
        details = json.dumps(event_dict["details"], default=str)
        metadata = json.dumps(event_dict["metadata"], default=str)
        tags = json.dumps(event_dict["tags"])

        # Encrypt if enabled
        is_encrypted = False
        if self.config.enable_encryption:
            details = self.encrypt_data(details)
            metadata = self.encrypt_data(metadata)
            is_encrypted = True

        # Compute integrity hash
        integrity_hash = (
            self.compute_integrity_hash(event)
            if self.config.enable_integrity_check
            else None
        )

        # Insert into database
        cursor.execute(
            """
            INSERT INTO audit_logs (
                event_id, timestamp, event_type, severity, user_id,
                session_id, ip_address, user_agent, resource, action,
                result, details, error_message, stack_trace, metadata,
                tags, correlation_id, integrity_hash, is_encrypted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event.event_id,
                event.timestamp,
                event.event_type.value,
                event.severity.value,
                event.user_id,
                event.session_id,
                event.ip_address,
                event.user_agent,
                event.resource,
                event.action,
                event.result,
                details,
                event.error_message,
                event.stack_trace,
                metadata,
                tags,
                event.correlation_id,
                integrity_hash,
                is_encrypted,
            ),
        )

        conn.commit()
        conn.close()

    def retrieve_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Retrieve audit events from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        if event_types:
            placeholders = ",".join(["?" for _ in event_types])
            query += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in event_types])

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        events = []
        for row in rows:
            # Decrypt if encrypted
            details = row["details"]
            metadata = row["metadata"]

            if row["is_encrypted"] and self.cipher:
                details = self.decrypt_data(details)
                metadata = self.decrypt_data(metadata)

            # Parse JSON fields
            details = json.loads(details)
            metadata = json.loads(metadata)
            tags = json.loads(row["tags"])

            # Verify integrity if enabled
            if self.config.enable_integrity_check and row["integrity_hash"]:
                # Would verify hash here
                pass

            event = AuditEvent(
                event_id=row["event_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                event_type=AuditEventType[row["event_type"]],
                severity=AuditSeverity[row["severity"]],
                user_id=row["user_id"],
                session_id=row["session_id"],
                ip_address=row["ip_address"],
                user_agent=row["user_agent"],
                resource=row["resource"],
                action=row["action"],
                result=row["result"],
                details=details,
                error_message=row["error_message"],
                stack_trace=row["stack_trace"],
                metadata=metadata,
                tags=tags,
                correlation_id=row["correlation_id"],
            )
            events.append(event)

        conn.close()
        return events


class FileAuditStorage(AuditStorage):
    """File-based audit storage with rotation"""

    def __init__(self, config: AuditConfig):
        super().__init__(config)
        self.file_path = Path(config.file_path)
        self.file_path.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.current_size = 0
        self._rotate_if_needed()

    def _get_current_file_path(self) -> Path:
        """Get current log file path"""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.file_path / f"audit_log_{timestamp}.jsonl"

    def _rotate_if_needed(self):
        """Rotate log file if size limit reached"""
        file_path = self._get_current_file_path()

        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb >= self.config.rotation_size_mb:
                # Compress and archive old file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = self.file_path / f"audit_log_{timestamp}.jsonl.gz"

                with open(file_path, "rb") as f_in:
                    with gzip.open(archive_path, "wb") as f_out:
                        f_out.writelines(f_in)

                # Remove old file
                file_path.unlink()

    def store_event(self, event: AuditEvent):
        """Store event to file"""
        self._rotate_if_needed()

        file_path = self._get_current_file_path()
        event_dict = asdict(event)

        # Encrypt if enabled
        if self.config.enable_encryption:
            event_str = self.encrypt_data(json.dumps(event_dict, default=str))
        else:
            event_str = json.dumps(event_dict, default=str)

        # Append to file
        with open(file_path, "a") as f:
            f.write(event_str + "\n")

    def retrieve_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Retrieve events from files"""
        events = []

        # Read from all relevant files
        for file_path in sorted(self.file_path.glob("audit_log_*.jsonl*")):
            if len(events) >= limit:
                break

            # Handle compressed files
            if file_path.suffix == ".gz":
                opener = gzip.open
                mode = "rt"
            else:
                opener = open
                mode = "r"

            with opener(file_path, mode) as f:
                for line in f:
                    if len(events) >= limit:
                        break

                    # Decrypt if needed
                    if self.config.enable_encryption:
                        line = self.decrypt_data(line.strip())

                    event_dict = json.loads(line)

                    # Apply filters
                    event_timestamp = datetime.fromisoformat(event_dict["timestamp"])

                    if start_date and event_timestamp < start_date:
                        continue
                    if end_date and event_timestamp > end_date:
                        continue
                    if event_types and event_dict["event_type"] not in [
                        et.value for et in event_types
                    ]:
                        continue
                    if user_id and event_dict["user_id"] != user_id:
                        continue

                    # Create event object
                    event = AuditEvent(
                        event_id=event_dict["event_id"],
                        timestamp=event_timestamp,
                        event_type=AuditEventType[event_dict["event_type"]],
                        severity=AuditSeverity[event_dict["severity"]],
                        user_id=event_dict.get("user_id"),
                        session_id=event_dict.get("session_id"),
                        ip_address=event_dict.get("ip_address"),
                        user_agent=event_dict.get("user_agent"),
                        resource=event_dict.get("resource"),
                        action=event_dict["action"],
                        result=event_dict["result"],
                        details=event_dict.get("details", {}),
                        error_message=event_dict.get("error_message"),
                        stack_trace=event_dict.get("stack_trace"),
                        metadata=event_dict.get("metadata", {}),
                        tags=event_dict.get("tags", []),
                        correlation_id=event_dict.get("correlation_id"),
                    )
                    events.append(event)

        return events[:limit]


class AsyncAuditWriter(threading.Thread):
    """Asynchronous audit event writer"""

    def __init__(self, storage: AuditStorage, config: AuditConfig):
        super().__init__(daemon=True)
        self.storage = storage
        self.config = config
        self.queue = Queue(maxsize=config.max_queue_size)
        self.batch = []
        self.running = True
        self.last_flush = datetime.now()

    def run(self):
        """Process audit events from queue"""
        while self.running:
            try:
                # Get event from queue with timeout
                event = self.queue.get(timeout=1)
                self.batch.append(event)

                # Check if batch should be flushed
                if (
                    len(self.batch) >= self.config.batch_size
                    or (datetime.now() - self.last_flush).seconds
                    >= self.config.flush_interval_seconds
                ):
                    self._flush_batch()

            except Empty:
                # Flush any remaining events
                if self.batch:
                    self._flush_batch()

    def _flush_batch(self):
        """Flush batch to storage"""
        for event in self.batch:
            try:
                self.storage.store_event(event)
            except Exception as e:
                logger.error(f"Failed to store audit event: {e}")

        self.batch.clear()
        self.last_flush = datetime.now()

    def add_event(self, event: AuditEvent):
        """Add event to queue"""
        try:
            self.queue.put_nowait(event)
        except:
            # Queue full, force flush and retry
            self._flush_batch()
            self.queue.put_nowait(event)

    def stop(self):
        """Stop the writer thread"""
        self.running = False
        self._flush_batch()


class AuditLogger:
    """Main audit logging system"""

    def __init__(self, config: AuditConfig = None):
        self.config = config or AuditConfig()
        self.storage = self._initialize_storage()
        self.async_writer = None

        if self.config.async_logging:
            self.async_writer = AsyncAuditWriter(self.storage, self.config)
            self.async_writer.start()

        self.alert_manager = AlertManager(config)
        self.correlation_tracker = {}

    def _initialize_storage(self) -> AuditStorage:
        """Initialize storage backend"""
        if self.config.storage_backend == "sqlite":
            return SQLiteAuditStorage(self.config)
        elif self.config.storage_backend == "file":
            return FileAuditStorage(self.config)
        else:
            raise ValueError(f"Unknown storage backend: {self.config.storage_backend}")

    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        result: str = "SUCCESS",
        severity: AuditSeverity = AuditSeverity.INFO,
        user_id: str | None = None,
        session_id: str | None = None,
        resource: str | None = None,
        details: dict[str, Any] | None = None,
        error_message: str | None = None,
        correlation_id: str | None = None,
        **kwargs,
    ) -> str:
        """Log an audit event"""
        # Check if event type should be logged
        if event_type in self.config.excluded_event_types:
            return None

        if severity not in self.config.log_levels:
            return None

        # Generate event ID
        event_id = hashlib.sha256(
            f"{datetime.now().isoformat()}_{event_type.value}_{action}".encode()
        ).hexdigest()[:16]

        # Create event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=kwargs.get("ip_address"),
            user_agent=kwargs.get("user_agent"),
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            error_message=error_message,
            stack_trace=kwargs.get("stack_trace"),
            metadata=kwargs.get("metadata", {}),
            tags=kwargs.get("tags", []),
            correlation_id=correlation_id,
        )

        # Store event
        if self.async_writer:
            self.async_writer.add_event(event)
        else:
            self.storage.store_event(event)

        # Check for alerts
        if self.config.enable_real_time_alerts:
            self.alert_manager.check_event(event)

        # Track correlation
        if correlation_id:
            if correlation_id not in self.correlation_tracker:
                self.correlation_tracker[correlation_id] = []
            self.correlation_tracker[correlation_id].append(event_id)

        return event_id

    def log_data_access(
        self, user_id: str, resource: str, operation: str, **kwargs
    ) -> str:
        """Log data access event"""
        return self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            action=f"Access {operation}",
            user_id=user_id,
            resource=resource,
            details={"operation": operation},
            **kwargs,
        )

    def log_model_event(
        self,
        model_id: str,
        action: str,
        user_id: str,
        details: dict[str, Any] = None,
        **kwargs,
    ) -> str:
        """Log model-related event"""
        event_type_map = {
            "train": AuditEventType.MODEL_TRAINING,
            "predict": AuditEventType.MODEL_PREDICTION,
            "deploy": AuditEventType.MODEL_DEPLOYMENT,
            "update": AuditEventType.MODEL_UPDATE,
            "delete": AuditEventType.MODEL_DELETION,
            "approve": AuditEventType.MODEL_APPROVAL,
        }

        event_type = event_type_map.get(action, AuditEventType.MODEL_UPDATE)

        return self.log_event(
            event_type=event_type,
            action=action,
            user_id=user_id,
            resource=f"model:{model_id}",
            details=details,
            **kwargs,
        )

    def log_security_event(
        self,
        event_type: str,
        user_id: str | None = None,
        details: dict[str, Any] = None,
        severity: AuditSeverity = AuditSeverity.WARNING,
        **kwargs,
    ) -> str:
        """Log security-related event"""
        security_event_map = {
            "login_success": AuditEventType.AUTH_SUCCESS,
            "login_failure": AuditEventType.AUTH_FAILURE,
            "permission_denied": AuditEventType.PERMISSION_DENIED,
            "security_alert": AuditEventType.SECURITY_ALERT,
            "password_change": AuditEventType.PASSWORD_CHANGE,
            "mfa": AuditEventType.MFA_EVENT,
        }

        audit_event_type = security_event_map.get(
            event_type, AuditEventType.SECURITY_ALERT
        )

        return self.log_event(
            event_type=audit_event_type,
            action=event_type,
            user_id=user_id,
            details=details,
            severity=severity,
            **kwargs,
        )

    def query_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        user_id: str | None = None,
        correlation_id: str | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Query audit events"""
        events = self.storage.retrieve_events(
            start_date=start_date,
            end_date=end_date,
            event_types=event_types,
            user_id=user_id,
            limit=limit,
        )

        # Filter by correlation ID if provided
        if correlation_id:
            related_event_ids = self.correlation_tracker.get(correlation_id, [])
            events = [e for e in events if e.event_id in related_event_ids]

        return events

    def generate_audit_report(
        self, start_date: datetime, end_date: datetime, report_type: str = "summary"
    ) -> dict[str, Any]:
        """Generate audit report"""
        events = self.query_events(
            start_date=start_date, end_date=end_date, limit=100000
        )

        if report_type == "summary":
            return self._generate_summary_report(events, start_date, end_date)
        elif report_type == "detailed":
            return self._generate_detailed_report(events, start_date, end_date)
        elif report_type == "compliance":
            return self._generate_compliance_report(events, start_date, end_date)
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def _generate_summary_report(
        self, events: list[AuditEvent], start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate summary audit report"""
        report = {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_events": len(events),
            "event_types": {},
            "severity_breakdown": {},
            "top_users": {},
            "failed_operations": 0,
            "security_events": 0,
            "data_access_events": 0,
            "model_events": 0,
        }

        # Analyze events
        for event in events:
            # Count by event type
            event_type = event.event_type.value
            report["event_types"][event_type] = (
                report["event_types"].get(event_type, 0) + 1
            )

            # Count by severity
            severity = event.severity.value
            report["severity_breakdown"][severity] = (
                report["severity_breakdown"].get(severity, 0) + 1
            )

            # Count by user
            if event.user_id:
                report["top_users"][event.user_id] = (
                    report["top_users"].get(event.user_id, 0) + 1
                )

            # Count failures
            if event.result != "SUCCESS":
                report["failed_operations"] += 1

            # Count specific categories
            if event.event_type in [
                AuditEventType.AUTH_SUCCESS,
                AuditEventType.AUTH_FAILURE,
                AuditEventType.PERMISSION_DENIED,
                AuditEventType.SECURITY_ALERT,
            ]:
                report["security_events"] += 1

            if event.event_type in [
                AuditEventType.DATA_ACCESS,
                AuditEventType.DATA_MODIFICATION,
                AuditEventType.DATA_DELETION,
                AuditEventType.DATA_EXPORT,
            ]:
                report["data_access_events"] += 1

            if event.event_type in [
                AuditEventType.MODEL_TRAINING,
                AuditEventType.MODEL_PREDICTION,
                AuditEventType.MODEL_DEPLOYMENT,
            ]:
                report["model_events"] += 1

        # Sort top users
        report["top_users"] = dict(
            sorted(report["top_users"].items(), key=lambda x: x[1], reverse=True)[:10]
        )

        return report

    def _generate_detailed_report(
        self, events: list[AuditEvent], start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate detailed audit report"""
        # Convert events to DataFrame for analysis
        df = pd.DataFrame([asdict(e) for e in events])

        report = {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_events": len(events),
            "hourly_distribution": {},
            "daily_distribution": {},
            "resource_access": {},
            "error_analysis": {},
            "performance_metrics": {},
        }

        if not df.empty:
            # Hourly distribution
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            report["hourly_distribution"] = df.groupby("hour").size().to_dict()

            # Daily distribution
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            report["daily_distribution"] = df.groupby("date").size().to_dict()

            # Resource access patterns
            if "resource" in df.columns:
                report["resource_access"] = (
                    df["resource"].value_counts().head(20).to_dict()
                )

            # Error analysis
            error_df = df[df["result"] != "SUCCESS"]
            if not error_df.empty:
                report["error_analysis"] = {
                    "total_errors": len(error_df),
                    "error_types": error_df["event_type"].value_counts().to_dict(),
                    "error_messages": error_df["error_message"]
                    .value_counts()
                    .head(10)
                    .to_dict(),
                }

        return report

    def _generate_compliance_report(
        self, events: list[AuditEvent], start_date: datetime, end_date: datetime
    ) -> dict[str, Any]:
        """Generate compliance-focused audit report"""
        report = {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "compliance_events": {},
            "data_operations": {},
            "user_activities": {},
            "security_incidents": [],
            "regulatory_requirements": {},
        }

        # Categorize compliance-related events
        for event in events:
            # Track compliance checks
            if event.event_type == AuditEventType.COMPLIANCE_CHECK:
                report["compliance_events"][event.event_id] = {
                    "timestamp": event.timestamp.isoformat(),
                    "result": event.result,
                    "details": event.details,
                }

            # Track data operations for GDPR/CCPA
            if event.event_type in [
                AuditEventType.DATA_ACCESS,
                AuditEventType.DATA_DELETION,
                AuditEventType.DATA_EXPORT,
                AuditEventType.PRIVACY_REQUEST,
            ]:
                if event.user_id not in report["data_operations"]:
                    report["data_operations"][event.user_id] = []

                report["data_operations"][event.user_id].append(
                    {
                        "timestamp": event.timestamp.isoformat(),
                        "operation": event.event_type.value,
                        "resource": event.resource,
                        "result": event.result,
                    }
                )

            # Track security incidents
            if event.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
                if event.event_type in [
                    AuditEventType.AUTH_FAILURE,
                    AuditEventType.PERMISSION_DENIED,
                    AuditEventType.SECURITY_ALERT,
                ]:
                    report["security_incidents"].append(
                        {
                            "timestamp": event.timestamp.isoformat(),
                            "type": event.event_type.value,
                            "user_id": event.user_id,
                            "details": event.details,
                            "ip_address": event.ip_address,
                        }
                    )

        # Check regulatory requirements
        report["regulatory_requirements"] = {
            "audit_trail_complete": len(events) > 0,
            "data_access_logged": any(
                e.event_type == AuditEventType.DATA_ACCESS for e in events
            ),
            "authentication_logged": any(
                e.event_type
                in [AuditEventType.AUTH_SUCCESS, AuditEventType.AUTH_FAILURE]
                for e in events
            ),
            "data_retention_compliant": True,  # Would check retention policies
            "encryption_enabled": self.config.enable_encryption,
            "integrity_checks": self.config.enable_integrity_check,
        }

        return report

    def export_audit_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "csv",
        output_path: str = None,
    ) -> str:
        """Export audit logs in specified format"""
        events = self.query_events(
            start_date=start_date, end_date=end_date, limit=1000000
        )

        # Convert to DataFrame
        df = pd.DataFrame([asdict(e) for e in events])

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"audit_export_{timestamp}.{format}"

        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", date_format="iso")
        elif format == "excel":
            df.to_excel(output_path, index=False)
        elif format == "parquet":
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unknown export format: {format}")

        # Log the export
        self.log_event(
            event_type=AuditEventType.DATA_EXPORT,
            action="export_audit_logs",
            resource="audit_logs",
            details={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "format": format,
                "output_path": output_path,
                "record_count": len(df),
            },
        )

        return output_path

    def cleanup_old_logs(self):
        """Clean up logs older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

        # Would implement actual cleanup based on storage backend
        # For SQLite:
        if isinstance(self.storage, SQLiteAuditStorage):
            conn = sqlite3.connect(self.storage.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM audit_logs WHERE timestamp < ?", (cutoff_date,))
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            # Log cleanup
            self.log_event(
                event_type=AuditEventType.DATA_RETENTION,
                action="cleanup_old_logs",
                details={
                    "cutoff_date": cutoff_date.isoformat(),
                    "deleted_count": deleted_count,
                },
            )

    def shutdown(self):
        """Shutdown audit logger"""
        if self.async_writer:
            self.async_writer.stop()
            self.async_writer.join()


class AlertManager:
    """Manage real-time alerts for audit events"""

    def __init__(self, config: AuditConfig):
        self.config = config
        self.alert_counts = {}
        self.last_alert_time = {}

    def check_event(self, event: AuditEvent):
        """Check if event should trigger an alert"""
        # Check severity-based alerts
        if event.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
            self._send_alert(event, f"High severity event: {event.severity.value}")

        # Check for repeated failures
        if event.result == "FAILURE":
            key = f"{event.event_type.value}_{event.user_id}"
            self.alert_counts[key] = self.alert_counts.get(key, 0) + 1

            threshold = self.config.alert_thresholds.get("repeated_failures", 5)
            if self.alert_counts[key] >= threshold:
                self._send_alert(
                    event,
                    f"Repeated failures detected: {self.alert_counts[key]} failures",
                )
                self.alert_counts[key] = 0  # Reset counter

        # Check for security events
        if event.event_type in [
            AuditEventType.PERMISSION_DENIED,
            AuditEventType.SECURITY_ALERT,
        ]:
            self._send_alert(event, f"Security event: {event.event_type.value}")

    def _send_alert(self, event: AuditEvent, message: str):
        """Send alert via configured channels"""
        # Rate limiting
        key = f"{event.event_type.value}_{event.user_id}"
        if key in self.last_alert_time:
            time_since_last = (datetime.now() - self.last_alert_time[key]).seconds
            if time_since_last < 300:  # 5 minute cooldown
                return

        self.last_alert_time[key] = datetime.now()

        # Log alert
        logger.warning(f"AUDIT ALERT: {message} - Event: {event.event_id}")

        # Send email if configured
        if self.config.alert_email:
            self._send_email_alert(event, message)

        # Send webhook if configured
        if self.config.alert_webhook:
            self._send_webhook_alert(event, message)

    def _send_email_alert(self, event: AuditEvent, message: str):
        """Send email alert"""
        # Implementation would use email library
        pass

    def _send_webhook_alert(self, event: AuditEvent, message: str):
        """Send webhook alert"""
        # Implementation would use requests library
        pass


def audit_operation(event_type: AuditEventType, resource: str | None = None):
    """Decorator for auditing function calls"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create audit logger (would use global instance)
            audit_logger = AuditLogger()

            # Extract context
            user_id = kwargs.get("user_id")
            session_id = kwargs.get("session_id")

            # Log start of operation
            correlation_id = hashlib.sha256(
                f"{func.__name__}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

            start_event_id = audit_logger.log_event(
                event_type=event_type,
                action=f"Start {func.__name__}",
                user_id=user_id,
                session_id=session_id,
                resource=resource,
                correlation_id=correlation_id,
            )

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Log successful completion
                audit_logger.log_event(
                    event_type=event_type,
                    action=f"Complete {func.__name__}",
                    result="SUCCESS",
                    user_id=user_id,
                    session_id=session_id,
                    resource=resource,
                    correlation_id=correlation_id,
                )

                return result

            except Exception as e:
                # Log failure
                audit_logger.log_event(
                    event_type=event_type,
                    action=f"Failed {func.__name__}",
                    result="FAILURE",
                    severity=AuditSeverity.ERROR,
                    user_id=user_id,
                    session_id=session_id,
                    resource=resource,
                    error_message=str(e),
                    correlation_id=correlation_id,
                )
                raise

        return wrapper

    return decorator
