"""Privacy Protection Module
Differential privacy, PII detection, data anonymization, and GDPR compliance
"""

import base64
import hashlib
import json
import logging
import re
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import phonenumbers

# PII detection
import spacy
from creditcard import CreditCard

# Encryption
from cryptography.fernet import Fernet

# Differential privacy
from diffprivlib import mechanisms
from diffprivlib.models import GaussianNB
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from diffprivlib.tools import mean, median
from diffprivlib.utils import PrivacyLeakWarning
from faker import Faker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker for anonymization
fake = Faker()

# Suppress privacy warnings for demo
warnings.filterwarnings("ignore", category=PrivacyLeakWarning)


class PIIType(Enum):
    """Types of Personally Identifiable Information"""

    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    IP_ADDRESS = "ip_address"
    MEDICAL_ID = "medical_id"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    CUSTOM = "custom"


@dataclass
class PrivacyConfig:
    """Privacy configuration"""

    # Differential privacy
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5  # Privacy parameter
    sensitivity: float = 1.0  # Query sensitivity

    # PII detection
    enable_pii_detection: bool = True
    pii_confidence_threshold: float = 0.8

    # Anonymization
    anonymization_method: str = (
        "generalization"  # generalization, suppression, masking, encryption
    )
    preserve_format: bool = True

    # GDPR
    enable_gdpr_compliance: bool = True
    data_retention_days: int = 365
    require_consent: bool = True
    allow_data_portability: bool = True
    enable_right_to_be_forgotten: bool = True

    # Audit
    enable_audit_logging: bool = True
    log_retention_days: int = 90


@dataclass
class PIIDetectionResult:
    """Result of PII detection"""

    column: str
    pii_type: PIIType
    confidence: float
    sample_values: list[str]
    detection_method: str


@dataclass
class AnonymizationResult:
    """Result of data anonymization"""

    original_shape: tuple[int, int]
    anonymized_shape: tuple[int, int]
    pii_columns_found: list[str]
    anonymization_methods_used: dict[str, str]
    k_anonymity: int | None = None
    l_diversity: int | None = None
    t_closeness: float | None = None


class DifferentialPrivacy:
    """Differential privacy implementation"""

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.privacy_budget = config.epsilon
        self.used_budget = 0.0

    def add_laplace_noise(
        self, data: np.ndarray, sensitivity: float = None
    ) -> np.ndarray:
        """Add Laplace noise for differential privacy"""
        if sensitivity is None:
            sensitivity = self.config.sensitivity

        # Calculate scale
        scale = sensitivity / self.config.epsilon

        # Add Laplace noise
        mechanism = mechanisms.Laplace(
            epsilon=self.config.epsilon, sensitivity=sensitivity
        )
        noisy_data = mechanism.randomise(data)

        # Update used budget
        self.used_budget += self.config.epsilon

        return noisy_data

    def add_gaussian_noise(
        self, data: np.ndarray, sensitivity: float = None
    ) -> np.ndarray:
        """Add Gaussian noise for differential privacy"""
        if sensitivity is None:
            sensitivity = self.config.sensitivity

        mechanism = mechanisms.Gaussian(
            epsilon=self.config.epsilon,
            delta=self.config.delta,
            sensitivity=sensitivity,
        )

        noisy_data = mechanism.randomise(data)
        self.used_budget += self.config.epsilon

        return noisy_data

    def private_mean(
        self, data: np.ndarray, bounds: tuple[float, float] = None
    ) -> float:
        """Calculate differentially private mean"""
        if bounds is None:
            bounds = (data.min(), data.max())

        dp_mean = mean(data, epsilon=self.config.epsilon, bounds=bounds)
        self.used_budget += self.config.epsilon

        return dp_mean

    def private_median(
        self, data: np.ndarray, bounds: tuple[float, float] = None
    ) -> float:
        """Calculate differentially private median"""
        if bounds is None:
            bounds = (data.min(), data.max())

        dp_median = median(data, epsilon=self.config.epsilon, bounds=bounds)
        self.used_budget += self.config.epsilon

        return dp_median

    def private_histogram(
        self, data: np.ndarray, bins: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create differentially private histogram"""
        # Get histogram
        counts, bin_edges = np.histogram(data, bins=bins)

        # Add noise to counts
        noisy_counts = self.add_laplace_noise(counts, sensitivity=1)

        # Ensure non-negative counts
        noisy_counts = np.maximum(noisy_counts, 0)

        return noisy_counts, bin_edges

    def private_query(
        self, query_func: Callable, data: Any, sensitivity: float = None
    ) -> Any:
        """Execute differentially private query"""
        result = query_func(data)

        if isinstance(result, (int, float)):
            result = self.add_laplace_noise(np.array([result]), sensitivity)[0]
        elif isinstance(result, np.ndarray):
            result = self.add_laplace_noise(result, sensitivity)

        return result

    def train_private_model(
        self, X: np.ndarray, y: np.ndarray, model_type: str = "logistic"
    ) -> Any:
        """Train differentially private ML model"""
        if model_type == "logistic":
            model = DPLogisticRegression(epsilon=self.config.epsilon)
        elif model_type == "naive_bayes":
            model = GaussianNB(epsilon=self.config.epsilon)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X, y)
        self.used_budget += self.config.epsilon

        return model

    def get_privacy_budget_status(self) -> dict[str, float]:
        """Get privacy budget status"""
        return {
            "total_budget": self.privacy_budget,
            "used_budget": self.used_budget,
            "remaining_budget": self.privacy_budget - self.used_budget,
            "usage_percentage": (self.used_budget / self.privacy_budget * 100)
            if self.privacy_budget > 0
            else 0,
        }


class PIIDetector:
    """PII detection and identification"""

    def __init__(self, config: PrivacyConfig):
        self.config = config

        # Load spaCy model for NER (install: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            logger.warning(
                "spaCy model not loaded. Some PII detection features disabled."
            )

        # Regex patterns for PII
        self.patterns = {
            PIIType.EMAIL: re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ),
            PIIType.PHONE: re.compile(
                r"(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
            ),
            PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b"),
            PIIType.CREDIT_CARD: re.compile(
                r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
            ),
            PIIType.IP_ADDRESS: re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            PIIType.DATE_OF_BIRTH: re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
        }

    def detect_pii(self, df: pd.DataFrame) -> list[PIIDetectionResult]:
        """Detect PII in DataFrame"""
        results = []

        for column in df.columns:
            # Skip numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                continue

            # Sample values for testing
            sample_values = df[column].dropna().astype(str).head(100)

            # Check each PII pattern
            for pii_type, pattern in self.patterns.items():
                matches = sample_values.apply(lambda x: bool(pattern.search(x)))
                match_ratio = matches.sum() / len(matches)

                if match_ratio > 0.5:  # More than 50% match
                    results.append(
                        PIIDetectionResult(
                            column=column,
                            pii_type=pii_type,
                            confidence=match_ratio,
                            sample_values=sample_values[matches].head(3).tolist(),
                            detection_method="regex",
                        )
                    )

            # NER-based detection
            if self.nlp:
                results.extend(self._detect_with_ner(column, sample_values))

        return results

    def _detect_with_ner(
        self, column: str, sample_values: pd.Series
    ) -> list[PIIDetectionResult]:
        """Detect PII using Named Entity Recognition"""
        results = []

        # Concatenate sample values
        text = " ".join(sample_values.head(20))
        doc = self.nlp(text)

        # Count entity types
        entity_counts = {}
        for ent in doc.ents:
            entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1

        # Map spaCy labels to PII types
        label_mapping = {
            "PERSON": PIIType.NAME,
            "GPE": PIIType.ADDRESS,
            "LOC": PIIType.ADDRESS,
            "ORG": PIIType.CUSTOM,
        }

        for label, count in entity_counts.items():
            if label in label_mapping and count > 3:
                results.append(
                    PIIDetectionResult(
                        column=column,
                        pii_type=label_mapping[label],
                        confidence=min(count / 20, 1.0),
                        sample_values=[
                            ent.text for ent in doc.ents if ent.label_ == label
                        ][:3],
                        detection_method="NER",
                    )
                )

        return results

    def validate_phone(self, phone: str, region: str = "US") -> bool:
        """Validate phone number"""
        try:
            parsed = phonenumbers.parse(phone, region)
            return phonenumbers.is_valid_number(parsed)
        except:
            return False

    def validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card number"""
        try:
            cc = CreditCard(card_number)
            return cc.is_valid()
        except:
            return False


class DataAnonymizer:
    """Data anonymization techniques"""

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.fake = Faker()
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.mapping_tables = {}

    def anonymize_dataframe(
        self, df: pd.DataFrame, pii_columns: list[str] | None = None
    ) -> tuple[pd.DataFrame, AnonymizationResult]:
        """Anonymize entire DataFrame"""
        # Detect PII if not specified
        if pii_columns is None:
            detector = PIIDetector(self.config)
            pii_results = detector.detect_pii(df)
            pii_columns = list(set([r.column for r in pii_results]))

        df_anon = df.copy()
        methods_used = {}

        for column in pii_columns:
            if column not in df.columns:
                continue

            # Determine anonymization method
            if self.config.anonymization_method == "generalization":
                df_anon[column] = self.generalize(df[column])
                methods_used[column] = "generalization"
            elif self.config.anonymization_method == "suppression":
                df_anon[column] = self.suppress(df[column])
                methods_used[column] = "suppression"
            elif self.config.anonymization_method == "masking":
                df_anon[column] = self.mask(df[column])
                methods_used[column] = "masking"
            elif self.config.anonymization_method == "encryption":
                df_anon[column] = self.encrypt(df[column])
                methods_used[column] = "encryption"
            else:
                df_anon[column] = self.pseudonymize(df[column])
                methods_used[column] = "pseudonymization"

        # Calculate privacy metrics
        k_anon = self.calculate_k_anonymity(df_anon)
        l_div = self.calculate_l_diversity(df_anon)
        t_close = self.calculate_t_closeness(df_anon)

        result = AnonymizationResult(
            original_shape=df.shape,
            anonymized_shape=df_anon.shape,
            pii_columns_found=pii_columns,
            anonymization_methods_used=methods_used,
            k_anonymity=k_anon,
            l_diversity=l_div,
            t_closeness=t_close,
        )

        return df_anon, result

    def generalize(self, series: pd.Series) -> pd.Series:
        """Generalize data to broader categories"""
        if pd.api.types.is_numeric_dtype(series):
            # Generalize numeric data to ranges
            bins = pd.qcut(series, q=5, duplicates="drop")
            return bins.astype(str)
        else:
            # Generalize text data
            return series.apply(
                lambda x: self._generalize_text(x) if pd.notna(x) else x
            )

    def _generalize_text(self, text: str) -> str:
        """Generalize text value"""
        text = str(text)

        # Email: keep domain only
        if "@" in text:
            return text.split("@")[1] if "@" in text else "email"

        # Phone: keep area code only
        if re.match(r"[\d\s\-\(\)]+", text):
            return text[:3] + "XXXX"

        # Default: keep first letter
        return text[0] + "***" if text else ""

    def suppress(self, series: pd.Series) -> pd.Series:
        """Suppress (remove) sensitive data"""
        return pd.Series(["*SUPPRESSED*"] * len(series), index=series.index)

    def mask(self, series: pd.Series) -> pd.Series:
        """Mask sensitive data"""

        def mask_value(val):
            if pd.isna(val):
                return val

            val_str = str(val)
            if len(val_str) <= 4:
                return "*" * len(val_str)

            # Keep first and last characters
            return val_str[0] + "*" * (len(val_str) - 2) + val_str[-1]

        return series.apply(mask_value)

    def encrypt(self, series: pd.Series) -> pd.Series:
        """Encrypt sensitive data"""

        def encrypt_value(val):
            if pd.isna(val):
                return val

            encrypted = self.fernet.encrypt(str(val).encode())
            return base64.b64encode(encrypted).decode()[:20] + "..."

        return series.apply(encrypt_value)

    def pseudonymize(self, series: pd.Series) -> pd.Series:
        """Replace with consistent pseudonyms"""
        # Create mapping table if not exists
        col_name = series.name or "default"
        if col_name not in self.mapping_tables:
            self.mapping_tables[col_name] = {}

        mapping = self.mapping_tables[col_name]

        def get_pseudonym(val):
            if pd.isna(val):
                return val

            val_str = str(val)
            if val_str not in mapping:
                # Generate consistent pseudonym
                if "@" in val_str:  # Email
                    mapping[val_str] = self.fake.email()
                elif re.match(r"[\d\s\-\(\)]+", val_str):  # Phone
                    mapping[val_str] = self.fake.phone_number()
                else:  # Name or other
                    mapping[val_str] = self.fake.name()

            return mapping[val_str]

        return series.apply(get_pseudonym)

    def synthetic_data_generation(
        self, df: pd.DataFrame, n_samples: int = None
    ) -> pd.DataFrame:
        """Generate synthetic data preserving statistical properties"""
        if n_samples is None:
            n_samples = len(df)

        synthetic_data = {}

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                # Generate from distribution
                mean = df[column].mean()
                std = df[column].std()
                synthetic_data[column] = np.random.normal(mean, std, n_samples)
            else:
                # Sample from existing values
                unique_values = df[column].dropna().unique()
                synthetic_data[column] = np.random.choice(unique_values, n_samples)

        return pd.DataFrame(synthetic_data)

    def calculate_k_anonymity(
        self, df: pd.DataFrame, quasi_identifiers: list[str] = None
    ) -> int:
        """Calculate k-anonymity value"""
        if quasi_identifiers is None:
            # Use all non-unique columns as quasi-identifiers
            quasi_identifiers = [
                col for col in df.columns if df[col].nunique() < len(df) * 0.9
            ]

        if not quasi_identifiers:
            return 1

        # Group by quasi-identifiers
        groups = df.groupby(quasi_identifiers).size()

        # k-anonymity is the minimum group size
        return int(groups.min()) if len(groups) > 0 else 1

    def calculate_l_diversity(
        self,
        df: pd.DataFrame,
        quasi_identifiers: list[str] = None,
        sensitive_attribute: str = None,
    ) -> int | None:
        """Calculate l-diversity value"""
        if quasi_identifiers is None or sensitive_attribute is None:
            return None

        # Group by quasi-identifiers
        groups = df.groupby(quasi_identifiers)[sensitive_attribute].apply(
            lambda x: x.nunique()
        )

        # l-diversity is the minimum number of distinct sensitive values
        return int(groups.min()) if len(groups) > 0 else None

    def calculate_t_closeness(
        self,
        df: pd.DataFrame,
        quasi_identifiers: list[str] = None,
        sensitive_attribute: str = None,
    ) -> float | None:
        """Calculate t-closeness value"""
        if quasi_identifiers is None or sensitive_attribute is None:
            return None

        # This is a simplified implementation
        # Full t-closeness requires Earth Mover's Distance calculation
        global_dist = df[sensitive_attribute].value_counts(normalize=True)

        max_distance = 0
        for _, group in df.groupby(quasi_identifiers):
            if len(group) > 0:
                group_dist = group[sensitive_attribute].value_counts(normalize=True)
                distance = abs(group_dist - global_dist).sum() / 2
                max_distance = max(max_distance, distance)

        return float(max_distance)


class GDPRCompliance:
    """GDPR compliance features"""

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.consent_records = {}
        self.data_subjects = {}
        self.audit_log = []

    def record_consent(
        self, subject_id: str, purposes: list[str], granted: bool = True
    ) -> str:
        """Record user consent"""
        consent_id = hashlib.sha256(
            f"{subject_id}{datetime.now()}".encode()
        ).hexdigest()[:16]

        self.consent_records[consent_id] = {
            "subject_id": subject_id,
            "purposes": purposes,
            "granted": granted,
            "timestamp": datetime.now(),
            "expires": datetime.now() + timedelta(days=365),
        }

        self._log_activity("consent_recorded", subject_id, {"consent_id": consent_id})

        return consent_id

    def check_consent(self, subject_id: str, purpose: str) -> bool:
        """Check if consent is granted for purpose"""
        for consent in self.consent_records.values():
            if (
                consent["subject_id"] == subject_id
                and purpose in consent["purposes"]
                and consent["granted"]
                and consent["expires"] > datetime.now()
            ):
                return True
        return False

    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw consent"""
        if consent_id in self.consent_records:
            self.consent_records[consent_id]["granted"] = False
            self.consent_records[consent_id]["withdrawn_at"] = datetime.now()

            subject_id = self.consent_records[consent_id]["subject_id"]
            self._log_activity(
                "consent_withdrawn", subject_id, {"consent_id": consent_id}
            )

            return True
        return False

    def right_to_access(self, subject_id: str) -> dict[str, Any]:
        """Implement right to access (data portability)"""
        if subject_id not in self.data_subjects:
            return {}

        self._log_activity("data_accessed", subject_id, {})

        return {
            "subject_id": subject_id,
            "data": self.data_subjects[subject_id],
            "consents": [
                c
                for c in self.consent_records.values()
                if c["subject_id"] == subject_id
            ],
            "export_date": datetime.now().isoformat(),
        }

    def right_to_rectification(
        self, subject_id: str, corrections: dict[str, Any]
    ) -> bool:
        """Implement right to rectification (correct data)"""
        if subject_id not in self.data_subjects:
            return False

        # Store original data for audit
        original_data = self.data_subjects[subject_id].copy()

        # Apply corrections
        self.data_subjects[subject_id].update(corrections)

        self._log_activity(
            "data_rectified",
            subject_id,
            {"original": original_data, "corrections": corrections},
        )

        return True

    def right_to_erasure(self, subject_id: str) -> bool:
        """Implement right to be forgotten"""
        if subject_id not in self.data_subjects:
            return False

        # Log before deletion
        self._log_activity(
            "data_erased",
            subject_id,
            {"data_categories": list(self.data_subjects[subject_id].keys())},
        )

        # Delete data
        del self.data_subjects[subject_id]

        # Delete consents
        consent_ids_to_delete = [
            cid
            for cid, consent in self.consent_records.items()
            if consent["subject_id"] == subject_id
        ]
        for cid in consent_ids_to_delete:
            del self.consent_records[cid]

        return True

    def data_portability_export(self, subject_id: str, format: str = "json") -> str:
        """Export data in portable format"""
        data = self.right_to_access(subject_id)

        if format == "json":
            return json.dumps(data, default=str, indent=2)
        elif format == "csv":
            df = pd.DataFrame([data["data"]])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def check_retention_policy(self) -> list[str]:
        """Check and apply data retention policy"""
        expired_subjects = []
        current_time = datetime.now()

        for subject_id, data in self.data_subjects.items():
            if "created_at" in data:
                created_at = data["created_at"]
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at)

                if (current_time - created_at).days > self.config.data_retention_days:
                    expired_subjects.append(subject_id)

        # Apply retention policy
        for subject_id in expired_subjects:
            self.right_to_erasure(subject_id)
            logger.info(f"Data for subject {subject_id} deleted per retention policy")

        return expired_subjects

    def _log_activity(self, action: str, subject_id: str, details: dict[str, Any]):
        """Log GDPR-related activity"""
        log_entry = {
            "timestamp": datetime.now(),
            "action": action,
            "subject_id": subject_id,
            "details": details,
        }

        self.audit_log.append(log_entry)

        # Rotate audit log
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]

    def get_audit_log(self, subject_id: str | None = None) -> list[dict]:
        """Get audit log entries"""
        if subject_id:
            return [
                entry for entry in self.audit_log if entry["subject_id"] == subject_id
            ]
        return self.audit_log


class PrivacyFramework:
    """Main privacy framework orchestrator"""

    def __init__(self, config: PrivacyConfig = None):
        self.config = config or PrivacyConfig()
        self.differential_privacy = DifferentialPrivacy(config)
        self.pii_detector = PIIDetector(config)
        self.anonymizer = DataAnonymizer(config)
        self.gdpr_compliance = GDPRCompliance(config)

    def apply_privacy_protection(
        self, df: pd.DataFrame, subject_id: str | None = None
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply comprehensive privacy protection"""
        report = {
            "timestamp": datetime.now(),
            "original_shape": df.shape,
            "privacy_methods_applied": [],
        }

        # Check GDPR consent if required
        if self.config.enable_gdpr_compliance and subject_id:
            if not self.gdpr_compliance.check_consent(subject_id, "data_processing"):
                raise PermissionError("No consent for data processing")

        # Detect PII
        pii_results = self.pii_detector.detect_pii(df)
        report["pii_detected"] = [asdict(r) for r in pii_results]

        # Anonymize PII columns
        pii_columns = list(set([r.column for r in pii_results]))
        df_anon, anon_result = self.anonymizer.anonymize_dataframe(df, pii_columns)
        report["anonymization"] = asdict(anon_result)
        report["privacy_methods_applied"].append("anonymization")

        # Apply differential privacy to numeric columns
        numeric_cols = df_anon.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in pii_columns:  # Don't add noise to already anonymized columns
                df_anon[col] = self.differential_privacy.add_laplace_noise(
                    df_anon[col].values
                )

        report["privacy_methods_applied"].append("differential_privacy")
        report["privacy_budget"] = self.differential_privacy.get_privacy_budget_status()

        # Log activity if GDPR enabled
        if self.config.enable_gdpr_compliance and subject_id:
            self.gdpr_compliance._log_activity("data_processed", subject_id, report)

        report["final_shape"] = df_anon.shape

        return df_anon, report

    def generate_privacy_report(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate comprehensive privacy report"""
        report = {
            "timestamp": datetime.now(),
            "dataset_shape": df.shape,
            "privacy_analysis": {},
        }

        # PII detection
        pii_results = self.pii_detector.detect_pii(df)
        report["privacy_analysis"]["pii"] = {
            "columns_with_pii": len(set([r.column for r in pii_results])),
            "pii_types_found": list(set([r.pii_type.value for r in pii_results])),
            "details": [asdict(r) for r in pii_results],
        }

        # Calculate privacy metrics
        k_anon = self.anonymizer.calculate_k_anonymity(df)
        report["privacy_analysis"]["k_anonymity"] = k_anon

        # Estimate required privacy budget
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        report["privacy_analysis"]["differential_privacy"] = {
            "numeric_columns": len(numeric_cols),
            "recommended_epsilon": 1.0,
            "recommended_delta": 1e-5,
        }

        # GDPR compliance status
        if self.config.enable_gdpr_compliance:
            report["gdpr_compliance"] = {
                "consent_records": len(self.gdpr_compliance.consent_records),
                "data_subjects": len(self.gdpr_compliance.data_subjects),
                "audit_log_entries": len(self.gdpr_compliance.audit_log),
            }

        return report


# Example usage
if __name__ == "__main__":
    # Create sample data with PII
    df = pd.DataFrame(
        {
            "name": ["John Doe", "Jane Smith", "Bob Johnson"],
            "email": ["john@example.com", "jane@example.com", "bob@example.com"],
            "phone": ["555-123-4567", "555-987-6543", "555-456-7890"],
            "age": [25, 30, 35],
            "salary": [50000, 60000, 70000],
            "ssn": ["123-45-6789", "987-65-4321", "456-78-9012"],
        }
    )

    # Initialize privacy framework
    config = PrivacyConfig(
        epsilon=1.0,
        anonymization_method="pseudonymization",
        enable_gdpr_compliance=True,
    )
    privacy = PrivacyFramework(config)

    # Apply privacy protection
    df_private, report = privacy.apply_privacy_protection(df)

    print("Original data:")
    print(df)
    print("\nPrivacy-protected data:")
    print(df_private)
    print("\nPrivacy report:")
    print(json.dumps(report, default=str, indent=2))
