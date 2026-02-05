"""
Security Framework
Comprehensive security implementation with input sanitization, injection prevention, and secret management
"""

import re
import logging
import hashlib
import hmac
import secrets
import json
import os
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import html
import bleach
from urllib.parse import quote, urlparse
import sqlparse
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import jwt
import base64
from pathlib import Path

# SQL injection prevention
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import psycopg2
from psycopg2 import sql

# Rate limiting
from functools import lru_cache
import time
from collections import defaultdict, deque

# Input validation
from pydantic import BaseModel, Field, validator, EmailStr
import validators

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration"""

    enable_input_sanitization: bool = True
    enable_sql_injection_prevention: bool = True
    enable_xss_protection: bool = True
    enable_csrf_protection: bool = True
    enable_rate_limiting: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: List[str] = field(
        default_factory=lambda: [".csv", ".json", ".xlsx", ".parquet"]
    )
    password_min_length: int = 12
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24


class InputSanitizer:
    """Input sanitization and validation"""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.sql_keywords = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "EXEC",
            "EXECUTE",
            "UNION",
            "SCRIPT",
        ]

        # XSS patterns
        self.xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),  # Event handlers
            re.compile(r"<iframe[^>]*>", re.IGNORECASE),
            re.compile(r"<object[^>]*>", re.IGNORECASE),
            re.compile(r"<embed[^>]*>", re.IGNORECASE),
        ]

    def sanitize_string(self, input_str: str, context: str = "general") -> str:
        """Sanitize string input based on context"""
        if not isinstance(input_str, str):
            input_str = str(input_str)

        # Remove null bytes
        input_str = input_str.replace("\x00", "")

        # Context-specific sanitization
        if context == "html":
            return self.sanitize_html(input_str)
        elif context == "sql":
            return self.sanitize_sql(input_str)
        elif context == "filename":
            return self.sanitize_filename(input_str)
        elif context == "email":
            return self.sanitize_email(input_str)
        elif context == "url":
            return self.sanitize_url(input_str)
        else:
            # General sanitization
            return self.sanitize_general(input_str)

    def sanitize_general(self, input_str: str) -> str:
        """General string sanitization"""
        # Remove control characters
        input_str = "".join(
            char for char in input_str if ord(char) >= 32 or char in "\n\r\t"
        )

        # HTML escape
        input_str = html.escape(input_str)

        # Limit length
        max_length = 10000
        if len(input_str) > max_length:
            input_str = input_str[:max_length]

        return input_str

    def sanitize_html(self, html_str: str) -> str:
        """Sanitize HTML content"""
        # Use bleach for HTML sanitization
        allowed_tags = [
            "p",
            "br",
            "strong",
            "em",
            "u",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "ul",
            "ol",
            "li",
            "blockquote",
            "a",
            "img",
        ]
        allowed_attributes = {"a": ["href", "title"], "img": ["src", "alt"]}

        cleaned = bleach.clean(
            html_str, tags=allowed_tags, attributes=allowed_attributes, strip=True
        )

        # Additional XSS prevention
        for pattern in self.xss_patterns:
            cleaned = pattern.sub("", cleaned)

        return cleaned

    def sanitize_sql(self, sql_str: str) -> str:
        """Sanitize SQL input to prevent injection"""
        # Remove SQL comments
        sql_str = re.sub(r"--.*", "", sql_str)
        sql_str = re.sub(r"/\*.*?\*/", "", sql_str, flags=re.DOTALL)

        # Escape special characters
        sql_str = sql_str.replace("'", "''")
        sql_str = sql_str.replace("\\", "\\\\")
        sql_str = sql_str.replace("\x00", "")
        sql_str = sql_str.replace("\n", " ")
        sql_str = sql_str.replace("\r", " ")
        sql_str = sql_str.replace("\x1a", "")

        # Check for dangerous keywords
        upper_str = sql_str.upper()
        for keyword in self.sql_keywords:
            if keyword in upper_str:
                logger.warning(f"Potentially dangerous SQL keyword detected: {keyword}")

        return sql_str

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent directory traversal"""
        # Remove path components
        filename = os.path.basename(filename)

        # Remove dangerous characters
        filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)

        # Prevent directory traversal
        filename = filename.replace("..", "")
        filename = filename.replace("/", "")
        filename = filename.replace("\\", "")

        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            filename = name[: max_length - len(ext)] + ext

        return filename

    def sanitize_email(self, email: str) -> Optional[str]:
        """Sanitize and validate email address"""
        email = email.strip().lower()

        # Basic email validation
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email):
            return None

        # Additional validation
        if not validators.email(email):
            return None

        return email

    def sanitize_url(self, url: str) -> Optional[str]:
        """Sanitize and validate URL"""
        # Basic URL validation
        if not validators.url(url):
            return None

        # Parse URL
        parsed = urlparse(url)

        # Check for allowed schemes
        allowed_schemes = ["http", "https"]
        if parsed.scheme not in allowed_schemes:
            return None

        # Prevent javascript: and data: URIs
        if parsed.scheme in ["javascript", "data", "vbscript"]:
            return None

        return url

    def validate_json(self, json_str: str) -> Optional[Dict]:
        """Validate and parse JSON safely"""
        try:
            # Limit JSON size
            if len(json_str) > self.config.max_request_size:
                logger.warning("JSON input exceeds maximum size")
                return None

            # Parse JSON
            data = json.loads(json_str)

            # Recursively sanitize strings in JSON
            return self._sanitize_json_recursive(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON input: {e}")
            return None

    def _sanitize_json_recursive(self, obj: Any) -> Any:
        """Recursively sanitize JSON object"""
        if isinstance(obj, str):
            return self.sanitize_general(obj)
        elif isinstance(obj, dict):
            return {k: self._sanitize_json_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_json_recursive(item) for item in obj]
        else:
            return obj


class SQLInjectionPrevention:
    """SQL injection prevention mechanisms"""

    def __init__(self):
        self.parameterized_queries = {}

    def create_safe_connection(self, connection_string: str):
        """Create a safe database connection"""
        # Use SQLAlchemy for safe connections
        engine = create_engine(
            connection_string, pool_pre_ping=True, pool_recycle=3600, echo=False
        )
        Session = sessionmaker(bind=engine)
        return Session()

    def execute_safe_query(self, session, query: str, params: Dict = None):
        """Execute query with parameterized inputs"""
        # Use parameterized queries
        if params:
            result = session.execute(text(query), params)
        else:
            result = session.execute(text(query))
        return result

    def build_safe_query(
        self, table: str, columns: List[str], conditions: Dict = None
    ) -> tuple:
        """Build safe SQL query with parameters"""
        # Validate table name
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
            raise ValueError("Invalid table name")

        # Validate column names
        for col in columns:
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", col):
                raise ValueError(f"Invalid column name: {col}")

        # Build query
        query = f"SELECT {', '.join(columns)} FROM {table}"
        params = {}

        if conditions:
            where_clauses = []
            for key, value in conditions.items():
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                    raise ValueError(f"Invalid condition key: {key}")
                param_name = f"param_{key}"
                where_clauses.append(f"{key} = :{param_name}")
                params[param_name] = value

            query += " WHERE " + " AND ".join(where_clauses)

        return query, params

    def sanitize_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifier (table/column name)"""
        # Only allow alphanumeric and underscore
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier}")
        return identifier

    def escape_like_pattern(self, pattern: str) -> str:
        """Escape special characters in LIKE patterns"""
        pattern = pattern.replace("\\", "\\\\")
        pattern = pattern.replace("%", "\\%")
        pattern = pattern.replace("_", "\\_")
        return pattern


class XSSProtection:
    """Cross-Site Scripting (XSS) protection"""

    def __init__(self):
        self.csp_policy = {
            "default-src": ["'self'"],
            "script-src": ["'self'", "'unsafe-inline'"],
            "style-src": ["'self'", "'unsafe-inline'"],
            "img-src": ["'self'", "data:", "https:"],
            "font-src": ["'self'"],
            "connect-src": ["'self'"],
            "frame-ancestors": ["'none'"],
            "base-uri": ["'self'"],
            "form-action": ["'self'"],
        }

    def get_csp_header(self) -> str:
        """Generate Content Security Policy header"""
        policies = []
        for directive, sources in self.csp_policy.items():
            policy = f"{directive} {' '.join(sources)}"
            policies.append(policy)
        return "; ".join(policies)

    def add_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses"""
        return {
            "Content-Security-Policy": self.get_csp_header(),
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

    def sanitize_user_content(self, content: str) -> str:
        """Sanitize user-generated content for display"""
        # Use bleach for comprehensive sanitization
        allowed_tags = ["p", "br", "strong", "em", "u", "li", "ul", "ol"]
        allowed_attributes = {}

        sanitized = bleach.clean(
            content, tags=allowed_tags, attributes=allowed_attributes, strip=True
        )

        # Additional encoding
        sanitized = html.escape(sanitized)

        return sanitized

    def generate_nonce(self) -> str:
        """Generate nonce for inline scripts"""
        return base64.b64encode(secrets.token_bytes(16)).decode("utf-8")


class SecretManager:
    """Secure secret management"""

    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = self._get_or_create_master_key()

        self.fernet = Fernet(self._derive_key(self.master_key))
        self.secrets_file = Path("secrets.enc")

    def _get_or_create_master_key(self) -> bytes:
        """Get or create master key from environment or file"""
        # Try environment variable first
        env_key = os.environ.get("MASTER_KEY")
        if env_key:
            return env_key.encode()

        # Try key file
        key_file = Path(".master_key")
        if key_file.exists():
            return key_file.read_bytes()

        # Generate new key
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        key_file.chmod(0o600)  # Restrict permissions
        return key

    def _derive_key(self, password: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"stable_salt",  # In production, use random salt
            iterations=100000,
            backend=default_backend(),
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def store_secret(self, key: str, value: str) -> None:
        """Store encrypted secret"""
        # Load existing secrets
        secrets = self._load_secrets()

        # Encrypt and store new secret
        encrypted_value = self.fernet.encrypt(value.encode())
        secrets[key] = encrypted_value.decode()

        # Save secrets
        self._save_secrets(secrets)

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve and decrypt secret"""
        secrets = self._load_secrets()

        if key not in secrets:
            return None

        encrypted_value = secrets[key].encode()
        decrypted_value = self.fernet.decrypt(encrypted_value)
        return decrypted_value.decode()

    def _load_secrets(self) -> Dict[str, str]:
        """Load secrets from file"""
        if not self.secrets_file.exists():
            return {}

        with open(self.secrets_file, "r") as f:
            return json.load(f)

    def _save_secrets(self, secrets: Dict[str, str]) -> None:
        """Save secrets to file"""
        with open(self.secrets_file, "w") as f:
            json.dump(secrets, f)

        # Restrict file permissions
        self.secrets_file.chmod(0o600)

    def rotate_secrets(self) -> None:
        """Rotate all secrets with new encryption"""
        old_fernet = self.fernet

        # Generate new key
        new_key = Fernet.generate_key()
        self.fernet = Fernet(new_key)

        # Re-encrypt all secrets
        secrets = self._load_secrets()
        rotated_secrets = {}

        for key, encrypted_value in secrets.items():
            # Decrypt with old key
            decrypted = old_fernet.decrypt(encrypted_value.encode())
            # Encrypt with new key
            new_encrypted = self.fernet.encrypt(decrypted)
            rotated_secrets[key] = new_encrypted.decode()

        # Save rotated secrets
        self._save_secrets(rotated_secrets)


class PasswordValidator:
    """Password strength validation"""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()

    def validate_password(self, password: str) -> tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []

        # Check length
        if len(password) < self.config.password_min_length:
            errors.append(
                f"Password must be at least {self.config.password_min_length} characters"
            )

        # Check for uppercase
        if self.config.password_require_uppercase and not any(
            c.isupper() for c in password
        ):
            errors.append("Password must contain at least one uppercase letter")

        # Check for lowercase
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        # Check for numbers
        if self.config.password_require_numbers and not any(
            c.isdigit() for c in password
        ):
            errors.append("Password must contain at least one number")

        # Check for special characters
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if self.config.password_require_special and not any(
            c in special_chars for c in password
        ):
            errors.append("Password must contain at least one special character")

        # Check against common passwords
        if self._is_common_password(password):
            errors.append("Password is too common")

        # Check for patterns
        if self._has_pattern(password):
            errors.append("Password contains predictable patterns")

        return len(errors) == 0, errors

    def _is_common_password(self, password: str) -> bool:
        """Check if password is in common passwords list"""
        common_passwords = [
            "password",
            "123456",
            "12345678",
            "qwerty",
            "abc123",
            "password123",
            "admin",
            "letmein",
            "welcome",
            "monkey",
        ]
        return password.lower() in common_passwords

    def _has_pattern(self, password: str) -> bool:
        """Check for keyboard patterns"""
        patterns = ["qwerty", "asdf", "zxcv", "1234", "4321"]
        password_lower = password.lower()
        return any(pattern in password_lower for pattern in patterns)

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        import bcrypt

        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        import bcrypt

        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))

    def generate_secure_password(self, length: int = 16) -> str:
        """Generate secure random password"""
        alphabet = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        )
        password = "".join(secrets.choice(alphabet) for _ in range(length))
        return password


class RateLimiter:
    """Rate limiting implementation"""

    def __init__(self):
        self.requests = defaultdict(lambda: deque(maxlen=100))
        self.blocked_ips = set()

    def check_rate_limit(
        self, identifier: str, max_requests: int = 100, window_seconds: int = 60
    ) -> bool:
        """Check if request should be rate limited"""
        if identifier in self.blocked_ips:
            return False

        current_time = time.time()
        request_times = self.requests[identifier]

        # Remove old requests outside window
        while request_times and request_times[0] < current_time - window_seconds:
            request_times.popleft()

        # Check if limit exceeded
        if len(request_times) >= max_requests:
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False

        # Add current request
        request_times.append(current_time)
        return True

    def block_identifier(self, identifier: str, duration_seconds: int = 3600):
        """Temporarily block an identifier"""
        self.blocked_ips.add(identifier)

        # Schedule unblock (in production, use proper scheduling)
        def unblock():
            time.sleep(duration_seconds)
            self.blocked_ips.discard(identifier)

        import threading

        threading.Thread(target=unblock, daemon=True).start()


class CSRFProtection:
    """Cross-Site Request Forgery protection"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token"""
        message = f"{session_id}:{time.time()}"
        signature = hmac.new(
            self.secret_key.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        token = base64.b64encode(f"{message}:{signature}".encode()).decode()
        return token

    def verify_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Verify CSRF token"""
        try:
            decoded = base64.b64decode(token).decode()
            message, signature = decoded.rsplit(":", 1)
            stored_session_id, timestamp = message.split(":", 1)

            # Check session ID
            if stored_session_id != session_id:
                return False

            # Check age
            if time.time() - float(timestamp) > max_age:
                return False

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(), message.encode(), hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False


class FileUploadValidator:
    """Secure file upload validation"""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()

        # MIME type mappings
        self.allowed_mimes = {
            ".csv": "text/csv",
            ".json": "application/json",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".parquet": "application/octet-stream",
        }

    def validate_file(self, filename: str, content: bytes) -> tuple[bool, str]:
        """Validate uploaded file"""
        # Check file extension
        _, ext = os.path.splitext(filename)
        if ext.lower() not in self.config.allowed_file_types:
            return False, f"File type {ext} not allowed"

        # Check file size
        if len(content) > self.config.max_request_size:
            return (
                False,
                f"File size exceeds maximum of {self.config.max_request_size} bytes",
            )

        # Check for malicious content
        if self._contains_malicious_content(content):
            return False, "File contains potentially malicious content"

        # Validate file signature (magic numbers)
        if not self._validate_file_signature(content, ext):
            return False, "File signature does not match extension"

        return True, "File validation passed"

    def _contains_malicious_content(self, content: bytes) -> bool:
        """Check for malicious patterns in file content"""
        # Check for executable signatures
        executable_signatures = [
            b"MZ",  # PE executable
            b"\x7fELF",  # ELF executable
            b"#!/bin/",  # Shell script
            b"#!/usr/bin/python",  # Python script
        ]

        for sig in executable_signatures:
            if content.startswith(sig):
                return True

        # Check for embedded scripts in CSV/JSON
        content_str = content.decode("utf-8", errors="ignore")
        dangerous_patterns = ["<script", "javascript:", "eval(", "exec("]
        for pattern in dangerous_patterns:
            if pattern.lower() in content_str.lower():
                return True

        return False

    def _validate_file_signature(self, content: bytes, extension: str) -> bool:
        """Validate file signature matches extension"""
        signatures = {
            ".xlsx": b"PK\x03\x04",  # ZIP archive (Excel uses ZIP)
            ".json": [b"{", b"["],  # JSON starts with { or [
            ".csv": None,  # CSV has no specific signature
            ".parquet": b"PAR1",  # Parquet signature
        }

        expected_sig = signatures.get(extension.lower())

        if expected_sig is None:
            return True  # No signature to check

        if isinstance(expected_sig, list):
            return any(content.startswith(sig) for sig in expected_sig)

        return content.startswith(expected_sig)


class SecurityFramework:
    """Main security framework orchestrator"""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.input_sanitizer = InputSanitizer(config)
        self.sql_protection = SQLInjectionPrevention()
        self.xss_protection = XSSProtection()
        self.secret_manager = SecretManager()
        self.password_validator = PasswordValidator(config)
        self.rate_limiter = RateLimiter()
        self.csrf_protection = CSRFProtection(secrets.token_urlsafe(32))
        self.file_validator = FileUploadValidator(config)

    def sanitize_request(self, request_data: Dict) -> Dict:
        """Sanitize entire request"""
        sanitized = {}

        for key, value in request_data.items():
            # Sanitize key
            clean_key = self.input_sanitizer.sanitize_string(key, context="general")

            # Sanitize value based on type
            if isinstance(value, str):
                sanitized[clean_key] = self.input_sanitizer.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[clean_key] = self.sanitize_request(value)
            elif isinstance(value, list):
                sanitized[clean_key] = [
                    self.input_sanitizer.sanitize_string(item)
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[clean_key] = value

        return sanitized

    def validate_api_request(
        self, request: Dict, schema: BaseModel
    ) -> tuple[bool, Any]:
        """Validate API request against schema"""
        try:
            # Sanitize input first
            sanitized = self.sanitize_request(request)

            # Validate against schema
            validated = schema(**sanitized)
            return True, validated
        except Exception as e:
            logger.warning(f"Request validation failed: {e}")
            return False, str(e)

    def secure_database_query(
        self, query: str, params: Dict = None
    ) -> tuple[str, Dict]:
        """Create secure database query"""
        # Sanitize parameters
        safe_params = {}
        if params:
            for key, value in params.items():
                if isinstance(value, str):
                    safe_params[key] = self.input_sanitizer.sanitize_sql(value)
                else:
                    safe_params[key] = value

        # Use parameterized query
        return query, safe_params

    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses"""
        return self.xss_protection.add_security_headers()

    def validate_file_upload(self, filename: str, content: bytes) -> tuple[bool, str]:
        """Validate file upload"""
        # Sanitize filename
        safe_filename = self.input_sanitizer.sanitize_filename(filename)

        # Validate file
        return self.file_validator.validate_file(safe_filename, content)


# Security decorators
def rate_limit(max_requests: int = 100, window: int = 60):
    """Rate limiting decorator"""
    limiter = RateLimiter()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier (e.g., IP address, user ID)
            identifier = kwargs.get("identifier", "unknown")

            if not limiter.check_rate_limit(identifier, max_requests, window):
                raise Exception("Rate limit exceeded")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_auth(func):
    """Authentication requirement decorator"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for authentication token
        token = kwargs.get("auth_token")
        if not token:
            raise Exception("Authentication required")

        # Verify token (implement actual verification)
        # This is a placeholder
        if not verify_token(token):
            raise Exception("Invalid authentication token")

        return func(*args, **kwargs)

    return wrapper


def sanitize_inputs(func):
    """Input sanitization decorator"""
    sanitizer = InputSanitizer()

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Sanitize string arguments
        sanitized_args = []
        for arg in args:
            if isinstance(arg, str):
                sanitized_args.append(sanitizer.sanitize_string(arg))
            else:
                sanitized_args.append(arg)

        # Sanitize keyword arguments
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                sanitized_kwargs[key] = sanitizer.sanitize_string(value)
            else:
                sanitized_kwargs[key] = value

        return func(*sanitized_args, **sanitized_kwargs)

    return wrapper


def verify_token(token: str) -> bool:
    """Placeholder for token verification"""
    # Implement actual token verification
    return True


# Example usage
if __name__ == "__main__":
    # Initialize security framework
    security = SecurityFramework()

    # Test input sanitization
    unsafe_input = "<script>alert('XSS')</script> SELECT * FROM users"
    safe_input = security.input_sanitizer.sanitize_string(unsafe_input, context="html")
    print(f"Sanitized: {safe_input}")

    # Test password validation
    password = "SecureP@ssw0rd123!"
    is_valid, errors = security.password_validator.validate_password(password)
    print(f"Password valid: {is_valid}, Errors: {errors}")

    # Test file validation
    filename = "data.csv"
    content = b"id,name,value\n1,test,100"
    is_valid, message = security.validate_file_upload(filename, content)
    print(f"File valid: {is_valid}, Message: {message}")
