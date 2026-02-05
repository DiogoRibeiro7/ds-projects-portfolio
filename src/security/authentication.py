"""Authentication and Authorization System"""

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

# For session management
from functools import wraps
from typing import Any

import bcrypt
import jwt
import redis
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for authorization"""

    ADMIN = "ADMIN"
    DATA_SCIENTIST = "DATA_SCIENTIST"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"
    API_USER = "API_USER"


class Permission(Enum):
    """System permissions"""

    # Model permissions
    MODEL_CREATE = "MODEL_CREATE"
    MODEL_READ = "MODEL_READ"
    MODEL_UPDATE = "MODEL_UPDATE"
    MODEL_DELETE = "MODEL_DELETE"
    MODEL_DEPLOY = "MODEL_DEPLOY"
    MODEL_APPROVE = "MODEL_APPROVE"

    # Data permissions
    DATA_READ = "DATA_READ"
    DATA_WRITE = "DATA_WRITE"
    DATA_DELETE = "DATA_DELETE"
    DATA_EXPORT = "DATA_EXPORT"

    # System permissions
    SYSTEM_ADMIN = "SYSTEM_ADMIN"
    USER_MANAGE = "USER_MANAGE"
    AUDIT_READ = "AUDIT_READ"
    CONFIG_MANAGE = "CONFIG_MANAGE"


@dataclass
class User:
    """User entity"""

    user_id: str
    username: str
    email: str
    roles: list[UserRole] = field(default_factory=list)
    permissions: list[Permission] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: datetime | None = None
    is_active: bool = True
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    password_hash: str | None = None
    failed_attempts: int = 0
    locked_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session"""

    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    refresh_token: str | None = None
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class AuthConfig:
    """Authentication configuration"""

    jwt_secret: str = None
    jwt_algorithm: str = "HS256"
    token_expiry_minutes: int = 60
    refresh_token_expiry_days: int = 7
    session_timeout_minutes: int = 30
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digits: bool = True
    password_require_special: bool = True
    mfa_required: bool = False
    session_store: str = "memory"  # memory, redis, database
    redis_host: str = "localhost"
    redis_port: int = 6379
    enable_api_keys: bool = True
    api_key_expiry_days: int = 365


class RolePermissionManager:
    """Manage role-based permissions"""

    def __init__(self):
        self.role_permissions = self._initialize_role_permissions()

    def _initialize_role_permissions(self) -> dict[UserRole, list[Permission]]:
        """Initialize default role-permission mappings"""
        return {
            UserRole.ADMIN: list(Permission),  # All permissions
            UserRole.DATA_SCIENTIST: [
                Permission.MODEL_CREATE,
                Permission.MODEL_READ,
                Permission.MODEL_UPDATE,
                Permission.MODEL_DEPLOY,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DATA_EXPORT,
            ],
            UserRole.ANALYST: [
                Permission.MODEL_READ,
                Permission.DATA_READ,
                Permission.DATA_EXPORT,
                Permission.AUDIT_READ,
            ],
            UserRole.VIEWER: [Permission.MODEL_READ, Permission.DATA_READ],
            UserRole.API_USER: [Permission.MODEL_READ, Permission.DATA_READ],
        }

    def get_permissions(self, roles: list[UserRole]) -> list[Permission]:
        """Get all permissions for given roles"""
        permissions = set()
        for role in roles:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])
        return list(permissions)

    def has_permission(self, user: User, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        # Get permissions from roles
        role_permissions = self.get_permissions(user.roles)

        # Combine with direct user permissions
        all_permissions = set(role_permissions) | set(user.permissions)

        return required_permission in all_permissions

    def require_permission(self, permission: Permission):
        """Decorator to require permission for function access"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract user from arguments
                user = kwargs.get("user") or (args[1] if len(args) > 1 else None)
                if not user or not isinstance(user, User):
                    raise PermissionError("User authentication required")

                if not self.has_permission(user, permission):
                    raise PermissionError(
                        f"Permission denied: {permission.value} required"
                    )

                return func(*args, **kwargs)

            return wrapper

        return decorator


class PasswordManager:
    """Secure password management"""

    def __init__(self, config: AuthConfig = None):
        self.config = config or AuthConfig()

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        # Validate password strength first
        if not self.validate_password_strength(password):
            raise ValueError("Password does not meet strength requirements")

        # Generate salt and hash
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"), password_hash.encode("utf-8")
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    def validate_password_strength(self, password: str) -> bool:
        """Validate password meets strength requirements"""
        if len(password) < self.config.password_min_length:
            return False

        if self.config.password_require_uppercase and not re.search(r"[A-Z]", password):
            return False

        if self.config.password_require_lowercase and not re.search(r"[a-z]", password):
            return False

        if self.config.password_require_digits and not re.search(r"\d", password):
            return False

        if self.config.password_require_special and not re.search(
            r'[!@#$%^&*(),.?":{}|<>]', password
        ):
            return False

        return True

    def generate_secure_password(self, length: int = 16) -> str:
        """Generate cryptographically secure password"""
        alphabet = ""
        if self.config.password_require_uppercase:
            alphabet += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if self.config.password_require_lowercase:
            alphabet += "abcdefghijklmnopqrstuvwxyz"
        if self.config.password_require_digits:
            alphabet += "0123456789"
        if self.config.password_require_special:
            alphabet += '!@#$%^&*(),.?":{}|<>'

        password = "".join(secrets.choice(alphabet) for _ in range(length))

        # Ensure all requirements are met
        while not self.validate_password_strength(password):
            password = "".join(secrets.choice(alphabet) for _ in range(length))

        return password


class TokenManager:
    """JWT token management"""

    def __init__(self, config: AuthConfig = None):
        self.config = config or AuthConfig()
        if not self.config.jwt_secret:
            self.config.jwt_secret = secrets.token_urlsafe(32)

    def generate_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "exp": datetime.utcnow()
            + timedelta(minutes=self.config.token_expiry_minutes),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16),  # JWT ID for token revocation
        }

        return jwt.encode(
            payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm
        )

    def generate_refresh_token(self, user: User) -> str:
        """Generate refresh token"""
        payload = {
            "user_id": user.user_id,
            "type": "refresh",
            "exp": datetime.utcnow()
            + timedelta(days=self.config.refresh_token_expiry_days),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16),
        }

        return jwt.encode(
            payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm
        )

    def verify_token(self, token: str) -> dict | None:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def refresh_access_token(self, refresh_token: str) -> str | None:
        """Generate new access token from refresh token"""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None

        # Create minimal user object for token generation
        user = User(
            user_id=payload["user_id"],
            username="",  # Would fetch from database
            email="",
            roles=[],
            permissions=[],
        )

        return self.generate_token(user)


class SessionManager:
    """Manage user sessions"""

    def __init__(self, config: AuthConfig = None):
        self.config = config or AuthConfig()
        self.sessions = {}  # In-memory store

        # Initialize session store
        if self.config.session_store == "redis":
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    decode_responses=True,
                )
                self.redis_client.ping()
            except:
                logger.warning("Redis connection failed, falling back to memory store")
                self.config.session_store = "memory"

        if self.config.session_store == "memory":
            self.cache = TTLCache(
                maxsize=10000, ttl=self.config.session_timeout_minutes * 60
            )

    def create_session(self, user: User, ip_address: str, user_agent: str) -> Session:
        """Create new user session"""
        session = Session(
            session_id=secrets.token_urlsafe(32),
            user_id=user.user_id,
            created_at=datetime.now(),
            expires_at=datetime.now()
            + timedelta(minutes=self.config.session_timeout_minutes),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Store session
        if self.config.session_store == "redis":
            self.redis_client.setex(
                f"session:{session.session_id}",
                self.config.session_timeout_minutes * 60,
                json.dumps(
                    {
                        "user_id": session.user_id,
                        "created_at": session.created_at.isoformat(),
                        "expires_at": session.expires_at.isoformat(),
                        "ip_address": session.ip_address,
                        "user_agent": session.user_agent,
                        "is_active": session.is_active,
                    }
                ),
            )
        else:
            self.cache[session.session_id] = session

        return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve session by ID"""
        if self.config.session_store == "redis":
            data = self.redis_client.get(f"session:{session_id}")
            if data:
                session_data = json.loads(data)
                return Session(
                    session_id=session_id,
                    user_id=session_data["user_id"],
                    created_at=datetime.fromisoformat(session_data["created_at"]),
                    expires_at=datetime.fromisoformat(session_data["expires_at"]),
                    ip_address=session_data["ip_address"],
                    user_agent=session_data["user_agent"],
                    is_active=session_data["is_active"],
                )
        else:
            return self.cache.get(session_id)

        return None

    def validate_session(self, session_id: str) -> bool:
        """Validate session is active and not expired"""
        session = self.get_session(session_id)
        if not session:
            return False

        if not session.is_active:
            return False

        if datetime.now() > session.expires_at:
            self.invalidate_session(session_id)
            return False

        # Update last activity
        session.last_activity = datetime.now()

        return True

    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        if self.config.session_store == "redis":
            self.redis_client.delete(f"session:{session_id}")
        else:
            if session_id in self.cache:
                del self.cache[session_id]

    def invalidate_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user"""
        if self.config.session_store == "redis":
            # Scan for user sessions
            for key in self.redis_client.scan_iter("session:*"):
                data = self.redis_client.get(key)
                if data:
                    session_data = json.loads(data)
                    if session_data["user_id"] == user_id:
                        self.redis_client.delete(key)
        else:
            # Remove from cache
            to_remove = [
                sid for sid, session in self.cache.items() if session.user_id == user_id
            ]
            for sid in to_remove:
                del self.cache[sid]


class MFAManager:
    """Multi-factor authentication manager"""

    def __init__(self):
        self.totp_window = 1  # Allow 1 time step before/after

    def generate_secret(self) -> str:
        """Generate TOTP secret"""
        return base64.b32encode(secrets.token_bytes(20)).decode("utf-8")

    def generate_qr_code(self, user: User, issuer: str = "ML Portfolio") -> str:
        """Generate QR code URL for TOTP setup"""
        if not user.mfa_secret:
            user.mfa_secret = self.generate_secret()

        # Generate provisioning URI
        uri = f"otpauth://totp/{issuer}:{user.username}?secret={user.mfa_secret}&issuer={issuer}"
        return uri

    def generate_totp(self, secret: str) -> str:
        """Generate current TOTP code"""
        # Implementation would use pyotp library
        # Simplified for demonstration
        timestamp = int(time.time() // 30)
        hash_value = hmac.new(
            base64.b32decode(secret), timestamp.to_bytes(8, "big"), hashlib.sha1
        ).digest()

        offset = hash_value[-1] & 0xF
        code = (
            int.from_bytes(hash_value[offset : offset + 4], "big") & 0x7FFFFFFF
        ) % 1000000

        return str(code).zfill(6)

    def verify_totp(self, secret: str, code: str) -> bool:
        """Verify TOTP code"""
        try:
            # Check current and adjacent time windows
            for i in range(-self.totp_window, self.totp_window + 1):
                timestamp = int(time.time() // 30) + i
                expected = self.generate_totp_for_time(secret, timestamp)
                if expected == code:
                    return True
            return False
        except:
            return False

    def generate_totp_for_time(self, secret: str, timestamp: int) -> str:
        """Generate TOTP for specific timestamp"""
        hash_value = hmac.new(
            base64.b32decode(secret), timestamp.to_bytes(8, "big"), hashlib.sha1
        ).digest()

        offset = hash_value[-1] & 0xF
        code = (
            int.from_bytes(hash_value[offset : offset + 4], "big") & 0x7FFFFFFF
        ) % 1000000

        return str(code).zfill(6)

    def generate_backup_codes(self, count: int = 10) -> list[str]:
        """Generate backup codes for account recovery"""
        return [secrets.token_hex(4) for _ in range(count)]


class APIKeyManager:
    """API key management for service authentication"""

    def __init__(self, config: AuthConfig = None):
        self.config = config or AuthConfig()
        self.api_keys = {}  # In production, use database

    def generate_api_key(
        self, user: User, name: str, permissions: list[Permission]
    ) -> tuple[str, str]:
        """Generate API key for user"""
        # Generate key components
        key_id = secrets.token_hex(8)
        key_secret = secrets.token_urlsafe(32)

        # Create hashed version for storage
        key_hash = hashlib.sha256(key_secret.encode()).hexdigest()

        # Store key information
        self.api_keys[key_id] = {
            "user_id": user.user_id,
            "name": name,
            "key_hash": key_hash,
            "permissions": [p.value for p in permissions],
            "created_at": datetime.now(),
            "expires_at": datetime.now()
            + timedelta(days=self.config.api_key_expiry_days),
            "last_used": None,
            "is_active": True,
        }

        # Return full key (only shown once)
        full_key = f"{key_id}.{key_secret}"
        return key_id, full_key

    def verify_api_key(self, api_key: str) -> dict | None:
        """Verify API key"""
        try:
            # Parse key
            if "." not in api_key:
                return None

            key_id, key_secret = api_key.split(".", 1)

            # Check if key exists
            if key_id not in self.api_keys:
                return None

            key_info = self.api_keys[key_id]

            # Check if active
            if not key_info["is_active"]:
                return None

            # Check expiration
            if datetime.now() > key_info["expires_at"]:
                return None

            # Verify secret
            key_hash = hashlib.sha256(key_secret.encode()).hexdigest()
            if key_hash != key_info["key_hash"]:
                return None

            # Update last used
            key_info["last_used"] = datetime.now()

            return key_info

        except Exception as e:
            logger.error(f"API key verification error: {e}")
            return None

    def revoke_api_key(self, key_id: str):
        """Revoke API key"""
        if key_id in self.api_keys:
            self.api_keys[key_id]["is_active"] = False

    def list_api_keys(self, user_id: str) -> list[dict]:
        """List user's API keys"""
        user_keys = []
        for key_id, key_info in self.api_keys.items():
            if key_info["user_id"] == user_id:
                # Don't include secret hash
                safe_info = {
                    "key_id": key_id,
                    "name": key_info["name"],
                    "created_at": key_info["created_at"],
                    "expires_at": key_info["expires_at"],
                    "last_used": key_info["last_used"],
                    "is_active": key_info["is_active"],
                }
                user_keys.append(safe_info)

        return user_keys


class AuthenticationService:
    """Main authentication service"""

    def __init__(self, config: AuthConfig = None):
        self.config = config or AuthConfig()
        self.password_manager = PasswordManager(config)
        self.token_manager = TokenManager(config)
        self.session_manager = SessionManager(config)
        self.mfa_manager = MFAManager()
        self.api_key_manager = APIKeyManager(config)
        self.role_manager = RolePermissionManager()
        self.users = {}  # In production, use database
        self.audit_log = []

    def register_user(
        self, username: str, email: str, password: str, roles: list[UserRole] = None
    ) -> User:
        """Register new user"""
        # Check if user exists
        if username in self.users:
            raise ValueError("Username already exists")

        # Create user
        user = User(
            user_id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            roles=roles or [UserRole.VIEWER],
            password_hash=self.password_manager.hash_password(password),
        )

        # Set permissions based on roles
        user.permissions = self.role_manager.get_permissions(user.roles)

        # Store user
        self.users[username] = user

        # Log registration
        self._log_event("USER_REGISTERED", user.user_id, {"username": username})

        return user

    def authenticate(
        self,
        username: str,
        password: str,
        mfa_code: str | None = None,
        ip_address: str = "127.0.0.1",
        user_agent: str = "Unknown",
    ) -> tuple[User | None, str | None, Session | None]:
        """Authenticate user and return user, token, and session"""
        # Get user
        user = self.users.get(username)
        if not user:
            self._log_event(
                "AUTH_FAILED", None, {"username": username, "reason": "user_not_found"}
            )
            return None, None, None

        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self._log_event("AUTH_FAILED", user.user_id, {"reason": "account_locked"})
            return None, None, None

        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash):
            user.failed_attempts += 1

            # Lock account if too many failed attempts
            if user.failed_attempts >= self.config.max_failed_attempts:
                user.locked_until = datetime.now() + timedelta(
                    minutes=self.config.lockout_duration_minutes
                )
                self._log_event(
                    "ACCOUNT_LOCKED",
                    user.user_id,
                    {"failed_attempts": user.failed_attempts},
                )

            self._log_event("AUTH_FAILED", user.user_id, {"reason": "invalid_password"})
            return None, None, None

        # Verify MFA if enabled
        if user.mfa_enabled or self.config.mfa_required:
            if not mfa_code:
                self._log_event("AUTH_FAILED", user.user_id, {"reason": "mfa_required"})
                return None, None, None

            if not self.mfa_manager.verify_totp(user.mfa_secret, mfa_code):
                self._log_event("AUTH_FAILED", user.user_id, {"reason": "invalid_mfa"})
                return None, None, None

        # Reset failed attempts
        user.failed_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()

        # Generate token
        token = self.token_manager.generate_token(user)

        # Create session
        session = self.session_manager.create_session(user, ip_address, user_agent)

        # Log successful authentication
        self._log_event(
            "AUTH_SUCCESS",
            user.user_id,
            {"ip_address": ip_address, "session_id": session.session_id},
        )

        return user, token, session

    def verify_request(
        self,
        token: str | None = None,
        session_id: str | None = None,
        api_key: str | None = None,
    ) -> User | None:
        """Verify request authentication"""
        # Try token authentication
        if token:
            payload = self.token_manager.verify_token(token)
            if payload:
                user_id = payload.get("user_id")
                # Get user from database
                for user in self.users.values():
                    if user.user_id == user_id:
                        return user

        # Try session authentication
        if session_id:
            if self.session_manager.validate_session(session_id):
                session = self.session_manager.get_session(session_id)
                if session:
                    # Get user from database
                    for user in self.users.values():
                        if user.user_id == session.user_id:
                            return user

        # Try API key authentication
        if api_key and self.config.enable_api_keys:
            key_info = self.api_key_manager.verify_api_key(api_key)
            if key_info:
                # Get user from database
                for user in self.users.values():
                    if user.user_id == key_info["user_id"]:
                        # Set permissions from API key
                        user.permissions = [
                            Permission[p] for p in key_info["permissions"]
                        ]
                        return user

        return None

    def logout(self, user: User, session_id: str | None = None):
        """Logout user"""
        if session_id:
            self.session_manager.invalidate_session(session_id)
        else:
            self.session_manager.invalidate_user_sessions(user.user_id)

        self._log_event("USER_LOGOUT", user.user_id, {"session_id": session_id})

    def change_password(self, user: User, old_password: str, new_password: str) -> bool:
        """Change user password"""
        # Verify old password
        if not self.password_manager.verify_password(old_password, user.password_hash):
            self._log_event(
                "PASSWORD_CHANGE_FAILED",
                user.user_id,
                {"reason": "invalid_old_password"},
            )
            return False

        # Update password
        user.password_hash = self.password_manager.hash_password(new_password)

        # Invalidate all sessions
        self.session_manager.invalidate_user_sessions(user.user_id)

        self._log_event("PASSWORD_CHANGED", user.user_id, {})
        return True

    def enable_mfa(self, user: User) -> str:
        """Enable MFA for user"""
        user.mfa_enabled = True
        user.mfa_secret = self.mfa_manager.generate_secret()

        self._log_event("MFA_ENABLED", user.user_id, {})

        return self.mfa_manager.generate_qr_code(user)

    def disable_mfa(self, user: User, password: str) -> bool:
        """Disable MFA for user"""
        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash):
            return False

        user.mfa_enabled = False
        user.mfa_secret = None

        self._log_event("MFA_DISABLED", user.user_id, {})
        return True

    def _log_event(self, event_type: str, user_id: str | None, details: dict):
        """Log authentication event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
        }
        self.audit_log.append(event)

        # Also log to file/database in production
        logger.info(f"Auth event: {event_type} for user {user_id}")

    def get_audit_log(
        self,
        user_id: str | None = None,
        event_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
        """Get filtered audit log"""
        filtered_log = self.audit_log

        if user_id:
            filtered_log = [e for e in filtered_log if e["user_id"] == user_id]

        if event_type:
            filtered_log = [e for e in filtered_log if e["event_type"] == event_type]

        if start_date:
            filtered_log = [
                e
                for e in filtered_log
                if datetime.fromisoformat(e["timestamp"]) >= start_date
            ]

        if end_date:
            filtered_log = [
                e
                for e in filtered_log
                if datetime.fromisoformat(e["timestamp"]) <= end_date
            ]

        return filtered_log


def require_auth(func):
    """Decorator to require authentication"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract authentication from context
        token = kwargs.get("token")
        session_id = kwargs.get("session_id")
        api_key = kwargs.get("api_key")

        if not any([token, session_id, api_key]):
            raise PermissionError("Authentication required")

        # Verify authentication (would use global auth service)
        # For now, just pass through
        return func(*args, **kwargs)

    return wrapper


def require_role(role: UserRole):
    """Decorator to require specific role"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = kwargs.get("user")
            if not user or role not in user.roles:
                raise PermissionError(f"Role {role.value} required")
            return func(*args, **kwargs)

        return wrapper

    return decorator
