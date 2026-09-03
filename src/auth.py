"""
auth.py - Enterprise Role-Based Authentication Engine for HyRAG.

Provides user validation, secure salted SHA-256 password hashing,
and role-based authorization (Employee vs Admin).
"""

import hashlib
import hmac
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("HyRAG.Auth")

# Static salt for reproducible password hashing across sessions
AUTH_SALT = "HyRAG_Enterprise_Security_Salt_2026"


def hash_password(password: str, salt: str = AUTH_SALT) -> str:
    """Generates a secure SHA-256 hash for a given password and salt."""
    salted_pwd = f"{salt}:{password}".encode("utf-8")
    return hashlib.sha256(salted_pwd).hexdigest()


# 10 Pre-configured Employee & Admin Accounts
USER_DATABASE: Dict[str, Dict[str, Any]] = {
    "EMP1001": {
        "user_id": "EMP1001",
        "name": "Alex Mercer",
        "email": "alex.mercer@enterprise.internal",
        "department": "Security Architecture",
        "role": "Employee",
        "password_hash": hash_password("HyRAG@1001")
    },
    "EMP1002": {
        "user_id": "EMP1002",
        "name": "Jordan Hayes",
        "email": "jordan.hayes@enterprise.internal",
        "department": "Cloud Engineering",
        "role": "Employee",
        "password_hash": hash_password("HyRAG@1002")
    },
    "EMP1003": {
        "user_id": "EMP1003",
        "name": "Elena Rostova",
        "email": "elena.rostova@enterprise.internal",
        "department": "Compliance & Audit",
        "role": "Employee",
        "password_hash": hash_password("HyRAG@1003")
    },
    "EMP1004": {
        "user_id": "EMP1004",
        "name": "Marcus Chen",
        "email": "marcus.chen@enterprise.internal",
        "department": "DevOps",
        "role": "Employee",
        "password_hash": hash_password("HyRAG@1004")
    },
    "EMP1005": {
        "user_id": "EMP1005",
        "name": "Sarah Jenkins",
        "email": "sarah.jenkins@enterprise.internal",
        "department": "Legal & Ethics",
        "role": "Employee",
        "password_hash": hash_password("HyRAG@1005")
    },
    "EMP1006": {
        "user_id": "EMP1006",
        "name": "David Kim",
        "email": "david.kim@enterprise.internal",
        "department": "Infrastructure",
        "role": "Employee",
        "password_hash": hash_password("HyRAG@1006")
    },
    "EMP1007": {
        "user_id": "EMP1007",
        "name": "Priya Sharma",
        "email": "priya.sharma@enterprise.internal",
        "department": "Data Engineering",
        "role": "Employee",
        "password_hash": hash_password("HyRAG@1007")
    },
    "EMP1008": {
        "user_id": "EMP1008",
        "name": "Liam O'Connor",
        "email": "liam.oconnor@enterprise.internal",
        "department": "Product Management",
        "role": "Employee",
        "password_hash": hash_password("HyRAG@1008")
    },
    "EMP1009": {
        "user_id": "EMP1009",
        "name": "Zoe Martinez",
        "email": "zoe.martinez@enterprise.internal",
        "department": "IT Operations",
        "role": "Employee",
        "password_hash": hash_password("HyRAG@1009")
    },
    "ADMIN001": {
        "user_id": "ADMIN001",
        "name": "Principal Administrator",
        "email": "admin@enterprise.internal",
        "department": "Global Infrastructure & Governance",
        "role": "Admin",
        "password_hash": hash_password("HyRAG@Admin01")
    },
}


def authenticate_user(user_id: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Validates user credentials against the secure user store.
    
    Returns:
        User record dictionary without password hash if authenticated, else None.
    """
    clean_id = user_id.strip().upper()
    if clean_id not in USER_DATABASE:
        return None

    user_record = USER_DATABASE[clean_id]
    target_hash = hash_password(password)

    # Constant-time comparison to prevent timing attacks
    if hmac.compare_digest(user_record["password_hash"], target_hash):
        safe_copy = {k: v for k, v in user_record.items() if k != "password_hash"}
        logger.info(f"User '{clean_id}' successfully authenticated as '{safe_copy['role']}'")
        return safe_copy

    logger.warning(f"Failed authentication attempt for user '{clean_id}'")
    return None


def get_user_info(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves safe user information by user ID."""
    clean_id = user_id.strip().upper()
    if clean_id in USER_DATABASE:
        return {k: v for k, v in USER_DATABASE[clean_id].items() if k != "password_hash"}
    return None


def list_all_users() -> List[Dict[str, Any]]:
    """Lists all active employee and admin accounts."""
    return [{k: v for k, v in u.items() if k != "password_hash"} for u in USER_DATABASE.values()]


def is_admin(user_id: str) -> bool:
    """Checks whether the given user ID has Admin privileges."""
    clean_id = user_id.strip().upper()
    user = USER_DATABASE.get(clean_id)
    return user is not None and user.get("role") == "Admin"
