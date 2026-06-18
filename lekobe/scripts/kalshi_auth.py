import base64
import json
import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.exceptions import InvalidSignature

import os
from dotenv import load_dotenv
load_dotenv()

KALSHI_KEY_ID = os.getenv("KALSHI_KEY_ID")
PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH")
if not PRIVATE_KEY_PATH:
    PRIVATE_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalshi_private_key.pem")

def load_private_key():
    if not KALSHI_KEY_ID:
        raise RuntimeError("KALSHI_KEY_ID is missing from the local environment")
    with open(PRIVATE_KEY_PATH, "rb") as key_file:
        return load_pem_private_key(
            key_file.read(),
            password=None,
        )

def sign_kalshi_request(method: str, path: str, body: dict = None) -> dict:
    """
    Generates the cryptographically signed headers required by the Kalshi API.
    """
    private_key = load_private_key()

    # 1. Generate current UTC timestamp in milliseconds
    current_time_milliseconds = int(time.time() * 1000)
    timestamp_str = str(current_time_milliseconds)

    # 2. Construct the message string to sign
    # Format: timestamp + method + path
    msg_string = timestamp_str + method.upper() + path

    # If there is a JSON payload, append it to the message string
    if body:
        msg_string += json.dumps(body)

    # 3. Cryptographically sign the message string using RSA-PSS
    signature = private_key.sign(
        msg_string.encode('utf-8'),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    # 4. Base64 encode the signature
    base64_signature = base64.b64encode(signature).decode('utf-8')

    # 5. Return the required HTTP Headers
    headers = {
        "Kalshi-Access-Key": KALSHI_KEY_ID,
        "Kalshi-Access-Signature": base64_signature,
        "Kalshi-Access-Timestamp": timestamp_str,
        "Content-Type": "application/json"
    }

    return headers
