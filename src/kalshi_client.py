"""Kalshi REST API client — public reads + authenticated order execution."""

import base64
import datetime
import logging
import random
import time
import uuid

import requests

from src import config

log = logging.getLogger(__name__)

_PUBLIC_MIN_INTERVAL_SECONDS = 0.20
_PUBLIC_MAX_RETRIES = 5
_PUBLIC_BACKOFF_BASE_SECONDS = 0.50
_PUBLIC_BACKOFF_MAX_SECONDS = 12.0


def _load_private_key(path: str):
    """Load RSA private key from PEM file. Returns key object or None."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend(),
            )
    except Exception as e:
        log.warning("Could not load private key from %s: %s", path, e)
        return None


def _sign_request(private_key, timestamp_ms: str, method: str, path: str) -> str:
    """Create Kalshi RSA-PSS SHA-256 signature."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

    path_no_query = path.split("?")[0]
    message = f"{timestamp_ms}{method}{path_no_query}".encode("utf-8")
    sig = private_key.sign(
        message,
        asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


class KalshiClient:
    def __init__(self, base_url: str = config.KALSHI_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        self._last_public_request_ts = 0.0
        self._private_key = None
        self._api_key_id = config.KALSHI_API_KEY_ID

    def _throttle_public_get(self):
        now = time.monotonic()
        elapsed = now - self._last_public_request_ts
        if elapsed < _PUBLIC_MIN_INTERVAL_SECONDS:
            time.sleep(_PUBLIC_MIN_INTERVAL_SECONDS - elapsed)
        self._last_public_request_ts = time.monotonic()

    @staticmethod
    def _parse_retry_after_seconds(resp: requests.Response) -> float | None:
        value = resp.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            return None

    def _public_get_with_retry(self, url: str, *, params: dict | None = None, timeout: int = 30) -> requests.Response:
        last_resp: requests.Response | None = None
        for attempt in range(_PUBLIC_MAX_RETRIES):
            self._throttle_public_get()
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                if attempt >= _PUBLIC_MAX_RETRIES - 1:
                    raise
                sleep_s = min(
                    _PUBLIC_BACKOFF_BASE_SECONDS * (2 ** attempt),
                    _PUBLIC_BACKOFF_MAX_SECONDS,
                ) + random.uniform(0.0, 0.25)
                log.warning("Kalshi GET network error (attempt %d/%d): %s; retrying in %.2fs",
                            attempt + 1, _PUBLIC_MAX_RETRIES, e, sleep_s)
                time.sleep(sleep_s)
                continue

            last_resp = resp
            if resp.status_code not in (429, 500, 502, 503, 504):
                resp.raise_for_status()
                return resp

            if attempt >= _PUBLIC_MAX_RETRIES - 1:
                break

            retry_after = self._parse_retry_after_seconds(resp)
            exp_backoff = min(
                _PUBLIC_BACKOFF_BASE_SECONDS * (2 ** attempt),
                _PUBLIC_BACKOFF_MAX_SECONDS,
            )
            sleep_s = max(retry_after or 0.0, exp_backoff) + random.uniform(0.0, 0.25)
            log.warning("Kalshi GET %s for %s (attempt %d/%d); retrying in %.2fs",
                        resp.status_code, url, attempt + 1, _PUBLIC_MAX_RETRIES, sleep_s)
            time.sleep(sleep_s)

        if last_resp is not None:
            last_resp.raise_for_status()
        raise requests.HTTPError(f"Kalshi GET failed for {url} without response")

    def _ensure_auth(self):
        if not self._api_key_id:
            raise RuntimeError(
                "Kalshi API key id is missing; set KALSHI_API_KEY_ID or add "
                '"kalshi_api_key_id" to your local config.json.'
            )
        if self._private_key is None:
            self._private_key = _load_private_key(config.KALSHI_PRIVATE_KEY_PATH)
        if self._private_key is None:
            raise RuntimeError(
                f"RSA private key not available at {config.KALSHI_PRIVATE_KEY_PATH}"
            )

    def _auth_headers(self, method: str, path: str) -> dict:
        self._ensure_auth()
        ts = str(int(datetime.datetime.now().timestamp() * 1000))
        sig = _sign_request(self._private_key, ts, method, path)
        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

    def get_balance(self) -> dict:
        path = "/trade-api/v2/portfolio/balance"
        r = self.session.get(
            self.base_url.rstrip("/").rsplit("/trade-api", 1)[0] + path,
            headers=self._auth_headers("GET", path),
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> dict:
        path = "/trade-api/v2/portfolio/positions"
        r = self.session.get(
            self.base_url.rstrip("/").rsplit("/trade-api", 1)[0] + path,
            headers=self._auth_headers("GET", path),
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        yes_price: int | None = None,
        no_price: int | None = None,
        order_type: str = "limit",
        time_in_force: str = "fill_or_kill",
        buy_max_cost: int | None = None,
    ) -> dict:
        path = "/trade-api/v2/portfolio/orders"
        body = {
            "ticker": ticker,
            "action": action.lower(),
            "side": side.lower(),
            "count": count,
            "type": order_type,
            "time_in_force": time_in_force,
            "client_order_id": str(uuid.uuid4()),
        }
        if yes_price is not None:
            body["yes_price"] = int(yes_price)
        if no_price is not None:
            body["no_price"] = int(no_price)
        if buy_max_cost is not None:
            body["buy_max_cost"] = int(buy_max_cost)

        r = self.session.post(
            self.base_url.rstrip("/").rsplit("/trade-api", 1)[0] + path,
            headers=self._auth_headers("POST", path),
            json=body,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def get_trades(self, cursor=None, limit=1000):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        r = self._public_get_with_retry(
            f"{self.base_url}/markets/trades", params=params, timeout=30,
        )
        data = r.json()
        return data.get("trades", []), data.get("cursor")

    def get_market(self, ticker: str) -> dict:
        r = self._public_get_with_retry(
            f"{self.base_url}/markets/{ticker}", timeout=15,
        )
        return r.json().get("market", r.json())

    def get_event(self, event_ticker: str) -> dict:
        r = self._public_get_with_retry(
            f"{self.base_url}/events/{event_ticker}", timeout=15,
        )
        data = r.json()
        return data.get("event", data)

    def get_recently_traded_tickers(self, pages: int = 3, per_page: int = 1000) -> list[str]:
        seen = []
        seen_set = set()
        cursor = None
        for _ in range(pages):
            try:
                trades, cursor = self.get_trades(cursor=cursor, limit=per_page)
            except Exception as e:
                log.warning("Failed to fetch trades page: %s", e)
                break
            for t in trades:
                tk = t.get("ticker", "")
                if tk and tk not in seen_set:
                    seen_set.add(tk)
                    seen.append(tk)
            if not cursor or not trades:
                break
        return seen
