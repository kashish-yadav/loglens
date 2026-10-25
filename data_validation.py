"""Load a log CSV and add cleaned columns. Bad values are flagged, never dropped."""
import io
from pathlib import Path

import pandas as pd

from common import INPUT_COLUMNS, LEVEL_ALIASES, REQUIRED_COLUMNS


class ValidationError(ValueError):
    """The file cannot be used at all (empty, wrong format, missing columns)."""


def load_csv(source, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV (path or bytes) as text. Raises ValidationError for unusable input."""
    name = filename or (str(source) if isinstance(source, (str, Path)) else "")
    if name and not name.lower().endswith(".csv"):
        raise ValidationError("Please upload a .csv file.")
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if not data.strip():
        raise ValidationError("The file is empty.")
    try:
        text = data.decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValidationError(f"Could not read this as a UTF-8 CSV: {exc}") from exc
    if df.empty:
        raise ValidationError("The CSV has a header but no records.")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and add: record_id, ts, level_clean, status_clean, is_valid, data_issues.

    A record is valid when its message is not empty. Original columns are untouched.
    """
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValidationError(f"Missing required column(s): {', '.join(missing)}. "
                              f"Expected: {', '.join(INPUT_COLUMNS)} (last four optional).")
    for col in INPUT_COLUMNS:
        out[col] = out[col].fillna("").astype(str) if col in out.columns else ""
    out.insert(0, "record_id", range(1, len(out) + 1))

    ts_raw = out["timestamp"].str.strip()
    out["ts"] = pd.to_datetime(ts_raw.where(ts_raw != ""), errors="coerce", utc=True,
                               format="ISO8601").dt.tz_convert(None)
    level = out["level"].str.strip().str.upper()
    out["level_clean"] = level.map(LEVEL_ALIASES).fillna("UNKNOWN")
    status = pd.to_numeric(out["status_code"].str.strip(), errors="coerce")
    out["status_clean"] = status.where(status == status.round()).astype("Int64")
    out["is_valid"] = out["message"].str.strip() != ""

    problems = {
        "empty_message": ~out["is_valid"],
        "bad_timestamp": out["ts"].isna(),
        "bad_level": out["level_clean"] == "UNKNOWN",
        "bad_status_code": out["status_clean"].isna() & (out["status_code"].str.strip() != ""),
    }
    out["data_issues"] = [";".join(k for k, m in problems.items() if m.iloc[i])
                          for i in range(len(out))]
    return out


def load_and_clean(source, filename: str | None = None) -> pd.DataFrame:
    return clean(load_csv(source, filename))
