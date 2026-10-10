from pathlib import Path
from io import StringIO
import logging

import pandas as pd
from lxml import html

from .config import settings


logger = logging.getLogger(__name__)


def dataframe_to_records(df):
    """
    Convert DataFrame to JSON-compatible records.
    Handles MultiIndex and duplicate column names.
    """
    df = df.copy()

    columns = []
    seen = {}

    for i, col in enumerate(df.columns, start=1):
        parts = col if isinstance(col, tuple) else (col,)

        parts = [
            str(value).strip()
            for value in parts
            if pd.notna(value)
            and not str(value).startswith("Unnamed:")
            and str(value).strip()
        ]

        name = " | ".join(parts) or f"column_{i}"

        count = seen.get(name, 0) + 1
        seen[name] = count

        if count > 1:
            name = f"{name} [{count}]"

        columns.append(name)

    df.columns = columns

    # Replace NaN with None
    df = df.astype(object).where(pd.notna(df), None)

    records = df.to_dict(orient="records")

    # Normalize special scalar types
    for record in records:
        for key, value in record.items():

            if isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()

            elif hasattr(value, "item"):
                try:
                    record[key] = value.item()
                except (TypeError, ValueError, AttributeError):
                    record[key] = str(value)

    return records


def parse_html_excel(file_path):
    """
    Read HTML-based CODAL Excel files.

    Supports:
    - Multiple HTML tables
    - Empty table rows
    - Multi-level headers
    - Explanatory text sections
    """

    document = html.parse(str(file_path))

    results = []

    tables = document.xpath("//table")

    for index, table in enumerate(tables, start=1):

        # Extract title and description
        container = table.xpath(
            'ancestor::div[contains(@class, "table-containet")][1]'
        )

        if container:
            titles = container[0].xpath(
                './div[contains(@class, "table-title")]/text()'
            )

            descriptions = container[0].xpath(
                './div[contains(@class, "table-description")]/text()'
            )
        else:
            titles = []
            descriptions = []

        title = (
            " ".join(titles).strip()
            if titles
            else f"table_{index}"
        )

        unit = (
            " ".join(descriptions).strip()
            if descriptions
            else None
        )

        # Remove empty TR elements
        for row in table.xpath(".//tr"):
            if not row.xpath("./th | ./td"):
                row.getparent().remove(row)

        rows = table.xpath(".//tr")

        if not rows:
            continue

        # Detect text-only tables
        is_text_table = (
            not table.xpath(".//th")
            and all(
                len(row.xpath("./td")) == 1
                for row in rows
            )
        )

        if is_text_table:
            notes = []

            for row in rows:
                for cell in row.xpath("./td"):
                    text = " ".join(
                        " ".join(cell.itertext()).split()
                    )

                    if text:
                        notes.append({
                            "text": text
                        })

            if notes:
                results.append({
                    "sheet": title,
                    "type": "notes",
                    "unit": unit,
                    "data": notes,
                })

            continue

        # Read each HTML table independently
        try:
            table_html = html.tostring(
                table,
                encoding="unicode",
            )

            dataframes = pd.read_html(
                StringIO(table_html),
                flavor="lxml",
            )

            for j, df in enumerate(dataframes, start=1):

                if df.empty:
                    continue

                sheet_name = (
                    title
                    if len(dataframes) == 1
                    else f"{title}_{j}"
                )

                results.append({
                    "sheet": sheet_name,
                    "type": "table",
                    "unit": unit,
                    "data": dataframe_to_records(df),
                })

        except (ValueError, IndexError) as e:
            logger.warning(
                "Error parsing HTML table %s in %s: %s",
                index,
                file_path,
                e,
            )

    return results


def read_codal_data(file_path: str):
    """
    Read all CODAL Excel files from a file or directory.

    Supported formats:
    - XLS
    - XLSX
    - HTML disguised as XLS
    - HTML

    Returns structured financial data and notes.
    """

    source_path = Path(settings.codal_data_source_path)
    full_path = source_path / file_path

    if not full_path.exists():
        raise FileNotFoundError(full_path)

    supported_extensions = {
        ".xls",
        ".xlsx",
        ".html",
        ".htm",
    }

    # Find files
    if full_path.is_file():
        excel_files = [full_path]
    else:
        excel_files = sorted(
            path
            for path in full_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in supported_extensions
        )

    results = []

    for excel_file in excel_files:

        if excel_file.suffix.lower() not in supported_extensions:
            continue

        try:
            with excel_file.open("rb") as f:
                header = f.read(4096)

            header_lower = header.lstrip().lower()

            # Detect HTML disguised as Excel
            is_html = any(
                token in header_lower
                for token in (
                    b"<html",
                    b"<table",
                    b"<!doctype html",
                )
            )

            if is_html:
                sections = parse_html_excel(excel_file)

            else:
                # Detect real Excel format
                engine = (
                    "openpyxl"
                    if header_lower.startswith(b"pk\x03\x04")
                    else "xlrd"
                )

                sheets = pd.read_excel(
                    excel_file,
                    sheet_name=None,
                    engine=engine,
                )

                sections = []

                for sheet_name, df in sheets.items():
                    sections.append({
                        "sheet": sheet_name,
                        "type": "table",
                        "unit": None,
                        "data": dataframe_to_records(df),
                    })

            for section in sections:
                results.append({
                    "file_path": str(excel_file),
                    **section,
                })

        except Exception:
            logger.exception(
                "Error reading CODAL file: %s",
                excel_file,
            )

    return results