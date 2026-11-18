import asyncio
import calendar
import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import UUID

import feedparser
import httpx
from app.infra.models import (
    IngestionRunModel,
    InstrumentModel,
    NewsArticleInstrumentModel,
    NewsArticleModel,
    NewsSourceModel,
    new_uuid7,
)
from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

RSS_HEADERS = {"User-Agent": "FinSightAI/1.1 (+https://github.com/finsight-ai)"}
MAX_SUMMARY_LEN = 2000

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_TRACKING_QUERY_PREFIXES = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref")

# Word-boundary rules so "method" never matches ETH and "spyware" never matches SPY
_TICKER_RULES: dict[str, list[tuple[re.Pattern[str], float]]] = {
    "BTCUSDT": [
        (re.compile(r"\b(bitcoin|btcusdt|\$?btc)\b", re.IGNORECASE), 1.0),
        (re.compile(r"\b(satoshi|lightning network|btc etf)\b", re.IGNORECASE), 0.7),
    ],
    "ETHUSDT": [
        (re.compile(r"\b(ethereum|ethusdt|\$?eth)\b", re.IGNORECASE), 1.0),
        (re.compile(r"\b(vitalik|ether|eip-\d+|eth etf)\b", re.IGNORECASE), 0.7),
    ],
    "SPY": [
        (re.compile(r"\b(s&p\s*500|sp500|spx|\$?spy)\b", re.IGNORECASE), 1.0),
        (
            re.compile(
                r"\b(wall street|federal reserve|fomc|u\.s\.\s+stocks?|nasdaq|dow jones)\b",
                re.IGNORECASE,
            ),
            0.6,
        ),
    ],
}


@dataclass(frozen=True)
class ParsedNewsItem:
    url: str
    url_hash: str
    content_hash: str
    title: str
    body: str
    language: str
    published_at: datetime
    ingested_at: datetime
    available_at: datetime
    availability_basis: str
    raw: dict[str, Any]


@dataclass
class FeedParseResult:
    items: list[ParsedNewsItem] = field(default_factory=list)
    rejected_count: int = 0
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class RssIngestSummary:
    sources_checked: int
    sources_failed: int
    rows_written: int
    rows_rejected: int
    links_created: int
    status: str


def canonicalize_url(raw_url: str) -> str:
    """Normalize URL and strip tracking query parameters and fragments."""
    parsed = urlparse(raw_url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    filtered_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if not any(k.lower().startswith(p) for p in _TRACKING_QUERY_PREFIXES)
    ]
    filtered_params.sort()
    query = urlencode(filtered_params)
    return urlunparse((scheme, netloc, path, "", query, ""))


def clean_rss_text(raw: str, max_len: int = MAX_SUMMARY_LEN) -> str:
    """Strip HTML tags, unescape entities, and collapse whitespace (short RSS summary only)."""
    unescaped = html.unescape(raw or "")
    stripped = _HTML_TAG_RE.sub(" ", unescaped)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()
    if len(collapsed) > max_len:
        return collapsed[:max_len].rstrip()
    return collapsed


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_entry_published_utc(entry: Any, fallback_utc: datetime) -> datetime:
    struct_t = entry.get("published_parsed") or entry.get("updated_parsed")
    if struct_t is not None:
        try:
            ts = calendar.timegm(struct_t)
            return datetime.fromtimestamp(ts, tz=UTC)
        except (TypeError, ValueError, OverflowError):
            pass
    return fallback_utc


def parse_rss_feed(
    xml_text: str,
    *,
    now: datetime | None = None,
    availability_basis: str = "live",
) -> FeedParseResult:
    """
    Parse an RSS/Atom XML document into normalized ParsedNewsItem records.
    - Stores title + short RSS summary in `body` (never fetches full HTML pages).
    - Sets available_at = ingested_at for live collection (point-in-time safe).
    - Counts entries missing URL or title in rejected_count.
    """
    ingested_at = (now or datetime.now(UTC)).astimezone(UTC)
    parsed = feedparser.parse(xml_text)
    feed_lang = str((parsed.feed or {}).get("language", "en"))[:10].split("-")[0].lower() or "en"

    result = FeedParseResult()
    seen_url_hashes: set[str] = set()

    for idx, entry in enumerate(parsed.entries):
        raw_link = str(entry.get("link") or "").strip()
        title = clean_rss_text(str(entry.get("title") or ""), max_len=500)

        if not raw_link or not title:
            result.rejected_count += 1
            if len(result.rejection_reasons) < 20:
                result.rejection_reasons.append(f"entry[{idx}]: missing link or title")
            continue

        canonical = canonicalize_url(raw_link)
        url_hash = sha256_hex(canonical)
        if url_hash in seen_url_hashes:
            continue
        seen_url_hashes.add(url_hash)

        raw_summary = str(entry.get("summary") or entry.get("description") or "")
        body = clean_rss_text(raw_summary, max_len=MAX_SUMMARY_LEN)
        content_hash = sha256_hex(f"{title.lower()} {body.lower()}".strip())
        published_at = _parse_entry_published_utc(entry, fallback_utc=ingested_at)

        # Point-in-time rule: live uses ingested_at; backfill proxy uses published_at + 5m
        if availability_basis == "live":
            available_at = ingested_at
        else:
            from datetime import timedelta

            available_at = published_at + timedelta(minutes=5)

        raw_meta = {
            "guid": str(entry.get("id") or ""),
            "author": str(entry.get("author") or ""),
            "tags": [str(t.get("term", "")) for t in (entry.get("tags") or []) if t.get("term")],
        }

        result.items.append(
            ParsedNewsItem(
                url=canonical,
                url_hash=url_hash,
                content_hash=content_hash,
                title=title,
                body=body,
                language=feed_lang,
                published_at=published_at,
                ingested_at=ingested_at,
                available_at=available_at,
                availability_basis=availability_basis,
                raw=raw_meta,
            )
        )

    return result


def match_instruments(
    title: str,
    body: str,
    symbol_to_id: dict[str, UUID],
) -> list[tuple[UUID, float]]:
    """Match an article's title + summary against tracked instruments using word boundaries."""
    text = f"{title} {body}"
    matches: list[tuple[UUID, float]] = []

    for sym, inst_id in symbol_to_id.items():
        rules = _TICKER_RULES.get(sym.upper(), [])
        best_rel = 0.0
        for pattern, relevance in rules:
            if pattern.search(text):
                best_rel = max(best_rel, relevance)
        if best_rel > 0.0:
            matches.append((inst_id, best_rel))

    return matches


async def _fetch_feed_xml(
    http: httpx.AsyncClient,
    url: str,
    max_retries: int = 2,
) -> str:
    for attempt in range(max_retries):
        try:
            resp = await http.get(url)
        except httpx.TransportError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1.0 * (attempt + 1))
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries - 1:
                resp.raise_for_status()
            await asyncio.sleep(1.0 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.text
    raise RuntimeError(f"Unreachable retry state for {url}")


async def ingest_rss_news(
    session: AsyncSession,
    *,
    client: httpx.AsyncClient | None = None,
) -> RssIngestSummary:
    """
    Poll all active RSS news_sources, deduplicate by url_hash, store short summary +
    available_at/availability_basis='live', link matched instruments, and ALWAYS write
    a heartbeat row into ingestion_runs (even when 0 new articles) for coverage tracking.
    """
    started_at = datetime.now(UTC)

    sources = list(
        (
            await session.execute(
                select(NewsSourceModel).where(
                    NewsSourceModel.kind == "rss",
                    NewsSourceModel.is_active.is_(True),
                )
            )
        )
        .scalars()
        .all()
    )
    active_instruments = list(
        (await session.execute(select(InstrumentModel).where(InstrumentModel.is_active.is_(True))))
        .scalars()
        .all()
    )
    symbol_to_id: dict[str, UUID] = {inst.symbol.upper(): inst.id for inst in active_instruments}

    owns_client = client is None
    http = client or httpx.AsyncClient(
        headers=RSS_HEADERS,
        follow_redirects=True,
        timeout=15.0,
    )

    total_written = 0
    total_rejected = 0
    total_links = 0
    failed_sources: list[str] = []

    try:
        for src in sources:
            try:
                xml_text = await _fetch_feed_xml(http, src.url)
                parsed = parse_rss_feed(xml_text, now=datetime.now(UTC), availability_basis="live")
                total_rejected += parsed.rejected_count

                if not parsed.items:
                    continue

                rows: list[dict[str, Any]] = []
                item_by_url_hash: dict[str, ParsedNewsItem] = {}
                for item in parsed.items:
                    rows.append(
                        {
                            "id": new_uuid7(),
                            "source_id": src.id,
                            "url": item.url,
                            "url_hash": item.url_hash,
                            "content_hash": item.content_hash,
                            "title": item.title,
                            "body": item.body,
                            "language": item.language,
                            "published_at": item.published_at,
                            "ingested_at": item.ingested_at,
                            "available_at": item.available_at,
                            "availability_basis": item.availability_basis,
                            "dedup_cluster_id": None,
                            "raw": item.raw,
                        }
                    )
                    item_by_url_hash[item.url_hash] = item

                stmt = (
                    pg_insert(NewsArticleModel)
                    .values(rows)
                    .on_conflict_do_nothing(index_elements=["url_hash"])
                    .returning(NewsArticleModel.id, NewsArticleModel.url_hash)
                )
                inserted = list((await session.execute(stmt)).all())
                total_written += len(inserted)

                link_rows: list[dict[str, Any]] = []
                for article_id, u_hash in inserted:
                    art_item = item_by_url_hash.get(u_hash)
                    if art_item is None:
                        continue
                    for inst_id, relevance in match_instruments(
                        art_item.title, art_item.body, symbol_to_id
                    ):
                        link_rows.append(
                            {
                                "article_id": article_id,
                                "instrument_id": inst_id,
                                "relevance": relevance,
                            }
                        )

                if link_rows:
                    link_stmt = (
                        pg_insert(NewsArticleInstrumentModel)
                        .values(link_rows)
                        .on_conflict_do_nothing(constraint="pk_news_article_instruments")
                    )
                    await session.execute(link_stmt)
                    total_links += len(link_rows)

            except Exception as exc:
                logger.warning("RSS fetch failed for {}: {}", src.name, exc)
                failed_sources.append(f"{src.name}: {exc}")

        if failed_sources and len(failed_sources) == len(sources) and sources:
            run_status = "failed"
        elif failed_sources or total_rejected > 0:
            run_status = "partial"
        else:
            run_status = "ok"

        error_msg = "; ".join(failed_sources)[:1000] if failed_sources else None

        # Always record heartbeat in ingestion_runs for uptime / coverage accounting
        session.add(
            IngestionRunModel(
                job_name="news_fetch",
                source="rss",
                instrument_id=None,
                timeframe=None,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                rows_written=total_written,
                rows_rejected=total_rejected,
                gaps_found=0,
                status=run_status,
                error=error_msg,
            )
        )
        await session.commit()

        logger.info(
            "news_fetch [rss]: sources={}/{} ok, written={}, rejected={}, links={}, status={}",
            len(sources) - len(failed_sources),
            len(sources),
            total_written,
            total_rejected,
            total_links,
            run_status,
        )

        return RssIngestSummary(
            sources_checked=len(sources),
            sources_failed=len(failed_sources),
            rows_written=total_written,
            rows_rejected=total_rejected,
            links_created=total_links,
            status=run_status,
        )
    except Exception as exc:
        await session.rollback()
        session.add(
            IngestionRunModel(
                job_name="news_fetch",
                source="rss",
                instrument_id=None,
                timeframe=None,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                rows_written=0,
                rows_rejected=0,
                gaps_found=0,
                status="failed",
                error=str(exc)[:1000],
            )
        )
        await session.commit()
        raise
    finally:
        if owns_client:
            await http.aclose()
