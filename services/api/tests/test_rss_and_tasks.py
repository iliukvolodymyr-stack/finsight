from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from app.infra.db import AsyncSessionLocal
from app.infra.models import (
    IngestionRunModel,
    NewsArticleInstrumentModel,
    NewsArticleModel,
    NewsSourceModel,
    new_uuid7,
)
from app.services.rss import (
    canonicalize_url,
    ingest_rss_news,
    match_instruments,
    parse_rss_feed,
)
from app.worker.celery_app import celery_app
from sqlalchemy import delete, select

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Crypto &amp; Markets Feed</title>
    <language>en-US</language>
    <item>
      <title>Bitcoin Breaks $92,000 as Institutional Inflows Accelerate</title>
      <link>https://example.com/news/btc-rally/?utm_source=rss&amp;ref=feed#section1</link>
      <description>&lt;p&gt;Spot &lt;b&gt;BTC&lt;/b&gt; ETFs saw strong inflows while Ethereum held steady.&lt;/p&gt;</description>
      <pubDate>Tue, 18 Nov 2026 10:15:00 GMT</pubDate>
      <guid>guid-btc-1</guid>
    </item>
    <item>
      <title>Bitcoin Breaks $92,000 Duplicate Tracking Link</title>
      <link>https://example.com/news/btc-rally?utm_campaign=twitter</link>
      <description>Duplicate URL with different tracking params.</description>
      <pubDate>Tue, 18 Nov 2026 10:16:00 GMT</pubDate>
    </item>
    <item>
      <title>S&amp;P 500 Futures Climb Ahead of Federal Reserve Minutes</title>
      <link>https://example.com/news/spy-futures</link>
      <description>Wall Street traders watch SPY ahead of FOMC release.</description>
      <pubDate>Tue, 18 Nov 2026 10:20:00 GMT</pubDate>
    </item>
    <item>
      <title>Corrupt Entry Missing Link</title>
      <description>No link element present.</description>
    </item>
  </channel>
</rss>
"""

EMPTY_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Empty</title></channel></rss>
"""


def test_canonicalize_url_strips_tracking_and_fragment():
    u1 = canonicalize_url("HTTPS://Example.COM/post/123/?utm_source=x&id=42&fbclid=abc#comments")
    u2 = canonicalize_url("https://example.com/post/123?id=42")
    assert u1 == u2 == "https://example.com/post/123?id=42"


def test_parse_rss_feed_fixture_available_at_and_rejections():
    ref_now = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)
    res = parse_rss_feed(SAMPLE_RSS_XML, now=ref_now, availability_basis="live")

    # 4 entries: 2 unique valid + 1 duplicate canonical URL skipped + 1 missing link rejected
    assert len(res.items) == 2
    assert res.rejected_count == 1

    first = res.items[0]
    assert first.url == "https://example.com/news/btc-rally"
    assert "<b>" not in first.body and "<p>" not in first.body
    assert "Spot BTC ETFs" in first.body
    assert first.published_at == datetime(2026, 11, 18, 10, 15, 0, tzinfo=UTC)
    assert first.ingested_at == ref_now
    assert first.available_at == ref_now
    assert first.availability_basis == "live"
    assert first.language == "en"


def test_match_instruments_word_boundaries():
    sym_map: dict[str, UUID] = {
        "BTCUSDT": new_uuid7(),
        "ETHUSDT": new_uuid7(),
        "SPY": new_uuid7(),
    }

    m1 = dict(
        match_instruments(
            "Bitcoin surges as Ethereum staking grows",
            "Crypto analysts track BTC and ETH.",
            sym_map,
        )
    )
    assert m1[sym_map["BTCUSDT"]] == 1.0
    assert m1[sym_map["ETHUSDT"]] == 1.0
    assert sym_map["SPY"] not in m1

    # Word-boundary protection: "method" != ETH, "spyware" != SPY
    m_false = match_instruments(
        "New cryptographic method blocks spyware attacks",
        "Whether researchers adopt this method remains unclear.",
        sym_map,
    )
    assert m_false == []

    m_spy = dict(match_instruments("S&P 500 hits record", "SPY volume climbs.", sym_map))
    assert m_spy[sym_map["SPY"]] == 1.0


@pytest.mark.asyncio
async def test_ingest_rss_news_dedup_links_and_heartbeat():
    test_source_id = new_uuid7()
    test_source_url = "https://mock.finsight.local/rss/test-feed.xml"
    before_test = datetime.now(UTC)

    def mock_handler(req: httpx.Request) -> httpx.Response:
        if str(req.url) == test_source_url:
            return httpx.Response(200, text=SAMPLE_RSS_XML)
        return httpx.Response(200, text=EMPTY_RSS_XML)

    async with AsyncSessionLocal() as session:
        session.add(
            NewsSourceModel(
                id=test_source_id,
                kind="rss",
                name="Mock Test RSS Feed",
                url=test_source_url,
                is_active=True,
                trust_score=0.9,
            )
        )
        await session.commit()

        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as http:
                # Run 1: inserts 2 articles, rejects 1 corrupt item, links tickers, writes heartbeat
                s1 = await ingest_rss_news(session, client=http)
                assert s1.sources_failed == 0
                assert s1.rows_written == 2
                assert s1.rows_rejected == 1
                assert s1.links_created >= 3  # BTC + ETH on art1, SPY on art2
                assert s1.status == "partial"

                # Verify stored articles have available_at & availability_basis='live'
                arts = list(
                    (
                        await session.execute(
                            select(NewsArticleModel).where(
                                NewsArticleModel.source_id == test_source_id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                assert len(arts) == 2
                assert all(a.availability_basis == "live" for a in arts)
                assert all(a.available_at == a.ingested_at for a in arts)

                art_ids = [a.id for a in arts]
                links = list(
                    (
                        await session.execute(
                            select(NewsArticleInstrumentModel).where(
                                NewsArticleInstrumentModel.article_id.in_(art_ids)
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                assert len(links) >= 3

                # Run 2 with same feed: 100% deduplicated (0 new rows), still writes heartbeat!
                s2 = await ingest_rss_news(session, client=http)
                assert s2.rows_written == 0
                assert s2.links_created == 0

                heartbeats = list(
                    (
                        await session.execute(
                            select(IngestionRunModel).where(
                                IngestionRunModel.job_name == "news_fetch",
                                IngestionRunModel.source == "rss",
                                IngestionRunModel.started_at >= before_test,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                assert len(heartbeats) >= 2
        finally:
            await session.execute(
                delete(NewsSourceModel).where(NewsSourceModel.id == test_source_id)
            )
            await session.execute(
                delete(IngestionRunModel).where(
                    IngestionRunModel.job_name == "news_fetch",
                    IngestionRunModel.started_at >= before_test,
                )
            )
            await session.commit()


def test_celery_beat_schedule_and_tasks_registered():
    # Ensure worker task module is loaded
    import app.worker.tasks  # noqa: F401

    schedule = celery_app.conf.beat_schedule
    assert "poll-closed-bars-1m" in schedule
    assert schedule["poll-closed-bars-1m"]["schedule"] == 60.0
    assert "fetch-rss-news-5m" in schedule
    assert schedule["fetch-rss-news-5m"]["schedule"] == 300.0
    assert "check-data-gaps-1h" in schedule
    assert schedule["check-data-gaps-1h"]["schedule"] == 3600.0

    assert "ingest.poll_closed_bars" in celery_app.tasks
    assert "ingest.fetch_rss_news" in celery_app.tasks
    assert "ingest.check_data_gaps" in celery_app.tasks
