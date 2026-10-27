"""Polite, resumable news scraper for building the manual tier.

Scraping is manual collection: we fetch from the primary source and do our own
extraction and cleaning. The course guidance is explicit -- "anything that
involves you getting data, processing cleaning and organizing is considered
Manual" -- so everything this module emits is tagged ``manual``.

The design is shaped by one fact: this has to run for days without
supervision, on a machine that may sleep, behind a connection that may drop.

**Sitemaps, not link crawling.**
    News sites publish sitemap indexes listing every article URL, usually
    partitioned by month. Walking those is dramatically cheaper than
    breadth-first crawling: no wasted fetches on navigation pages, no
    politeness budget spent discovering what a sitemap would have told us for
    free, and archives going back years become reachable directly.

**robots.txt is honoured per user-agent.**
    ``urllib.robotparser`` is checked for our own agent string before every
    fetch, and the site's declared ``Crawl-delay`` overrides our default rate
    when it is slower. Lecture 3 slide 25 is blunt that crawled text is not
    necessarily redistributable; we ship the scraper and the cleaned
    derivative, never the raw HTML, and record the licence position per domain
    in sources.yaml.

**Resume is per-URL, not per-run.**
    Every fetched URL is appended to a ``seen`` file as it completes. Restart
    picks up exactly where it stopped, so an interrupted three-day scrape
    loses seconds of work rather than hours.

**A cheap script gate runs before storage.**
    Full language identification happens later, but storing English pages from
    an Assamese domain wastes disk and slows every later stage, so pages whose
    target-script ratio is hopeless are dropped at fetch time. The threshold is
    deliberately loose -- this is a storage optimisation, not the language
    filter.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import threading
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urljoin, urlparse

import httpx

from common.io import Doc, ShardWriter

__all__ = ["ScrapeConfig", "RobotsPolicy", "SiteScraper", "script_ratio"]

USER_AGENT = (
    "CL3410-coursework/1.0 (IIIT Hyderabad student project; "
    "monolingual corpus collection; contact via course staff)"
)

# Unicode blocks used by the script gate.
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
}

_SITEMAP_LOC_RE = re.compile(rb"<loc>\s*([^<\s]+)\s*</loc>", re.I)
_IS_SITEMAP_INDEX_RE = re.compile(rb"<sitemapindex", re.I)


# Sitemap names that carry article text, versus ones that list landing pages.
# Pratidin Time's category-sitemap.xml yields section fronts with no prose;
# its webcontent-sitemap.xml is an index of 2,930 date-partitioned files that
# hold the actual articles.
_GOOD_SITEMAP_HINTS = ("webcontent", "post", "article", "news", "sitemap_2", "story")
_POOR_SITEMAP_HINTS = ("category", "tag", "author", "page-sitemap", "webstory", "video")


_SITEMAP_DATE_RE = re.compile(r"(20\d{2})[-_/]?(\d{2})?[-_/]?(\d{2})?")


def _sitemap_priority(url: str) -> tuple[int, int, str]:
    """Sort key: article sitemaps first, then newest first, then by name.

    The date component matters more than it looks. News sites partition their
    sitemap index by day, and a plain alphabetical sort walks the archive
    oldest-first -- straight into years of sparse early sitemaps. Measured on
    asomiyapratidin: recent days carry ~55 articles each while the earliest
    carry almost none, and an alphabetical walk spent thirty minutes visiting
    ~2,500 near-empty sitemaps to yield 39 articles. Newest-first puts the
    dense partitions in front, which matters because the crawl is always
    stopped by a time budget rather than by exhausting the site.
    """
    low = url.lower()
    m = _SITEMAP_DATE_RE.search(low)
    if m:
        year, month, day = m.group(1), m.group(2) or "00", m.group(3) or "00"
        recency = -int(f"{year}{month}{day}")  # negative => newest first
    else:
        recency = 0
    if any(h in low for h in _POOR_SITEMAP_HINTS):
        return (2, recency, low)
    if any(h in low for h in _GOOD_SITEMAP_HINTS):
        return (0, recency, low)
    return (1, recency, low)


def script_ratio(text: str, script: str) -> float:
    """Fraction of letters that belong to the target Unicode block.

    Digits, punctuation and whitespace are ignored, so a page of Assamese
    prose containing numbers is not penalised for them.
    """
    lo, hi = SCRIPT_RANGES[script]
    total = in_script = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            if lo <= ord(ch) <= hi:
                in_script += 1
    return in_script / total if total else 0.0


@dataclass
class ScrapeConfig:
    """Politeness and yield settings.

    Attributes:
        requests_per_second: Our own cap. A site's ``Crawl-delay`` wins if it
            is slower.
        timeout: Per-request timeout in seconds.
        min_chars: Shortest extracted article worth storing.
        min_script_ratio: Storage-time script gate; deliberately loose.
        max_urls_per_site: Ceiling on sitemap URLs collected per domain.
        max_seconds_per_site: Wall-clock budget per domain per run.
        retries: Attempts per URL before giving up.
        shard_bytes: Uncompressed JSON per shard. Far smaller than the 128 MB
            default used for dump extraction: a shard is only durable once it
            commits, and this job runs for many hours, so the window of work
            at risk has to stay small. 8 MB is roughly 20 minutes of crawling.
    """

    requests_per_second: float = 2.0
    timeout: float = 20.0
    min_chars: int = 400
    min_script_ratio: float = 0.30
    max_urls_per_site: int = 400_000
    max_seconds_per_site: float = 86_400.0
    retries: int = 2
    shard_bytes: int = 8 * 1024 * 1024


class RobotsPolicy:
    """robots.txt fetching, caching and enforcement for one domain."""

    def __init__(self, base_url: str, user_agent: str = USER_AGENT) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.parser = urllib.robotparser.RobotFileParser()
        self.sitemaps: list[str] = []
        self.crawl_delay: float | None = None
        self._load()

    def _load(self) -> None:
        url = f"{self.base_url}/robots.txt"
        try:
            r = httpx.get(url, headers={"User-Agent": self.user_agent}, timeout=20.0,
                          follow_redirects=True)
            body = r.text if r.status_code == 200 else ""
        except Exception:
            body = ""
        self.parser.parse(body.splitlines())
        self.sitemaps = re.findall(r"(?im)^\s*Sitemap:\s*(\S+)", body)
        try:
            delay = self.parser.crawl_delay(self.user_agent)
            self.crawl_delay = float(delay) if delay else None
        except Exception:
            self.crawl_delay = None

    def allowed(self, url: str) -> bool:
        """True if robots.txt permits our agent to fetch this URL."""
        try:
            return self.parser.can_fetch(self.user_agent, url)
        except Exception:
            return True  # unparseable robots: default to the site's own links


class SiteScraper:
    """Scrape one domain into shards, resumably.

    Example::

        s = SiteScraper("https://time8.in", out_dir="data/assamese/raw/news",
                        source="time8", doc_prefix="as_news_time8",
                        script="bengali")
        s.run()
    """

    def __init__(
        self,
        base_url: str,
        out_dir: Path | str,
        source: str,
        doc_prefix: str,
        script: str,
        config: ScrapeConfig | None = None,
        licence: str = "unknown-see-sources.yaml",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.domain = urlparse(self.base_url).netloc
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.doc_prefix = doc_prefix
        self.script = script
        self.cfg = config or ScrapeConfig()
        self.licence = licence

        self.robots = RobotsPolicy(self.base_url)
        self.delay = max(
            1.0 / self.cfg.requests_per_second,
            self.robots.crawl_delay or 0.0,
        )
        self.seen_path = self.out_dir / f"{doc_prefix}.seen"
        self.seen: set[str] = self._load_seen()
        self._pending_seen: list[str] = []
        self.counts = {
            "urls_from_sitemap": 0,
            "already_seen": 0,
            "robots_blocked": 0,
            "fetch_failed": 0,
            "no_extract": 0,
            "too_short": 0,
            "wrong_script": 0,
            "emitted": 0,
            "words": 0,
        }
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=self.cfg.timeout,
            follow_redirects=True,
        )

    # -- resume ------------------------------------------------------------

    def _load_seen(self) -> set[str]:
        if not self.seen_path.exists():
            return set()
        with self.seen_path.open("r", encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}

    def _mark_seen(self, url: str) -> None:
        """Record a URL as done. Only safe when no document depends on it.

        Used for URLs that produced nothing worth keeping -- blocked, failed,
        too short, wrong script. Re-fetching those on resume would be pure
        waste, and there is no data to lose.
        """
        self.seen.add(url)
        with self.seen_path.open("a", encoding="utf-8") as fh:
            fh.write(url + "\n")

    def _defer_seen(self, url: str) -> None:
        """Hold a URL back until the shard holding its document is durable.

        A shard is only committed -- renamed into place and marked done -- when
        it fills or the writer closes. Marking a URL seen before that point
        means a crash loses the text *and* skips the URL on resume, so the
        document is gone for good. Deferring costs at most a re-fetch of the
        current shard's URLs.
        """
        self._pending_seen.append(url)

    def _flush_pending_seen(self) -> None:
        """Commit deferred URLs once their shard is on disk."""
        if not self._pending_seen:
            return
        with self.seen_path.open("a", encoding="utf-8") as fh:
            for url in self._pending_seen:
                self.seen.add(url)
                fh.write(url + "\n")
        self._pending_seen.clear()

    # -- sitemaps ----------------------------------------------------------

    def _fetch_bytes(self, url: str) -> bytes | None:
        for attempt in range(self.cfg.retries + 1):
            try:
                r = self._client.get(url)
                if r.status_code == 200:
                    body = r.content
                    if url.endswith(".gz") or body[:2] == b"\x1f\x8b":
                        try:
                            body = gzip.decompress(body)
                        except Exception:
                            pass
                    return body
                if r.status_code in (429, 503):
                    time.sleep(5 * (attempt + 1))
                    continue
                return None
            except Exception:
                time.sleep(2 * (attempt + 1))
        return None

    def sitemap_urls(self) -> Iterator[str]:
        """Walk sitemap indexes and yield article URLs.

        Sitemap indexes nest, so this is a queue rather than a loop: an index
        yields sub-sitemaps, which may themselves be indexes.
        """
        # Sites declare sitemaps in robots.txt that sometimes 404 (time8.in
        # advertises /sitemap_index.xml and serves an HTML error page), so the
        # conventional locations are appended as fallbacks rather than trusted
        # only when robots.txt is silent.
        queue: list[str] = list(self.robots.sitemaps)
        queue += [
            f"{self.base_url}{path}"
            for path in (
                "/sitemap.xml",
                "/sitemap_index.xml",
                "/wp-sitemap.xml",
                "/sitemap-index.xml",
                "/post-sitemap.xml",
                "/news-sitemap.xml",
            )
        ]
        visited: set[str] = set()
        produced = 0
        queue.sort(key=_sitemap_priority)

        while queue and produced < self.cfg.max_urls_per_site:
            sm = queue.pop(0)
            if sm in visited:
                continue
            visited.add(sm)
            time.sleep(self.delay)
            body = self._fetch_bytes(sm)
            if not body:
                continue
            locs = [m.decode("utf-8", "replace") for m in _SITEMAP_LOC_RE.findall(body)]
            if _IS_SITEMAP_INDEX_RE.search(body):
                # Depth-first into article sitemaps. Without the sort, a
                # site's category-sitemap is popped first and the URL budget
                # is spent on landing pages that carry no article text.
                queue = sorted(locs, key=_sitemap_priority) + queue
                continue
            for loc in locs:
                if produced >= self.cfg.max_urls_per_site:
                    break
                produced += 1
                self.counts["urls_from_sitemap"] += 1
                yield loc

    # -- scraping ----------------------------------------------------------

    def _extract(self, html: str, url: str) -> str | None:
        import trafilatura

        return trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
            no_fallback=False,
        )

    def run(self, urls: Iterable[str] | None = None) -> dict:
        """Fetch, extract and store. Safe to interrupt and re-run."""
        started = time.time()
        source_urls = urls if urls is not None else self.sitemap_urls()

        with ShardWriter(
            self.out_dir, prefix=self.doc_prefix, target_bytes=self.cfg.shard_bytes
        ) as writer:
            shards_done = 0
            for url in source_urls:
                if time.time() - started > self.cfg.max_seconds_per_site:
                    break
                if url in self.seen or url in self._pending_seen:
                    self.counts["already_seen"] += 1
                    continue
                if not self.robots.allowed(url):
                    self.counts["robots_blocked"] += 1
                    self._mark_seen(url)
                    continue

                time.sleep(self.delay)
                body = self._fetch_bytes(url)
                if not body:
                    self._mark_seen(url)
                    self.counts["fetch_failed"] += 1
                    continue

                try:
                    text = self._extract(body.decode("utf-8", "replace"), url)
                except Exception:
                    text = None
                if not text:
                    self.counts["no_extract"] += 1
                    self._mark_seen(url)
                    continue
                if len(text) < self.cfg.min_chars:
                    self.counts["too_short"] += 1
                    self._mark_seen(url)
                    continue
                if script_ratio(text, self.script) < self.cfg.min_script_ratio:
                    self.counts["wrong_script"] += 1
                    self._mark_seen(url)
                    continue

                doc = Doc(
                    # Content-derived, not counter-derived: a per-run counter
                    # resets to 0 on every restart, and this collector gets
                    # restarted often (session teardown, Task Scheduler
                    # handoff). Two restarts each starting their own count at
                    # 0000000 collide -- measured on real output: 1,641
                    # duplicate ids across 19,600 Assamese news documents,
                    # which later broke MinHash/LSH's insert (it rejects a
                    # duplicate key outright) and silently corrupted the
                    # split manifests before that. A URL hash is stable
                    # across any number of restarts because it depends only
                    # on what was fetched, never on how many times the
                    # process has been launched.
                    id=f"{self.doc_prefix}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]}",
                    text=text,
                    source=self.source,
                    collection_method="manual",
                    url=url,
                    retrieved=time.strftime("%Y-%m-%d"),
                    licence=self.licence,
                    meta={"domain": self.domain},
                )
                writer.write(doc)
                self._defer_seen(url)
                self.counts["emitted"] += 1
                self.counts["words"] += doc.n_words()

                # A shard just committed, so its URLs are now safe to record.
                if len(writer.shards_written) > shards_done:
                    shards_done = len(writer.shards_written)
                    self._flush_pending_seen()

        # The writer closed cleanly, committing the final shard.
        self._flush_pending_seen()
        self._client.close()
        return self.counts
