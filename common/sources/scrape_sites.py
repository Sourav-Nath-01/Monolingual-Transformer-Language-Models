"""Site inventory and runner for the news scrape.

Every domain here was probed before inclusion: robots.txt fetched, sitemaps
located, and a sample article extracted and script-checked. Sites that
advertise sitemaps which 404 (time8.in), that publish in English under an
Assamese-sounding name (pratidintime.com is Pratidin Time's English edition),
or whose sitemaps list two URLs (niyomiyabarta.org) are recorded here as
rejected rather than silently omitted -- the report needs to show which
sources were considered, not just which were used.

Run with::

    python -m common.sources.scrape_sites --lang assamese
    python -m common.sources.scrape_sites --lang hindi --hours 12
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Site", "SITES", "REJECTED"]


@dataclass(frozen=True)
class Site:
    """One scrape target.

    Attributes:
        base_url: Scheme and host, no trailing slash.
        key: Short identifier; becomes the source key and id prefix.
        script: Which Unicode block its text should be in.
        licence: Licence position, carried into sources.yaml. Almost always
            all-rights-reserved for news, which is why we ship the scraper and
            the cleaned derivative rather than raw HTML.
        note: Why it is here.
    """

    base_url: str
    key: str
    script: str
    licence: str = "all-rights-reserved (news); derivative text only"
    note: str = ""


SITES: dict[str, list[Site]] = {
    "assamese": [
        Site(
            "https://www.asomiyapratidin.in",
            "asomiyapratidin",
            "bengali",
            note="Assamese daily. webcontent-sitemap.xml indexes 2,930 "
            "date-partitioned sitemaps; sampled articles 96-100% Assamese "
            "script. The single largest scrapable Assamese source found.",
        ),
        Site(
            "https://www.dainikjanambhumi.co.in",
            "dainikjanambhumi",
            "bengali",
            note="Assamese daily. No sitemap declared; conventional paths "
            "are tried as fallbacks.",
        ),
        Site(
            "https://niyomiyabarta.org",
            "niyomiyabarta",
            "bengali",
            note="Assamese daily. Sitemap lists only a handful of URLs, so "
            "yield is expected to be negligible; included because Assamese "
            "text is scarce enough that even small sources are worth the "
            "few minutes of crawl budget.",
        ),
    ],
    "hindi": [
        Site(
            "https://www.livehindustan.com",
            "livehindustan",
            "devanagari",
            note="8 sitemaps declared, no blanket disallow.",
        ),
        Site(
            "https://www.prabhatkhabar.com",
            "prabhatkhabar",
            "devanagari",
            note="4 sitemaps declared, no blanket disallow.",
        ),
        Site(
            "https://www.jagran.com",
            "jagran",
            "devanagari",
            note="3 sitemaps declared, no blanket disallow.",
        ),
        Site(
            "https://www.patrika.com",
            "patrika",
            "devanagari",
            note="3 sitemaps declared, no blanket disallow.",
        ),
        Site(
            "https://www.navjivanindia.com",
            "navjivanindia",
            "devanagari",
            note="5 sitemaps declared, no blanket disallow.",
        ),
    ],
}

# Probed and excluded. Kept so the report can state what was tried.
REJECTED = {
    "time8.in": "robots.txt advertises /sitemap_index.xml; it and all five "
    "conventional paths return 404 as HTML. Would need link crawling.",
    "pratidintime.com": "English-language edition. Sampled articles scored "
    "0.00 on the Bengali-script gate. The Assamese sister title is "
    "asomiyapratidin.in, which is included above.",
    "amarasom.com": "unreachable during probing",
    "asomiyakhabar.com": "unreachable during probing",
    "sadinsamachar.com": "unreachable during probing",
    "bhaskar.com": "28 blanket Disallow rules; too much of the site is "
    "off-limits to be worth the crawl budget.",
    "amarujala.com": "16 blanket Disallow rules; same reasoning.",
}


def main(argv: list[str] | None = None) -> int:
    """Scrape every site for one language. Resumable; safe to re-run."""
    import argparse
    import json
    from pathlib import Path

    from common.io import enable_utf8_console
    from common.sources.scrape import ScrapeConfig, SiteScraper

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=sorted(SITES))
    p.add_argument("--hours", type=float, default=24.0, help="budget per site")
    p.add_argument("--rps", type=float, default=3.0, help="requests per second")
    p.add_argument("--max-urls", type=int, default=400_000)
    p.add_argument("--out", default="", help="defaults to data/<lang>/raw/news")
    args = p.parse_args(argv)

    out = Path(args.out or f"data/{args.lang}/raw/news")
    cfg = ScrapeConfig(
        requests_per_second=args.rps,
        max_urls_per_site=args.max_urls,
        max_seconds_per_site=args.hours * 3600,
    )

    totals = {}
    for site in SITES[args.lang]:
        print(f"\n=== {site.key}  ({site.base_url})", flush=True)
        scraper = SiteScraper(
            site.base_url,
            out_dir=out,
            source=site.key,
            doc_prefix=f"{args.lang[:2]}_news_{site.key}",
            script=site.script,
            config=cfg,
            licence=site.licence,
        )
        counts = scraper.run()
        totals[site.key] = counts
        print(f"    {json.dumps(counts)}", flush=True)

    stats_path = Path(args.lang) / "stats" / "collect_news.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(totals)
    merged["_total_words"] = sum(c["words"] for c in totals.values())
    merged["_total_docs"] = sum(c["emitted"] for c in totals.values())
    stats_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ntotal: {merged['_total_docs']:,} docs, {merged['_total_words']:,} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
