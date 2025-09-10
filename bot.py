# Requirements: pip install atproto feedparser requests beautifulsoup4 python-dateutil
import os, re, json, requests, feedparser
from datetime import date, datetime, timezone, timedelta
from bs4 import BeautifulSoup
from atproto import Client

# ---------- Chargement config ----------
CFG = json.load(open("config.json", "r", encoding="utf-8"))
BELIEFS = CFG["belief_weights"]
THRESH = CFG["score_threshold"]
MAX_PER_FEED = CFG.get("max_items_per_feed", 12)
SOURCES = CFG["sources"]
SOCIAL = CFG.get("social_rules", {})

# ---------- Hugging Face ----------
HF_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HF_HEADERS = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN','')}"}

# ---------- Utilitaires ----------
def clean_html(s: str) -> str:
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)

def score(text: str) -> int:
    t = text.lower()
    return sum(w for k, w in BELIEFS.items() if k in t)

def pick_best_item():
    items = []
    for url in SOURCES:
        feed = feedparser.parse(url)
        for e in feed.entries[:MAX_PER_FEED]:
            title = e.title or ""
            summary = clean_html(getattr(e, "summary", ""))
            s = score(f"{title} {summary}")
            items.append((s, title, summary, e.link))
    if not items:
        return None
    items.sort(key=lambda x: x[0], reverse=True)
    best = items[0]
    return best if best[0] >= THRESH else None

def summarize_5_lines(title: str, summary: str) -> str:
    # Texte 4–5 lignes, ton “éclairage” (réfléchi, optimisme mesuré, IA/espace)
    base = f"{title}. {summary}"
    prompt = (
        "Lis et écris un court éclairage financier en 4 à 5 lignes. "
        "Style: réfléchi, concret, optimisme mesuré, orienté IA/spatial/innovation. "
        "Chaque ligne commence par '-'. Évite le jargon inutile.\n\n" + base[:4000]
    )
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 180},
        "options": {"wait_for_model": True}
    }
    try:
        r = requests.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        # HF peut renvoyer différents formats; on vise generated_text
        if isinstance(data, list) and data and "generated_text" in data[0]:
            out = data[0]["generated_text"]
        else:
            out = ""
    except Exception:
        out = ""
    out = re.sub(r"\n{3,}", "\n\n", (out or "").strip())
    lines = [l.strip(" -") for l in out.split("\n") if l.strip()]
    if len(lines) < 4:  # fallback extractif si HF renvoie peu
        lines = extractive_fallback(summary, k=5)
    return "\n".join(f"- {l}" for l in lines[:5])

def extractive_fallback(text: str, k=5):
    sents = re.split(r"(?<=[.!?])\s+", text or "")
    scored = []
    for s in sents:
        sl = s.strip()
        if 40 <= len(sl) <= 240:
            scored.append((score(sl) + min(len(sl)//60, 3), sl))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [s for _, s in scored[:k]]
    if not out:
        out = [ (text or "")[:200] or "Point clé à surveiller." ]
    return out

# ---------- Bluesky ----------
def bsky_client() -> Client:
    handle = os.environ.get("BSKY_HANDLE")
    pwd = os.environ.get("BSKY_PASSWORD")
    if not handle or not pwd:
        raise SystemExit("Missing BSKY_HANDLE or BSKY_PASSWORD")
    c = Client()
    c.login(handle, pwd)
    return c

def post_bsky(client: Client, text: str):
    # Bluesky recommande ~300 chars; on tronque proprement
    t = text.strip()
    if len(t) > 295:
        t = t[:292].rstrip() + "…"
    client.send_post(text=t)

# (Optionnel) Quelques actions sociales light
def search_and_like(client: Client, cfg: dict) -> int:
    rules = cfg.get("social_rules", {})
    if not rules or not rules.get("enable_likes", False):
        return 0
    max_likes = int(rules.get("max_likes_per_run", 0))
    if max_likes <= 0:
        return 0

    keywords = rules.get("keywords_for_social", [])
    whitelist = set(rules.get("accounts_whitelist", []))
    min_like = int(rules.get("min_engagement_like", 5))
    liked = 0

    # API wrapper names can vary by version; we try/except to stay robust
    for kw in keywords:
        if liked >= max_likes: break
        try:
            res = client.app.bsky.feed.search_posts({"q": kw, "limit": 10})
        except Exception:
            continue
        posts = res.get("posts", []) if isinstance(res, dict) else []
        for post in posts:
            if liked >= max_likes: break
            try:
                author = post["author"]["handle"]
                like_count = post.get("likeCount", 0) or 0
                uri, cid = post.get("uri"), post.get("cid")
                if not uri or not cid:
                    continue
                if (author in whitelist) or (like_count >= min_like):
                    try:
                        client.like(uri, cid)
                        liked += 1
                    except Exception:
                        pass
            except Exception:
                pass
    return liked

def maybe_follow_accounts(client: Client, cfg: dict) -> int:
    rules = cfg.get("social_rules", {})
    if not rules or not rules.get("enable_follows", False):
        return 0
    max_follows = int(rules.get("max_follows_per_run", 0))
    if max_follows <= 0:
        return 0

    whitelist = rules.get("accounts_whitelist", [])
    hours = int(rules.get("follow_only_if_posted_recently_hours", 72))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    followed = 0

    for handle in whitelist:
        if followed >= max_follows: break
        try:
            profile = client.get_profile(handle)
            if profile.get("viewer", {}).get("following"):
                continue  # déjà suivi

            feed = client.get_author_feed(handle, {"limit": 5})
            has_recent = False
            for item in feed.get("feed", []):
                ts = item.get("post", {}).get("record", {}).get("createdAt")
                if ts:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt >= cutoff:
                        has_recent = True
                        break
            if has_recent:
                client.follow(profile["did"])
                followed += 1
        except Exception:
            pass
    return followed

# ---------- Main ----------
def main():
    # Garde-fous secrets HF
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("Missing HF_TOKEN")

    best = pick_best_item()
    if not best:
        # Rien d'assez pertinent aujourd'hui
        return

    score_val, title, summary, link = best
    even_day = (date.today().toordinal() % 2 == 0)

    c = bsky_client()

    if even_day:
        # Jour "éclairage"
        wakeup = summarize_5_lines(title, summary)
        post_bsky(c, f"{wakeup}\n{link}")
    else:
        # Jour "repost/teaser"
        teaser = f"Éclairage utile 👇 {title}"
        post_bsky(c, f"{teaser}\n{link}")

    # Actions sociales légères (facultatif, sûr)
    try:
        liked = search_and_like(c, CFG)
        followed = maybe_follow_accounts(c, CFG)
        # print(f"Social actions: liked={liked}, followed={followed}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
