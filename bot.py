# Requirements: pip install atproto feedparser requests beautifulsoup4 python-dateutil
import os, re, json, requests, feedparser
from datetime import date, datetime, timezone, timedelta
from bs4 import BeautifulSoup
from atproto import Client
from atproto.rich_text import RichText  # pour liens cliquables via facets

# =======================
# Chargement configuration
# =======================
CFG = json.load(open("config.json", "r", encoding="utf-8"))
BELIEFS = CFG["belief_weights"]
THRESH = CFG["score_threshold"]
MAX_PER_FEED = CFG.get("max_items_per_feed", 12)
SOURCES = CFG["sources"]
SOCIAL = CFG.get("social_rules", {})
POSTING = CFG.get("posting", {})

# ===============
# Hugging Face API
# ===============
HF_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
TRANS_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-fr"
HF_HEADERS = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"}

# =============
# Utilitaires
# =============
def clean_html(s: str) -> str:
    """Supprime les balises HTML et normalise l'espace."""
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)

def strip_boilerplate(txt: str) -> str:
    """Retire les signatures de flux ('The post ... appeared first on ...') et les trackers."""
    if not txt:
        return ""
    txt = re.sub(r"The post .*? appeared first on .*?$", "", txt, flags=re.IGNORECASE | re.DOTALL)
    txt = re.sub(r"(\?|&)(utm_[^=]+|fbclid)=[^&\s]+", "", txt)
    return txt.strip()

def score(text: str) -> int:
    """Score par mots-clés selon les convictions."""
    t = (text or "").lower()
    return sum(w for k, w in BELIEFS.items() if k in t)

def pick_best_item():
    """Parcourt les flux et retourne le meilleur item au-dessus du seuil."""
    items = []
    for url in SOURCES:
        feed = feedparser.parse(url)
        for e in feed.entries[:MAX_PER_FEED]:
            title = e.title or ""
            summary_raw = clean_html(getattr(e, "summary", ""))
            summary = strip_boilerplate(summary_raw)
            s = score(f"{title} {summary}")
            link = getattr(e, "link", "") or ""
            items.append((s, title, summary, link))
    if not items:
        return None
    items.sort(key=lambda x: x[0], reverse=True)
    best = items[0]
    return best if best[0] >= THRESH else None

def extractive_fallback(text: str, k=5):
    """Fallback simple: sélectionne 4–5 phrases informatives si le modèle IA répond mal."""
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

# ---- Détection FR / EN + traduction secours
def is_likely_english(txt: str) -> bool:
    if not txt:
        return False
    en_markers = re.findall(r"\b(the|and|of|to|in|for|with|on|from|by|is|are|this|that|it|as)\b", txt.lower())
    return len(en_markers) >= 3

def translate_to_fr(text: str) -> str:
    try:
        r = requests.post(TRANS_URL, headers=HF_HEADERS, json={"inputs": text[:4000]}, timeout=45)
        data = r.json()
        if isinstance(data, list) and data and "translation_text" in data[0]:
            return data[0]["translation_text"].strip()
    except Exception:
        pass
    return text  # fallback

def summarize_5_lines(title: str, summary: str) -> str:
    """
    Produit 4–5 puces en FR :
    - Fait marquant
    - Pourquoi c’est important
    - Implications (marché/technos)
    - Risque / point de vigilance
    - Prochaine étape / à surveiller
    """
    base = f"{title}. {summary}"
    prompt = (
        "Tu écris UNIQUEMENT en FRANÇAIS. "
        "Produis 4 à 5 puces courtes (≤ ~25 mots), ton posé, analytique, optimisme mesuré, "
        "orienté IA / spatial / innovation. Varie légèrement les tournures à chaque réponse.\n"
        "Structure :\n"
        "- Fait marquant\n- Pourquoi c’est important\n- Implications (marché/technos)\n"
        "- Risque / point de vigilance\n- Prochaine étape / à surveiller\n\n"
        f"{base[:4000]}"
    )
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 220, "temperature": 0.7},
        "options": {"wait_for_model": True}
    }
    out = ""
    try:
        r = requests.post(HF_URL, headers=HF_HEADERS, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        out = data[0]["generated_text"] if isinstance(data, list) and data and "generated_text" in data[0] else ""
    except Exception:
        out = ""

    out = re.sub(r"\n{3,}", "\n\n", (out or "").strip())
    lines = [l.strip(" -*•\t") for l in out.split("\n") if l.strip()]
    if len(lines) < 4:
        lines = extractive_fallback(summary, k=5)
    text = "\n".join(f"- {l}" for l in lines[:5])

    # Garde-fou: si ça ressemble à de l’anglais, traduis en FR
    if is_likely_english(text):
        text = translate_to_fr(text)
    return text

# =============
# Bluesky client
# =============
def bsky_client() -> Client:
    handle = os.environ.get("BSKY_HANDLE")
    pwd = os.environ.get("BSKY_PASSWORD")
    if not handle or not pwd:
        raise SystemExit("Missing BSKY_HANDLE or BSKY_PASSWORD")
    c = Client()
    c.login(handle, pwd)
    return c

def post_bsky(client: Client, text: str):
    """Poste sur Bluesky en respectant ~300 caractères et en rendant les liens cliquables (facets)."""
    if os.environ.get("DRY_RUN") == "1":
        print("[DRY_RUN] Aurait posté:\n", (text or "")[:295])
        return
    t = (text or "").strip()
    if len(t) > 295:
        t = t[:292].rstrip() + "…"
    rt = RichText(t)
    try:
        rt.detect_facets(client)  # URLs/mentions -> facets cliquables
    except Exception:
        pass
    client.send_post(text=rt.text, facets=rt.facets)

# ==========================
# Actions sociales (optionnel)
# ==========================
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

    for kw in keywords:
        if liked >= max_likes:
            break
        try:
            res = client.app.bsky.feed.search_posts({"q": kw, "limit": 10})
        except Exception:
            continue
        posts = res.get("posts", []) if isinstance(res, dict) else []
        for post in posts:
            if liked >= max_likes:
                break
            try:
                author = post["author"]["handle"]
                like_count = post.get("likeCount", 0) or 0
                uri, cid = post.get("uri"), post.get("cid")
                if not uri or not cid:
                    continue
                if (author in whitelist) or (like_count >= min_like):
                    try:
                        if os.environ.get("DRY_RUN") == "1":
                            print(f"[DRY_RUN] Aurait liké: {uri}")
                        else:
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
        if followed >= max_follows:
            break
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
                if os.environ.get("DRY_RUN") == "1":
                    print(f"[DRY_RUN] Aurait suivi: {handle}")
                else:
                    client.follow(profile["did"])
                followed += 1
        except Exception:
            pass
    return followed

# =====
# Main
# =====
def main():
    # Secrets requis
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("Missing HF_TOKEN")

    best = pick_best_item()
    if not best:
        print("Aucun item au-dessus du seuil, fin du run.")
        return

    _, title, summary, link = best
    even_day = (date.today().toordinal() % 2 == 0)
    include_link = bool(POSTING.get("include_link_in_main", True))

    c = bsky_client()

    if even_day:
        # Jour "éclairage" (IA en FR)
        wakeup = summarize_5_lines(title, summary)
        text = f"{wakeup}\n{link}" if include_link and link else wakeup
        post_bsky(c, text)
    else:
        # Jour "repost/teaser" (FR)
        teaser = f"Lecture conseillée 👇 {title}"
        text = f"{teaser}\n{link}" if include_link and link else teaser
        post_bsky(c, text)

    # Actions sociales légères (facultatif)
    try:
        liked = search_and_like(c, CFG)
        followed = maybe_follow_accounts(c, CFG)
        print(f"Social actions: liked={liked}, followed={followed}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
