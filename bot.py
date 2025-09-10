# Requirements: pip install atproto feedparser requests beautifulsoup4 python-dateutil
import os, re, json, requests, feedparser
from datetime import date, datetime, timezone, timedelta
from bs4 import BeautifulSoup
from atproto import Client, models
from difflib import SequenceMatcher

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
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)

def strip_boilerplate(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"The post .*? appeared first on .*?$", "", txt, flags=re.IGNORECASE | re.DOTALL)
    txt = re.sub(r"(\?|&)(utm_[^=]+|fbclid)=[^&\s]+", "", txt)
    return txt.strip()

def score(text: str) -> int:
    t = (text or "").lower()
    return sum(w for k, w in BELIEFS.items() if k in t)

def pick_best_item():
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

def fetch_article_text(url: str) -> str:
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(ps)
        return text[:8000]
    except Exception:
        return ""

def build_context(title: str, summary: str, link: str) -> tuple[str, str]:
    article = fetch_article_text(link)
    base = f"{title}. {summary}".strip()
    ctx = (base + ("\n\n" + article if article else "")).strip()
    ctx = ctx[:4000] if ctx else base[:1000]
    return ctx, article

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
        out = [(text or "")[:200] or "Point clé à surveiller."]
    return out

# ---- Détection FR/EN + traduction secours
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
    return text

# ---- Anti-copie & dédup ----
def too_similar(a: str, b: str, thresh=0.82):
    na = re.sub(r"\W+", " ", (a or "").lower()).strip()
    nb = re.sub(r"\W+", " ", (b or "").lower()).strip()
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= thresh

def filter_copies(lines: list[str], source_head: str) -> list[str]:
    kept = []
    for l in lines:
        if too_similar(l, source_head):
            continue
        kept.append(l)
    return kept

def dedupe_lines(lines: list[str]) -> list[str]:
    out, seen = [], set()
    for l in lines:
        norm = re.sub(r"[\W_]+", " ", l.lower()).strip()
        norm = re.sub(r"\s+", " ", norm)
        if norm in seen or not norm:
            continue
        seen.add(norm)
        out.append(l)
    return out

# -------- Extraction de signaux (acteurs / chiffres / dates) ----------
MONTHS = r"(janv\.?|févr\.?|mars|avr\.?|mai|juin|juil\.?|août|sept\.?|oct\.?|nov\.?|déc\.?|january|february|march|april|may|june|july|august|september|october|november|december)"
YEAR = r"(20\d{2}|19\d{2})"
PCT = r"\b\d{1,3}(?:[.,]\d+)?\s?%\b"
MONEY = r"(?:\$|€)\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s?(?:bn|billion|milliard|milliards|m|million|millions|k)?"
PLAIN_NUM = r"\b\d{2,4}(?:[.,]\d+)?\b"

def extract_actors(text: str, max_items=6):
    cand = re.findall(r"\b([A-Z][a-zA-Z0-9&\-]+(?:\s+[A-Z][a-zA-Z0-9&\-]+){0,3})\b", text or "")
    bad = {"The","This","That","And","For","With","From","By","In","On","Of","At","An","A","To","As","It","Its","We","He","She"}
    out, seen = [], set()
    for c in cand:
        if any(w in bad for w in c.split()):
            continue
        key = c.strip()
        if len(key) < 3:
            continue
        knorm = key.lower()
        if knorm in seen:
            continue
        seen.add(knorm)
        out.append(key)
        if len(out) >= max_items:
            break
    return out

def extract_signals(text: str) -> dict:
    if not text:
        return {"actors": [], "numbers": [], "dates": []}
    actors = extract_actors(text)
    nums = set(re.findall(MONEY, text, flags=re.IGNORECASE)) | set(re.findall(PCT, text))
    if len(nums) < 3:
        nums |= set(re.findall(PLAIN_NUM, text))
    dates = set(re.findall(MONTHS + r"\s+" + YEAR, text, flags=re.IGNORECASE))
    years = set(re.findall(r"\b" + YEAR + r"\b", text))
    numbers = [n.strip() for n in list(nums) if n.strip()][:6]
    dates_fmt = []
    for d in list(dates):
        if isinstance(d, tuple):
            dates_fmt.append(" ".join(d).strip())
        else:
            dates_fmt.append(str(d).strip())
    dates_fmt += [y for y in list(years) if y not in dates_fmt]
    dates_fmt = dates_fmt[:6]
    return {"actors": actors[:6], "numbers": numbers, "dates": dates_fmt}

def make_fact_hints(signals: dict) -> str:
    if not signals:
        return ""
    parts = []
    if signals.get("actors"):
        parts.append("Acteurs: " + ", ".join(signals["actors"][:4]))
    if signals.get("numbers"):
        parts.append("Chiffres: " + ", ".join(signals["numbers"][:4]))
    if signals.get("dates"):
        parts.append("Échéances: " + ", ".join(signals["dates"][:4]))
    txt = " ; ".join(parts)
    return (txt[:300] + "…") if len(txt) > 300 else txt

# =================
# Résumé en français
# =================
def _gen_bullets(context: str, fact_hints: str, strict: bool) -> list[str]:
    header = (
        "Tu écris UNIQUEMENT en FRANÇAIS. Produis 4 à 5 puces courtes (≤ ~25 mots), "
        "ton posé, analytique, optimisme mesuré, orienté IA / spatial / innovation. "
        "Privilégie chiffres, acteurs, échéances si disponibles. Varie les tournures. "
        "N'énonce PAS deux fois la même idée. INTERDIT: reprendre mot à mot la phrase d'ouverture; reformule."
    )
    if strict:
        header += " Interdiction d'utiliser toute séquence de ≥6 mots consécutifs issue du texte source."

    prompt = (
        header + "\nStructure :\n- Fait marquant\n- Pourquoi c’est important\n- Implications (marché/technos)\n"
        "- Risque / point de vigilance\n- Prochaine étape / à surveiller\n\n"
        f"Faits détectés (à intégrer si pertinents): {fact_hints}\n\n"
        f"{context[:4000]}"
    )
    params = {
        "max_new_tokens": 260,
        "temperature": 0.8 if strict else 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.18 if strict else 1.12,
        "do_sample": True,
        "no_repeat_ngram_size": 5
    }
    try:
        r = requests.post(HF_URL, headers=HF_HEADERS,
                          json={"inputs": prompt, "parameters": params, "options": {"wait_for_model": True}},
                          timeout=90)
        r.raise_for_status()
        data = r.json()
        out = data[0]["generated_text"] if isinstance(data, list) and data and "generated_text" in data[0] else ""
    except Exception:
        out = ""
    raw = [l.strip(" -*•\t") for l in re.split(r"\n+", (out or "").strip()) if l.strip()]
    return raw

def summarize_5_lines(context: str, fact_hints: str) -> str:
    head = (context or "")[:400]  # zone où se trouve souvent la phrase d'ouverture
    # 1ère passe
    raw_lines = _gen_bullets(context, fact_hints, strict=False)
    lines = dedupe_lines(raw_lines)
    lines = filter_copies(lines, head)

    # Si trop proche/maigre, 2e passe plus stricte
    if len(lines) < 4 or any(too_similar(l, head) for l in lines):
        raw_lines2 = _gen_bullets(context, fact_hints, strict=True)
        lines2 = filter_copies(dedupe_lines(raw_lines2), head)
        if len(lines2) >= 4:
            lines = lines2

    if len(lines) < 4:
        lines = extractive_fallback(context, k=5)

    text = "\n".join(f"- {l}" for l in lines[:5])
    if is_likely_english(text):
        text = translate_to_fr(text)
    return text

# =============
# Bluesky client + anti-redite cross-jour
# =============
def bsky_client() -> Client:
    handle = os.environ.get("BSKY_HANDLE")
    pwd = os.environ.get("BSKY_PASSWORD")
    if not handle or not pwd:
        raise SystemExit("Missing BSKY_HANDLE or BSKY_PASSWORD")
    c = Client()
    c.login(handle, pwd)
    return c

def already_posted_recently(client: Client, link: str, self_handle: str, lookback: int = 25) -> bool:
    """Vérifie si ce lien a déjà été posté récemment (évite re-post jours suivants)."""
    try:
        feed = client.get_author_feed(self_handle, {"limit": lookback})
        for item in feed.get("feed", []):
            rec = item.get("post", {}).get("record", {})
            txt = (rec.get("text") or "").strip()
            # Regarde aussi dans les embeds des réponses "Source :" précédentes
            if link and link in txt:
                return True
            embed = item.get("post", {}).get("embed", {})
            ext = embed.get("external", {}) if isinstance(embed, dict) else {}
            if isinstance(ext, dict) and link and (ext.get("uri") == link):
                return True
    except Exception:
        pass
    return False

def post_bsky(client: Client, text: str, link: str | None = None, include_link: bool = False):
    if os.environ.get("DRY_RUN") == "1":
        preview = text if len(text) <= 295 else (text[:292].rstrip() + "…")
        print("[DRY_RUN] POST:", preview, "| LINK:", link if include_link else "(none)")
        return None
    t = (text or "").strip()
    if len(t) > 295:
        t = t[:292].rstrip() + "…"
    embed = None
    if include_link and link:
        try:
            embed = models.AppBskyEmbedExternal.Main(
                external=models.AppBskyEmbedExternal.External(
                    uri=link, title="Source", description=""
                )
            )
        except Exception as e:
            print("[WARN] embed externe KO:", e)
            embed = None
    return client.send_post(text=t, embed=embed)

def reply_with_link_card(client: Client, parent_post: dict, link: str):
    if not parent_post or not link:
        print("[INFO] Pas de parent ou pas de lien pour reply.")
        return
    try:
        embed = models.AppBskyEmbedExternal.Main(
            external=models.AppBskyEmbedExternal.External(uri=link, title="Source", description="")
        )
        reply_ref = {"root": {"uri": parent_post["uri"], "cid": parent_post["cid"]},
                     "parent": {"uri": parent_post["uri"], "cid": parent_post["cid"]}}
        client.send_post(text="Source :", embed=embed, reply_to=reply_ref)
        print("[OK] Reply Source posté.")
    except Exception as e:
        print("[ERROR] Reply Source KO:", e)

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
        if followed >= max_follows: break
        try:
            profile = client.get_profile(handle)
            if profile.get("viewer", {}).get("following"):
                continue
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
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("Missing HF_TOKEN")

    best = pick_best_item()
    if not best:
        print("Aucun item au-dessus du seuil, fin du run.")
        return

    _, title, summary, link = best
    even_day = (date.today().toordinal() % 2 == 0)
    include_link_main = bool(POSTING.get("include_link_in_main", False))
    include_link_reply = bool(POSTING.get("include_link_in_reply", True))

    c = bsky_client()

    # Anti-redite cross-jour: ne pas reposter le même lien récemment
    if link and already_posted_recently(c, link, os.environ.get("BSKY_HANDLE", ""), lookback=25):
        print("Lien déjà posté récemment — on saute ce run.")
        return

    if even_day:
        context, fulltext = build_context(title, summary, link)
        signals = extract_signals(fulltext or context)
        hints = make_fact_hints(signals)

        # Si article très pauvre ET aucun signal -> mieux vaut s'abstenir
        if (len((fulltext or "")) < 800) and (not signals["actors"] and not signals["numbers"]):
            print("Article pauvre (peu de matière). Sortie sans poster.")
            return

        wakeup = summarize_5_lines(context, hints)
        res = post_bsky(c, wakeup, link=link, include_link=include_link_main)
    else:
        teaser = f"Lecture conseillée 👇 {title}"
        res = post_bsky(c, teaser, link=link, include_link=include_link_main)

    if include_link_reply and link and res:
        reply_with_link_card(c, res, link)

    try:
        liked = search_and_like(c, CFG)
        followed = maybe_follow_accounts(c, CFG)
        print(f"Social actions: liked={liked}, followed={followed}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
