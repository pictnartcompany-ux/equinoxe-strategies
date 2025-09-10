# Requirements: pip install atproto feedparser requests beautifulsoup4 python-dateutil
import os, re, json, requests, feedparser
from datetime import date, datetime, timezone, timedelta
from bs4 import BeautifulSoup
from atproto import Client, models  # pas de RichText (compat large)

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
    """Récupère le corps de l'article pour donner de la matière à l'IA."""
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
    """Construit un contexte riche + retourne aussi le plein texte pour l'extraction des signaux."""
    article = fetch_article_text(link)
    base = f"{title}. {summary}".strip()
    ctx = (base + ("\n\n" + article if article else "")).strip()
    ctx = ctx[:4000] if ctx else base[:1000]
    return ctx, article

# -------- Extraction de signaux (acteurs / chiffres / dates) ----------
MONTHS = r"(janv\.?|févr\.?|mars|avr\.?|mai|juin|juil\.?|août|sept\.?|oct\.?|nov\.?|déc\.?|january|february|march|april|may|june|july|august|september|october|november|december)"
YEAR = r"(20\d{2}|19\d{2})"
PCT = r"\b\d{1,3}(?:[.,]\d+)?\s?%\b"
MONEY = r"(?:\$|€)\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s?(?:bn|billion|milliard|milliards|m|million|millions|k)?"
PLAIN_NUM = r"\b\d{2,4}(?:[.,]\d+)?\b"

def extractive_fallback(text: str, k=5):
    """Fallback: sélectionne 4–5 phrases informatives si l'IA répond mal."""
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


def extract_actors(text: str, max_items=6):
    # Séquences de mots Capitalisés (2 à 4), filtre basique
    cand = re.findall(r"\b([A-Z][a-zA-Z0-9&\-]+(?:\s+[A-Z][a-zA-Z0-9&\-]+){0,3})\b", text or "")
    # Filtre bruits communs
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
    # privilégie les montants/%, sinon quelques nombres nus
    if len(nums) < 3:
        nums |= set(re.findall(PLAIN_NUM, text))
    # dates (mois + année, ou année seule)
    dates = set(re.findall(MONTHS + r"\s+" + YEAR, text, flags=re.IGNORECASE))
    years = set(re.findall(r"\b" + YEAR + r"\b", text))
    # normalise
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
    """Construit une petite section 'Faits détectés' à injecter dans le prompt."""
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

def dedupe_lines(lines: list[str]) -> list[str]:
    """Supprime les doublons ou quasi-doublons (normalisation simple)."""
    out, seen = [], set()
    for l in lines:
        norm = re.sub(r"[\W_]+", " ", l.lower()).strip()
        norm = re.sub(r"\s+", " ", norm)
        if norm in seen or not norm:
            continue
        seen.add(norm)
        out.append(l)
    return out

# ---- FR/EN + traduction secours
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

# =================
# Résumé en français
# =================
def summarize_5_lines(context: str, fact_hints: str) -> str:
    prompt = (
        "Tu écris UNIQUEMENT en FRANÇAIS. Produis 4 à 5 puces courtes (≤ ~25 mots), "
        "ton posé, analytique, optimisme mesuré, orienté IA / spatial / innovation. "
        "Si disponibles, privilégie données chiffrées, noms d'acteurs, échéances. "
        "Varie légèrement les tournures. N'énonce PAS deux fois la même idée. "
        "INTERDIT: copier mot à mot la source; reformule.\n"
        "Structure :\n- Fait marquant\n- Pourquoi c’est important\n- Implications (marché/technos)\n"
        "- Risque / point de vigilance\n- Prochaine étape / à surveiller\n\n"
        f"Faits détectés (à intégrer si pertinents): {fact_hints}\n\n"
        f"{context[:4000]}"
    )
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 260, "temperature": 0.7},
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
    raw_lines = [l.strip(" -*•\t") for l in out.split("\n") if l.strip()]
    lines = dedupe_lines(raw_lines)
    if len(lines) < 4:
        # fallback extractif sur le contexte riche
        lines = extractive_fallback(context, k=5)
    text = "\n".join(f"- {l}" for l in lines[:5])

    # Garde-fou FR
    if is_likely_english(text):
        text = translate_to_fr(text)
    return text

# =============
# Bluesky client + post / reply
# =============
def bsky_client() -> Client:
    handle = os.environ.get("BSKY_HANDLE")
    pwd = os.environ.get("BSKY_PASSWORD")
    if not handle or not pwd:
        raise SystemExit("Missing BSKY_HANDLE or BSKY_PASSWORD")
    c = Client()
    c.login(handle, pwd)
    return c

def post_bsky(client: Client, text: str, link: str | None = None, include_link: bool = False):
    """Poste et retourne l'objet {'uri','cid'} du post créé. Si include_link, ajoute une carte cliquable."""
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
    """Répond au post parent avec une carte 'Source :', en dict brut (compat large)."""
    if not parent_post or not link:
        print("[INFO] Pas de parent ou pas de lien pour reply.")
        return
    try:
        embed = models.AppBskyEmbedExternal.Main(
            external=models.AppBskyEmbedExternal.External(
                uri=link, title="Source", description=""
            )
        )
        reply_ref = {
            "root": {"uri": parent_post["uri"], "cid": parent_post["cid"]},
            "parent": {"uri": parent_post["uri"], "cid": parent_post["cid"]},
        }
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
    include_link_main = bool(POSTING.get("include_link_in_main", False))   # post principal: par défaut sans lien
    include_link_reply = bool(POSTING.get("include_link_in_reply", True))  # réponse "Source :" activée

    c = bsky_client()

    if even_day:
        # Contexte riche + signaux
        context, fulltext = build_context(title, summary, link)
        signals = extract_signals(fulltext or context)
        hints = make_fact_hints(signals)
        wakeup = summarize_5_lines(context, hints)
        res = post_bsky(c, wakeup, link=link, include_link=include_link_main)
    else:
        teaser = f"Lecture conseillée 👇 {title}"
        res = post_bsky(c, teaser, link=link, include_link=include_link_main)

    # Publier la source en réponse (carte cliquable)
    if include_link_reply and link and res:
        reply_with_link_card(c, res, link)

    # Social léger (optionnel)
    try:
        liked = search_and_like(c, CFG)
        followed = maybe_follow_accounts(c, CFG)
        print(f"Social actions: liked={liked}, followed={followed}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
