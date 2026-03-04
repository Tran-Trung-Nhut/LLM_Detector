import re
from bs4 import BeautifulSoup

_BOILERPLATE_PATTERNS = [
    r"\bbug fixes?\b",
    r"\bperformance improvements?\b",
    r"\bstability improvements?\b",
    r"\bminor improvements?\b",
    r"\bwe fixed\b",
]

def html_to_text(html: str) -> str:
    if html is None:
        return ""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_boilerplate(text: str) -> bool:
    t = (text or "").lower()
    if len(t) < 10:
        return True
    for pat in _BOILERPLATE_PATTERNS:
        if re.search(pat, t):
            # nếu toàn câu chỉ là boilerplate thì coi là boilerplate
            # heuristic đơn giản: độ dài ngắn + chứa pattern
            if len(t) < 80:
                return True
    return False

def build_app_text(title: str, description: str, recent_changes: str, category: str) -> str:
    title = title or ""
    description = description or ""
    recent_changes = recent_changes or ""
    category = category or ""
    # dùng marker token để model học cấu trúc
    return (
        f"[TITLE] {title}\n"
        f"[CATEGORY] {category}\n"
        f"[DESCRIPTION]\n{description}\n"
        f"[RECENT_CHANGES]\n{recent_changes}\n"
    ).strip()