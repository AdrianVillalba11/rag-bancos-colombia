"""Utilidades de URL para el rastreo."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

_EXTENSIONES_NO_HTML = re.compile(
    r"\.(pdf|jpe?g|png|gif|svg|webp|ico|css|js|json|xml|zip|rar|xlsx?|docx?|pptx?|mp[34]|avi|mov)$",
    re.IGNORECASE,
)


def normalize_url(url: str) -> str:
    """Quita fragmento y query, pasa a minúsculas el host y elimina la barra final."""
    url, _ = urldefrag(url)
    p = urlsplit(url)
    ruta = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), ruta, "", ""))


def resolve(base: str, href: str) -> str | None:
    """Convierte un enlace relativo en URL absoluta http(s) normalizada."""
    href = href.strip()
    if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
        return None
    absoluta = urljoin(base, href)
    if urlsplit(absoluta).scheme not in ("http", "https"):
        return None
    return normalize_url(absoluta)


def same_site(url: str, base_url: str) -> bool:
    def host(u: str) -> str:
        return urlsplit(u).netloc.lower().removeprefix("www.")

    return host(url) == host(base_url)


def is_html_candidate(url: str) -> bool:
    return not _EXTENSIONES_NO_HTML.search(urlsplit(url).path)


def diversify(urls: Iterable[str], depth: int = 2) -> list[str]:
    """Reparte las URLs en turnos entre secciones del sitio (primeros `depth` segmentos).

    Así, al limitar el número de páginas, la muestra cubre distintos productos y no solo la
    primera sección que aparezca en el sitemap.
    """
    grupos: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        segmentos = [s for s in urlsplit(url).path.split("/") if s]
        grupos[("/".join(segmentos[:depth]))].append(url)

    resultado: list[str] = []
    colas = list(grupos.values())
    while colas:
        siguientes = []
        for cola in colas:
            resultado.append(cola.pop(0))
            if cola:
                siguientes.append(cola)
        colas = siguientes
    return resultado
