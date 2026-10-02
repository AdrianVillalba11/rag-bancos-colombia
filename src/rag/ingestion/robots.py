"""Intérprete de robots.txt con soporte de comodines (`*` y `$`).

`urllib.robotparser` de la biblioteca estándar no entiende comodines, y los sitios bancarios los
usan (por ejemplo `Disallow: /*pdf*`). Se aplica la regla de precedencia por coincidencia más
larga y, ante un empate, gana `Allow`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class _Rule:
    allow: bool
    pattern: str
    regex: re.Pattern[str]

    @classmethod
    def build(cls, allow: bool, pattern: str) -> _Rule:
        expresion = re.escape(pattern).replace(r"\*", ".*")
        if expresion.endswith(r"\$"):
            expresion = expresion[:-2] + "$"
        return cls(allow, pattern, re.compile(expresion))


@dataclass
class _Group:
    agents: list[str] = field(default_factory=list)
    rules: list[_Rule] = field(default_factory=list)


class RobotsPolicy:
    def __init__(self, rules: list[_Rule] | None = None, sitemaps: list[str] | None = None) -> None:
        self._rules = rules or []
        self.sitemaps = sitemaps or []

    @classmethod
    def allow_all(cls) -> RobotsPolicy:
        return cls()

    @classmethod
    def parse(cls, text: str, user_agent: str) -> RobotsPolicy:
        token = user_agent.split("/")[0].split()[0].lower()
        grupos: list[_Group] = []
        sitemaps: list[str] = []
        actual: _Group | None = None
        leyendo_agentes = False

        for linea in text.splitlines():
            linea = linea.split("#", 1)[0].strip()
            if ":" not in linea:
                continue
            clave, valor = (parte.strip() for parte in linea.split(":", 1))
            clave = clave.lower()
            if clave == "user-agent":
                if not leyendo_agentes or actual is None:
                    actual = _Group()
                    grupos.append(actual)
                actual.agents.append(valor.lower())
                leyendo_agentes = True
            elif clave in ("allow", "disallow"):
                leyendo_agentes = False
                if actual is not None and valor:
                    actual.rules.append(_Rule.build(clave == "allow", valor))
            elif clave == "sitemap":
                sitemaps.append(valor)

        # Un grupo específico para nuestro agente tiene prioridad sobre el comodín
        especificos = [g for g in grupos if any(a != "*" and a in token for a in g.agents)]
        generales = [g for g in grupos if "*" in g.agents]
        elegidos = especificos or generales
        return cls([regla for g in elegidos for regla in g.rules], sitemaps)

    def allowed(self, url: str) -> bool:
        partes = urlsplit(url)
        ruta = partes.path or "/"
        if partes.query:
            ruta += "?" + partes.query

        mejor: _Rule | None = None
        for regla in self._rules:
            if not regla.regex.match(ruta):
                continue
            if (
                mejor is None
                or len(regla.pattern) > len(mejor.pattern)
                or (len(regla.pattern) == len(mejor.pattern) and regla.allow)
            ):
                mejor = regla
        return True if mejor is None else mejor.allow
