"""Almacenamiento local de las páginas crudas.

Cada página se guarda comprimida (`data/raw/<banco>/<id>.html.gz`) y un `manifest.json` por banco
indica la URL, la fecha, el hash y el tamaño de cada archivo. La compresión es determinista
(sin marca de tiempo) para que reejecutar el scraping no genere cambios espurios en git.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from rag.domain.models import RawPage

logger = logging.getLogger(__name__)


class RawStore:
    def __init__(self, raw_dir: Path) -> None:
        self._raw_dir = raw_dir

    def _bank_dir(self, bank: str) -> Path:
        return self._raw_dir / bank

    @staticmethod
    def _file_id(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    def save_all(self, bank: str, pages: Iterator[RawPage]) -> int:
        """Guarda las páginas a medida que llegan y escribe el manifiesto al final.

        Si se guardó al menos una página, elimina las de ejecuciones previas que ya no
        aparecen; si el scraping no produjo nada, conserva lo anterior.
        """
        directorio = self._bank_dir(bank)
        entradas: list[dict[str, object]] = []
        try:
            for pagina in pages:
                directorio.mkdir(parents=True, exist_ok=True)
                nombre = f"{self._file_id(pagina.url)}.html.gz"
                datos = gzip.compress(pagina.html.encode("utf-8"), mtime=0)
                (directorio / nombre).write_bytes(datos)
                entradas.append(
                    {
                        "url": pagina.url,
                        "file": nombre,
                        "sha256": pagina.hash,
                        "status_code": pagina.status_code,
                        "fetched_at": pagina.fetched_at.isoformat(),
                        "bytes_compressed": len(datos),
                    }
                )
        finally:
            # Si no se guardó nada, se conserva lo que hubiera de ejecuciones anteriores.
            # El manifiesto refleja lo guardado aunque el scraping se interrumpa a medias.
            if entradas:
                self._write_manifest(directorio, bank, entradas)
        return len(entradas)

    @staticmethod
    def _write_manifest(directorio: Path, bank: str, entradas: list[dict[str, object]]) -> None:
        vigentes = {str(e["file"]) for e in entradas}
        for archivo in directorio.glob("*.html.gz"):
            if archivo.name not in vigentes:
                archivo.unlink()
        entradas.sort(key=lambda e: str(e["url"]))
        (directorio / "manifest.json").write_text(
            json.dumps({"bank": bank, "pages": entradas}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, bank: str) -> Iterator[RawPage]:
        """Lee las páginas guardadas de un banco."""
        manifiesto = self._bank_dir(bank) / "manifest.json"
        if not manifiesto.exists():
            return
        datos = json.loads(manifiesto.read_text(encoding="utf-8"))
        for entrada in datos["pages"]:
            ruta = self._bank_dir(bank) / entrada["file"]
            if not ruta.exists():
                logger.warning("Archivo crudo ausente", extra={"file": str(ruta)})
                continue
            yield RawPage(
                bank=bank,
                url=entrada["url"],
                html=gzip.decompress(ruta.read_bytes()).decode("utf-8"),
                status_code=entrada["status_code"],
                fetched_at=datetime.fromisoformat(entrada["fetched_at"]),
            )

    def banks(self) -> list[str]:
        if not self._raw_dir.exists():
            return []
        return sorted(p.name for p in self._raw_dir.iterdir() if (p / "manifest.json").exists())
