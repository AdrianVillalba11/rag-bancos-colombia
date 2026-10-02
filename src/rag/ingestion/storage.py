"""Almacenamiento local de páginas crudas, documentos limpios y chunks.

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

from rag.domain.models import Chunk, CleanDocument, RawPage, Section

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


class CleanStore:
    """Documentos limpios y chunks por banco, en JSON legible.

    Estructura: `data/clean/<banco>/documents/<id>.json` y `data/clean/<banco>/chunks.jsonl`.
    """

    def __init__(self, clean_dir: Path) -> None:
        self._clean_dir = clean_dir

    def _docs_dir(self, bank: str) -> Path:
        return self._clean_dir / bank / "documents"

    def save_documents(self, bank: str, docs: list[CleanDocument]) -> int:
        directorio = self._docs_dir(bank)
        directorio.mkdir(parents=True, exist_ok=True)
        vigentes = set()
        for doc in docs:
            nombre = f"{doc.id}.json"
            vigentes.add(nombre)
            registro = {
                "id": doc.id,
                "bank": doc.bank,
                "url": doc.url,
                "title": doc.title,
                "fetched_at": doc.fetched_at.isoformat(),
                "source_hash": doc.source_hash,
                "sections": [{"heading": s.heading, "text": s.text} for s in doc.sections],
            }
            (directorio / nombre).write_text(
                json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        for archivo in directorio.glob("*.json"):
            if archivo.name not in vigentes:
                archivo.unlink()
        return len(docs)

    def load_documents(self, bank: str) -> list[CleanDocument]:
        directorio = self._docs_dir(bank)
        if not directorio.exists():
            return []
        docs = []
        for archivo in sorted(directorio.glob("*.json")):
            r = json.loads(archivo.read_text(encoding="utf-8"))
            docs.append(
                CleanDocument(
                    bank=r["bank"],
                    url=r["url"],
                    title=r["title"],
                    sections=tuple(Section(s["heading"], s["text"]) for s in r["sections"]),
                    fetched_at=datetime.fromisoformat(r["fetched_at"]),
                    source_hash=r["source_hash"],
                )
            )
        return docs

    def save_chunks(self, bank: str, chunks: list[Chunk]) -> int:
        directorio = self._clean_dir / bank
        directorio.mkdir(parents=True, exist_ok=True)
        lineas = [
            json.dumps({"id": c.id, "text": c.text, **c.metadata()}, ensure_ascii=False)
            for c in chunks
        ]
        (directorio / "chunks.jsonl").write_text("\n".join(lineas) + "\n", encoding="utf-8")
        return len(chunks)

    def load_chunks(self, bank: str) -> list[Chunk]:
        archivo = self._clean_dir / bank / "chunks.jsonl"
        if not archivo.exists():
            return []
        chunks = []
        for linea in archivo.read_text(encoding="utf-8").splitlines():
            r = json.loads(linea)
            chunks.append(
                Chunk(
                    document_id=r["document_id"],
                    bank=r["bank"],
                    url=r["url"],
                    title=r["title"],
                    heading=r["heading"],
                    text=r["text"],
                    position=r["position"],
                )
            )
        return chunks

    def banks(self) -> list[str]:
        if not self._clean_dir.exists():
            return []
        return sorted(p.name for p in self._clean_dir.iterdir() if (p / "chunks.jsonl").exists())
