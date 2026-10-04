# RAG Bancos Colombia

Asistente conversacional basado en **RAG** (*Retrieval-Augmented Generation*) que responde preguntas
sobre la información pública publicada en los sitios web de **BBVA Colombia**, **Bancolombia** y
**Davivienda**. El sistema extrae el contenido con web scraping, lo limpia, lo vectoriza, lo indexa
y lo expone en una interfaz web de chat con respuestas en streaming, citas de las fuentes, historial
de conversaciones y un módulo de analítica de uso.

Todo corre **dentro de Docker con un solo comando** y con herramientas **gratuitas y de código
abierto** (modelos locales, sin APIs de pago).

## Contenido

1. [Requisitos previos](#requisitos-previos)
2. [Puesta en marcha](#puesta-en-marcha)
3. [Uso de la interfaz](#uso-de-la-interfaz)
4. [Operaciones habituales](#operaciones-habituales)
5. [Arquitectura](#arquitectura)
6. [Patrones de diseño](#patrones-de-diseño)
7. [Stack tecnológico](#stack-tecnológico)
8. [Configuración](#configuración)
9. [Seguridad y manejo de errores](#seguridad-y-manejo-de-errores)
10. [Analítica del histórico](#analítica-del-histórico)
11. [Pruebas y calidad](#pruebas-y-calidad)
12. [Rendimiento medido](#rendimiento-medido)
13. [Decisiones y supuestos](#decisiones-y-supuestos)
14. [Limitaciones conocidas](#limitaciones-conocidas)
15. [Mejoras futuras](#mejoras-futuras)

## Requisitos previos

| Requisito | Detalle |
|---|---|
| **Docker** con **Docker Compose v2** | Docker Desktop (Windows/macOS) o Docker Engine con el plugin de Compose. |
| **Memoria para Docker** | **Perfil GPU:** 16 GB de RAM en el equipo. **Perfil CPU:** al menos **8 GB asignados a Docker** (en Docker Desktop, *Settings → Resources*; por defecto suele asignar la mitad de la RAM). |
| **Disco libre** | ~20 GB: imágenes (~13 GB, la de Ollama es grande) y modelos (3–8 GB según el perfil). |
| **Puerto libre** | `8080` en el host (configurable con `APP_PORT`). Ningún otro servicio publica puertos. |
| **GPU NVIDIA** (recomendada) | 8 GB de VRAM o más y Docker con acceso a la GPU. Con ella las respuestas tardan ~3–9 s; sin ella funciona en CPU con un modelo más ligero. |
| **Git** | Para clonar el repositorio. |

No hace falta instalar Python, Ollama ni ninguna base de datos en tu máquina: todo está en los
contenedores. La **primera ejecución descarga** las imágenes y los modelos (el de lenguaje del perfil
elegido, `bge-m3` y `bge-reranker-v2-m3`), por lo que necesita conexión a internet y tarda varios
minutos.

## Puesta en marcha

```bash
# 1. Clonar el repositorio
git clone https://github.com/AdrianVillalba11/rag-bancos-colombia.git
cd rag-bancos-colombia

# 2. Elegir el perfil de configuración (editar POSTGRES_PASSWORD es recomendable)
cp .env.gpu.example .env      # con GPU NVIDIA (recomendado: llama3, mucho más rápido)
# cp .env.example .env        # sin GPU (modelo ligero llama3.2:3b, funciona en cualquier equipo)

# 3. Levantar todo
docker compose up -d
```

| Perfil | Archivo | Modelo | Reranker | Para quién |
|---|---|---|---|---|
| **GPU** (recomendado) | `.env.gpu.example` | `llama3` (8B) | GPU, 20 candidatos | Equipos con GPU NVIDIA de 8 GB de VRAM o más. |
| **CPU** | `.env.example` | `llama3.2:3b` | CPU, 5 candidatos | Cualquier equipo con ≥ 8 GB para Docker. |

Eso es todo. Al ejecutar el paso 3 ocurre, en orden:

1. Se **construye la imagen** de la aplicación (Python 3.12; PyTorch para CUDA o para CPU según el
   perfil).
2. `ollama` arranca y `ollama-init` **descarga los modelos** del perfil y `bge-m3` (una sola vez;
   quedan en un volumen).
3. `bootstrap` **indexa en ChromaDB** los 3.235 chunks ya limpios de `data/clean`, solo si el índice
   está vacío (en los siguientes arranques no hace nada). Como los **embeddings ya vienen
   calculados** en el repositorio (`data/clean/<banco>/embeddings.npz`, ~6 MB), este paso tarda
   segundos y no usa el modelo de embeddings.
4. `app` arranca, **precarga los modelos en segundo plano** y queda disponible.

Sigue el progreso con `docker compose logs -f` y comprueba el estado con
`docker compose ps`. Cuando `app` figure como `healthy`, abre:

- **Chat:** <http://localhost:8080> (o el puerto que definas en `APP_PORT`)
- **Analítica:** <http://localhost:8080/analytics>
- **Salud:** <http://localhost:8080/api/health>

> Mientras los modelos se precargan (`"warm": false` en `/api/health`), la primera respuesta puede
> tardar más. La interfaz lo avisa.

### Cambiar de perfil

Los dos perfiles solo difieren en lo propio de cada uno (modelo, dispositivo del reranker y, en el de
GPU, las variables que activan la GPU en Docker). Para cambiar, copia el otro archivo sobre `.env`
y vuelve a construir: `docker compose up -d --build`. Una prueba automática garantiza que ambos
archivos no se desincronicen.

En una RTX 4060 de 8 GB caben `llama3`, `bge-m3` y el reranker en media precisión a la vez.

### Detener y limpiar

```bash
docker compose down        # detiene y elimina los contenedores (conserva los datos)
docker compose down -v     # además elimina los volúmenes: modelos, índice e historial
```

## Uso de la interfaz

1. **Pregunta** en el cuadro inferior (Enter envía, Shift+Enter hace un salto de línea) o pulsa una
   de las sugerencias. La respuesta aparece en *streaming*.
2. **Filtra por banco** con el selector superior si quieres limitar la consulta a BBVA Colombia,
   Bancolombia o Davivienda. Con "Todos" se busca en los tres.
3. Las respuestas llevan **citas numeradas** `[1]`, `[2]`… y la lista de **fuentes** enlazadas debajo.
   Si no hay información suficiente, el asistente lo dice en vez de inventar.
4. **Preguntas de seguimiento:** el sistema recuerda los últimos *N* mensajes de la conversación
   (`HISTORY_MAX_MESSAGES`, por defecto 6) y reescribe preguntas como "¿y cuánto cuesta?" usando ese
   contexto.
5. **Historial:** el panel izquierdo lista tus conversaciones; al pulsar una se recupera completa.
   "Nueva conversación" empieza otra. La ✕ de cada entrada solo la quita de este navegador.
6. **Feedback:** 👍 / 👎 bajo cada respuesta, que alimenta la analítica.
7. **Analítica:** el enlace superior abre el dashboard con las métricas del histórico.

La API también puede usarse directamente:

```bash
# Respuesta completa en JSON
curl -s localhost:8080/api/chat -H "Content-Type: application/json" \
  -d '{"message": "¿Qué requisitos necesito para abrir una cuenta de ahorros?", "stream": false}'
```

| Método y ruta | Descripción |
|---|---|
| `POST /api/chat` | Responde una pregunta. Por defecto en streaming SSE (`meta`, `token`, `done`, `error`). |
| `GET /api/sessions?ids=...` | Resumen de las conversaciones cuyos ID envía el cliente. |
| `GET /api/sessions/{id}/messages` | Mensajes de una conversación. |
| `POST /api/messages/{id}/feedback` | Valora una respuesta (`{"value": 1}` o `-1`). |
| `GET /api/config` | Bancos disponibles y límites. |
| `GET /api/health` | Estado de Postgres, ChromaDB y Ollama. |
| `GET /api/analytics?days=N` | Informe de analítica (token opcional). |

## Operaciones habituales

Todo se ejecuta con Docker (servicios `ingest` y `tests`, perfil `tools`):

```bash
# Volver a vectorizar los datos limpios (sin descargar nada)
docker compose run --rm ingest python scripts/ingest.py --index-only

# Volver a descargar los sitios, limpiar y reindexar (tarda ~12 min por el ritmo cortés de scraping)
docker compose run --rm ingest python scripts/ingest.py --scrape

# Solo un banco
docker compose run --rm ingest python scripts/ingest.py --banks bancolombia --scrape

# Inspeccionar la recuperación sin LLM (modos vector, hybrid y rerank)
docker compose run --rm ingest python scripts/search.py "¿cuota de manejo?" --mode rerank

# Chat por consola con historial persistente
docker compose run --rm ingest python scripts/chat.py

# Informe de analítica en consola (Markdown o JSON)
docker compose run --rm ingest python scripts/analytics_report.py --days 7

# Generar conversaciones de demostración con el sistema real
docker compose run --rm ingest python scripts/seed_demo.py

# Pruebas unitarias y de integración (incluye Postgres y ChromaDB reales)
docker compose run --rm tests
```

## Arquitectura

```text
                         ┌────────────────────────── Docker Compose ───────────────────────────┐
  Navegador ── HTTP ───► │  app (FastAPI + UI)                                                 │
                         │    │  ① reescribe la pregunta con el historial                      │
                         │    │  ② búsqueda híbrida: vectorial + BM25  ──► RRF                 │
                         │    │  ③ reranker (cross-encoder)                                    │
                         │    │  ④ guardrail de relevancia                                     │
                         │    │  ⑤ generación en streaming ──► citas                           │
                         │    ▼                                                                │
                         │  chroma (vectores)   ollama (LLM + bge-m3)       postgres (historial)│
                         │                                                                     │
                         │  bootstrap: indexa data/clean en Chroma la primera vez              │
                         └─────────────────────────────────────────────────────────────────────┘

  Ingesta (offline):  scraping ─► data/raw ─► limpieza ─► data/clean ─► chunking ─► embeddings ─► Chroma
```

**Flujo de una pregunta** ([rag_service.py](src/rag/generation/rag_service.py)):

1. Se lee el historial reciente y, si hay seguimiento, el LLM **reescribe** la pregunta para que se
   entienda sola (respetando el banco elegido en la interfaz).
2. **Recuperación híbrida:** los 20 mejores fragmentos por similitud vectorial (`bge-m3`) y por
   coincidencia léxica (BM25) se **fusionan con RRF** (*Reciprocal Rank Fusion*). BM25 encuentra
   siglas, cifras y nombres de producto ("4x1000", "CDT") que los embeddings pueden confundir.
3. **Reranker** (`bge-reranker-v2-m3`): puntúa pregunta y fragmento juntos y deja los 5 mejores.
4. **Guardrail:** si la relevancia máxima queda bajo el umbral, no se consulta al modelo y se
   responde que no hay información.
5. **Generación** con el modelo del perfil (`llama3` o `llama3.2:3b`): el prompt obliga a responder solo con el contexto y a citar con `[n]`.
   Las fuentes mostradas son las que el modelo realmente citó.
6. Se guarda cada turno en Postgres con sus métricas (latencia por etapa, relevancia, fuentes).

### Estructura del repositorio

```text
├── docker-compose.yml, docker-compose.gpu.yml, Dockerfile, .env.example, .env.gpu.example
├── data/raw/<banco>/      páginas descargadas (HTML comprimido + manifest)
├── data/clean/<banco>/    documentos limpios (JSON), chunks (JSONL) y embeddings (.npz)
├── scripts/               ingest, scrape, search, chat, analytics_report, seed_demo
├── src/rag/
│   ├── config/            Settings (pydantic-settings, Singleton)
│   ├── domain/            entidades, interfaces (contratos) y errores de dominio
│   ├── ingestion/         scrapers/, robots, sitemap, cleaning, chunking, storage, pipeline
│   ├── retrieval/         embeddings, vector_store (Chroma), lexical (BM25), hybrid, rerankers/
│   ├── generation/        llm/ (Ollama), prompts, rag_service (Facade)
│   ├── conversation/      repository (Postgres), memory (ventana de N mensajes), schema
│   ├── analytics/         metrics (7 métricas), report
│   ├── api/               main (FastAPI), routes/, schemas, errors, dependencies (inyección)
│   ├── infra/             logging JSON, reintentos, timeouts, rate limit, db, http
│   └── web/               interfaz (HTML, CSS y JS sin dependencias externas)
└── tests/                 unit/ e integration/
```

Los datos limpios (`data/raw`, `data/clean`) están **versionados** para poder revisar el contenido
sin ejecutar Docker, junto con los **embeddings ya calculados** (`embeddings.npz`, ~6 MB en media
precisión). El índice de ChromaDB **no** se versiona: se regenera en el primer arranque a partir de
esos embeddings. La caché solo se usa si coinciden el modelo y el texto exacto de cada chunk; si no,
se recalcula automáticamente.

## Patrones de diseño

Se aplican más de los tres exigidos. Los contratos viven en
[domain/interfaces.py](src/rag/domain/interfaces.py), de modo que las capas altas dependen de
abstracciones y no de implementaciones concretas.

| Patrón | Dónde | Por qué |
|---|---|---|
| **Strategy** | `Scraper` ([BBVA](src/rag/ingestion/scrapers/bbva.py), [Bancolombia](src/rag/ingestion/scrapers/bancolombia.py), [Davivienda](src/rag/ingestion/scrapers/davivienda.py)); `Reranker` ([BGE](src/rag/retrieval/rerankers/bge.py) / [Noop](src/rag/retrieval/rerankers/base.py)); `LLMClient` ([Ollama](src/rag/generation/llm/ollama.py)) | Cada sitio tiene su propia configuración de rastreo; el reranker se puede desactivar o sustituir; el modelo de lenguaje puede cambiarse sin tocar el pipeline. |
| **Factory** | [ScraperFactory](src/rag/ingestion/scrapers/factory.py), [RerankerFactory](src/rag/retrieval/rerankers/factory.py), [LLMFactory](src/rag/generation/llm/factory.py) | Elige la estrategia a partir de la configuración. Añadir un banco es registrar una clase, sin modificar la fábrica. |
| **Repository** | [`ChromaVectorStore`](src/rag/retrieval/vector_store.py) y [`PostgresConversationRepository`](src/rag/conversation/repository.py) | Aíslan el almacenamiento: el dominio habla con `VectorStore` y `ConversationRepository`, no con SQL ni con el cliente de Chroma. Permite cambiar de motor y probar con dobles en memoria. |
| **Facade** | [`RagService`](src/rag/generation/rag_service.py) | Una única llamada oculta reescritura, recuperación, reranking, guardrail, generación y citas. La API no conoce esas piezas. |
| **Singleton / Inyección de dependencias** | [`get_settings()`](src/rag/config/settings.py) y [`AppState`](src/rag/api/dependencies.py) | Una sola configuración validada y un único conjunto de servicios pesados (modelos, índice BM25, pool de conexiones), construidos una vez y compartidos. |
| **Template Method** | [`BaseScraper`](src/rag/ingestion/scrapers/base.py) | Define el flujo común (robots → descubrimiento → descarga → deduplicación) y cada banco solo declara su configuración. |

## Stack tecnológico

| Componente | Elección | Justificación |
|---|---|---|
| Lenguaje | Python 3.12 | Requisito; ecosistema de ML y de scraping. |
| API y UI | FastAPI + HTML/CSS/JS | Streaming SSE nativo, validación con pydantic y una UI sin dependencias externas (válida con una CSP estricta). |
| LLM | `llama3` (8B, perfil GPU) o `llama3.2:3b` (perfil CPU) en **Ollama** | Local y gratuito; buen español; el 8B cabe en una GPU de 8 GB y el 3B en 8 GB de RAM. Se cambia con `LLM_MODEL`. |
| Embeddings | `bge-m3` (vía Ollama) | Multilingüe, muy buen resultado en español y reutiliza el mismo servicio que el LLM. |
| Base vectorial | **ChromaDB** | Self-hosted, simple, persistente y con filtros por metadatos (banco). |
| Búsqueda léxica | `rank_bm25` | Complementa lo semántico con coincidencia exacta de siglas y cifras. |
| Reranker | `bge-reranker-v2-m3` (sentence-transformers) | Cross-encoder multilingüe, de la misma familia que los embeddings; mejora la precisión de los fragmentos finales. |
| Historial | **PostgreSQL** | Datos relacionales (conversaciones, mensajes, citas), concurrencia real y consultas analíticas. |
| Scraping | `httpx` + BeautifulSoup/lxml | Los sitios son HTML estático o con contenido en el HTML; evita la complejidad de un navegador. |
| Orquestación | Docker Compose | Un solo comando; servicios aislados y reproducibles. |
| Calidad | pytest, ruff | 194 pruebas; análisis estático. |

## Configuración

Toda la configuración está externalizada en `.env`, con dos plantillas: [.env.example](.env.example)
(perfil CPU) y [.env.gpu.example](.env.gpu.example) (perfil GPU). Las más relevantes:

| Variable | Por defecto | Descripción |
|---|---|---|
| `HISTORY_MAX_MESSAGES` | `6` | Mensajes previos que recuerda cada conversación (**N**). |
| `LLM_MODEL` / `LLM_NUM_CTX` | según perfil (`llama3` / `8192` en GPU) | Modelo y ventana de contexto. |
| `EMBEDDING_MODEL` | `bge-m3` | Modelo de embeddings. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | Tamaño y solape de los chunks (caracteres). |
| `RETRIEVAL_TOP_K` | `20` | Candidatos de la recuperación. |
| `RERANK_CANDIDATES` / `RERANK_TOP_N` | según perfil / `5` | Candidatos que puntúa el reranker y fragmentos finales. |
| `RERANKER_ENABLED` / `HYBRID_SEARCH_ENABLED` | `true` | Activan el reranker y la parte léxica. |
| `MIN_RELEVANCE_SCORE` | `0.15` | Umbral del guardrail ("no sé"). |
| `SCRAPE_BANKS`, `SCRAPE_MAX_PAGES_PER_BANK` | los tres, `250` | Alcance del scraping. |
| `RATE_LIMIT_PER_MINUTE`, `MAX_CONCURRENT_CHATS` | `30`, `4` | Protección de la API. |
| `ANALYTICS_TOKEN` | vacío | Si se define, protege la analítica. |
| `APP_PORT` | `8080` | Puerto del host. |

## Seguridad y manejo de errores

**Manejo de errores**

- **Reintentos con backoff exponencial y jitter** en scraping, embeddings, modelo y bases de datos
  ([retry.py](src/rag/infra/retry.py)).
- **Timeouts explícitos** en todas las llamadas: HTTP, Ollama, ChromaDB y sentencias de Postgres.
- **Excepciones de dominio** ([errors.py](src/rag/domain/errors.py)) traducidas a respuestas HTTP
  coherentes, sin filtrar detalles internos. En streaming, un fallo viaja como evento `error`.
- **Degradación elegante:** si cae el reranker se usa el orden de la recuperación; si falla la
  reescritura se usa la pregunta original; si cae el modelo se responde con un mensaje de
  contingencia.
- **Scraping defensivo:** respeta `robots.txt` (con comodines), un ritmo cortés, descarta duplicados
  y un fallo puntual no detiene el resto. Ante un 401/403 se detiene sin intentar evadir el bloqueo.
- **Ingesta idempotente:** los IDs de los chunks son deterministas; reindexar no duplica.
- **Logging estructurado** en JSON y **`/api/health`** sobre las tres dependencias.

**Seguridad**

- **Defensa contra *prompt injection*:** el contenido scrapeado y la pregunta son *datos*. Las reglas
  van en el mensaje de sistema, el contexto se delimita y se neutralizan en él los caracteres de
  etiqueta. Se probó con una página envenenada que intentaba cerrar el contexto y dar órdenes: el
  modelo la ignoró.
- **Guardrail anti-alucinación:** sin contexto relevante no se consulta al modelo.
- **Validación de entradas:** longitud y caracteres de control de la pregunta, banco permitido, ID de
  sesión con formato estricto.
- **Rate limiting** por cliente (429 + `Retry-After`), cupo de respuestas simultáneas (503), tope de
  tamaño de petición (413).
- **Cabeceras de seguridad** y **CSP estricta**; la interfaz construye el DOM sin `innerHTML`.
- **Privacidad del historial:** no existe un listado global; cada navegador solo consulta las
  conversaciones cuyos ID guarda, porque el ID de sesión actúa como credencial.

## Analítica del histórico

El módulo [analytics](src/rag/analytics/metrics.py) recorre todo el histórico guardado en Postgres y
calcula, en una pasada, las siguientes métricas (dashboard en `/analytics`, endpoint
`/api/analytics` y script `analytics_report.py`):

1. **Volumen de uso:** conversaciones, preguntas y serie diaria.
2. **Temas y preguntas frecuentes:** clasificación por reglas de palabras clave, términos más
   consultados y preguntas repetidas (agrupadas sin distinguir acentos ni mayúsculas).
3. **Latencia:** media, p50, p90 y p95, y tiempo medio por etapa (recuperación, reranker, generación).
4. **Huecos de conocimiento:** preguntas sin respuesta, con baja relevancia o valoradas 👎, que
   indican qué contenido falta.
5. **Bancos y fuentes:** citas por banco, páginas más citadas y uso del filtro.
6. **Feedback:** satisfacción y cobertura de valoraciones.
7. **Perfil de las conversaciones:** longitud, tasa de seguimiento, conversaciones de una sola
   pregunta (posible abandono) y seguimientos reescritos.

**Valores de impacto:** tasa de resolución, satisfacción, latencia media y **tiempo ahorrado
estimado**. Este último se basa en un *supuesto configurable* (`ANALYTICS_MANUAL_SEARCH_MINUTES`,
4 minutos de búsqueda manual por consulta resuelta) y se indica como estimación, no como medición.

## Pruebas y calidad

```bash
docker compose run --rm tests
```

Incluye pruebas unitarias (scraping con un sitio simulado, limpieza, chunking, recuperación,
generación, API, analítica, endurecimiento) y de **integración contra ChromaDB y PostgreSQL
reales**. La API y la interfaz también se verificaron manualmente en el navegador.

## Rendimiento medido

Mediciones reales en un portátil con **16 GB de RAM, GPU RTX 4060 (8 GB de VRAM)** y Docker Desktop
con **7,6 GB** de memoria para su VM. El arranque se probó **desde cero** (clon limpio, volúmenes
nuevos, imagen construida sin caché).

| Medición | Perfil GPU (`llama3`) | Perfil CPU (`llama3.2:3b`) |
|---|---|---|
| Construcción de la imagen (sin caché) | no medida desde cero (la imagen CUDA es mayor) | 3–7 min |
| Descarga de modelos (primera vez) | 8–11 min (≈6 GB) | ≈7 min (≈3,2 GB) |
| Indexación inicial con la caché de embeddings | ≈40 s | < 1 min |
| *Referencia: indexar sin la caché de embeddings* | ≈3 min | **≈28 min** |
| Calentamiento de modelos al arrancar | 1–4 min (observado) | ≈5 min |
| **`docker compose up -d` completo, desde cero** | — | **≈15 min** |
| Respuesta de una pregunta, en caliente | **3–9 s** | **60–90 s** |
| Reranker | 0,3 s (20 candidatos) | 4–5 s (5 candidatos) |
| Memoria en uso | ≈6,6 GB de VRAM | ≈5,3 GB de RAM |

Notas:

- El perfil CPU **funciona, pero es lento**: una respuesta tarda entre uno y un minuto y medio, sobre
  todo por la generación del modelo. El perfil GPU es el recomendado.
- Los tiempos de descarga dependen de la conexión (entre 3 y 22 MB/s durante las pruebas) y no
  incluyen las imágenes base de Docker (la de Ollama pesa ≈9 GB).
- Con `llama3` de 8B en CPU **no es viable** en una VM de 7,6 GB: la carga de modelos supera la memoria
  y el sistema se vuelve inutilizable. Por eso existen los dos perfiles.

## Decisiones y supuestos

- **Sitios:** se usan los tres bancos (BBVA Colombia, Bancolombia y Davivienda), con hasta 250
  páginas por banco para que el scraping dure minutos. En total: 574 páginas descargadas, 433
  documentos tras la limpieza y 3.235 chunks.
- **Scraping responsable:** identificación honesta (`User-Agent` propio), `robots.txt` respetado
  mediante un intérprete propio con soporte de comodines (el de la biblioteca estándar no los
  entiende) y 1 s entre solicitudes.
- **Datos versionados:** se suben los datos crudos y limpios para que el repositorio pueda revisarse
  sin Docker; los crudos se guardan comprimidos y de forma determinista para no ensuciar el historial.
- **Limpieza en dos etapas:** estructural por página y de corpus por banco (elimina texto repetido en
  muchas páginas, páginas vacías y duplicados), lo que resuelve las plantillas renderizadas con
  JavaScript.
- **Chunking por estructura:** respeta párrafos y encabezados, con solape y contexto
  (`título — encabezado`) antepuesto al embedding.
- **Ollama dentro de Docker** (no en el host), como exige el enunciado.
- **Sin cuentas de usuario:** el historial se asocia a un ID de sesión que guarda el navegador.
- **Ambigüedades del enunciado:** donde algo no estaba definido (métricas concretas, N por defecto,
  alcance del scraping) se eligió un valor razonable y se dejó configurable.

## Limitaciones conocidas

- **CPU sin GPU es lento:** el reranker tarda ~10 s con 20 candidatos y el modelo genera pocos tokens
  por segundo. Con GPU el reranker tarda ~0,3 s y una respuesta unos 3–9 s. Para equipos modestos
  conviene un modelo menor (`LLM_MODEL=llama3.2:3b`) y `RERANK_CANDIDATES=5`.
- **Primera ejecución pesada:** descarga las imágenes y los modelos (3–8 GB según el perfil).
- **GPU de 8 GB y varias copias del reranker:** con la app activa ya hay ~6,6 GB de VRAM en uso. Si
  se lanza a la vez otro proceso que cargue su propio reranker (por ejemplo, un script auxiliar),
  la VRAM se agota, Windows pasa a usar la RAM del sistema como memoria de vídeo y todo se vuelve muy
  lento; en un equipo con poca RAM libre llegó a colgar el motor de Docker. Conviene detener la app
  antes de ejecutar herramientas que usen el reranker.
- **Sin evaluación sistemática de la recuperación:** la elección de la búsqueda híbrida y del reranker
  se justificó con ejemplos concretos y con mediciones de latencia, no con métricas como *hit-rate* o
  MRR sobre un conjunto de preguntas de referencia (ver mejoras futuras).
- **Perfil CPU y memoria:** con `llama3` 8B en CPU se necesitan ~9 GB solo para los modelos; con
  menos de ~12 GB para Docker la carga de modelos se vuelve inviable. Por eso el perfil CPU usa el
  modelo de 3B.
- **BBVA y el WAF:** el sitio devolvió 403 a algunos clientes (por ejemplo, `curl` en Windows) y 200
  a `httpx`. Es un filtrado del lado del servidor; el scraper no intenta evadirlo y se detiene si lo
  recibe.
- **Davivienda:** sus encabezados visibles no usan `<h2>`, por lo que cada documento queda en una sola
  sección y el chunking cae a división por tamaño (menos precisión de tema).
- **Contenido dinámico:** no se ejecuta JavaScript; solo se indexa lo que viene en el HTML. Por eso
  tarifas cargadas por JavaScript pueden no estar disponibles (el asistente responde que no tiene la
  información).
- **Frescura:** el índice es una fotografía del momento del scraping; no hay actualización
  automática.
- **Historial por navegador:** sin login, un usuario no puede recuperar sus conversaciones desde otro
  dispositivo.
- **Respuesta interrumpida:** si el navegador se cierra a mitad de una respuesta en streaming, esa
  respuesta no se guarda.
- **Rate limiting en memoria:** válido para un solo proceso; con varias réplicas haría falta Redis.
- **Contenedores como root** y credenciales de ejemplo en `.env.example`: aceptable para una demo,
  no para producción.
- **Temas por palabras clave:** la clasificación de temas de la analítica es por reglas, no por un
  modelo.
- **Python local 3.13:** `chromadb-client` no tiene ruedas para Windows con 3.13, por eso las
  pruebas que usan ChromaDB y Postgres se ejecutan dentro de Docker.

## Mejoras futuras

- **Actualización programada** del scraping (re-scraping incremental con detección de cambios).
- **Evaluación sistemática de la recuperación:** generar preguntas sintéticas a partir de los chunks
  (con el propio LLM, pidiendo parafrasear y descartando las que copian el texto), usar la página de
  origen como respuesta correcta y medir *hit-rate@k* y MRR para la búsqueda vectorial, la híbrida y
  con o sin reranker; compararla con otros modelos de lenguaje e integrarla en CI. Se deja fuera de
  esta entrega por tiempo y por la carga que supone en una GPU de 8 GB.
- **Autenticación y cuentas de usuario**, con historial entre dispositivos.
- **Rate limiting distribuido** con Redis y observabilidad (métricas Prometheus, trazas).
- **Clasificación de temas con el LLM** y detección automática de huecos de contenido.
- **Soporte de páginas dinámicas** (renderizado con navegador sin cabeza) y de documentos PDF.
- **Reranker y embeddings afinados** con el dominio bancario colombiano.
- **Contenedores sin privilegios**, gestión de secretos y endurecimiento de la imagen.
- **Resumen de conversaciones largas** en lugar de una ventana fija de mensajes.

## Aviso sobre los datos

El contenido indexado proviene de sitios web públicos de terceros y se usa únicamente con fines
educativos y de evaluación técnica. El asistente es una herramienta de apoyo para consultar
información publicada; **no constituye asesoría financiera**.

## Licencia

Código distribuido bajo licencia [MIT](LICENSE).
