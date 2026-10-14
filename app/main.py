
"""Backend FastAPI de Archivo TIC IA.

Compatible con ejecución local y despliegue en Vercel.
Las credenciales de OpenAI permanecen en el servidor.
La sincronización de Google Drive solo está permitida en localhost.
"""

from pathlib import Path
from ipaddress import ip_address

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.models.documento import AskRequest, AskResponse
from app.services.sync_service import SyncService
from app.services.agent_service import AgentService


# ============================================================
# CONFIGURACIÓN DE FASTAPI
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="Archivo TIC IA",
    version="1.2.0",
    description="Agente documental con Google Drive y OpenAI File Search",
)

app.mount(
    "/assets",
    StaticFiles(directory=ROOT / "assets"),
    name="assets",
)


# ============================================================
# SEGURIDAD DE LA SINCRONIZACIÓN
# ============================================================

def local_only(request: Request):
    """Permite sincronizar únicamente desde localhost.

    En Vercel, la sincronización permanece bloqueada
    porque OAuth de escritorio requiere ejecución local.
    """

    if settings.vercel_mode:
        raise HTTPException(
            status_code=403,
            detail="La sincronización de Drive solo está disponible en localhost.",
        )

    host = request.client.host if request.client else ""

    try:
        if not ip_address(host).is_loopback:
            raise ValueError()

    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Sincronización disponible solo desde localhost",
        )

    key = settings.local_admin_key

    if key and request.headers.get("X-Admin-Key") != key:
        raise HTTPException(
            status_code=401,
            detail="Clave administrativa incorrecta",
        )


# ============================================================
# INTERFAZ WEB
# ============================================================

@app.get("/")
def index():
    """Muestra Archivo TIC IA sin solicitar contraseña."""
    return FileResponse(ROOT / "index.html")


# ============================================================
# ESTADO DEL SERVIDOR
# ============================================================

@app.get("/api/health")
def health():
    """Verifica que la configuración principal esté disponible."""

    return {
        "status": "ok",
        "configured": bool(
            settings.openai_api_key and settings.drive_folder_id
        ),
    }


# ============================================================
# INVENTARIO DOCUMENTAL
# ============================================================

@app.get("/api/documents")
def documents():
    """Devuelve el inventario de documentos indexados."""

    manifest = SyncService.load_manifest()

    docs = manifest.get("documents", {})

    return {
        "documents": list(docs.values()),
        "indexed": len(docs),
        "vector_store_configured": bool(
            settings.vector_store_id
            or manifest.get("vector_store_id")
        ),
    }


# ============================================================
# CONSULTAS A OPENAI CON O SIN RAG
# ============================================================

@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    """Responde consultas utilizando OpenAI.

    Cuando use_search=True, emplea File Search y
    el Vector Store configurado.
    """

    try:
        if not settings.openai_api_key:
            raise ValueError("Falta OPENAI_API_KEY")

        return AgentService().ask(
            payload.question,
            payload.use_search,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Error al consultar OpenAI. Revisa los registros del servidor.",
        )


# ============================================================
# SINCRONIZACIÓN CON GOOGLE DRIVE
# ============================================================

@app.post(
    "/api/sync",
    dependencies=[Depends(local_only)],
)
def sync():
    """Sincroniza documentos desde Google Drive.

    Disponible únicamente durante la ejecución local.
    """

    try:
        if not settings.openai_api_key:
            raise ValueError("Falta OPENAI_API_KEY")

        return SyncService().sync()

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Error durante la sincronización de documentos.",
        )
