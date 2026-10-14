
"""Aplicación documental FastAPI con protección del despliegue remoto."""
import base64
import binascii
import secrets
from pathlib import Path
from ipaddress import ip_address

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.models.documento import AskRequest, AskResponse
from app.services.sync_service import SyncService
from app.services.agent_service import AgentService

ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="Archivo TIC IA", version="1.1.0")
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")


@app.middleware("http")
async def protect_remote(request: Request, call_next):
    """Protege todas las rutas en Vercel usando HTTP Basic sobre HTTPS."""
    if not settings.vercel_mode:
        return await call_next(request)

    if not settings.app_access_password:
        return Response(
            "La autenticacion del servidor no esta configurada.",
            status_code=503,
        )

    authorization = request.headers.get("authorization", "")
    authenticated = False

    if authorization.startswith("Basic "):
        try:
            encoded = authorization[6:].strip()
            decoded = base64.b64decode(
                encoded, validate=True
            ).decode("utf-8")
            username, password = decoded.split(":", 1)

            authenticated = (
                secrets.compare_digest(username, "profesor")
                and secrets.compare_digest(
                    password, settings.app_access_password
                )
            )
        except (ValueError, UnicodeDecodeError, binascii.Error):
            authenticated = False

    if not authenticated:
        return Response(
            "Se requiere autenticacion.",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Archivo TIC IA"'},
        )

    return await call_next(request)


def local_only(request: Request):
    """La sincronización original solo se permite desde localhost."""
    if settings.vercel_mode:
        raise HTTPException(
            status_code=403,
            detail="Sincronización remota no habilitada."
        )

    host = request.client.host if request.client else ""
    try:
        if not ip_address(host).is_loopback:
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Sincronización disponible solo desde localhost"
        )

    key = settings.local_admin_key
    if key and request.headers.get("X-Admin-Key") != key:
        raise HTTPException(
            status_code=401,
            detail="Clave administrativa incorrecta"
        )


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "configured": bool(
            settings.openai_api_key and settings.drive_folder_id
        ),
    }


@app.get("/api/documents")
def documents():
    doc = SyncService.load_manifest()
    return {
        "documents": list(doc.get("documents", {}).values()),
        "indexed": len(doc.get("documents", {})),
        "vector_store_configured": bool(
            settings.vector_store_id or doc.get("vector_store_id")
        ),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        if not settings.openai_api_key:
            raise ValueError("Falta OPENAI_API_KEY")

        return AgentService().ask(
            payload.question, payload.use_search
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Error al consultar OpenAI."
        )


@app.post("/api/sync", dependencies=[Depends(local_only)])
def sync():
    try:
        if not settings.openai_api_key:
            raise ValueError("Falta OPENAI_API_KEY")

        return SyncService().sync()

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Error durante la sincronizacion."
        )
