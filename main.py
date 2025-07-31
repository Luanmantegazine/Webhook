import json
import logging
import hmac
import hashlib
from typing import Optional, Dict

from fastapi import FastAPI, Request, Header, HTTPException, status
from fastapi.responses import JSONResponse
import uvicorn

WEBHOOK_SECRET = "12345"

logger = logging.getLogger("generic_webhook")
logging.basicConfig(level=logging.INFO)

def verify_signature(body: bytes, header_sig: str) -> bool:
    if not header_sig:
        return False
    if header_sig.startswith("sha256="):
        header_sig = header_sig.split("sha256=")[1]

    expected = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header_sig)

app = FastAPI()

@app.get("/", response_model=Dict[str, str])
def home():
    return {"message": "Webhook subscriber is running"}

# ────────────────────────────────
# APENAS UM endpoint POST
# ────────────────────────────────
EVENT_MESSAGES = {
    "expression_update": "Expression recebido com sucesso!",
    "review_update":     "Review recebido com sucesso!",
    "question_update":   "Question recebido com sucesso!",
    "answer_update":     "Answer recebido com sucesso!",
}

@app.post("/notification")
async def webhook_handler(
    request: Request,
    x_vurdere_event: Optional[str] = Header(None),
    x_vurdere_signature_256: Optional[str] = Header(None),
):
    body_bytes = await request.body()
    logger.info("Recebido evento=%s bytes=%d", x_vurdere_event, len(body_bytes))

    # ─── Validação básica ───────────────────────────────
    if not x_vurdere_event:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Missing X-Webhook-Event")
    if not x_vurdere_signature_256:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Missing X-Webhook-Signature-256")
    if not verify_signature(body_bytes, x_vurdere_signature_256):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    # ─── Corpo JSON ─────────────────────────────────────
    try:
        payload = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Body is not valid JSON")

    # Log opcional para rastreamento
    if isinstance(payload, dict) and "interactionId" in payload:
        logger.info("interactionId=%s", payload["interactionId"])

    event = x_vurdere_event.lower()

    # ─── Responde de acordo com o evento ────────────────
    if event in EVENT_MESSAGES:
        return JSONResponse({"message": EVENT_MESSAGES[event]})

    logger.warning("Evento não suportado: %s", event)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"message": f"Evento {event} aceito sem processamento específico"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
