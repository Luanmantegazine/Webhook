import json
import logging
import hmac
import hashlib
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Header, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import uvicorn

WEBHOOK_SECRET = "12345"
logger = logging.getLogger("generic_webhook")
logging.basicConfig(level=logging.INFO)

app = FastAPI()


def verify_signature(body: bytes, header_sig: str) -> bool:
    if not header_sig or not header_sig.startswith("sha256="):
        return False

    signature = header_sig.split("sha256=")[1]
    expected_signature = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_signature, signature)

async def verify_and_parse_payload(
        request: Request,
        x_vurdere_event: Optional[str] = Header(None),
        x_vurdere_signature_256: Optional[str] = Header(None),
) -> Dict[str, Any]:
    if not x_vurdere_event:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Header X-Vurdere-Event ausente")
    if not x_vurdere_signature_256:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Header X-Vurdere-Signature-256 ausente")
    body_bytes = await request.body()
    if not verify_signature(body_bytes, x_vurdere_signature_256):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Assinatura inválida")
    try:
        payload = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Corpo da requisição não é um JSON válido")

    logger.info("Evento '%s' recebido e validado com sucesso.", x_vurdere_event)
    if isinstance(payload, dict) and "interactionId" in payload:
        logger.info("interactionId=%s", payload["interactionId"])

    return payload


# ─── Endpoints ────────────────────────────────────────────────────────
@app.get("/", response_model=Dict[str, str])
def home():
    return {"message": "Webhook subscriber is running"}


@app.post("/notification/expression", status_code=status.HTTP_200_OK)
async def webhook_expression_handler(
        payload: Dict[str, Any] = Depends(verify_and_parse_payload)
):
    return JSONResponse({"message": "Expression recebido com sucesso!"})


INTERACTION_MESSAGES = {
    "review_update": "Review recebido com sucesso!",
    "question_update": "Question recebido com sucesso!",
    "answer_update": "Answer recebido com sucesso!",
}

@app.post("/notification/interaction", status_code=status.HTTP_200_OK)
async def webhook_interaction_handler(
        x_vurdere_event: str = Header(...),
        payload: Dict[str, Any] = Depends(verify_and_parse_payload)
):
    event = x_vurdere_event.lower()

    if event in INTERACTION_MESSAGES:
        return JSONResponse({"message": INTERACTION_MESSAGES[event]})

    logger.warning("Evento '%s' não suportado neste endpoint, mas foi aceito.", event)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"message": f"Evento {event} aceito, mas sem processamento específico neste endpoint."}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)