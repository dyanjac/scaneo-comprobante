
import re
import io
import json
import os
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import DonutProcessor, VisionEncoderDecoderModel

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "6")))
torch.set_num_interop_threads(int(os.getenv("TORCH_NUM_INTEROP_THREADS", "2")))

APP_NAME = "donut-service"

# Recomendado para recibos / documentos tipo CORD
DEFAULT_MODEL_ID = os.getenv("DONUT_MODEL_ID", "naver-clova-ix/donut-base-finetuned-cord-v2")

# Prompt por defecto (CORD v2)
DEFAULT_TASK_PROMPT = os.getenv("DONUT_TASK_PROMPT", "<s_cord-v2>")

# En vez de max_length (peligroso), controlamos con max_new_tokens
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DONUT_MAX_NEW_TOKENS", "256"))

app = FastAPI(title=APP_NAME)

processor: Optional[DonutProcessor] = None
model: Optional[VisionEncoderDecoderModel] = None
device: str = "cpu"


def _load_model() -> None:
    global processor, model, device
    if processor is not None and model is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = DonutProcessor.from_pretrained(DEFAULT_MODEL_ID, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(DEFAULT_MODEL_ID)
    model.to(device)
    model.eval()


def _read_images_from_upload(data: bytes, filename: str) -> List[Image.Image]:
    lower = (filename or "").lower()
    if lower.endswith(".pdf"):
        try:
            pages = convert_from_bytes(data, fmt="png")
            return [p.convert("RGB") for p in pages]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Error convirtiendo PDF a imagen. En Windows suele faltar Poppler (pdftoppm) en PATH. "
                    f"Detalle: {e}"
                ),
            )

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return [img]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Archivo no soportado o corrupto: {e}")


def _try_parse_json(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _prompt_to_decoder_start_id(task_prompt: str) -> int:
    """Convierte task_prompt a decoder_start_token_id. Si no es válido, devuelve 400."""
    assert processor is not None

    ids = processor.tokenizer(task_prompt, add_special_tokens=False).input_ids
    if not ids:
        raise HTTPException(
            status_code=400,
            detail=f"task_prompt inválido o vacío: '{task_prompt}'. Prueba con '{DEFAULT_TASK_PROMPT}'.",
        )
    return int(ids[0])


def _get_safe_max_new_tokens(requested: int) -> int:
    """
    Evita reventar el positional embedding del decoder.
    Usamos max_position_embeddings del decoder y dejamos un margen.
    """
    assert model is not None

    # Algunos configs lo exponen aquí:
    dec_cfg = getattr(model.config, "decoder", None)
    max_pos = getattr(dec_cfg, "max_position_embeddings", None)

    # Si no está, usa un default conservador
    if not isinstance(max_pos, int) or max_pos <= 0:
        max_pos = 512  # conservador

    # Como arrancamos con 1 token, limitamos new tokens a (max_pos - margen)
    # margen 8 para evitar offsets internos (mBART usa offset)
    safe_limit = max(32, max_pos - 8)

    return max(1, min(int(requested), safe_limit))


@torch.inference_mode()
def _run_donut_on_image(img: Image.Image, task_prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    assert processor is not None and model is not None

    # --------------------------------------------------
    # 🔍 1️⃣ UPSCALE INTELIGENTE PARA MEJORAR DETALLE
    # --------------------------------------------------
    w, h = img.size
    max_side = max(w, h)

    # Objetivo ideal para documentos: ~1400-1800 px
    TARGET_MAX_SIDE = 1600
    MAX_ALLOWED_SIDE = 2200  # límite duro para no matar CPU/RAM

    if max_side < TARGET_MAX_SIDE:
        scale = TARGET_MAX_SIDE / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Evita que pase del límite duro
        if max(new_w, new_h) > MAX_ALLOWED_SIDE:
            scale = MAX_ALLOWED_SIDE / max_side
            new_w = int(w * scale)
            new_h = int(h * scale)

        img = img.resize((new_w, new_h), Image.BICUBIC)

    # --------------------------------------------------
    # 2️⃣ PROCESAMIENTO NORMAL DONUT
    # --------------------------------------------------
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)

    decoder_start_token_id = _prompt_to_decoder_start_id(task_prompt)

    bad_words_ids = (
        [[processor.tokenizer.unk_token_id]]
        if processor.tokenizer.unk_token_id is not None
        else None
    )

    safe_max_new = _get_safe_max_new_tokens(max_new_tokens)

    outputs = model.generate(
        pixel_values,
        decoder_start_token_id=decoder_start_token_id,
        max_new_tokens=safe_max_new,   # pon 384

        num_beams=1,
        do_sample=False,

        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        length_penalty=0.9,

        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        bad_words_ids=bad_words_ids,
    )

    raw = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    parsed = _try_parse_json(raw)

    fields = _extract_invoice_fields_from_raw(raw)
    items = _extract_items_from_raw(raw)
    return {
        "raw": raw,
        "json": parsed,
        "used_max_new_tokens": safe_max_new,
        "fields": fields,
        "items": items,
    }


def _normalize_amount(s: str) -> Optional[float]:
    """
    Normaliza montos tipo:
      "4,279"
      "1,234.56"
      "123,45"
      "21186"
      "256986"
    
    Heurística optimizada para facturas LATAM:
    - Si tiene punto y coma -> detecta formato europeo vs US.
    - Si tiene solo coma:
        * 2 decimales -> decimal
        * 3 decimales:
            - si total corto (4 dígitos) -> asumir centavos (4279 -> 42.79)
            - si más largo -> miles
    - Si es entero largo sin separadores -> asumir 2 decimales.
    """

    if not s:
        return None

    s = s.strip()

    # deja solo dígitos, coma y punto
    import re
    s2 = re.sub(r"[^0-9\.,]", "", s)
    if not s2:
        return None

    # --------------------------------------------------
    # Caso 1: tiene punto y coma (ej 1,234.56 o 1.234,56)
    # --------------------------------------------------
    if "," in s2 and "." in s2:
        # Si el punto aparece después de la coma → formato europeo (1.234,56)
        if s2.rfind(".") < s2.rfind(","):
            s2 = s2.replace(".", "").replace(",", ".")
        else:
            s2 = s2.replace(",", "")
        try:
            return float(s2)
        except:
            return None

    # --------------------------------------------------
    # Caso 2: solo coma
    # --------------------------------------------------
    if "," in s2 and "." not in s2:
        parts = s2.split(",")

        # Caso decimal típico: 123,45
        if len(parts[-1]) == 2:
            s2 = s2.replace(",", ".")
            try:
                return float(s2)
            except:
                return None

        # Caso ambiguo: 4,279
        if len(parts[-1]) == 3:
            digits = s2.replace(",", "")

            # Si total tiene 4 dígitos (4279) → probablemente 42.79
            if len(digits) == 4:
                try:
                    return float(digits[:-2] + "." + digits[-2:])
                except:
                    return None

            # Si más largo → tratar coma como miles
            try:
                return float(digits)
            except:
                return None

    # --------------------------------------------------
    # Caso 3: solo punto (decimal estándar)
    # --------------------------------------------------
    if "." in s2 and "," not in s2:
        try:
            return float(s2)
        except:
            return None

    # --------------------------------------------------
    # Caso 4: entero largo sin separadores
    # 21186 -> 211.86
    # 256986 -> 2569.86
    # --------------------------------------------------
    if s2.isdigit() and len(s2) >= 4:
        try:
            return float(s2[:-2] + "." + s2[-2:])
        except:
            return None

    # --------------------------------------------------
    # Fallback final
    # --------------------------------------------------
    try:
        return float(s2)
    except:
        return None


def _extract_invoice_fields_from_raw(raw: str) -> Dict[str, Any]:
    """
    Extrae campos comunes de facturas/boletas peruanas desde el RAW de Donut.
    Heurístico (porque CORD no está entrenado para SUNAT).
    """
    out: Dict[str, Any] = {"ruc": None, "series_number": None, "issue_date": None, "document_type": None,
                           "subtotal": None, "tax": None, "currency": None, "notes": []}

    # Documento
    if "FACTURA" in raw.upper():
        out["document_type"] = "FACTURA"
    if "BOLETA" in raw.upper():
        out["document_type"] = "BOLETA"

    # RUC (11 dígitos)
    rucs = re.findall(r"\b(\d{11})\b", raw)
    if rucs:
        # prioriza el primero
        out["ruc"] = rucs[0]

    # Serie-número (ej FFA1-6419)
    m = re.search(r"\b([A-Z0-9]{3,6}-\d{1,10})\b", raw)
    if m:
        out["series_number"] = m.group(1)

    # Fecha YYYY-MM-DD
    d = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", raw)
    if d:
        out["issue_date"] = d.group(1)

    # Subtotal / tax: usa tags CORD si existen
    # <s_subtotal_price> 21186 </s_subtotal_price>
    sm = re.search(r"<s_subtotal_price>\s*([^<]+)\s*</s_subtotal_price>", raw)
    if sm:
        out["subtotal"] = _normalize_amount(sm.group(1))

    tm = re.search(r"<s_tax_price>\s*([^<]+)\s*</s_tax_price>", raw)
    if tm:
        out["tax"] = _normalize_amount(tm.group(1))

    # Moneda: buscar "SOLES" o "PEN"
    if "SOLES" in raw.upper() or "S/." in raw.upper() or "PEN" in raw.upper():
        out["currency"] = "PEN"

    # Notas por “ruido” del modelo
    if any(ch in raw for ch in ["突破", "美國"]) or re.search(r"[가-힣]", raw):
        out["notes"].append("Salida contiene texto no-latino (modelo CORD puede meter ruido en facturas SUNAT).")

    return out

def _extract_items_from_raw(raw: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    # separa por <sep/> (CORD lo usa como separador de línea)
    chunks = raw.split("<sep/>")
    for ch in chunks:
        # tomar valores dentro de tags comunes
        desc = re.findall(r"<s_nm>\s*([^<]+)\s*</s_nm>", ch)
        nums = re.findall(r"<s_num>\s*([^<]+)\s*</s_num>", ch)
        prices = re.findall(r"<s_price>\s*([^<]+)\s*</s_price>", ch)

        # limpia
        desc = [d.strip() for d in desc if d.strip()]
        nums = [n.strip() for n in nums if n.strip()]
        prices = [p.strip() for p in prices if p.strip()]

        # heurística: si parece fila de item -> tiene descripción + al menos 1 número y 1 precio
        if desc and (nums or prices):
            item = {
                "description": desc[0],
                "nums": nums[:3],      # por si hay cantidad/und/códigos
                "prices": prices[:3],  # por si hay unitario/importe
                "raw_chunk": ch.strip()[:250]
            }
            items.append(item)

    return items

@app.on_event("startup")
def startup_event():
    _load_model()


@app.get("/health")
def health():
    dec_cfg = getattr(model.config, "decoder", None) if model is not None else None
    max_pos = getattr(dec_cfg, "max_position_embeddings", None) if dec_cfg is not None else None
    return {
        "service": APP_NAME,
        "model_id": DEFAULT_MODEL_ID,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "default_task_prompt": DEFAULT_TASK_PROMPT,
        "decoder_max_position_embeddings": max_pos,
    }


@app.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    task_prompt: str = Form(DEFAULT_TASK_PROMPT),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
) -> Dict[str, Any]:
    if max_new_tokens <= 0 or max_new_tokens > 4096:
        raise HTTPException(status_code=400, detail="max_new_tokens inválido (1..4096)")

    data = await file.read()
    images = _read_images_from_upload(data, file.filename or "")

    results: List[Dict[str, Any]] = []
    for i, img in enumerate(images, start=1):
        out = _run_donut_on_image(img, task_prompt=task_prompt, max_new_tokens=max_new_tokens)
        results.append({"page": i, **out})

    return {
        "filename": file.filename,
        "pages": len(images),
        "task_prompt": task_prompt,
        "results": results,
    }