"""FastAPI app with lifespan initialization and async-safe execution."""

import os
import json
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool  # ← КРИТИЧЕСКИ ВАЖНО ДЛЯ ПОТОКОВ

from core.config import TEXT_CFG, VISION_CFG, EMBED_CFG
from models.model_router import ModelRouter
from models.llm import LLM
from models.generator import ImageGenerator
from memory import MemoryStore

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading models...")
    app.state.llm = LLM()
    app.state.embed = ModelRouter(EMBED_CFG)
    app.state.documents = {}

    try:
        app.state.generator = ImageGenerator()
        print("[startup] SD 1.5 ready")
    except Exception as e:
        print(f"[startup] Image gen disabled: {e}")
        app.state.generator = None

    print(f"[startup] Text:   {TEXT_CFG['model']} ({TEXT_CFG['quant']})")
    print(f"[startup] Vision: {VISION_CFG['model']} ({VISION_CFG['quant']})")
    print(f"[startup] Embed:  {EMBED_CFG['model']} ({EMBED_CFG['quant']})")

    yield
    print("[shutdown] Cleaning up...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "chat.html")


@app.post("/stream")
async def stream(request: Request):
    data = await request.form()
    text = data.get("text", "")
    session_id = request.headers.get("X-Session-Id", "default")

    llm = request.app.state.llm
    embed = request.app.state.embed

    # Выносим работу с базой данных ChromaDB в пул потоков
    memory = MemoryStore(session_id, embed_router=embed)
    context = await run_in_threadpool(memory.get_context, text, recent_n=5, relevant_n=3)

    # Исправлено: в LLM-классе должен быть метод system_prompt (или замените на строку)
    system_content = llm.system_prompt() if hasattr(llm, "system_prompt") else "You are a helpful assistant."
    messages = [{"role": "system", "content": system_content}]
    messages.extend(context)

    # Attach uploaded document if any
    doc_text = request.app.state.documents.get(session_id)
    if doc_text:
        if text:
            messages.append({"role": "user", "content": f"[Uploaded Document]\n{doc_text}\n\nQuestion: {text}"})
        else:
            messages.append({"role": "user", "content": f"[Uploaded Document]\n{doc_text}"})

    if not doc_text and text:
        messages.append({"role": "user", "content": text})

    # Проверка на генерацию картинок (вынесено в пул потоков)
    is_img, img_prompt = False, ""
    if hasattr(llm, "is_image_request") and text:
        is_img, img_prompt = llm.is_image_request(text)

    if is_img and request.app.state.generator:
        # Тяжелая генерация SD 1.5 уходит в тред-пул
        img_b64 = await run_in_threadpool(request.app.state.generator.generate_to_base64, img_prompt)
        if img_b64:
            await run_in_threadpool(memory.add, "user", text)
            await run_in_threadpool(memory.add, "assistant", f"[Generated image: {img_prompt}]")

            def img_stream():
                yield f"data: {json.dumps({'token': img_prompt, 'generated': True, 'image': img_b64})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(img_stream(), media_type="text/event-stream")

    # Асинхронный генератор для LiteLLM стриминга без блокировки сервера
    async def event_generator():
        full = ""

        # Обертка над синхронным итератором генератора LiteLLM
        def get_chunks():
            return list(llm.chat_stream(messages))

        chunks = await run_in_threadpool(get_chunks)

        for chunk in chunks:
            full += chunk  # Исправлен баг: теперь текст накапливается корректно
            yield f"data: {json.dumps({'token': chunk})}\n\n"

        # Сохранение финального контекста в базу (в тред-пуле)
        await run_in_threadpool(memory.add, "user", text or "[document question]")
        await run_in_threadpool(memory.add, "assistant", full)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/vision")
async def vision(request: Request, image: UploadFile = File(...), prompt: str = Form("")):
    llm = request.app.state.llm
    contents = await image.read()
    b64 = base64.b64encode(contents).decode()

    p = prompt or (llm.vision_prompt() if hasattr(llm, "vision_prompt") else "Analyze this image.")

    # Запрос к Vision-модели LiteLLM уходит в тред-пул
    response = await run_in_threadpool(llm.vision_chat, b64, p)
    return {"response": response}


@app.post("/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...)):
    # Исправлен путь импорта в соответствии с вашим предыдущим кодом (класс AudioPipeline)
    from models.audio import AudioPipeline
    contents = await audio.read()

    # Инициализация и локальный инференс Whisper в тред-пуле
    pipeline = AudioPipeline()
    text = await run_in_threadpool(pipeline.transcribe, contents)
    return {"text": text}


@app.post("/upload")
async def upload(
        request: Request,
        file: UploadFile = File(...),
        mode: str = Form("auto"),
        ask_question: str = Form("")
):
    # Исправлен путь импорта в соответствии с кодом парсера (DocumentParser)
    from models.document_parser import DocumentParser
    llm = request.app.state.llm
    parser = DocumentParser(llm)

    session_id = request.headers.get("X-Session-Id", "default")
    contents = await file.read()

    # Тяжелый парсинг (PDF/OCR) уходит в тред-пул
    result = await run_in_threadpool(parser.parse, contents, file.filename, mode)

    if result.get("type") == "error":
        return result

    doc_text = parser.to_chat_messages(result)
    request.app.state.documents[session_id] = doc_text

    if ask_question:
        system_content = llm.system_prompt() if hasattr(llm, "system_prompt") else "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Document:\n{doc_text}\n\nQuestion: {ask_question}"}
        ]
        answer = await run_in_threadpool(llm.chat, messages)
        return {**result, "question": ask_question, "answer": answer}

    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
