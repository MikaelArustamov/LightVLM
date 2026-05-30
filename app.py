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
from dotenv import load_dotenv
load_dotenv()

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Loading models...")


    from dotenv import load_dotenv
    load_dotenv()


    app.state.llm = LLM()


    app.state.embed = ModelRouter(EMBED_CFG)
    app.state.documents = {}

    # Image generation
    try:
        app.state.generator = ImageGenerator()
        print("[startup] SD 1.5 ready")
    except Exception as e:
        print(f"[startup] Image gen disabled: {e}")
        import traceback
        traceback.print_exc()
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


    text_lower = text.lower().strip()
    trigger_words = ["нарисуй", "сгенерируй", "draw", "generate", "image", "картинка", "picture"]


    is_img = any(word in text_lower for word in trigger_words)

    if is_img and request.app.state.generator:
        try:
            print(f"[Image Route Triggered] Direct intercept for prompt: {text}")


            clean_prompt = text_lower
            for word in trigger_words:
                clean_prompt = clean_prompt.replace(word, "")
            clean_prompt = clean_prompt.strip(",. ")


            img_b64 = await run_in_threadpool(request.app.state.generator.generate_to_base64, clean_prompt)

            if img_b64:

                memory = MemoryStore(session_id, embed_router=embed)
                await run_in_threadpool(memory.add, "user", text)
                await run_in_threadpool(memory.add, "assistant", f"[Generated image: {clean_prompt}]")


                if session_id in request.app.state.documents:
                    del request.app.state.documents[session_id]

                def img_stream():
                    yield f"data: {json.dumps({'token': clean_prompt, 'generated': True, 'image': img_b64})}\n\n"
                    yield "data: [DONE]\n\n"


                return StreamingResponse(img_stream(), media_type="text/event-stream")

        except Exception as e:
            print(f"[Image Error] Stable Diffusion execution failed: {e}")
            is_img = False


    memory = MemoryStore(session_id, embed_router=embed)
    context = await run_in_threadpool(memory.get_context, text, recent_n=5, relevant_n=3)

    system_content = llm.system_prompt() if hasattr(llm, "system_prompt") else "You are a helpful assistant."
    messages = [{"role": "system", "content": system_content}]
    messages.extend(context)


    doc_text = request.app.state.documents.get(session_id)
    if doc_text:
        if text:
            messages.append({"role": "user", "content": f"[Uploaded Document]\n{doc_text}\n\nQuestion: {text}"})
        else:
            messages.append({"role": "user", "content": f"[Uploaded Document]\n{doc_text}"})
    elif text:
        messages.append({"role": "user", "content": text})


    async def event_generator():
        full = ""
        try:

            def get_chunks():
                return list(llm.chat_stream(messages))

            chunks = await run_in_threadpool(get_chunks)

            for chunk in chunks:
                full += chunk
                yield f"data: {json.dumps({'token': chunk})}\n\n"


            await run_in_threadpool(memory.add, "user", text or "[document question]")
            await run_in_threadpool(memory.add, "assistant", full)

        except Exception as e:

            print(f"[LiteLLM Error] Text streaming crashed: {e}")
            yield f"data: {json.dumps({'token': f' [Core Error: {str(e)}]. Проверьте TEXT_MODEL в .env.'})}\n\n"
        finally:
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

    from models.audio import AudioPipeline
    contents = await audio.read()


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

    from models.document_parser import DocumentParser
    llm = request.app.state.llm
    parser = DocumentParser(llm)

    session_id = request.headers.get("X-Session-Id", "default")
    contents = await file.read()


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
