"""
============================================================
 LINE Chatbot x OpenRouter API (FastAPI, deploy บน Vercel)
 แบบ Stateless — ไม่มี Chat History, ไม่มี RAG
 - ทุกคำตอบพ่วงชื่อผู้ใช้ + userId ต่อท้ายเสมอ
 - ถ้า AI ตอบว่าไม่ทราบ (นอกขอบเขต knowledge) จะ push แจ้งแอดมินอัตโนมัติ
============================================================

วิธีติดตั้ง (สรุป)
----------------------------------------------------------
1) คัดลอก .env.example เป็น .env แล้วใส่ค่าจริง:
     LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET,
     OPENROUTER_API_KEY, ADMIN_USER_ID
2) แก้ system prompt ได้ที่ไฟล์ prompt.txt (โหลดตอน cold start)
3) รันทดสอบในเครื่อง:
     pip install -r requirements-dev.txt
     uvicorn api.index:app --reload --port 8000
4) Deploy ขึ้น Vercel:
     vercel            # ครั้งแรก
     vercel --prod     # ขึ้น production
   แล้วนำค่าตัวแปรใน .env ไปตั้งใน Vercel Project Settings > Environment Variables
5) เอา URL ที่ได้ (เช่น https://xxx.vercel.app/webhook) ไปวางใน
   LINE Developers Console > Messaging API > Webhook settings > Webhook URL
   แล้วกด Verify และเปิดสวิตช์ "Use webhook"
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse

load_dotenv()

# ===== path ของ prompt.txt (อยู่ที่ root ของโปรเจกต์ นอกโฟลเดอร์ api/) =====
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt.txt"


def load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"[warn] ไม่พบไฟล์ prompt.txt ที่ {PROMPT_PATH}")
        return ""


# โหลดครั้งเดียวตอน cold start ของฟังก์ชัน
SYSTEM_PROMPT = load_system_prompt()

# ===== ตั้งค่าที่แก้ไขได้ผ่าน .env (มีค่า default ให้) =====
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
# ต้องตรงกับข้อความที่ system prompt สั่งให้ AI ตอบตอนไม่มีความรู้ (ใช้เช็คแบบ exact match)
FALLBACK_MESSAGE = os.getenv("FALLBACK_MESSAGE", "คำถามอยู่นอกขอบเขตที่ Bot จะตอบได้ โปรดรอ Admin ตอบกลับ")

# ===== ค่าลับ ดึงจาก environment เท่านั้น (ตั้งใน .env หรือ Vercel env vars) =====
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")

LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_PROFILE_URL = "https://api.line.me/v2/bot/profile/{user_id}"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI(title="LINE OA x OpenRouter Webhook")


def verify_line_signature(body: bytes, signature: Optional[str]) -> bool:
    """ตรวจลายเซ็น x-line-signature ด้วย LINE_CHANNEL_SECRET (HMAC-SHA256)."""
    if not LINE_CHANNEL_SECRET:
        # ยังไม่ได้ตั้ง secret ไว้ -> ข้ามการตรวจ (ไม่แนะนำสำหรับ production)
        print("[warn] ยังไม่ได้ตั้งค่า LINE_CHANNEL_SECRET จึงข้ามการตรวจลายเซ็น webhook")
        return True
    if not signature:
        return False
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "line-oa-openrouter-webhook"}


@app.post("/webhook")
async def webhook(request: Request, x_line_signature: Optional[str] = Header(default=None)):
    """รับ Webhook จาก LINE ทุกครั้งที่มีข้อความเข้ามา"""
    body = await request.body()

    if not verify_line_signature(body, x_line_signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    events = payload.get("events", [])

    # ใช้ client เดียวร่วมกันต่อ 1 request เพื่อลด overhead การเปิด connection
    async with httpx.AsyncClient(timeout=20.0) as client:
        for event in events:
            if event.get("type") == "message" and event.get("message", {}).get("type") == "text":
                try:
                    await handle_text_message(client, event)
                except Exception as exc:  # กันไม่ให้ 1 event ที่พังทำให้ทั้ง request fail
                    print(f"[error] handle_text_message: {exc}")

    # หมายเหตุ: บน serverless (Vercel) ต้อง await งานทั้งหมดให้เสร็จก่อน return response
    # เพราะ background task ที่ค้างอยู่หลัง response อาจไม่ถูกรันต่อให้จบ
    return PlainTextResponse("OK")


async def handle_text_message(client: httpx.AsyncClient, event: dict) -> None:
    """
    ประมวลผลข้อความ 1 ข้อความ:
      1) ส่งไป OpenRouter, ดึงชื่อผู้ใช้, ตอบกลับ LINE เสมอ
      2) ถ้าคำตอบตรงกับ FALLBACK_MESSAGE (AI ตอบไม่ได้) ให้ push แจ้งแอดมินเพิ่ม
    """
    user_text = event["message"]["text"]
    reply_token = event["replyToken"]
    user_id = event["source"]["userId"]

    ai_reply = await call_openrouter(client, user_text)
    display_name = await get_user_display_name(client, user_id)

    # พ่วง [คุณ: displayName, ID: userId] ต่อท้ายคำตอบทุกครั้ง แล้วตอบผู้ใช้เสมอ
    final_reply = f"{ai_reply}\n\n[คุณ: {display_name}, ID: {user_id}]"
    await reply_to_line(client, reply_token, final_reply)

    # ถ้า AI ตอบว่าไม่ทราบ ให้แจ้งแอดมินแยกต่างหาก (คนละเส้นจากการตอบผู้ใช้ด้านบน)
    if ai_reply.strip() == FALLBACK_MESSAGE:
        await notify_admin(client, display_name, user_id, user_text)


async def call_openrouter(client: httpx.AsyncClient, user_text: str) -> str:
    """เรียก OpenRouter Chat Completions API"""
    if not OPENROUTER_API_KEY:
        print("[error] ยังไม่ได้ตั้งค่า OPENROUTER_API_KEY")
        return "ขออภัยค่ะ ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะคะ"

    payload = {
        "model": OPENROUTER_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    try:
        response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        result = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        print(f"[error] OpenRouter request failed: {exc}")
        return "ขออภัยค่ะ ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะคะ"

    if response.status_code == 200 and result.get("choices"):
        return result["choices"][0]["message"]["content"].strip()

    print(f"[error] OpenRouter error ({response.status_code}): {response.text}")
    return "ขออภัยค่ะ ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะคะ"


async def get_user_display_name(client: httpx.AsyncClient, user_id: str) -> str:
    """
    ดึงชื่อผู้ใช้ (displayName) จาก LINE Get Profile API โดยใช้ userId
    ถ้าดึงไม่สำเร็จ (เช่น ผู้ใช้ยังไม่ยินยอมให้เข้าถึงโปรไฟล์) จะคืนค่า 'ไม่ทราบชื่อ' แทน
    """
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return "ไม่ทราบชื่อ"

    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    try:
        response = await client.get(LINE_PROFILE_URL.format(user_id=user_id), headers=headers)
    except httpx.HTTPError as exc:
        print(f"[error] Get profile request failed: {exc}")
        return "ไม่ทราบชื่อ"

    if response.status_code == 200:
        return response.json().get("displayName", "ไม่ทราบชื่อ")

    print(f"[error] Get profile error: {response.text}")
    return "ไม่ทราบชื่อ"


async def reply_to_line(client: httpx.AsyncClient, reply_token: str, text: str) -> None:
    """ตอบกลับข้อความไปยัง LINE ด้วย Reply API"""
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("[error] ยังไม่ได้ตั้งค่า LINE_CHANNEL_ACCESS_TOKEN จึงตอบกลับไม่ได้")
        return

    # LINE จำกัดความยาวข้อความต่อก้อนไว้ที่ 5000 ตัวอักษร
    safe_text = text if len(text) <= 4900 else text[:4900] + "..."

    payload = {"replyToken": reply_token, "messages": [{"type": "text", "text": safe_text}]}
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}

    try:
        response = await client.post(LINE_REPLY_URL, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"[error] LINE reply error: {response.text}")
    except httpx.HTTPError as exc:
        print(f"[error] LINE reply request failed: {exc}")


async def notify_admin(client: httpx.AsyncClient, display_name: str, user_id: str, user_text: str) -> None:
    """
    แจ้งเตือนแอดมินว่ามีคำถามที่ AI ตอบไม่ได้ (ส่งผ่าน Push API ไปยัง ADMIN_USER_ID)
    แนบชื่อ, userId ของผู้ถาม และคำถามจริงไปด้วย เพื่อให้แอดมินรู้ทันทีว่าต้องตอบอะไร
    """
    if not ADMIN_USER_ID:
        print("[error] ยังไม่ได้ตั้งค่า ADMIN_USER_ID ใน environment จึงแจ้งเตือนไม่ได้")
        return

    message = (
        f'คุณ: {display_name}, ID: {user_id} ถามคำถามที่ AI ตอบไม่ได้\n'
        f'คำถาม: "{user_text}"'
    )
    await push_to_user(client, ADMIN_USER_ID, message)


async def push_to_user(client: httpx.AsyncClient, user_id: str, text: str) -> None:
    """
    ส่งข้อความหาผู้ใช้คนใดก็ได้แบบเจาะจง (Push API)
    ใช้ทั้งตอนแจ้งแอดมินอัตโนมัติ และตอนแอดมินต้องการตอบกลับผู้ใช้ด้วยตนเอง
    ไม่ติดข้อจำกัดเรื่อง reply token ที่หมดอายุใน 1 นาที
    """
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("[error] ยังไม่ได้ตั้งค่า LINE_CHANNEL_ACCESS_TOKEN จึง push ไม่ได้")
        return

    safe_text = text if len(text) <= 4900 else text[:4900] + "..."
    payload = {"to": user_id, "messages": [{"type": "text", "text": safe_text}]}
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}

    try:
        response = await client.post(LINE_PUSH_URL, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"[error] LINE push error: {response.text}")
    except httpx.HTTPError as exc:
        print(f"[error] LINE push request failed: {exc}")
