"""
==================================================================
 LINE Chatbot x OpenRouter — เวอร์ชันอธิบายละเอียด (สำหรับมือใหม่)
==================================================================

ไฟล์นี้คือ "เว็บเซิร์ฟเวอร์" หนึ่งตัว ที่คอยนั่งรอรับข้อความจาก LINE
แล้วเอาไปถาม AI (ผ่าน OpenRouter) จากนั้นส่งคำตอบกลับไปหาผู้ใช้ทาง LINE

ภาพรวมทีละขั้นตอน เมื่อมีคนพิมพ์คุยกับบอทใน LINE:
  1. LINE จะยิง HTTP POST request มาที่ endpoint "/webhook" ของเรา พร้อมข้อมูลแบบ JSON
  2. เราตรวจก่อนว่า request นี้มาจาก LINE จริง ไม่ใช่คนปลอมแปลงยิงเข้ามาเอง (ตรวจลายเซ็น)
  3. เราแกะ JSON เพื่อดึง "ข้อความที่ผู้ใช้พิมพ์มา" ออกมา
  4. เราส่งข้อความนั้นไปถาม OpenRouter พร้อมกับ "system prompt" (คำสั่ง/บุคลิกของ AI จาก prompt.txt)
  5. เราเอาคำตอบที่ได้ ส่งกลับไปหาผู้ใช้ทาง LINE Reply API
  6. ถ้า AI ตอบว่า "นอกขอบเขต/ไม่ทราบ" เราจะส่งข้อความแจ้งเตือนแยกไปหาแอดมินอีกที
     (ข้อความที่ส่งกลับผู้ใช้ กับข้อความที่ส่งแจ้งแอดมิน เป็นคนละข้อความ คนละ API call กัน)

คำศัพท์ที่เจอบ่อยในไฟล์นี้ (อธิบายครั้งเดียวตรงนี้ จะได้ไม่ต้องอธิบายซ้ำทุกจุด):
  - "async def"  = ประกาศฟังก์ชันที่ "รอ" งานบางอย่างได้โดยไม่ค้างทั้งโปรแกรม
  - "await"      = แปลว่า "รอจนกว่าจะได้ผลลัพธ์ตรงนี้ก่อน แล้วค่อยไปทำบรรทัดถัดไป"
                   ใช้ตอนต้องรอเน็ตเวิร์ก เช่น เรียก API ของ LINE หรือ OpenRouter
  - "-> str"     = บอกว่าฟังก์ชันนี้จะ "คืนค่า" (return) เป็น string
                   ": httpx.AsyncClient" หลังชื่อพารามิเตอร์ = บอกว่าพารามิเตอร์ตัวนั้นควรเป็นชนิดข้อมูลอะไร
                   สิ่งเหล่านี้เป็นแค่ "คำอธิบายชนิดข้อมูล" ช่วยให้อ่านง่ายขึ้น ไม่ได้บังคับให้โปรแกรมทำงานต่าง
  - dict         = ข้อมูลแบบ key-value เหมือน {"ชื่อ": "ค่า"} ส่วน JSON ที่ส่งไปมาระหว่างเซิร์ฟเวอร์
                   ก็คือ dict ที่ถูกแปลงเป็น string เพื่อส่งผ่านเน็ตเวิร์ก

ข้อสังเกตสำคัญ: ในไฟล์นี้ทุกที่ที่ต้องอ่านค่าจาก HTTP header (เช่น ลายเซ็นจาก LINE)
เราจะเขียนโค้ดดึงค่าออกมาเองแบบตรงไปตรงมา (request.headers.get(...))
แทนที่จะใช้ความสามารถ "auto-inject พารามิเตอร์" ของ FastAPI ที่ทำให้ค่าโผล่มาในพารามิเตอร์ของฟังก์ชัน
โดยที่เราไม่เห็นว่ามันไปดึงมาจากไหน — เขียนแบบตรงไปตรงมาจะได้เห็นชัดว่าค่าต่างๆ มาจากที่ใดบ้าง
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

# โหลดค่าจากไฟล์ .env เข้ามาเป็น environment variable ของโปรแกรม (ใช้ตอนรันในเครื่องตัวเอง)
# เวลา deploy จริงบน Vercel, Vercel จะอ่านค่าจาก "Environment Variables" ที่ตั้งไว้ใน dashboard แทน
# ไม่ได้อ่านจากไฟล์ .env (เพราะเราไม่ได้ upload ไฟล์ .env ขึ้นไปด้วย ตาม .gitignore)
load_dotenv()


# ==================================================================
# ส่วนที่ 1: อ่านค่า config ต่างๆ จาก environment variable
# ==================================================================
# os.getenv("ชื่อตัวแปร") = ไปหาค่า environment variable ชื่อนั้น ถ้าไม่เจอจะได้ None
# os.getenv("ชื่อตัวแปร", "ค่า default") = ถ้าไม่เจอ ให้ใช้ "ค่า default" แทน

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# os.getenv คืนค่าออกมาเป็น string เสมอ ต่อให้ค่าจริงๆ ควรเป็นตัวเลข
# เราจึงต้องแปลงเป็นตัวเลขเองด้วย int(...)
MAX_TOKENS_AS_TEXT = os.getenv("MAX_TOKENS", "512")
MAX_TOKENS = int(MAX_TOKENS_AS_TEXT)

# ข้อความนี้ต้องตรงกับที่เราสั่งไว้ใน prompt.txt ว่า "ถ้าไม่รู้ ให้ตอบว่าอะไร" แบบตัวอักษรเป๊ะๆ
# เพราะเราจะเอาคำตอบของ AI มาเทียบ (==) กับข้อความนี้ เพื่อรู้ว่าต้องแจ้งแอดมินหรือเปล่า
FALLBACK_MESSAGE = os.getenv(
    "FALLBACK_MESSAGE",
    "คำถามอยู่นอกขอบเขตที่ Bot จะตอบได้ โปรดรอ Admin ตอบกลับ",
)

# ที่อยู่ (URL) ของ API ต่างๆ ที่เราจะเรียกใช้ เขียนแยกไว้เป็นตัวแปรชื่อสื่อความหมาย
# {user_id} ใน LINE_GET_PROFILE_URL_TEMPLATE เป็นแค่ "ช่องว่าง" ที่จะถูกแทนที่ด้วยค่าจริงทีหลัง
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_GET_PROFILE_URL_TEMPLATE = "https://api.line.me/v2/bot/profile/{user_id}"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


# ==================================================================
# ส่วนที่ 2: โหลด system prompt จากไฟล์ prompt.txt มาเก็บไว้ในตัวแปร
# ==================================================================
# Path(__file__)      = ที่อยู่ของไฟล์ index.py นี้เอง
# .resolve()          = แปลงให้เป็น path แบบเต็ม (absolute path) ไม่ว่าจะรันจากที่ไหน
# .parent             = โฟลเดอร์ที่ไฟล์นี้อยู่ (คือโฟลเดอร์ api/)
# .parent (อีกที)     = โฟลเดอร์แม่ของ api/ นั่นคือ root ของโปรเจกต์ ที่ prompt.txt วางอยู่จริง
CURRENT_FILE_PATH = Path(__file__).resolve()
API_FOLDER_PATH = CURRENT_FILE_PATH.parent
PROJECT_ROOT_PATH = API_FOLDER_PATH.parent
PROMPT_FILE_PATH = PROJECT_ROOT_PATH / "prompt.txt"


def read_system_prompt_from_file(file_path: Path) -> str:
    """
    อ่านไฟล์ prompt.txt ทั้งไฟล์ออกมาเป็น string เดียว

    พารามิเตอร์:
      file_path = ที่อยู่ของไฟล์ที่จะอ่าน (เป็น object ชนิด Path ไม่ใช่ string ธรรมดา)

    ค่าที่คืนกลับ (return):
      ข้อความในไฟล์ทั้งหมด (string) — ถ้าหาไฟล์ไม่เจอ จะคืนค่าเป็น string ว่าง "" แทน
    """
    try:
        file_content_text = file_path.read_text(encoding="utf-8")
        return file_content_text.strip()  # .strip() ตัดช่องว่าง/บรรทัดว่างที่หัวและท้ายไฟล์ออก
    except FileNotFoundError:
        print("[warn] หาไฟล์ prompt.txt ไม่เจอที่ตำแหน่ง:", file_path)
        return ""


# เรียกฟังก์ชันด้านบนแค่ครั้งเดียวตอนเซิร์ฟเวอร์เริ่มทำงาน (ไม่ใช่ทุกครั้งที่มีข้อความเข้ามา)
# แล้วเก็บผลลัพธ์ไว้ในตัวแปรระดับบนสุดของไฟล์นี้ ให้ทุกฟังก์ชันด้านล่างเรียกใช้ได้
SYSTEM_PROMPT_TEXT = read_system_prompt_from_file(PROMPT_FILE_PATH)


# ==================================================================
# ส่วนที่ 3: สร้างเว็บแอป FastAPI
# ==================================================================
# FastAPI() สร้าง "เว็บแอปพลิเคชัน" ขึ้นมา 1 ตัว เก็บไว้ในตัวแปรชื่อ app
# Vercel จะมองหาตัวแปรชื่อ app ในไฟล์นี้โดยเฉพาะ แล้วเอาไปรันเป็นเว็บเซิร์ฟเวอร์ให้อัตโนมัติ
app = FastAPI(title="LINE OA x OpenRouter Webhook (ฉบับอธิบายละเอียด)")


# ==================================================================
# ส่วนที่ 4: ฟังก์ชันตรวจสอบว่า request มาจาก LINE จริง ไม่ถูกปลอมแปลง
# ==================================================================
def is_request_signature_valid(request_body_bytes: bytes, signature_from_header: str | None) -> bool:
    """
    ตรวจลายเซ็นดิจิทัลที่ LINE แนบมากับทุก request (อยู่ใน HTTP header ชื่อ "x-line-signature")

    หลักการทำงาน:
      LINE เข้ารหัสเนื้อหาของ request ด้วยวิธี HMAC-SHA256 โดยใช้ "channel secret"
      ที่เรารู้ค่าเหมือนกันเป็นคีย์ลับ แล้วส่งผลลัพธ์มาใน header
      ฝั่งเราก็เอา channel secret ของเรา มาเข้ารหัสเนื้อหา request แบบเดียวกัน
      แล้วเทียบผลลัพธ์ว่าตรงกับที่ LINE ส่งมาไหม ถ้าตรง แปลว่า request นี้มาจาก LINE จริง

    พารามิเตอร์:
      request_body_bytes    = เนื้อหาดิบของ request ทั้งก้อน (ชนิด bytes ไม่ใช่ string หรือ dict)
      signature_from_header = ค่าที่ LINE แนบมาใน header "x-line-signature"
                               (เป็น None ได้ ถ้า request นั้นไม่มี header นี้)

    ค่าที่คืนกลับ:
      True  = ลายเซ็นถูกต้อง (หรือเรายังไม่ได้ตั้งค่า secret ไว้เลย จึงข้ามการตรวจ)
      False = ลายเซ็นไม่ถูกต้อง หรือไม่มีลายเซ็นมาเลย
    """
    if not LINE_CHANNEL_SECRET:
        # ยังไม่ได้ตั้งค่า LINE_CHANNEL_SECRET ไว้ใน .env -> ข้ามการตรวจไปก่อน
        # (ทำแบบนี้เพื่อให้ตอนหัดทำครั้งแรกยังทดสอบได้โดยไม่ต้องมี secret ก่อน
        #  แต่ตอนใช้งานจริงกับผู้ใช้จริง ควรตั้งค่านี้เสมอ เพื่อความปลอดภัย)
        print("[warn] ยังไม่ได้ตั้งค่า LINE_CHANNEL_SECRET จึงข้ามการตรวจลายเซ็น")
        return True

    if signature_from_header is None:
        return False

    # แปลง channel secret จาก string ให้เป็น bytes ก่อน เพราะฟังก์ชัน hmac ต้องการ bytes
    channel_secret_as_bytes = LINE_CHANNEL_SECRET.encode("utf-8")

    # คำนวณค่า HMAC-SHA256 ของเนื้อหา request โดยใช้ channel secret เป็นคีย์เข้ารหัส
    hmac_calculator = hmac.new(channel_secret_as_bytes, request_body_bytes, hashlib.sha256)
    our_signature_as_bytes = hmac_calculator.digest()  # ผลลัพธ์ดิบเป็น bytes

    # LINE ส่งลายเซ็นมาเป็น base64 string เราจึงต้องแปลงผลลัพธ์ของเราให้เป็น base64 string ด้วย
    # ถึงจะเอามาเทียบกับของ LINE ได้ตรงๆ
    our_signature_as_base64_text = base64.b64encode(our_signature_as_bytes).decode("utf-8")

    # hmac.compare_digest ใช้เทียบ string สองตัวแบบ "ปลอดภัย" (กันการโจมตีแบบจับเวลา)
    # ดีกว่าการเทียบด้วย == ธรรมดา เวลาที่ข้อมูลเกี่ยวข้องกับความปลอดภัยแบบนี้
    return hmac.compare_digest(our_signature_as_base64_text, signature_from_header)


# ==================================================================
# ส่วนที่ 5: endpoint (เส้นทาง URL) ที่เปิดให้เรียกเข้ามาได้
# ==================================================================
@app.get("/")
async def health_check():
    """
    Endpoint นี้ไว้เช็คเฉยๆ ว่าเซิร์ฟเวอร์ยังทำงานอยู่ไหม (เข้าผ่าน browser ธรรมดาได้เลย)
    ไม่เกี่ยวข้องกับ LINE โดยตรง แค่ไว้ทดสอบว่า deploy สำเร็จหรือยัง
    """
    return {"status": "ok", "service": "line-oa-openrouter-webhook"}


@app.post("/webhook")
async def receive_line_webhook(request: Request):
    """
    Endpoint หลัก — LINE จะยิง POST request มาที่นี่ทุกครั้งที่มีคนคุยกับบอทของเรา

    พารามิเตอร์:
      request = object ที่ FastAPI ส่งเข้ามาให้ แทน "คำขอ HTTP ทั้งก้อน" ที่เพิ่งเข้ามา
                 เราจะดึงข้อมูลที่ต้องการออกมาจาก object นี้เองทีละอย่าง (ไม่ใช้ทางลัดของ FastAPI
                 ที่จะเนียนแทรกค่าเข้าพารามิเตอร์ให้อัตโนมัติ เพื่อให้เห็นชัดว่าแต่ละค่ามาจากไหน)
    """

    # ขั้นที่ 1: อ่านเนื้อหาดิบของ request ออกมาเป็น bytes (ยังไม่แปลงเป็น JSON)
    #           ต้องอ่านเป็น bytes ดิบไว้ก่อน เพราะการตรวจลายเซ็นต้องใช้ข้อมูลตรงตัวเป๊ะๆ ตามที่ LINE ส่งมา
    #           ถ้าแปลงเป็น JSON แล้วค่อยแปลงกลับ ตัวอักษรอาจไม่ตรงเป๊ะแล้วตรวจลายเซ็นไม่ผ่าน
    raw_body_bytes = await request.body()

    # ขั้นที่ 2: ดึงค่า header ชื่อ "x-line-signature" ออกมาด้วยตัวเอง
    #           request.headers ทำงานคล้าย dictionary หนึ่งใบ
    #           .get("x-line-signature") = "ถ้ามี key นี้ให้เอาค่ามา ถ้าไม่มีให้คืนค่า None แทนการ error"
    signature_header_value = request.headers.get("x-line-signature")

    # ขั้นที่ 3: เอา bytes กับ signature ไปตรวจสอบด้วยฟังก์ชันที่เขียนไว้ในส่วนที่ 4
    signature_ok = is_request_signature_valid(raw_body_bytes, signature_header_value)
    if not signature_ok:
        # HTTPException คือวิธีบอก FastAPI ว่า "ให้ตอบกลับเป็นสถานะ error นี้ พร้อมข้อความนี้"
        raise HTTPException(status_code=403, detail="ลายเซ็นไม่ถูกต้อง (signature invalid)")

    # ขั้นที่ 4: แปลง bytes ให้เป็น dict ของ python ด้วย json.loads
    try:
        webhook_payload_dict = json.loads(raw_body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="รูปแบบ JSON ไม่ถูกต้อง")

    # โครงสร้างข้อมูลที่ LINE ส่งมา หน้าตาประมาณนี้ (ตัวอย่าง):
    #   {
    #     "events": [
    #        {
    #          "type": "message",
    #          "replyToken": "xxxxxxxx",
    #          "message": {"type": "text", "text": "สวัสดีครับ"},
    #          "source": {"userId": "Uxxxxxxxx"}
    #        }
    #        (อาจมีมากกว่า 1 event ใน request เดียว)
    #     ]
    #   }
    # .get("events", []) = ถ้าไม่มี key "events" เลย ให้ใช้ list ว่างแทน กันโปรแกรม error
    list_of_events = webhook_payload_dict.get("events", [])

    # เปิดการเชื่อมต่อ HTTP ไว้ 1 ชุด (http_client) แล้วใช้ยิง request หลายๆ ครั้งร่วมกัน
    # ประหยัดเวลากว่าเปิดการเชื่อมต่อใหม่ทุกครั้งที่จะคุยกับ LINE หรือ OpenRouter
    async with httpx.AsyncClient(timeout=20.0) as http_client:
        for one_event in list_of_events:
            event_type = one_event.get("type")
            message_info_dict = one_event.get("message", {})
            message_type = message_info_dict.get("type")

            # เราสนใจแค่ event ที่เป็น "ข้อความตัวอักษร" เท่านั้น
            # (ไม่สนใจสติกเกอร์ รูปภาพ วิดีโอ ฯลฯ ในตัวอย่างนี้)
            is_text_message = event_type == "message" and message_type == "text"
            if is_text_message:
                try:
                    await process_one_text_message(http_client, one_event)
                except Exception as error:
                    # ถ้า event นี้พังระหว่างทาง อย่าให้ทั้ง request ล้มไปด้วย
                    # แค่ log ข้อผิดพลาดไว้ แล้วไปทำ event ถัดไปต่อ
                    print("[error] ประมวลผลข้อความล้มเหลว:", error)

    # LINE ต้องการแค่คำตอบ HTTP status code 200 กลับไปเฉยๆ ไม่ได้สนใจเนื้อหาที่ตอบ
    # หมายเหตุ: เพราะรันบน serverless (Vercel) เราต้อง await ทุกงานให้เสร็จก่อน return ตรงนี้
    # เพราะพองาน "หลัง return response" อาจไม่ถูกรันต่อจนจบ (เครื่องอาจถูกพักไปแล้ว)
    return PlainTextResponse("OK")


# ==================================================================
# ส่วนที่ 6: ฟังก์ชันจัดการข้อความ 1 ข้อความ (แกนกลางของ logic ทั้งหมด)
# ==================================================================
async def process_one_text_message(http_client: httpx.AsyncClient, event: dict) -> None:
    """
    จัดการข้อความ 1 ข้อความจากผู้ใช้ ทำตามลำดับนี้:
      ก) ส่งคำถามไปถาม OpenRouter แล้วเอาคำตอบกลับไปตอบผู้ใช้ทาง LINE
      ข) ถ้าคำตอบตรงกับข้อความ fallback เป๊ะๆ (แปลว่า AI ไม่รู้เรื่องนี้)
         ให้ส่งข้อความแจ้งเตือนแยกไปหาแอดมินเพิ่มอีกที (คนละข้อความกับที่ตอบผู้ใช้)

    พารามิเตอร์:
      http_client = ตัวเชื่อมต่อ HTTP ที่เปิดไว้ใช้ร่วมกัน (ส่งต่อมาจากฟังก์ชันที่เรียกเรา)
      event       = dict ข้อมูลของ event เดียวที่ได้มาจาก JSON ของ LINE (ดูตัวอย่างโครงสร้างด้านบน)
    """
    # ดึงค่าจาก dict ที่ซ้อนกันหลายชั้น มาเก็บไว้ในตัวแปรชื่อสื่อความหมาย จะได้อ่านง่ายขึ้น
    user_message_text = event["message"]["text"]
    line_reply_token = event["replyToken"]
    line_user_id = event["source"]["userId"]

    # ก) ถามคำถามไปที่ AI แล้วรอคำตอบ (await = รอจนกว่าจะได้คำตอบก่อนไปบรรทัดถัดไป)
    ai_answer_text = await ask_openrouter(http_client, user_message_text)

    # ข) ส่งคำตอบของ AI กลับไปหาผู้ใช้ตรงๆ
    #    หมายเหตุ: ผู้ใช้จะเห็นแค่คำตอบของ AI เท่านั้น "ไม่มี" ชื่อหรือ userId ของตัวเองแนบไปด้วย
    await send_line_reply(http_client, line_reply_token, ai_answer_text)

    # ค) เช็คว่าคำตอบของ AI ตรงกับข้อความ fallback เป๊ะๆ หรือไม่
    #    .strip() คือตัดช่องว่าง/บรรทัดว่างหัวท้ายออกก่อนเทียบ กันกรณี AI ตอบมามีช่องว่างเกินมาแถม
    is_out_of_scope_answer = ai_answer_text.strip() == FALLBACK_MESSAGE

    if is_out_of_scope_answer:
        # ต้องรู้ "ชื่อที่แสดง" ของผู้ใช้ก่อน ถึงจะเอาไปใส่ในข้อความแจ้งเตือนแอดมินได้
        # เราจะเรียก API นี้เฉพาะตอนจำเป็น (ตอนต้องแจ้งแอดมิน) เท่านั้น ไม่เรียกทุกข้อความ
        user_display_name = await get_line_user_display_name(http_client, line_user_id)

        # ส่งข้อความแจ้งเตือนไปหาแอดมิน — ข้อความนี้จะมีชื่อ + userId ของผู้ถามอยู่ด้วย
        # แต่ข้อความนี้จะไม่ถูกส่งไปหาผู้ใช้ที่ถามเลย มันไปหาแอดมินเท่านั้น (คนละ API call กับข้อ ข)
        await send_admin_alert(http_client, user_display_name, line_user_id, user_message_text)


# ==================================================================
# ส่วนที่ 7: ฟังก์ชันเรียก OpenRouter (ตัว AI)
# ==================================================================
async def ask_openrouter(http_client: httpx.AsyncClient, question_text: str) -> str:
    """
    ส่งคำถามไปที่ OpenRouter Chat Completions API แล้วคืนคำตอบกลับมาเป็น string

    พารามิเตอร์:
      http_client   = ตัวเชื่อมต่อ HTTP ที่ใช้ยิง request ออกไป
      question_text = คำถามของผู้ใช้ (string ธรรมดา)

    ค่าที่คืนกลับ:
      คำตอบของ AI (string) หรือข้อความแจ้งขัดข้องชั่วคราว ถ้าเรียก API ไม่สำเร็จ
    """
    if not OPENROUTER_API_KEY:
        print("[error] ยังไม่ได้ตั้งค่า OPENROUTER_API_KEY")
        return "ขออภัยค่ะ ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะคะ"

    # สร้าง dict ที่เป็น "เนื้อหา request" ตามรูปแบบที่ OpenRouter กำหนดไว้
    #   role "system" = คำสั่ง/บุคลิกของ AI (มาจากไฟล์ prompt.txt ที่โหลดไว้ตอนต้นไฟล์)
    #   role "user"   = คำถามจริงของผู้ใช้ที่พิมพ์เข้ามา
    request_body_dict = {
        "model": OPENROUTER_MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_TEXT},
            {"role": "user", "content": question_text},
        ],
    }

    # ต้องแนบ API key ไปใน HTTP header ชื่อ "Authorization" ตามรูปแบบ "Bearer <api key>"
    # (เป็นมาตรฐานที่ OpenRouter กำหนด ไม่ใช่สิ่งที่เราคิดขึ้นเอง)
    request_headers_dict = {
        "Authorization": "Bearer " + OPENROUTER_API_KEY,
    }

    try:
        # http_client.post(url, json=..., headers=...) คือการยิง HTTP POST request ออกไป
        #   ตัวแรก (OPENROUTER_CHAT_URL) = จะยิงไปที่ไหน
        #   json=request_body_dict       = เนื้อหาที่จะส่งไป (httpx จะแปลง dict -> ข้อความ JSON ให้เราเอง)
        #   headers=request_headers_dict = header ที่จะแนบไปด้วย (ในที่นี้คือ API key)
        http_response = await http_client.post(
            OPENROUTER_CHAT_URL,
            json=request_body_dict,
            headers=request_headers_dict,
        )
        # http_response.json() = แปลงเนื้อหา JSON ที่ตอบกลับมา ให้กลายเป็น dict ของ python
        response_body_dict = http_response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        print("[error] เรียก OpenRouter ไม่สำเร็จ:", error)
        return "ขออภัยค่ะ ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะคะ"

    request_succeeded = http_response.status_code == 200
    response_has_choices = bool(response_body_dict.get("choices"))

    if request_succeeded and response_has_choices:
        # โครงสร้างคำตอบของ OpenRouter หน้าตาประมาณนี้ (ตัวอย่าง):
        #   {"choices": [ {"message": {"role": "assistant", "content": "คำตอบอยู่ตรงนี้"}} ]}
        first_choice_dict = response_body_dict["choices"][0]
        message_dict = first_choice_dict["message"]
        answer_text = message_dict["content"]
        return answer_text.strip()

    print(
        f"[error] OpenRouter ตอบกลับผิดปกติ (status={http_response.status_code}):",
        http_response.text,
    )
    return "ขออภัยค่ะ ระบบขัดข้องชั่วคราว ลองใหม่อีกครั้งนะคะ"


# ==================================================================
# ส่วนที่ 8: ฟังก์ชันเรียก LINE API (ดึงโปรไฟล์ / ตอบกลับ / ส่งหาแอดมิน)
# ==================================================================
async def get_line_user_display_name(http_client: httpx.AsyncClient, user_id: str) -> str:
    """
    เรียก LINE Get Profile API เพื่อขอ "ชื่อที่แสดง" (displayName) ของผู้ใช้ตาม user_id
    ถ้าเรียกไม่สำเร็จ (เช่น ผู้ใช้ไม่ได้ยินยอมให้เข้าถึงโปรไฟล์) จะคืนค่า "ไม่ทราบชื่อ" แทน

    ฟังก์ชันนี้ถูกเรียกใช้เฉพาะตอนที่ต้องแจ้งเตือนแอดมินเท่านั้น (ดูส่วนที่ 6) ไม่ได้เรียกทุกข้อความ
    """
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return "ไม่ทราบชื่อ"

    # .format(user_id=user_id) แทนที่ {user_id} ใน template URL ด้วยค่าจริงที่ส่งเข้ามา
    profile_api_url = LINE_GET_PROFILE_URL_TEMPLATE.format(user_id=user_id)
    request_headers_dict = {"Authorization": "Bearer " + LINE_CHANNEL_ACCESS_TOKEN}

    try:
        http_response = await http_client.get(profile_api_url, headers=request_headers_dict)
    except httpx.HTTPError as error:
        print("[error] ดึงโปรไฟล์ผู้ใช้ไม่สำเร็จ:", error)
        return "ไม่ทราบชื่อ"

    if http_response.status_code == 200:
        profile_dict = http_response.json()
        return profile_dict.get("displayName", "ไม่ทราบชื่อ")

    print(
        f"[error] LINE ตอบกลับผิดปกติตอนขอโปรไฟล์ (status={http_response.status_code}):",
        http_response.text,
    )
    return "ไม่ทราบชื่อ"


async def send_line_reply(http_client: httpx.AsyncClient, reply_token: str, text_to_send: str) -> None:
    """
    ตอบกลับผู้ใช้ทาง LINE Reply API — นี่คือข้อความที่ "ผู้ใช้ที่ถามจะเห็น" โดยตรง
    จึงส่งแค่คำตอบของ AI ไปตรงๆ ไม่แนบชื่อหรือ userId ของผู้ใช้ปนไปด้วย

    ข้อควรรู้: reply_token ใช้ได้แค่ครั้งเดียว และหมดอายุภายใน 1 นาที
    ต้องใช้ทันทีหลังได้รับ event มา ใช้ซ้ำหรือใช้ช้าเกินไปจะไม่สำเร็จ
    """
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("[error] ยังไม่ได้ตั้งค่า LINE_CHANNEL_ACCESS_TOKEN จึงตอบกลับไม่ได้")
        return

    # LINE จำกัดความยาวข้อความ 1 ก้อนไว้ที่ 5000 ตัวอักษร เราตัดไว้ก่อนที่ 4900 ตัวอักษร กันชนขอบพอดี
    if len(text_to_send) > 4900:
        text_to_send = text_to_send[:4900] + "..."

    request_body_dict = {
        "replyToken": reply_token,
        "messages": [
            {"type": "text", "text": text_to_send},
        ],
    }
    request_headers_dict = {"Authorization": "Bearer " + LINE_CHANNEL_ACCESS_TOKEN}

    http_response = await http_client.post(LINE_REPLY_URL, json=request_body_dict, headers=request_headers_dict)
    if http_response.status_code != 200:
        print(f"[error] ตอบกลับ LINE ไม่สำเร็จ (status={http_response.status_code}):", http_response.text)


async def send_admin_alert(
    http_client: httpx.AsyncClient,
    user_display_name: str,
    user_id: str,
    original_question_text: str,
) -> None:
    """
    ส่งข้อความแจ้งเตือนไปหาแอดมิน (ผ่าน LINE Push API) ว่ามีคำถามที่ AI ตอบไม่ได้

    ข้อความที่ส่งไปหาแอดมินนี้ "มี" ชื่อ + userId ของผู้ถามอยู่ด้วย เพื่อให้แอดมินรู้ว่า
    ต้องเข้าไปตอบใคร แต่ข้อความนี้ไปถึงแค่แอดมินคนเดียว (ตาม ADMIN_USER_ID) ไม่ได้ส่งไปหาผู้ถามเลย
    """
    if not ADMIN_USER_ID:
        print("[error] ยังไม่ได้ตั้งค่า ADMIN_USER_ID จึงแจ้งเตือนไม่ได้")
        return

    alert_text = (
        "คุณ: " + user_display_name + ", ID: " + user_id + " ถามคำถามที่ AI ตอบไม่ได้\n"
        "คำถาม: \"" + original_question_text + "\""
    )

    await send_line_push_message(http_client, ADMIN_USER_ID, alert_text)


async def send_line_push_message(http_client: httpx.AsyncClient, target_user_id: str, text_to_send: str) -> None:
    """
    ส่งข้อความหาผู้ใช้คนใดคนหนึ่งโดยตรง ด้วย LINE Push API

    ต่างจาก Reply API ตรงที่ Push API ไม่ต้องใช้ reply_token และไม่มีเวลาหมดอายุ
    แค่ระบุ user_id ที่ต้องการส่งหาได้เลย (ในไฟล์นี้ใช้ส่งหาแอดมินเท่านั้น)
    """
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("[error] ยังไม่ได้ตั้งค่า LINE_CHANNEL_ACCESS_TOKEN จึง push ไม่ได้")
        return

    if len(text_to_send) > 4900:
        text_to_send = text_to_send[:4900] + "..."

    request_body_dict = {
        "to": target_user_id,
        "messages": [
            {"type": "text", "text": text_to_send},
        ],
    }
    request_headers_dict = {"Authorization": "Bearer " + LINE_CHANNEL_ACCESS_TOKEN}

    http_response = await http_client.post(LINE_PUSH_URL, json=request_body_dict, headers=request_headers_dict)
    if http_response.status_code != 200:
        print(f"[error] push ข้อความไม่สำเร็จ (status={http_response.status_code}):", http_response.text)
