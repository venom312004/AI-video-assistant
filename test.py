# from core.transcriber import transcribe_all
# from utils.audio_processor import  process_input

# source = ""

# chunks=process_input(source)
# print(transcribe_all(chunks))


# from dotenv import load_dotenv
# load_dotenv()
# import os

# from utils.audio_processor import process_input
# from core.transcriber import transcribe_all

# #print("KEY LOADED:",os.getenv("SARVAM_API_KEY"))   #API Key will print
# print("KEY LOADED:",bool(os.getenv("SARVAM_API_KEY"))) #True or False for existing of key
# print("CWD:",os.getcwd())
# source="https://www.youtube.com/watch?v=tplWXd_T7YQ"
# language="hinglish"

# chunk=process_input(source)
# transcript=transcribe_all(chunk,language)
# print("\n=== TRANSCRIPT ===\n")
# print(transcript)

from dotenv import load_dotenv
load_dotenv()   # MUST be before any core/ imports

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summary import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions


source = "https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s"
language = "english"   # "english" → Whisper, "hinglish" → Sarvam



chunks = process_input(source)


transcript = transcribe_all(chunks, language=language)
print("\n" + "=" * 60)
print("📝 TRANSCRIPT")
print("=" * 60)
print(transcript[:500] + "..." if len(transcript) > 500 else transcript)


title = generate_title(transcript)
summary = summarize(transcript)

print("\n" + "=" * 60)
print(f"📌 TITLE: {title}")
print("=" * 60)
print("\n📋 SUMMARY")
print("-" * 60)
print(summary)



action_items = extract_action_items(transcript)
decisions = extract_key_decisions(transcript)
questions = extract_questions(transcript)

print("\n" + "=" * 60)
print("✅ ACTION ITEMS")
print("=" * 60)
print(action_items)

print("\n" + "=" * 60)
print("🔑 KEY DECISIONS")
print("=" * 60)
print(decisions)

print("\n" + "=" * 60)
print("❓ OPEN QUESTIONS")
print("=" * 60)
print(questions)
