# src/alerts.py
import time
import pyttsx3
from . import config

engine = pyttsx3.init()
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

last_alert = ""
last_time = 0


def speak(text):
    engine.say(text)
    engine.runAndWait()


def alert(text, class_name):
    global last_alert, last_time
    now = time.time()

    # Evitar repetir la misma alerta demasiado seguido
    if class_name == last_alert and (now - last_time) < config.DEBOUNCE_TIME:
        print(f"[DEBUG] alerta '{class_name}' ignorada por debounce ({now - last_time:.2f}s)")
        return

    print("🔊 ALERTA:", text)
    speak(text)

    last_alert = class_name
    last_time = now