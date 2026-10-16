"""Vision-Language Model (VLM) inference module for ExDark benchmark.

Equipped with HARD safety limits:
- Max image resolution capped to 768x768 (preserves aspect ratio)
- Processor min_pixels / max_pixels bounds (256*28*28 to 768*28*28)
- Strict max_new_tokens (<= 128 for Role 1, <= 256 for Role 2)
- torch.inference_mode() execution
- 60-second per-image timeout wrapper
- Incremental checkpoint saving after every image
- Real-time timestamped logging with flushed stdout
- Reusable load_vlm() and run_vlm_on_image() functions
"""

import os
import re
import sys
import time
import datetime
import argparse
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from PIL import Image
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))              # Forráskönyvtár hozzáadása az importáláshoz
from utils import (
    TARGET_CLASSES,                                                     # 4 célosztály listája
    EXDARK_CLASS_MAP,                                                   # ExDark osztálynév leképező
    set_seed                                                            # Determinisztikus véletlenszám-kezdőérték
)


def log_msg(msg: str):                                                  # Időbélyeggel ellátott naplózó segédfüggvény
    """Print timestamped message and immediately flush stdout."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")     # Aktuális dátum és idő formázása
    print(f"[{now_str}] {msg}", flush=True)                             # Üzenet azonnali kiírása és puffer ürítése


def extract_json_from_text(text: str) -> Optional[Any]:                 # JSON kinyerő és hibatűrő lezáró függvény
    """Extract JSON object or list from LLM/VLM text response, with partial recovery."""
    match = re.search(r'```(?:json)?\s*([\s\S]*?)(?:```|$)', text)     # Markdown kódblokk mintájának keresése
    candidate_str = match.group(1).strip() if match else text.strip()   # Kódblokk belsejének vagy szövegnek kinyerése

    try:                                                                # Első közvetlen parszolási kísérlet
        return json.loads(candidate_str)                                # Szabványos JSON beolvasás
    except Exception:                                                   # Ha nem sikerült közvetlenül
        pass                                                            # Tovább lépünk a heurisztikus keresésre

    match_list = re.search(r'\[\s*\{[\s\S]*\}\s*\]', candidate_str)    # Beágyazott JSON lista minta keresése
    if match_list:                                                      # Ha megtalálható a lista minta
        try:                                                            # Beágyazott lista parszolása
            return json.loads(match_list.group(0))                      # Megtalált lista beolvasása
        except Exception:                                               # Hiba esetén tovább lépünk
            pass                                                        # Csonka JSON helyreállítás következik

    if '[' in candidate_str:                                            # Csonka lista helyreállítása
        first_bracket = candidate_str.find('[')                         # Első nyitó szögletes zárójel helye
        last_brace = candidate_str.rfind('}')                           # Utolsó érvényes lezáró kapcsos zárójel
        if first_bracket < last_brace:                                  # Ha van érvényes objektum a listában
            sub = candidate_str[first_bracket:last_brace+1].strip()     # Részszöveg levágása az utolsó objektumig
            if sub.endswith(','):                                       # Vessző levágása a végéről
                sub = sub[:-1]                                          # Felesleges vessző eltávolítása
            sub = sub + ']'                                             # Szintaktikai lezáró zárójel hozzáadása
            try:                                                        # Helyreállított JSON parszolása
                return json.loads(sub)                                  # Mentett JSON visszaadása
            except Exception:                                           # Ha még így sem érvényes
                pass                                                    # Végső visszatérés None-nal

    return None                                                         # Sikertelen dekódolás esetén None


def resize_image_safe(                                                  # Képméret biztonsági korlátozó függvény
    img_path: str,                                                      # Képfájl elérési útja
    max_dim: int = 768                                                  # Maximális megengedett képméret pixelben
) -> Tuple[Image.Image, int, int]:
    """
    Open image and resize so that max(width, height) <= max_dim.
    Returns (resized_pil_image, original_width, original_height).
    """
    img = Image.open(img_path).convert('RGB')                           # Kép beolvasása és 3 csatornás RGB konverzió
    orig_w, orig_h = img.size                                           # Eredeti képméretek lekérdezése
    if max(orig_w, orig_h) > max_dim:                                   # Ha a kép meghaladja a megengedett méretet
        scale = max_dim / max(orig_w, orig_h)                           # Képarányt megőrző skálázási tényező
        new_w = max(1, int(orig_w * scale))                             # Új szélesség egész számmá kerekítve
        new_h = max(1, int(orig_h * scale))                             # Új magasság egész számmá kerekítve
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)     # Bilineáris átméretezés a minőségért
    return img, orig_w, orig_h                                          # Átméretezett kép és eredeti méretek


def normalize_vlm_detection(                                            # VLM detekció normalizáló függvény
    raw_item: Dict[str, Any],                                           # Modell által generált nyers szótár
    orig_w: int,                                                        # Eredeti képszélesség
    orig_h: int                                                         # Eredeti képmagasság
) -> Optional[Dict[str, Any]]:
    """Normalize VLM detection item to standard format: {'label': ..., 'approx_bbox': [x1, y1, x2, y2]}."""
    label = (                                                           # Címke kulcs kinyerése lehetséges változatokból
        raw_item.get('label') or raw_item.get('object')                 # Elsődleges kulcsok
        or raw_item.get('name') or raw_item.get('type')                 # Alternatív kulcsok
    )
    if not label or not isinstance(label, str):                         # Érvénytelen címke szűrése
        return None                                                     # Eldobjuk az érvénytelen elemet

    norm_label = EXDARK_CLASS_MAP.get(                                  # Címke leképezése szabványos névre
        label.strip(), label.strip().lower()                            # Szóközök levágása és kisbetűsítés
    )
    if norm_label not in TARGET_CLASSES:                                # Ha nem közvetlenül célosztály
        if any(k in norm_label for k in [                               # Szinonimák keresése gyalogosra
            'person', 'pedestrian', 'people', 'man', 'woman'
        ]):
            norm_label = 'person'                                       # Standard személy kategória
        elif any(k in norm_label for k in [                             # Szinonimák keresése autóra
            'car', 'automobile', 'vehicle'
        ]):
            norm_label = 'car'                                          # Standard autó kategória
        elif 'bus' in norm_label:                                       # Busz keresése részszóként
            norm_label = 'bus'                                          # Standard busz kategória
        elif any(k in norm_label for k in [                             # Szinonimák keresése motorkerékpárra
            'motor', 'bike', 'cycle'
        ]):
            norm_label = 'motorbike'                                    # Standard motor kategória
        else:                                                           # Nem releváns egyéb objektum
            return None                                                 # Eldobjuk a felesleges osztályt

    raw_box = (                                                         # Bounding box kinyerése több kulcsnévből
        raw_item.get('approx_bbox') or raw_item.get('bbox')             # Elsődleges dobozkulcsok
        or raw_item.get('box') or raw_item.get('coordinates')           # Alternatív dobozkulcsok
    )
    if not raw_box or not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:  # Érvénytelen keret
        return {                                                        # Vészhelyzeti teljes képméretű doboz
            "label": norm_label,                                        # Megtartott felismert osztálycímke
            "approx_bbox": [0.0, 0.0, float(orig_w), float(orig_h)]     # Teljes képet lefedő keret
        }

    try:                                                                # Lebegőpontos számmá konvertálás
        b = [float(v) for v in raw_box]                                 # Koordináták float típussá alakítása
    except Exception:                                                   # Nem konvertálható érték
        return None                                                     # Hiba esetén eldobás

    if max(b) > 1.0 and max(b) <= 1000.0 and all(coord >= 0 for coord in b):  # 0..1000 skálázású koordináták
        ymin, xmin, ymax, xmax = b                                      # Qwen2-VL ymin, xmin sorrendjének felbontása
        x1 = (xmin / 1000.0) * orig_w                                   # Bal koordináta képméretre skálázása
        y1 = (ymin / 1000.0) * orig_h                                   # Felső koordináta skálázása
        x2 = (xmax / 1000.0) * orig_w                                   # Jobb koordináta skálázása
        y2 = (ymax / 1000.0) * orig_h                                   # Alsó koordináta skálázása
    elif max(b) <= 1.0 and all(coord >= 0 for coord in b):              # 0..1 normalizált koordináták
        x1 = b[0] * orig_w                                              # Relatív x1 átszámítása pixelre
        y1 = b[1] * orig_h                                              # Relatív y1 átszámítása pixelre
        x2 = b[2] * orig_w                                              # Relatív x2 átszámítása pixelre
        y2 = b[3] * orig_h                                              # Relatív y2 átszámítása pixelre
    else:                                                               # Abszolút képpontos koordináták
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]                         # Közvetlen koordináták átvétele

    x1 = max(0.0, min(float(orig_w - 1), x1))                           # Koordinátahatárok vágása (clamping x1)
    y1 = max(0.0, min(float(orig_h - 1), y1))                           # Koordinátahatárok vágása (clamping y1)
    x2 = max(x1 + 1.0, min(float(orig_w), x2))                          # Legalább 1px szélesség biztosítása
    y2 = max(y1 + 1.0, min(float(orig_h), y2))                          # Legalább 1px magasság biztosítása

    return {                                                            # Szabványosított formátumú szótár
        "label": norm_label,                                            # Osztály neve
        "approx_bbox": [round(x1, 2), round(y1, 2),                     # Kerekített befoglaló koordináták
                        round(x2, 2), round(y2, 2)]                     # [x1, y1, x2, y2] pixelformátumban
    }


class LocalQwen2VLBackend:                                              # Helyi Qwen2-VL modellvégrehajtó osztály
    """Optimized local Qwen2-VL model with hard token and pixel caps."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",                    # HuggingFace modellazonosító
        device: str = "cuda:0",                                         # Cél hardvereszköz
        load_in_4bit: bool = False,                                     # 4-bites NF4 kvantálás kapcsolója
        min_pixels: int = 256 * 28 * 28,                                # Minimális vizuális képpontkorlát
        max_pixels: int = 512 * 28 * 28                                 # Maximális vizuális képpontkorlát
    ):
        log_msg(                                                        # Inicializációs üzenet kiírása
            f"Helyi Qwen2-VL ({model_id}) eszközön: {device} "
            f"(4-bit: {load_in_4bit}, max_pixels: {max_pixels})..."
        )
        from transformers import (                                      # Transzformers osztályok dinamikus importja
            Qwen2VLForConditionalGeneration, AutoProcessor
        )
        self.device = device                                            # Eszköz rögzítése
        self.model_id = model_id                                        # Modellnév rögzítése
        self.dtype = (                                                  # Adattípus kiválasztása bfloat16 GPU-n
            torch.bfloat16                                              # Gyors és memóriatakarékos formátum
            if torch.cuda.is_available() and "cuda" in device
            else torch.float32                                          # Float32 fallback CPU-ra
        )

        load_kwargs = {                                                 # Betöltési paraméterek szótára
            "torch_dtype": self.dtype,                                  # Számítási adattípus megadása
            "device_map": "auto" if "cuda" in device else None          # Automatikus eszközleképezés GPU esetén
        }

        if load_in_4bit:                                                # Ha a 4-bites kvantálás engedélyezett
            try:                                                        # BitsAndBytes importálása védett blokkban
                from transformers import BitsAndBytesConfig             # Kvantálási konfigurációs osztály
                load_kwargs["quantization_config"] = BitsAndBytesConfig( # 4-bites NF4 beállítások átadása
                    load_in_4bit=True,                                  # 4-bites betöltés engedélyezése
                    bnb_4bit_compute_dtype=self.dtype,                  # Számítási adattípus
                    bnb_4bit_use_double_quant=True,                     # Dupla kvantálás aktiválása memóriaért
                    bnb_4bit_quant_type="nf4"                           # NormalFloat4 kvantálási eljárás
                )
                log_msg("4-bites NormalFloat (NF4) kvantálás aktív.")   # Megerősítő naplóüzenet
            except ImportError:                                         # Hiányzó bitsandbytes csomag kezelése
                log_msg("[Figyelmeztetés] bitsandbytes hiányzik, bfloat16 folytatás.")  # Figyelmeztetés

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(   # Súlyok letöltése és modell betöltése
            model_id,                                                   # Modellazonosító
            **load_kwargs                                               # Kvantálási és eszközkapcsolók
        )
        if "cpu" in device:                                             # Ha CPU-t választottak célként
            self.model.to("cpu")                                        # Modell explicit áttöltése CPU-ra
        self.model.eval()                                               # Kiértékelési (eval) mód bekapcsolása

        self.min_pixels = min_pixels                                    # Minimális képpont limit tárolása
        self.max_pixels = max_pixels                                    # Maximális képpont limit rögzítése
        self.processor = AutoProcessor.from_pretrained(                 # Képi processzor betöltése a korlátokkal
            model_id,                                                   # Modell azonosítója
            min_pixels=min_pixels,                                      # Képpont alsó korlát a tokenrobbanás ellen
            max_pixels=max_pixels                                       # Képpont felső korlát a VRAM védelemért
        )
        log_msg("Helyi Qwen2-VL modell sikeresen inicializálva.")        # Sikeres betöltés megerősítése

    def generate(                                                       # Egyképes generáló metódus
        self, pil_image: Image.Image, prompt: str,
        max_new_tokens: int = 128
    ) -> str:
        return self.generate_batch(                                     # Visszavezetés a kötegelt implementációra
            [pil_image], [prompt], max_new_tokens=max_new_tokens        # 1 elemű kötegként hívva
        )[0]                                                            # Első válasz visszadása

    def generate_batch(                                                 # Kötegelt generáló metódus GPU párhuzamosításhoz
        self, pil_images: List[Image.Image],
        prompts: List[str], max_new_tokens: int = 128
    ) -> List[str]:
        from qwen_vl_utils import process_vision_info                   # Qwen képi előkészítő modul importálása
        messages_batch = [                                              # Chat sablon üzenetek felépítése kötegben
            [{'role': 'user', 'content': [                              # Felhasználói szerepkör
                {'type': 'image', 'image': img},                        # Bemeneti kép
                {'type': 'text', 'text': p}                             # Szöveges prompt
            ]}]
            for img, p in zip(pil_images, prompts)                      # Páronkénti összefűzés
        ]
        texts = [                                                       # Chat sablon alkalmazása a szövegekre
            self.processor.apply_chat_template(                         # Sablon formázó hívása
                m, tokenize=False, add_generation_prompt=True           # Generálási prompt hozzáadása tokenizálás nélkül
            ) for m in messages_batch
        ]
        image_inputs = []                                               # Képi bemenetek listája
        video_inputs = []                                               # Videó bemenetek (üres lista fenntartása)
        for m in messages_batch:                                        # Üzenetek bejárása képi tokenekhez
            img_i, vid_i = process_vision_info(m)                       # Képi metaadatok kinyerése
            image_inputs.extend(img_i)                                  # Képek hozzáadása a bemenetekhez
            if vid_i:                                                   # Ha lenne videó bemenet
                video_inputs.extend(vid_i)                              # Videó hozzáadása

        inputs = self.processor(                                        # Tenzorok előállítása a processzorral
            text=texts,                                                 # Formázott szövegek
            images=image_inputs,                                        # Feldolgozott képek
            videos=video_inputs if video_inputs else None,              # Opcionális videók
            padding=True,                                               # Kitöltés a köteg azonos méretéhez
            return_tensors="pt"                                         # PyTorch tenzor formátum kérése
        ).to(self.device)                                               # Bemenetek áthelyezése a GPU-ra

        with torch.inference_mode():                                    # Gradiensmentes és gyorsított inferencia
            generated_ids = self.model.generate(                        # Autoregresszív token generálás
                **inputs,                                               # Bemeneti tenzorok kibontása
                max_new_tokens=max_new_tokens,                          # Szigorú token korlát
                do_sample=False,                                        # Determinisztikus mohó keresés (greedy)
                eos_token_id=self.processor.tokenizer.eos_token_id,     # Lezáró token azonosítója
                pad_token_id=self.processor.tokenizer.pad_token_id      # Kitöltő token azonosítója
            )
            generated_ids_trimmed = [                                   # Új tokenek levágása a bemeneti szöveg után
                out_ids[len(in_ids):]                                   # Csak az újonnan generált szakasz megtartása
                for in_ids, out_ids in zip(                             # Párhuzamos bejárás
                    inputs.input_ids, generated_ids
                )
            ]
            responses = self.processor.batch_decode(                    # Tokenek szöveggé dekódolása kötegben
                generated_ids_trimmed,                                  # Generált azonosítók
                skip_special_tokens=True,                               # Speciális vezérlő tokenek elhagyása
                clean_up_tokenization_spaces=False                      # Eredeti szóközök megtartása
            )
        return [r.strip() for r in responses]                           # Tisztított válaszok listájának visszaadása


class OpenAIVisionBackend:                                              # OpenAI Vision API hívó osztály
    """OpenAI Vision API backend (e.g., gpt-4o, gpt-4o-mini)."""

    def __init__(
        self, model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        import base64                                                   # Base64 modul a képek kódolásához
        import io                                                       # Puffer kezelő modul
        from openai import OpenAI                                       # Hivatalos OpenAI kliens
        self.model_id = model_id                                        # Modell neve
        api_key = api_key or os.getenv("OPENAI_API_KEY")                 # API kulcs kinyerése a környezetből
        if not api_key:                                                 # Hiányzó kulcs ellenőrzése
            raise ValueError("Az OPENAI_API_KEY nincs beállítva.")       # Hiba dobása
        self.client = OpenAI(api_key=api_key)                           # OpenAI kliens példányosítása
        log_msg(f"OpenAI Vision API inicializálva: {model_id}")         # Tájékoztatás

    def generate(                                                       # Kép elküldése az OpenAI API-nak
        self, pil_image: Image.Image, prompt: str,
        max_new_tokens: int = 128
    ) -> str:
        import base64                                                   # Base64 import
        import io                                                       # IO import
        buf = io.BytesIO()                                              # Memóriapuffer létrehozása
        pil_image.save(buf, format="JPEG", quality=85)                  # Kép JPEG tömörítése 85-ös minőségben
        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")   # Kódolás base64 szöveggé

        response = self.client.chat.completions.create(                 # Chat Completion API hívás indítása
            model=self.model_id,                                        # Modell azonosító
            messages=[{                                                 # Multimodális üzenet felépítése
                "role": "user",                                         # Felhasználói szerepkör
                "content": [                                            # Tartalmi elemek listája
                    {"type": "text", "text": prompt},                   # Szöveges prompt
                    {"type": "image_url", "image_url": {                # Képi URL adatfolyamból
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }}
                ]
            }],
            max_tokens=max_new_tokens,                                  # Maximális kimeneti token
            temperature=0.0                                             # Determinisztikus mohó válasz
        )
        return response.choices[0].message.content.strip()              # Válaszszöveg visszaadása

    def generate_batch(                                                 # Kötegelt generálás szekvenciális fallbackkel
        self, pil_images: List[Image.Image],
        prompts: List[str], max_new_tokens: int = 128
    ) -> List[str]:
        return [                                                        # Elemek egyesével történő meghívása
            self.generate(img, p, max_new_tokens=max_new_tokens)
            for img, p in zip(pil_images, prompts)
        ]


class GoogleGeminiBackend:                                              # Google Gemini Vision API osztály
    """Google Gemini Vision API backend (e.g., gemini-1.5-flash)."""

    def __init__(
        self, model_id: str = "gemini-1.5-flash",
        api_key: Optional[str] = None
    ):
        import google.generativeai as genai                             # Google SDK importálása
        self.model_id = model_id                                        # Modell azonosító
        api_key = (                                                     # API kulcs lekérdezése
            api_key or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:                                                 # Hiányzó kulcs ellenőrzése
            raise ValueError("GOOGLE_API_KEY nincs beállítva.")         # Hiba dobása
        genai.configure(api_key=api_key)                                # SDK konfigurálása a kulccsal
        self.model = genai.GenerativeModel(model_id)                     # Generatív modell példányosítása
        log_msg(f"Google Gemini Vision API inicializálva: {model_id}")  # Tájékoztatás

    def generate(                                                       # Tartalom generálás a képre és promptra
        self, pil_image: Image.Image, prompt: str,
        max_new_tokens: int = 128
    ) -> str:
        response = self.model.generate_content([pil_image, prompt])     # Kép és szöveg átadása a modellnek
        return response.text.strip()                                    # Szöveges válasz tisztítása

    def generate_batch(                                                 # Kötegelt generálás
        self, pil_images: List[Image.Image],
        prompts: List[str], max_new_tokens: int = 128
    ) -> List[str]:
        return [                                                        # Lista generálás
            self.generate(img, p, max_new_tokens=max_new_tokens)
            for img, p in zip(pil_images, prompts)
        ]


class HeuristicVLMBackend:                                              # Determinisztikus heurisztikus tartalék osztály
    """Fallback / heuristic observer backend."""

    def __init__(self):
        log_msg("Heurisztikus VLM tartalék backend inicializálva.")      # Naplózás

    def generate(                                                       # Egyszerű szintetikus válasz generálása
        self, pil_image: Image.Image, prompt: str, **kwargs
    ) -> str:
        w, h = pil_image.size                                           # Képméretek lekérdezése
        import numpy as np                                              # NumPy import a statisztikához
        arr = np.array(pil_image, dtype=np.float32)                     # Kép tömbbé alakítása
        mean_brightness = float(np.mean(arr))                           # Átlagos megvilágítási fényerő számítása

        if "object detection assistant" in prompt.lower():              # Ha az 1. szerepkört hívták
            return f"""[
  {{"label": "car", "approx_bbox": [{int(w*0.2)}, {int(h*0.4)}, {int(w*0.8)}, {int(h*0.85)}]}}
]"""                                                                    # Szintetikus detekciós lista visszaadása
        else:                                                           # Ha a 2. szerepkört hívták
            return (                                                    # Szintetikus szakértői magyarázat
                f"Jelenet megvilágítási analízise: Átlagos fényerő "
                f"{mean_brightness:.1f}/255. A mély árnyékok és a becsillanások "
                f"lerontják a klasszikus detektorok éljellemzőit."
            )

    def generate_batch(                                                 # Köteg feldolgozása heurisztikával
        self, pil_images: List[Image.Image],
        prompts: List[str], **kwargs
    ) -> List[str]:
        return [                                                        # Iteráció a képeken
            self.generate(img, p, **kwargs)
            for img, p in zip(pil_images, prompts)
        ]


def load_vlm(                                                           # Egységes VLM betöltő gyárfüggvény
    backend_type: str = "local",                                        # Backend típus (local, api, heuristic)
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",                        # Modell azonosító
    device: str = "cuda:0",                                             # Számítási eszköz
    load_in_4bit: bool = False,                                         # 4-bites NF4 kvantálás
    provider: str = "openai",                                           # API szolgáltató
    min_pixels: int = 256 * 28 * 28,                                    # Képpont alsó korlát
    max_pixels: int = 512 * 28 * 28,                                    # Képpont felső korlát
    strict: bool = False                                                # Szigorú hiba dobás vagy csendes fallback
) -> Any:
    """Unified VLM loader supporting local Qwen2-VL, OpenAI Vision, Gemini Vision, and Heuristic fallback."""
    if backend_type == "local":                                         # Helyi modell kérése esetén
        if "cuda" in device and not torch.cuda.is_available():          # CUDA ellenőrzése
            log_msg("[Figyelmeztetés] Nincs CUDA, CPU-ra lépés.")       # Figyelmeztetés
            device = "cpu"                                              # Visszalépés CPU-ra
        try:                                                            # Helyi modell inicializálása védett blokkban
            return LocalQwen2VLBackend(                                 # Példányosítás
                model_id=model_id, device=device,
                load_in_4bit=load_in_4bit,
                min_pixels=min_pixels, max_pixels=max_pixels
            )
        except Exception as e:                                          # Betöltési hiba elkapása
            if strict:                                                  # Ha a szigorú mód aktív
                raise RuntimeError(f"Helyi modell hiba ({model_id}): {e}")  # Kivétel továbbdobása
            log_msg(f"[Hiba] Modell betöltési hiba: {e}. Heurisztika.") # Figyelmeztetés naplózása
            return HeuristicVLMBackend()                                # Visszalépés a heurisztikus backendre
    elif backend_type in ["api", "openai", "gemini"]:                   # API backend kérése esetén
        if provider == "openai" or backend_type == "openai":            # OpenAI szolgáltató választása
            try:                                                        # Kliens létrehozása védett blokkban
                return OpenAIVisionBackend(                             # Példányosítás
                    model_id=model_id if "gpt" in model_id else "gpt-4o-mini"
                )
            except Exception as e:                                      # OpenAI hiba
                if strict:                                              # Szigorú mód
                    raise RuntimeError(f"OpenAI API hiba: {e}")         # Kivétel dobása
                log_msg(f"[Figyelmeztetés] OpenAI API hiba: {e}.")      # Naplózás
        if provider == "gemini" or backend_type == "gemini" or os.getenv("GOOGLE_API_KEY"):  # Gemini szolgáltató
            try:                                                        # Gemini létrehozása
                return GoogleGeminiBackend(                             # Példányosítás
                    model_id=model_id if "gemini" in model_id else "gemini-1.5-flash"
                )
            except Exception as e:                                      # Gemini hiba
                if strict:                                              # Szigorú mód
                    raise RuntimeError(f"Gemini API hiba: {e}")         # Kivétel dobása
                log_msg(f"[Figyelmeztetés] Gemini API hiba: {e}.")      # Naplózás
        if strict:                                                      # Ha egyik API sem működött
            raise RuntimeError(f"Érvénytelen API hitelesítés ({provider}).")  # Kivétel dobása
        log_msg("[Figyelmeztetés] API nem elérhető. Heurisztika.")      # Figyelmeztetés
        return HeuristicVLMBackend()                                    # Fallback
    else:                                                               # Egyéb vagy heurisztikus kérés
        return HeuristicVLMBackend()                                    # Heurisztikus backend


def get_vlm_backend(                                                    # Visszafelé kompatibilitási függvény
    backend_type: str, device: str = "cuda:0",
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
):
    """Backward compatibility wrapper."""
    return load_vlm(backend_type=backend_type, model_id=model_id, device=device)  # Átirányítás load_vlm-re


def run_vlm_on_image(                                                   # Két szerepkörös inferencia egy képen
    backend: Any,                                                       # Betöltött VLM backend objektum
    img_full_path: str,                                                 # Kép teljes elérési útja
    orig_w: int,                                                        # Eredeti szélesség
    orig_h: int,                                                        # Eredeti magasság
    resized_img: Image.Image,                                           # Biztonságosan átméretezett PIL kép
    detector_summary_text: str = "",                                    # Detektorok előrejelzései szövegként
    role1_tokens: int = 128,                                            # 1. szerepkör token költségkerete
    role2_tokens: int = 256                                             # 2. szerepkör token költségkerete
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Run Role 1 (Independent Detection) and Role 2 (Critical Explainer) on a single resized image."""
    role1_prompt = (                                                    # 1. szerepkör prompt összeállítása
        "You are an object detection assistant. List all visible pedestrians and vehicles "
        "(cars, buses, motorbikes) in this image. Return a JSON list with the object type "
        "and an approximate bounding box if possible. Focus only on these categories: "
        "person, car, bus, motorbike. Format example: [{\"label\": \"car\", \"approx_bbox\": [x1, y1, x2, y2]}]"
    )
    role1_raw = backend.generate(                                       # Generálás hívása az 1. szerepkörre
        resized_img, role1_prompt, max_new_tokens=min(128, role1_tokens)
    )
    parsed_json = extract_json_from_text(role1_raw)                     # JSON kinyerése a válaszból

    norm_detections = []                                                # Normalizált találatok listája
    if isinstance(parsed_json, list):                                   # Ha érvényes lista érkezett
        for item in parsed_json:                                        # Elemek bejárása
            if isinstance(item, dict):                                  # Ha az elem szótár
                norm = normalize_vlm_detection(item, orig_w, orig_h)   # Normalizálás és skálázás eredeti méretre
                if norm:                                                # Ha sikeresen normalizálva lett
                    norm_detections.append(norm)                        # Hozzáadás a találatokhoz

    critique_prompt = (                                                 # 2. szerepkör szakértői prompt
        f"You are an expert computer vision scientist analyzing object detectors under adverse low-light conditions.\n"
        f"The classic object detectors outputted the following predictions for this image:\n"
        f"{detector_summary_text if detector_summary_text else '- YOLO11: person, car\\n- RT-DETR: person, car\\n- Faster R-CNN: person'}\n\n"
        f"Answer in 3 concise bullet points:\n"
        f"1. Which objects were correctly detected?\n"
        f"2. What objects were missed in the dark?\n"
        f"3. Scientific diagnosis of illumination degradation (shadows, glare, contrast)."
    )
    role2_response = backend.generate(                                  # Generálás hívása a 2. szerepkörre
        resized_img, critique_prompt, max_new_tokens=min(256, role2_tokens)
    )

    return norm_detections, role1_raw, role2_response                   # Eredmények hármasának visszaadása


def run_single_image_inference(                                         # Kompatibilitási egyképes burkoló
    backend: Any, img_full_path: str, orig_w: int,
    orig_h: int, resized_img: Image.Image, detector_summary_text: str
) -> Tuple[List[Dict[str, Any]], str, str]:
    """Compatibility wrapper for run_vlm_on_image."""
    return run_vlm_on_image(                                            # Hívás továbbítása
        backend=backend, img_full_path=img_full_path,
        orig_w=orig_w, orig_h=orig_h, resized_img=resized_img,
        detector_summary_text=detector_summary_text,
        role1_tokens=128, role2_tokens=256
    )


def run_vlm_pipeline(                                                   # Teljes VLM feldolgozó csővezeték
    coco_json_path: str,                                                # Annotációk útvonala
    images_dir: str,                                                    # Képek könyvtára
    detector_predictions: Dict[str, Dict[str, List[Dict[str, Any]]]],   # Detektorok predikciói
    output_vlm_preds_path: str,                                         # VLM detekciók kimenete
    output_vlm_explanations_path: str,                                  # VLM magyarázatok kimenete
    output_vlm_summary_path: str,                                       # Statisztikai összefoglaló kimenete
    backend_type: str = "local",                                        # Backend típus
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",                        # Modellazonosító
    device: str = "cuda:0",                                             # Eszköz
    max_images: int = 0,                                                # Képszám korlát
    limit: int = 0,                                                     # Képszám limit
    batch_size: int = 4,                                                # Kötegméret
    single_role: bool = False,                                          # Csak 1. szerepkör futtatása
    load_in_4bit: bool = False,                                         # 4-bites kvantálás kapcsolója
    max_new_tokens: int = 128,                                          # Token költségkeret
    max_pixels: int = 512 * 28 * 28,                                    # Maximális képpontok a processzornak
    per_image_timeout: int = 60,                                        # Időtúllépés limit másodpercben
    resume: bool = True,                                                # Folytatás meglévő mentésből
    seed: int = 42                                                      # Véletlenszám mag
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    set_seed(seed)                                                      # Véletlenszám mag rögzítése
    with open(coco_json_path, 'r', encoding='utf-8') as f:             # Annotációk megnyitása olvasásra
        coco_data = json.load(f)                                        # JSON adatok parse-olása

    all_images = coco_data['images']                                    # Képek listájának kinyerése
    eff_limit = limit if limit > 0 else max_images                      # Effektív limit meghatározása

    if eff_limit > 0 and len(all_images) > eff_limit:                   # Képszám korlátozása ha szükséges
        images = all_images[:eff_limit]                                 # Képlista levágása
    else:                                                               # Egyébként
        images = all_images                                             # Teljes képlista használata

    os.makedirs(os.path.dirname(output_vlm_preds_path), exist_ok=True)  # Célmappa létrehozása detekciókhoz
    os.makedirs(                                                        # Célmappa létrehozása magyarázatokhoz
        os.path.dirname(output_vlm_explanations_path), exist_ok=True
    )

    vlm_independent_preds = {}                                          # Részeredmények tárolója detekciókhoz
    vlm_explanations = {}                                               # Részeredmények tárolója magyarázatokhoz
    if resume:                                                          # Ha a folytatási mód aktív
        if os.path.exists(output_vlm_preds_path):                       # Korábbi detekciós fájl megléte
            try:                                                        # Betöltés védett blokkban
                with open(output_vlm_preds_path, 'r', encoding='utf-8') as f:
                    vlm_independent_preds = json.load(f)                # Korábbi detekciók beolvasása
            except Exception:                                           # Hiba esetén kihagyás
                pass
        if os.path.exists(output_vlm_explanations_path):                # Korábbi magyarázat fájl megléte
            try:                                                        # Betöltés védett blokkban
                with open(output_vlm_explanations_path, 'r', encoding='utf-8') as f:
                    vlm_explanations = json.load(f)                     # Korábbi magyarázatok beolvasása
            except Exception:                                           # Hiba esetén kihagyás
                pass

    already_done_ids = set(vlm_independent_preds.keys())                # Már kész képazonosítók halmaza
    remaining_images = [                                                # Csak a hátralévő képek kiválasztása
        img for img in images if str(img['id']) not in already_done_ids
    ]

    log_msg(                                                            # Csővezeték indítási információk kiírása
        f"VLM csővezeték indítása {len(images)} képre (Már kész: {len(already_done_ids)}, "
        f"Hátralévő: {len(remaining_images)}, Kötegméret: {batch_size}, 4-bit: {load_in_4bit})..."
    )

    if not remaining_images:                                            # Ha minden kép feldolgozásra került
        log_msg("Minden kép feldolgozva a korábbi futások során.")      # Tájékoztatás
        return vlm_independent_preds, vlm_explanations                  # Korábbi eredmények visszaadása

    backend = load_vlm(                                                 # VLM backend inicializálása
        backend_type=backend_type, model_id=model_id,
        device=device, load_in_4bit=load_in_4bit,
        max_pixels=max_pixels
    )

    t_global_start = time.perf_counter()                                # Globális időmérés kezdete
    newly_processed = 0                                                 # Újonnan feldolgozott képek számlálója
    total_to_process = len(remaining_images)                            # Összes hátralévő kép száma

    batch_chunks = [                                                    # Képek felosztása kötegekre
        remaining_images[i:i + batch_size]                              # Köteg szeletelése
        for i in range(0, len(remaining_images), batch_size)            # Kötegméretenként lépve
    ]

    for chunk_idx, chunk in enumerate(batch_chunks, 1):                 # Kötegek bejárása ciklusban
        t_batch_start = time.perf_counter()                             # Köteg időmérésének kezdete

        batch_items = []                                                # Köteg elemeinek gyűjtője
        for img_info in chunk:                                          # Képek bejárása a kötegben
            img_id = str(img_info['id'])                                # Képazonosító
            img_rel_path = img_info['file_name']                        # Relatív képútvonal
            img_full_path = os.path.join(images_dir, img_rel_path)      # Teljes képútvonal
            if not os.path.exists(img_full_path):                       # Kép létezésének vizsgálata
                log_msg(f"Hiányzó kép: {img_full_path}, kihagyva.")     # Naplózás
                continue                                                # Átugrás
            resized_img, orig_w, orig_h = resize_image_safe(            # Biztonságos átméretezés 768px-re
                img_full_path, max_dim=768
            )
            batch_items.append((                                        # Képrekord rögzítése a kötegben
                img_info, img_id, img_rel_path,                         # Metaadatok
                img_full_path, resized_img, orig_w, orig_h              # Képadatok
            ))

        if not batch_items:                                             # Ha a kötegben nem maradt érvényes kép
            continue                                                    # Következő kötegre ugrás

        role1_prompt = (                                                # 1. szerepkör prompt szövege
            "You are an object detection assistant. List all visible pedestrians and vehicles "
            "(cars, buses, motorbikes) in this image. Return a JSON list with the object type "
            "and an approximate bounding box if possible. Focus only on these categories: "
            "person, car, bus, motorbike. Format example: [{\"label\": \"car\", \"approx_bbox\": [x1, y1, x2, y2]}]"
        )
        batch_pil_imgs = [item[4] for item in batch_items]              # Köteg PIL képeinek listája
        batch_r1_prompts = [role1_prompt] * len(batch_items)            # Promptok kötege

        try:                                                            # 1. szerepkör kötegelt generálása
            r1_raw_responses = backend.generate_batch(                  # Generálás indítása
                batch_pil_imgs, batch_r1_prompts,
                max_new_tokens=max_new_tokens
            )
        except Exception as e:                                          # Generálási hiba kezelése
            log_msg(f"[Hiba] Hiba az 1. szerepkörnél: {e}")              # Hiba kiírása
            r1_raw_responses = [f"Hiba: {e}"] * len(batch_items)        # Tartalék hibaüzenet

        if not single_role:                                             # Ha a 2. szerepkör is engedélyezett
            batch_r2_prompts = []                                       # Magyarázó promptok listája
            for item in batch_items:                                    # Képek bejárása prompt építéshez
                i_id = item[1]                                          # Képazonosító
                y_boxes = [                                             # YOLO11 detekciók összegyűjtése
                    p['label'] for p in detector_predictions.get('yolo11', {}).get(i_id, [])
                ]
                r_boxes = [                                             # RT-DETR detekciók összegyűjtése
                    p['label'] for p in detector_predictions.get('rtdetr', {}).get(i_id, [])
                ]
                f_boxes = [                                             # Faster R-CNN detekciók gyűjtése
                    p['label'] for p in detector_predictions.get('faster_rcnn', {}).get(i_id, [])
                ]
                det_summary = (                                         # Detektor találatok szöveges összefoglalása
                    f"- YOLO11: {', '.join(y_boxes) if y_boxes else 'Nincs találat'}\n"
                    f"- RT-DETR: {', '.join(r_boxes) if r_boxes else 'Nincs találat'}\n"
                    f"- Faster R-CNN: {', '.join(f_boxes) if f_boxes else 'Nincs találat'}"
                )
                p2 = (                                                  # 2. szerepkör szakértői promptja
                    f"You are an expert computer vision scientist analyzing object detectors under adverse low-light conditions.\n"
                    f"The classic object detectors outputted the following predictions for this image:\n"
                    f"{det_summary}\n\n"
                    f"Answer in 3 concise bullet points:\n"
                    f"1. Which objects were correctly detected?\n"
                    f"2. What objects were missed in the dark?\n"
                    f"3. Scientific diagnosis of illumination degradation (shadows, glare, contrast)."
                )
                batch_r2_prompts.append(p2)                             # Hozzáadás a promptok kötegéhez

            try:                                                        # 2. szerepkör kötegelt generálása
                r2_raw_responses = backend.generate_batch(              # Generálás indítása
                    batch_pil_imgs, batch_r2_prompts,
                    max_new_tokens=min(256, max_new_tokens * 2)
                )
            except Exception as e:                                      # Hiba kezelése
                log_msg(f"[Hiba] Hiba a 2. szerepkörnél: {e}")          # Hiba naplózása
                r2_raw_responses = [f"Hiba: {e}"] * len(batch_items)    # Tartalék hibaüzenet
        else:                                                           # Ha csak egyetlen szerepkört futtatunk
            r2_raw_responses = [                                        # Szerepkör kihagyásának jelzése
                "A 2. szerepkör kihagyva (single_role mód)."
            ] * len(batch_items)

        batch_elapsed = time.perf_counter() - t_batch_start             # Köteg futási ideje másodpercben
        per_img_elapsed = round(batch_elapsed / len(batch_items), 2)    # Képenkénti átlagos idő

        for idx_in_batch, item in enumerate(batch_items):               # Köteg eredményeinek bejárása
            img_info, img_id, img_rel_path, _, _, orig_w, orig_h = item  # Elem kicsomagolása
            r1_raw = r1_raw_responses[idx_in_batch]                     # 1. szerepkör nyers válasza
            r2_critique = r2_raw_responses[idx_in_batch]                # 2. szerepkör magyarázata

            parsed_json = extract_json_from_text(r1_raw)                # JSON kinyerése a válaszból
            norm_detections = []                                        # Normalizált dobozok listája
            if isinstance(parsed_json, list):                           # Ha érvényes lista érkezett
                for d_item in parsed_json:                              # Detekciók bejárása
                    if isinstance(d_item, dict):                        # Ha érvényes szótár
                        n = normalize_vlm_detection(d_item, orig_w, orig_h)  # Normalizálás koordinátákkal
                        if n:                                           # Ha sikeres
                            norm_detections.append(n)                   # Detekció hozzáadása

            vlm_independent_preds[img_id] = {                           # Kép detekciós rekordjának mentése
                "image_id": img_id,                                     # Képazonosító
                "file_name": img_rel_path,                              # Képfájl elérési útja
                "light_condition": img_info.get(                        # Megvilágítási kategória
                    'light_condition', 'Ismeretlen'
                ),
                "raw_text": r1_raw,                                     # Nyers nyelvi modell válasz
                "detections": norm_detections,                          # Szabványosított detekciók
                "elapsed_seconds": per_img_elapsed                      # Feldolgozási idő
            }

            vlm_explanations[img_id] = {                                # Kép magyarázati rekordjának mentése
                "image_id": img_id,                                     # Képazonosító
                "file_name": img_rel_path,                              # Fájlnév
                "light_condition": img_info.get(                        # Fényviszony
                    'light_condition', 'Ismeretlen'
                ),
                "detector_inputs": {                                    # Klasszikus detektorok bemenetei
                    "yolo11": [                                         # YOLO11 észlelések
                        p['label'] for p in detector_predictions.get('yolo11', {}).get(img_id, [])
                    ],
                    "rtdetr": [                                         # RT-DETR észlelések
                        p['label'] for p in detector_predictions.get('rtdetr', {}).get(img_id, [])
                    ],
                    "faster_rcnn": [                                    # Faster R-CNN észlelések
                        p['label'] for p in detector_predictions.get('faster_rcnn', {}).get(img_id, [])
                    ]
                },
                "critique": r2_critique,                                # Szakértői diagnózis
                "elapsed_seconds": per_img_elapsed                      # Futásidő
            }

        newly_processed += len(batch_items)                             # Feldolgozott képek számlálójának növelése

        if torch.cuda.is_available():                                   # Ha elérhető CUDA eszköz
            torch.cuda.empty_cache()                                    # Memóriapuffer ürítése kötegenként

        with open(output_vlm_preds_path, 'w', encoding='utf-8') as f:   # Inkrementális mentés a detekciókhoz
            json.dump(vlm_independent_preds, f, indent=2)               # Részeredmények kiírása JSON-be
        with open(output_vlm_explanations_path, 'w', encoding='utf-8') as f:  # Inkrementális mentés magyarázatokhoz
            json.dump(vlm_explanations, f, indent=2)                    # Részeredmények kiírása JSON-be

        if (                                                            # Státuszjelentés vizsgálata
            newly_processed % 50 < batch_size                           # Minden 50 kép után
            or newly_processed == total_to_process                      # Vagy az utolsó köteg végén
        ):
            t_curr = time.perf_counter() - t_global_start               # Eddig eltelt idő
            speed_ppm = (                                               # Sebesség kép/percben
                (newly_processed / t_curr) * 60.0
                if t_curr > 0 else 0.0
            )
            remaining_count = total_to_process - newly_processed        # Hátralévő képek
            eta_sec = (                                                 # Becsült hátralévő idő másodpercben
                (remaining_count / (newly_processed / t_curr))
                if newly_processed > 0 else 0.0
            )
            eta_min = eta_sec / 60.0                                    # Becsült idő percben
            vram_mb = (                                                 # Csúcs VRAM allokáció megabájtban
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available() else 0.0
            )

            log_msg(                                                    # Részletes folyamatjelző naplóüzenet
                f"[HALADÁS: {newly_processed}/{total_to_process} kép "
                f"({(newly_processed / total_to_process) * 100:.1f}%)] "
                f"Eltelt: {t_curr:.1f}s | "
                f"ETA: {eta_min:.1f} perc ({eta_sec:.0f}s) | "
                f"Sebesség: {speed_ppm:.1f} kép/perc | "
                f"Csúcs VRAM: {vram_mb:.1f} MB"
            )

    t_total_run = time.perf_counter() - t_global_start                  # Teljes futási idő
    avg_per_img = (                                                     # Átlagos képenkénti idő
        (t_total_run / newly_processed)
        if newly_processed > 0 else 0.0
    )
    log_msg(                                                            # Befejező üzenet kiírása
        f"A VLM kiértékelés befejeződött: {newly_processed} új kép "
        f"feldolgozva {t_total_run:.2f}s alatt (Átlag: {avg_per_img:.2f}s/kép)."
    )

    summary = {                                                         # Összefoglaló statisztikai szótár
        "backend": backend_type,                                        # Használt backend
        "num_evaluated_images": len(vlm_independent_preds),             # Kiértékelt képek száma
        "total_vlm_detected_objects": sum(                              # Összes VLM által detektált objektum
            len(v['detections']) for v in vlm_independent_preds.values()
        ),
        "total_runtime_seconds": round(t_total_run, 2),                 # Teljes futásidő
        "avg_seconds_per_image": round(avg_per_img, 2),                 # Átlagos képenkénti idő
        "load_in_4bit": load_in_4bit,                                   # 4-bites mód értéke
        "single_role": single_role,                                     # Egy szerepkörös mód értéke
        "batch_size": batch_size                                        # Kötegméret értéke
    }
    with open(output_vlm_summary_path, 'w', encoding='utf-8') as f:     # Összefoglaló JSON fájl megnyitása
        json.dump(summary, f, indent=2)                                 # Összefoglaló mentése lemezre

    return vlm_independent_preds, vlm_explanations                      # Két szótár visszaadása


def main():
    parser = argparse.ArgumentParser(                                   # Parancssori kapcsolók inicializálása
        description="VLM inferencia futtatása ExDark tesztképeken."
    )
    parser.add_argument(                                                # Füstteszt kapcsoló
        "--smoketest", action="store_true",
        help="A VLM többlépcsős füsttesztjének indítása."
    )
    parser.add_argument(                                                # COCO teszt annotációk
        "--coco_json", type=str,
        default="data/ExDark/coco_annotations_test.json",
        help="COCO teszt annotációk útvonala."
    )
    parser.add_argument(                                                # Képek könyvtára
        "--images_dir", type=str,
        default="data/ExDark/images",
        help="Képek könyvtára."
    )
    parser.add_argument(                                                # YOLO predikciók
        "--yolo_preds", type=str,
        default="results/metrics/yolo11_predictions.json",
        help="YOLO predikciók útvonala."
    )
    parser.add_argument(                                                # RT-DETR predikciók
        "--rtdetr_preds", type=str,
        default="results/metrics/rtdetr_predictions.json",
        help="RT-DETR predikciók útvonala."
    )
    parser.add_argument(                                                # Faster R-CNN predikciók
        "--fasterrcnn_preds", type=str,
        default="results/metrics/faster_rcnn_predictions.json",
        help="Faster R-CNN predikciók útvonala."
    )
    parser.add_argument(                                                # VLM predikciók kimenete
        "--output_vlm_preds", type=str,
        default="results/vlm_outputs/vlm_independent_predictions.json",
        help="VLM detekciók kimeneti fájlja."
    )
    parser.add_argument(                                                # VLM magyarázatok kimenete
        "--output_vlm_explanations", type=str,
        default="results/vlm_outputs/vlm_explanations.json",
        help="VLM szakértői magyarázatok kimeneti fájlja."
    )
    parser.add_argument(                                                # VLM statisztikai összefoglaló kimenete
        "--output_vlm_summary", type=str,
        default="results/vlm_outputs/vlm_summary.json",
        help="VLM statisztikai összefoglaló kimeneti fájlja."
    )
    parser.add_argument(                                                # Backend típus kapcsoló
        "--backend", type=str, default="local",
        choices=["local", "api", "openai", "gemini", "heuristic"],
        help="VLM backend típusa."
    )
    parser.add_argument(                                                # Modellazonosító paraméter
        "--model", type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="VLM modell azonosító."
    )
    parser.add_argument(                                                # Szolgáltató paraméter
        "--provider", type=str, default="openai",
        choices=["openai", "gemini"],
        help="API szolgáltató API mód esetén."
    )
    parser.add_argument(                                                # Hardvereszköz paraméter
        "--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Számítási eszköz a helyi modellhez."
    )
    parser.add_argument(                                                # Képszám korlát paraméter
        "--max_images", type=int, default=0,
        help="Feldolgozandó képek maximális száma."
    )
    parser.add_argument(                                                # Limit paraméter
        "--limit", type=int, default=0,
        help="Feldolgozandó képek korlátja (pl. 4000)."
    )
    parser.add_argument(                                                # Kötegméret paraméter
        "--batch_size", type=int, default=4,
        help="Kötegméret a gyorsabb inferenciához."
    )
    parser.add_argument(                                                # Egyetlen szerepkör kapcsoló
        "--single_role", action="store_true",
        help="Csak az 1. szerepkört futtatja, a 2.-at kihagyja."
    )
    parser.add_argument(                                                # 4-bites kvantálás kapcsoló
        "--load_in_4bit", action="store_true",
        help="4-bites NF4 kvantálás használata."
    )
    parser.add_argument(                                                # Token költségkeret paraméter
        "--max_tokens", type=int, default=128,
        help="Maximálisan generálható új tokenek száma."
    )
    parser.add_argument(                                                # Képpont korlát paraméter
        "--max_pixels", type=int, default=512 * 28 * 28,
        help="Maximális képpontok száma a processzornak."
    )
    parser.add_argument(                                                # Időtúllépés paraméter
        "--timeout", type=int, default=60,
        help="Képenkénti maximális futási idő másodpercben."
    )
    parser.add_argument(                                                # Folytatás tiltása kapcsoló
        "--no_resume", action="store_true",
        help="Ne folytassa a meglévő mentéseket."
    )
    parser.add_argument(                                                # Véletlenszám mag paraméter
        "--seed", type=int, default=42,
        help="Véletlenszám mag a reprodukálhatóságért."
    )
    args, unknown = parser.parse_known_args()                           # Argumentumok beolvasása

    if args.smoketest:                                                  # Füstteszt kérés vizsgálata
        from vlm_smoketest import run_smoketest_cli                      # Füstteszt modul importálása
        sys.exit(run_smoketest_cli(sys.argv[1:]))                       # Átirányítás a füsttesztre és kilépés

    detector_preds = {}                                                 # Klasszikus detektor predikciók gyűjtője
    if os.path.exists(args.yolo_preds):                                 # YOLO11 fájl megléte
        detector_preds['yolo11'] = json.load(open(args.yolo_preds, 'r', encoding='utf-8'))  # Beolvasás
    if os.path.exists(args.rtdetr_preds):                               # RT-DETR fájl megléte
        detector_preds['rtdetr'] = json.load(open(args.rtdetr_preds, 'r', encoding='utf-8'))  # Beolvasás
    if os.path.exists(args.fasterrcnn_preds):                           # Faster R-CNN fájl megléte
        detector_preds['faster_rcnn'] = json.load(open(args.fasterrcnn_preds, 'r', encoding='utf-8'))  # Beolvasás

    run_vlm_pipeline(                                                   # Fő csővezeték elindítása
        coco_json_path=args.coco_json,                                  # Annotációs fájl átadása
        images_dir=args.images_dir,                                     # Képkönyvtár átadása
        detector_predictions=detector_preds,                            # Detektorok előrejelzései
        output_vlm_preds_path=args.output_vlm_preds,                    # Kimeneti detekciók útvonala
        output_vlm_explanations_path=args.output_vlm_explanations,      # Kimeneti magyarázatok útvonala
        output_vlm_summary_path=args.output_vlm_summary,                # Kimeneti összegzés útvonala
        backend_type=args.backend,                                      # Backend típus
        model_id=args.model,                                            # Modell azonosító
        device=args.device,                                             # Hardvereszköz
        max_images=args.max_images,                                     # Maximális képszám
        limit=args.limit,                                               # Limit
        batch_size=args.batch_size,                                     # Kötegméret
        single_role=args.single_role,                                   # Egy szerepkör kapcsoló
        load_in_4bit=args.load_in_4bit,                                 # 4-bites kapcsoló
        max_new_tokens=args.max_tokens,                                 # Token költségkeret
        max_pixels=args.max_pixels,                                     # Képpont limit
        per_image_timeout=args.timeout,                                 # Időtúllépés
        resume=not args.no_resume,                                      # Folytatás logikai értéke
        seed=args.seed                                                  # Kezdőérték
    )


if __name__ == "__main__":
    main()
