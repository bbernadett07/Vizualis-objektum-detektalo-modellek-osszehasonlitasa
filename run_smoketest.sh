#!/usr/bin/env bash

# Azonnali leállás hiba esetén: ha bármelyik parancs (pl. a Python szkript) nem nulla státuszkóddal tér vissza, a bash szkript azonnal megszakad.
# Ez megakadályozza a hibák észrevétlen továbbgördülését.
set -e

# Átlépünk a szkript tényleges könyvtárába (a projekt gyökerébe), függetlenül attól, hogy a felhasználó honnan hívta meg.
# Miért van itt: elkerüli a relatív elérési utak (data/, results/) elcsúszását.
# Mellékhatás: módosítja a munkakönyvtárat a bash futás idejére.
cd "$(dirname "$0")"

# Tájékoztató fejléc kiírása a felhasználónak
echo "=== VLM Füstteszt (Smoke Test) Indítása ==="

# A füstteszt Python modul futtatása.
# -u kapcsoló: kikapcsolja a standard kimenet (stdout/stderr) pufferelését.
# Miért van itt: valós idejű naplózást biztosít; fagyás vagy hiba esetén nem vesznek el az utolsó sorok a pufferben.
# "$@": továbbadja a felhasználó által megadott összes parancssori argumentumot (pl. --backend api, --load_in_4bit, stb.).
python -u src/vlm_smoketest.py "$@"

# Sikeres lefutás jelzése
echo "=== VLM Füstteszt Sikeresen Befejeződött ==="
