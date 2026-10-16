"""Script to download and unpack the ExDark dataset and ground truth annotations."""

import os
import sys
import shutil
import zipfile
import argparse
import urllib.request
import gdown

EXDARK_IMAGES_DRIVE_ID = "1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx"           # Google Drive ID a képekhez
EXDARK_GROUNDTRUTH_DRIVE_ID = "1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i"      # Google Drive ID az annotációkhoz
IMAGECLASSLIST_URL = (                                                  # GitHub URL a metaadatokhoz
    "https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image"
    "-Dataset/master/Groundtruth/imageclasslist.txt"
)

TARGET_FOLDERS = ['People', 'Car', 'Bus', 'Motorbike']                  # Csak a 4 vizsgált célosztály


def download_file(drive_id: str, output_path: str):
    """Download file from Google Drive using gdown."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:  # Meglévő archívum ellenőrzése
        print(f"[Infó] Megtalálható a korábbi archívum itt: {output_path} ({os.path.getsize(output_path) / (1024*1024):.1f} MB), letöltés kihagyása.")  # Tájékoztatás
        return                                                          # Ismételt letöltés kihagyása
    print(f"[Infó] Google Drive fájl letöltése (Azonosító: {drive_id}) ide: {output_path}...")  # Letöltési üzenet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)             # Célkönyvtár létrehozása
    gdown.download(id=drive_id, output=output_path, quiet=False)        # Letöltés gdown könyvtárral


def extract_annotations(zip_path: str, extract_dir: str):
    """Extract ExDark annotations into target directory."""
    print(f"[Infó] Annotációk kibontása innen: {zip_path} ide: {extract_dir}...")  # Kibontás indítása
    os.makedirs(extract_dir, exist_ok=True)                             # Kimeneti mappa biztosítása
    with zipfile.ZipFile(zip_path, 'r') as z:                           # ZIP archívum megnyitása olvasásra
        for member in z.namelist():                                     # Bejegyzések bejárása
            if member.startswith('__MACOSX') or member.endswith('/'):   # macOS meta és mappa kihagyása
                continue                                                # Következő elemre lépés
            parts = member.split('/')                                   # Útvonal felbontása részekre
            if len(parts) >= 3 and parts[0] == 'ExDark_Annno':          # Formátum ellenőrzése
                class_folder = parts[1]                                 # Osztály alkönyvtár neve
                file_name = parts[2]                                    # Annotációs fájlnév
                dest_dir = os.path.join(extract_dir, class_folder)      # Cél mappa elérési útja
                os.makedirs(dest_dir, exist_ok=True)                     # Osztálymappa létrehozása
                dest_file = os.path.join(dest_dir, file_name)           # Célfájl teljes elérési útja
                if not os.path.exists(dest_file):                       # Csak ha még nem létezik
                    with open(dest_file, 'wb') as f_out:                # Bináris kimeneti fájl nyitása
                        f_out.write(z.read(member))                     # Tömörített adat kiírása
    print(f"[Infó] Annotációk sikeresen kibontva a következő könyvtárba: {extract_dir}.")  # Sikeres kibontás


def extract_images(zip_path: str, extract_dir: str, target_classes_only: bool = False):
    """Extract ExDark images into target directory."""
    print(f"[Infó] Képek kibontása innen: {zip_path} ide: {extract_dir} (target_classes_only={target_classes_only})...")  # Kibontás indítása
    os.makedirs(extract_dir, exist_ok=True)                             # Képek célmappájának létrehozása
    with zipfile.ZipFile(zip_path, 'r') as z:                           # Kép ZIP archívum megnyitása
        for member in z.namelist():                                     # Képelemek bejárása
            if member.startswith('__MACOSX') or member.endswith('/'):   # Szemétfájlok kiszűrése
                continue                                                # Átugorjuk a nem kép elemeket
            parts = member.split('/')                                   # Elérési út szétbontása
            if len(parts) >= 3 and parts[0] == 'ExDark':                # Kép mappastruktúra ellenőrzése
                class_folder = parts[1]                                 # Osztálymappa azonosítása
                if target_classes_only and class_folder not in TARGET_FOLDERS:  # Nem célosztályok eldobása
                    continue                                            # Tárhelykímélő kihagyás
                file_name = parts[2]                                    # Képfájl neve
                dest_dir = os.path.join(extract_dir, class_folder)      # Célkönyvtár összeállítása
                os.makedirs(dest_dir, exist_ok=True)                     # Osztálymappa biztosítása
                dest_file = os.path.join(dest_dir, file_name)           # Teljes célútvonal
                if not os.path.exists(dest_file):                       # Új képek lemezre írása
                    with open(dest_file, 'wb') as f_out:                # Képfájl megnyitása írásra
                        f_out.write(z.read(member))                     # Képadatok kicsomagolása
    print(f"[Infó] Képek sikeresen kibontva a következő könyvtárba: {extract_dir}.")  # Befejezési üzenet


def download_imageclasslist(output_path: str):
    """Download imageclasslist.txt metadata file."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:  # Ha már létezik és nem üres
        print(f"[Infó] Megtalálható a meglévő {output_path}, letöltés kihagyása.")  # Kihagyás jelzése
        return                                                          # Letöltés mellőzése
    print(f"[Infó] imageclasslist.txt letöltése innen: {IMAGECLASSLIST_URL}...")  # Letöltési infó
    os.makedirs(os.path.dirname(output_path), exist_ok=True)             # Szülőkönyvtár létrehozása
    urllib.request.urlretrieve(IMAGECLASSLIST_URL, output_path)         # HTTP letöltés GitHubról
    print(f"[Infó] imageclasslist.txt letöltve ide: {output_path}.")    # Sikeres letöltés kiírása


def main():
    parser = argparse.ArgumentParser(                                   # Argumentum parser létrehozása
        description="ExDark adathalmaz letöltése és előkészítése."      # Parser leírása
    )
    parser.add_argument("--data_dir", type=str, default="data/ExDark",  # Kimeneti könyvtár paraméter
                        help="Kimeneti adatkönyvtár elérési útja.")     # Súgó szöveg
    parser.add_argument("--cache_dir", type=str, default="../",         # Gyorsítótár keresési útvonal
                        help="Gyorsítótár könyvtár a zip fájlokhoz.")   # Cache súgó
    parser.add_argument("--target_classes_only", action="store_true",   # Csak célosztályok kapcsoló
                        default=True, help="Csak a 4 célosztály.")      # Célosztály szűrés
    args = parser.parse_args()                                          # Parancssori argumentumok feldolgozása

    os.makedirs(args.data_dir, exist_ok=True)                            # Adatkönyvtár létrehozása

    gt_zip_name = "test_groundtruth.zip"                                # Annotációs ZIP fájlnév
    img_zip_name = "exdark_images.zip"                                  # Kép ZIP fájlnév

    candidate_gt = None                                                 # Annotáció cache jelölt kezdetben None
    candidate_img = None                                                # Kép cache jelölt kezdetben None

    for check_dir in [args.data_dir, args.cache_dir, ".", ".."]:        # Lehetséges mappák bejárása
        p_gt = os.path.join(check_dir, gt_zip_name)                     # Annotációs archívum vizsgálata
        if os.path.exists(p_gt) and os.path.getsize(p_gt) > 1024:       # Ha létezik és nem sérült
            candidate_gt = p_gt                                         # Megtalált cache beállítása
            break                                                       # Keresés leállítása

    for check_dir in [args.data_dir, args.cache_dir, ".", ".."]:        # Mappák bejárása a képekhez
        p_img = os.path.join(check_dir, img_zip_name)                   # Képarchívum vizsgálata
        if os.path.exists(p_img) and os.path.getsize(p_img) > 1024:     # Ha létezik és érvényes méretű
            candidate_img = p_img                                       # Megtalált kép cache beállítása
            break                                                       # Keresési ciklus megszakítása

    dest_gt_zip = candidate_gt or os.path.join(args.data_dir, gt_zip_name)   # Meglévő vagy letöltési GT útvonal
    dest_img_zip = candidate_img or os.path.join(args.data_dir, img_zip_name) # Meglévő vagy letöltési Kép útvonal

    if not os.path.exists(dest_gt_zip):                                 # Ha hiányzik az annotáció
        download_file(EXDARK_GROUNDTRUTH_DRIVE_ID, dest_gt_zip)         # Letöltés Google Drive-ról
    else:
        print(f"[Infó] Gyorsítótárazott (cache) annotációs archívum használata: {dest_gt_zip}")  # Cache infó

    if not os.path.exists(dest_img_zip):                                # Ha hiányzik a képarchívum
        download_file(EXDARK_IMAGES_DRIVE_ID, dest_img_zip)             # Képek letöltése Google Drive-ról
    else:
        print(f"[Infó] Gyorsítótárazott (cache) kép-archívum használata: {dest_img_zip}")  # Kép cache infó

    img_class_list_file = os.path.join(args.data_dir, "imageclasslist.txt")  # Metaadat fájl útvonala
    download_imageclasslist(img_class_list_file)                        # imageclasslist.txt letöltése

    annotations_dir = os.path.join(args.data_dir, "annotations")        # Annotációk célkönyvtára
    images_dir = os.path.join(args.data_dir, "images")                  # Képek célkönyvtára

    extract_annotations(dest_gt_zip, annotations_dir)                   # Annotációk kicsomagolása
    extract_images(dest_img_zip, images_dir,                            # Képek kicsomagolása
                   target_classes_only=args.target_classes_only)        # Csak a 4 célosztály kibontása

    print("\n[Kész] Az ExDark adathalmaz letöltése és kibontása sikeresen befejeződött!")  # Végső siker


if __name__ == "__main__":
    main()                                                              # Fő belépési pont meghívása
