import os
import shutil
from PIL import Image
from ultralytics import YOLO
import random
import sys
import subprocess

def main():
    # Proverava da li su potrebni paketi instalirani
    try:
        import pillow
        import ultralytics
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "ultralytics"])

    # Definiše putanje
    zip_file_path = "luna.v1-dataset-v1.yolov8.zip"
    extracted_folder_path = "dataset"
    trained_model_path = "bestCiggareteButt.pt"  # ili koristite "lastCiggareteButt.pt"
    results_save_path = "results"
    expected_dataset_path = 'datasets/dataset'

    # Raspakuje zip fajl ako već nije raspakovan
    if not os.path.exists(extracted_folder_path):
        os.makedirs(extracted_folder_path)
        shutil.unpack_archive(zip_file_path, extracted_folder_path)
        print("Fajl je uspešno raspakovan i premešten!")
    else:
        print("Dataset je već raspakovan.")

    # Funkcija za promenu veličine slika
    def resize_images(directory):
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    filepath = os.path.join(subdir, file)
                    with Image.open(filepath) as img:
                        img = img.resize((640, 480))
                        img.save(filepath)

    # Menja veličinu slika u trenirajućem, testnom i validacionom direktorijumu
    resize_images('dataset/train/images')
    resize_images('dataset/test/images')
    resize_images('dataset/valid/images')

    # Proverava da li data.yaml fajl postoji
    data_yaml_path = 'dataset/data.yaml'
    if not os.path.exists(data_yaml_path):
        print(f"Fajl {data_yaml_path} ne postoji.")
        print(f"Trenutni radni direktorijum: {os.getcwd()}")
        print(f"Sadržaj dataset direktorijuma: {os.listdir('dataset')}")
        print(f"Sadržaj validacionog direktorijuma: {os.listdir('dataset/valid')}")

    # Premesti dataset na očekivanu lokaciju ako je potrebno
    if not os.path.exists(expected_dataset_path):
        os.makedirs('datasets', exist_ok=True)
        shutil.move('dataset', 'datasets/')
        print("Dataset je premešten na očekivanu lokaciju.")
    else:
        print("Dataset je već na očekivanoj lokaciji.")

    # Proverava da li je putanja do data.yaml fajla ispravna
    data_yaml_path = os.path.join(expected_dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        print(f"Fajl {data_yaml_path} ne postoji.")
        print(f"Trenutni radni direktorijum: {os.getcwd()}")
        print(f"Sadržaj dataset direktorijuma: {os.listdir(expected_dataset_path)}")
        print(f"Sadržaj validacionog direktorijuma: {os.listdir(os.path.join(expected_dataset_path, 'valid'))}")

    # Ponovo menja veličinu slika nakon premještanja dataset-a
    resize_images(os.path.join(expected_dataset_path, 'train/images'))
    resize_images(os.path.join(expected_dataset_path, 'test/images'))
    resize_images(os.path.join(expected_dataset_path, 'valid/images'))

    # Koristi unapred obučeni model za inferenciju
    model = YOLO(trained_model_path)

    # Definiše putanju do test slika
    test_images_path = os.path.join(expected_dataset_path, 'test/images')

    # Kreira direktorijum za rezultate ako ne postoji
    os.makedirs(results_save_path, exist_ok=True)

    # Funkcija za izbor nasumične slike
    def get_random_image(directory):
        images = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
        return os.path.join(directory, random.choice(images))

    # Dobija nasumičnu sliku iz test seta za inferenciju
    inference_image_path = get_random_image(test_images_path)
    print(f"Odabrana slika za inferenciju: {inference_image_path}")

    # Izvršava inferenciju
    results = model(inference_image_path)

    # Prikazuje i čuva rezultate
    for result in results:
        result.plot(show=True)
    for result in results:
        result_index = len(os.listdir(results_save_path))
        result_save_path = os.path.join(results_save_path, f"result_{result_index}.jpg")
        result.save(result_save_path)
        print(f"Rezultat je sačuvan u {result_save_path}")

if __name__ == "__main__":
    main()
