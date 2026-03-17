# scripts/download_data.py
import os
import shutil
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
REQUIRED_FILES = [
    "articles.csv",
    "customers.csv",
    "transactions_train.csv",
    "sample_submission.csv",
]


def download_competition_data(competition_name: str, destination: Path) -> None:
    """
    Скачивает файлы соревнования Kaggle через kagglehub и копирует их в destination.
    """
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "Библиотека kagglehub не установлена. Установите её: pip install kagglehub"
        )

    print(f"⏳ Загрузка данных для соревнования '{competition_name}'...")
    try:
        # kagglehub.competition_download возвращает путь к папке с файлами (уже распакованными)
        downloaded_path = kagglehub.competition_download(competition_name)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки через kagglehub: {e}")

    # Создаём директорию назначения, если её нет
    destination.mkdir(parents=True, exist_ok=True)

    # Копируем все файлы из временной папки в destination
    for file_name in os.listdir(downloaded_path):
        src = os.path.join(downloaded_path, file_name)
        dst = destination / file_name
        shutil.copy2(src, dst)  # copy2 сохраняет метаданные
        print(f"   Скопирован {file_name}")

    print(f"✅ Все файлы скопированы в {destination}")


def main():
    # Если все необходимые файлы уже есть — выходим
    if all((RAW_DATA_DIR / f).exists() for f in REQUIRED_FILES):
        print("✅ Все CSV уже скачаны в data/raw/. Пропускаем загрузку.")
        return

    # Иначе — качаем через Kaggle API
    competition = "h-and-m-personalized-fashion-recommendations"
    try:
        download_competition_data(competition, RAW_DATA_DIR)
    except Exception as e:
        print(f"❌ Не удалось загрузить данные: {e}")
        print("Убедитесь, что у вас есть доступ к интернету и настроены учетные данные Kaggle.")
        print("Инструкция по настройке: https://www.kaggle.com/docs/api#authentication")
        raise

    # Проверяем, что все файлы действительно появились
    missing = [f for f in REQUIRED_FILES if not (RAW_DATA_DIR / f).exists()]
    if missing:
        print(f"⚠️ После загрузки не найдены файлы: {missing}")
    else:
        print("✅ Загрузка завершена успешно.")


if __name__ == "__main__":
    main()