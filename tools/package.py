"""Сборка пакета модели для «Контура».

Пакет должен проверяться до распаковки, поэтому контрольные суммы
считаются и для каждого файла, и для архива целиком: приложение
скачивает его по сети, а всё, что пришло по сети, доверия не заслуживает.
"""
import hashlib, json, sys, zipfile
from pathlib import Path

FILES = [
    "model_quantized.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]

NOTICE = """Пакет модели для приложения «Контур»

Содержит производную работу от модели
  viktor-shcherb/sberbank-rubert-base-collection3
  https://huggingface.co/viktor-shcherb/sberbank-rubert-base-collection3
  Лицензия: Apache License 2.0

Исходная модель, в свою очередь, основана на
  ai-forever/ruBert-base (SberDevices), Apache License 2.0
и обучена на корпусе RCC-MSU/collection3.

Внесённые изменения:
  1. Экспорт из формата PyTorch в ONNX.
  2. Динамическое квантование весов в int8.

Ни архитектура, ни обученные веса иным образом не изменялись. Текст
лицензии Apache 2.0 приложен в файле LICENSE.
"""

def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def main(build_dir: str, out_dir: str, license_path: str, version: str) -> None:
    build, out = Path(build_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "id": "ru-ner-collection3",
        "version": version,
        "task": "token-classification",
        "language": "ru",
        "entities": ["PER", "ORG", "LOC"],
        "runtime": "onnxruntime",
        "source": "viktor-shcherb/sberbank-rubert-base-collection3",
        "license": "Apache-2.0",
        "quantization": "int8-dynamic",
        "files": {name: {"sha256": digest(build / name), "bytes": (build / name).stat().st_size} for name in FILES},
    }

    archive = out / f"ru-ner-collection3-{version}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for name in FILES:
            zip_file.write(build / name, name)
        zip_file.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zip_file.writestr("NOTICE", NOTICE)
        zip_file.write(license_path, "LICENSE")

    index = {
        **{key: manifest[key] for key in ("id", "version", "task", "language", "entities", "runtime", "source", "license", "quantization")},
        "archive": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": digest(archive),
    }
    (out / "manifest.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
