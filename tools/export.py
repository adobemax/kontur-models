"""Экспорт модели поиска сущностей в ONNX с квантованием.

Приложение работает без Python, поэтому модель нужна в формате, который
исполняет onnxruntime прямо в Node. Динамическое квантование в int8
уменьшает файл вчетверо — иначе пакет весил бы больше самого приложения.
"""
import json, shutil, sys
from pathlib import Path

from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

SOURCE = "viktor-shcherb/sberbank-rubert-base-collection3"

def main(out_dir: str) -> None:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    raw = out / "onnx-fp32"
    model = ORTModelForTokenClassification.from_pretrained(SOURCE, export=True)
    model.save_pretrained(raw)
    AutoTokenizer.from_pretrained(SOURCE).save_pretrained(raw)

    quantizer = ORTQuantizer.from_pretrained(raw)
    # Динамическое квантование не привязано к набору инструкций процессора:
    # один файл работает и на Apple Silicon, и на обычных серверах.
    quantizer.quantize(
        save_dir=out,
        quantization_config=AutoQuantizationConfig.arm64(is_static=False, per_channel=False),
    )
    AutoTokenizer.from_pretrained(SOURCE).save_pretrained(out)

    for name in ("config.json",):
        shutil.copy(raw / name, out / name)

    sizes = {p.name: p.stat().st_size for p in sorted(out.glob("*")) if p.is_file()}
    print(json.dumps(sizes, ensure_ascii=False, indent=2))
    print("исходный fp32:", sum(p.stat().st_size for p in raw.glob("*.onnx")))

if __name__ == "__main__":
    main(sys.argv[1])
