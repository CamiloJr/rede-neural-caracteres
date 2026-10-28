"""Dados, exemplos didáticos e persistência em JSON (sem executar código)."""
from __future__ import annotations

import json
import os
import string
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from modelo import ENTRADAS, RedeNeural

COLUNAS, LINHAS = 5, 8
CARACTERES = string.ascii_uppercase + string.digits
LIMITE_AMOSTRAS = 2000


def padrao(*rows: str) -> list[int]:
    if len(rows) != LINHAS or any(len(row) != COLUNAS or set(row) - {"0", "1"} for row in rows):
        raise ValueError("Padrão de exemplo inválido.")
    return [int(c) for row in rows for c in row]


def exemplos_demonstracao() -> list[dict]:
    # Variações manuais para mostrar que uma classe pode ter várias amostras.
    padroes = {
        "A": [
            ("01110","10001","10001","11111","10001","10001","10001","00000"),
            ("00100","01010","10001","11111","10001","10001","10001","00000"),
            ("01110","10001","10001","11111","10001","10001","00001","00000"),
            ("01100","10010","10010","11110","10010","10010","10010","00000"),
        ],
        "B": [
            ("11110","10001","10001","11110","10001","10001","11110","00000"),
            ("11100","10010","10010","11100","10010","10010","11100","00000"),
            ("11110","10001","10001","11110","10001","10001","11111","00000"),
            ("11110","10001","10001","11110","10001","10001","11100","00000"),
        ],
        "C": [
            ("01111","10000","10000","10000","10000","10000","01111","00000"),
            ("00111","01000","10000","10000","10000","01000","00111","00000"),
            ("01110","10001","10000","10000","10000","10001","01110","00000"),
            ("01111","10000","10000","10000","10000","10000","00111","00000"),
        ],
    }
    return [{"rotulo": k, "pixels": padrao(*rows)} for k, variants in padroes.items() for rows in variants]


def validar_amostra(item: dict) -> dict:
    if not isinstance(item, dict):
        raise ValueError("Amostra inválida.")
    label = item.get("rotulo")
    pixels = item.get("pixels")
    if not isinstance(label, str) or len(label) != 1 or label not in CARACTERES:
        raise ValueError("Cada rótulo deve ser uma letra maiúscula de A a Z ou um dígito.")
    if not isinstance(pixels, list) or len(pixels) != ENTRADAS or any(type(x) is not int or x not in (0, 1) for x in pixels):
        raise ValueError("Cada desenho deve ter 40 pixels binários.")
    if not any(pixels):
        raise ValueError("Um desenho de treinamento não pode ficar em branco.")
    return {"rotulo": label, "pixels": list(pixels)}


def matrizes(amostras: list[dict], classes: list[str] | None = None):
    classes = list(classes) if classes is not None else sorted({a["rotulo"] for a in amostras})
    indices = {c: i for i, c in enumerate(classes)}
    X = np.array([a["pixels"] for a in amostras], dtype=np.float64)
    y = np.array([indices[a["rotulo"]] for a in amostras], dtype=int)
    return X, y, classes


def contradicoes(amostras: list[dict]) -> int:
    labels_by_picture: dict[tuple[int, ...], set[str]] = {}
    for item in amostras:
        labels_by_picture.setdefault(tuple(item["pixels"]), set()).add(item["rotulo"])
    return sum(len(labels) > 1 for labels in labels_by_picture.values())


def contagens(amostras: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(a["rotulo"] for a in amostras).items()))


def salvar_projeto(path: str, amostras, rede, historico, config):
    payload = {"formato": "rede-caracteres-5x8", "versao": 1,
               "amostras": amostras, "rede": rede.para_dict() if rede else None,
               "historico": historico, "configuracao": config}
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Gravação atômica: uma falha não destrói o arquivo anterior.
    fd, temp_name = tempfile.mkstemp(prefix=".rede_", suffix=".tmp", dir=out.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temp_name, out)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def carregar_projeto(path: str):
    if Path(path).stat().st_size > 20 * 1024 * 1024:
        raise ValueError("O arquivo excede 20 MB.")
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or obj.get("formato") != "rede-caracteres-5x8" or obj.get("versao") != 1:
        raise ValueError("Este arquivo não é um projeto compatível.")
    raw = obj.get("amostras")
    if not isinstance(raw, list) or len(raw) > LIMITE_AMOSTRAS:
        raise ValueError("Lista de amostras inválida.")
    amostras = [validar_amostra(item) for item in raw]
    net_data = obj.get("rede")
    rede = RedeNeural.de_dict(net_data) if net_data is not None else None
    if rede and (set(rede.classes) != set(a["rotulo"] for a in amostras) or
                 any(len(c) != 1 or c not in CARACTERES for c in rede.classes)):
        raise ValueError("As classes da rede não correspondem às amostras.")
    hist = obj.get("historico", [])
    if not isinstance(hist, list) or len(hist) > 100000 or (hist and not rede):
        raise ValueError("Histórico de treinamento inválido.")
    history = []
    for i, entry in enumerate(hist, 1):
        if not isinstance(entry, dict) or entry.get("epoca") != i:
            raise ValueError("Sequência de épocas inválida.")
        loss, accuracy = float(entry["erro"]), float(entry["acuracia"])
        if not np.isfinite(loss) or not np.isfinite(accuracy) or loss < 0 or not (0 <= accuracy <= 1):
            raise ValueError("Métricas de treinamento inválidas.")
        history.append({"epoca": i, "erro": loss, "acuracia": accuracy})
    config = obj.get("configuracao", {})
    if not isinstance(config, dict):
        raise ValueError("Configuração inválida.")
    return amostras, rede, history, config
