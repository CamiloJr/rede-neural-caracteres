"""Rede neural pequena, implementada com NumPy para fins didáticos.

Entrada binária -> camada oculta ReLU -> saída softmax. O treinamento usa
entropia cruzada e gradiente descendente em lote, uma atualização por época.
"""
from __future__ import annotations

import numpy as np

ENTRADAS = 40


class RedeNeural:
    def __init__(self, classes: list[str], ocultos: int = 16, seed: int = 42):
        if not 2 <= len(classes) <= 36 or len(set(classes)) != len(classes):
            raise ValueError("A rede precisa de 2 a 36 classes distintas.")
        if not 2 <= ocultos <= 64:
            raise ValueError("Quantidade de neurônios ocultos inválida.")
        self.classes = list(classes)
        self.ocultos = int(ocultos)
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / ENTRADAS), (ENTRADAS, ocultos))
        self.b1 = np.zeros(ocultos)
        self.W2 = rng.normal(0, np.sqrt(2 / ocultos), (ocultos, len(classes)))
        self.b2 = np.zeros(len(classes))

    def propagar(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != ENTRADAS or not np.isfinite(X).all():
            raise ValueError("A entrada deve conter 40 valores finitos por amostra.")
        z1 = X @ self.W1 + self.b1
        h = np.maximum(0.0, z1)  # ReLU
        z2 = h @ self.W2 + self.b2
        z2 = z2 - z2.max(axis=1, keepdims=True)  # softmax estável
        exp = np.exp(z2)
        return h, exp / exp.sum(axis=1, keepdims=True)

    def avaliar(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        y = np.asarray(y, dtype=int)
        if len(y) == 0:
            raise ValueError("A base não pode estar vazia.")
        _, probs = self.propagar(X)
        loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1)).mean()
        acc = np.mean(np.argmax(probs, axis=1) == y)
        return float(loss), float(acc)

    def gradientes(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) == 0 or len(y) != len(X):
            raise ValueError("Base de treinamento inválida.")
        h, probs = self.propagar(X)
        if np.any(y < 0) or np.any(y >= len(self.classes)):
            raise ValueError("Índice de classe inválido.")
        dz2 = probs.copy()
        dz2[np.arange(len(y)), y] -= 1
        dz2 /= len(y)
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.W2.T) * (h > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return dW1, db1, dW2, db2

    def treinar_epoca(self, X: np.ndarray, y: np.ndarray, taxa: float):
        if not (0 < taxa <= 1):
            raise ValueError("A taxa de aprendizado deve estar entre 0 e 1.")
        dW1, db1, dW2, db2 = self.gradientes(X, y)
        self.W1 -= taxa * dW1
        self.b1 -= taxa * db1
        self.W2 -= taxa * dW2
        self.b2 -= taxa * db2
        # Métricas são calculadas após a atualização dos pesos.
        return self.avaliar(X, y)

    def para_dict(self) -> dict:
        return {"classes": self.classes, "ocultos": self.ocultos,
                "W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist()}

    @classmethod
    def de_dict(cls, obj: dict):
        rede = cls(obj["classes"], int(obj["ocultos"]))
        shapes = {"W1": (ENTRADAS, rede.ocultos), "b1": (rede.ocultos,),
                  "W2": (rede.ocultos, len(rede.classes)), "b2": (len(rede.classes),)}
        for nome, shape in shapes.items():
            array = np.asarray(obj[nome], dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Parâmetros inválidos da rede: {nome}.")
            setattr(rede, nome, array)
        return rede
