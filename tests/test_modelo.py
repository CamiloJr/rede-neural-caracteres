import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelo import RedeNeural
from dados import (validar_amostra, exemplos_demonstracao, matrizes, contradicoes,
                   salvar_projeto, carregar_projeto)


class ModeloTest(unittest.TestCase):
    def setUp(self):
        self.samples = exemplos_demonstracao()
        self.X, self.y, self.classes = matrizes(self.samples)
        self.net = RedeNeural(self.classes, 12)

    def test_softmax_and_shapes(self):
        h, p = self.net.propagar(self.X)
        self.assertEqual(h.shape, (12, 12))
        self.assertEqual(p.shape, (12, 3))
        np.testing.assert_allclose(p.sum(axis=1), 1)
        self.assertTrue(np.all(h >= 0))
        self.assertTrue(np.all(p >= 0))
        self.assertTrue(np.isfinite(p).all())

    def test_backprop_matches_numerical_gradient(self):
        gradients = self.net.gradientes(self.X, self.y)
        candidates = [("W1", (2, 3), gradients[0]), ("b1", (4,), gradients[1]),
                      ("W2", (4, 1), gradients[2]), ("b2", (1,), gradients[3])]
        eps = 1e-6
        for name, index, analytic in candidates:
            with self.subTest(parameter=name):
                arr = getattr(self.net, name)
                original = arr[index]
                arr[index] = original + eps
                plus = self.net.avaliar(self.X, self.y)[0]
                arr[index] = original - eps
                minus = self.net.avaliar(self.X, self.y)[0]
                arr[index] = original
                self.assertAlmostEqual((plus - minus) / (2 * eps), analytic[index], places=6)

    def test_training_reduces_error_and_recognizes_training_samples(self):
        initial_loss, _ = self.net.avaliar(self.X, self.y)
        for _ in range(300):
            loss, accuracy = self.net.treinar_epoca(self.X, self.y, .1)
        self.assertLess(loss, initial_loss * .1)
        self.assertEqual(accuracy, 1.0)
        _, p = self.net.propagar(self.X)
        self.assertEqual([self.classes[i] for i in np.argmax(p, axis=1)], [x["rotulo"] for x in self.samples])

    def test_contributions_reconstruct_layer_sums(self):
        x = self.X[0]
        h, _ = self.net.propagar(x)
        np.testing.assert_allclose((x[:, None] * self.net.W1).sum(axis=0) + self.net.b1, x @ self.net.W1 + self.net.b1)
        np.testing.assert_allclose((h[0][:, None] * self.net.W2).sum(axis=0) + self.net.b2, h[0] @ self.net.W2 + self.net.b2)

    def test_project_round_trip_and_continue(self):
        history = []
        for epoch in range(8):
            loss, acc = self.net.treinar_epoca(self.X, self.y, .1)
            history.append({"epoca": epoch + 1, "erro": loss, "acuracia": acc})
        config = {"ocultos": 12, "epocas": 300, "taxa": .1, "ritmo": 1}
        with tempfile.TemporaryDirectory() as d:
            f = str(Path(d) / "projeto.json")
            salvar_projeto(f, self.samples, self.net, history, config)
            loaded_samples, loaded_net, loaded_history, loaded_config = carregar_projeto(f)
            self.assertEqual(loaded_samples, self.samples)
            self.assertEqual(loaded_history, history)
            self.assertEqual(loaded_config, config)
            np.testing.assert_allclose(loaded_net.propagar(self.X)[1], self.net.propagar(self.X)[1], atol=0, rtol=0)
            a = self.net.treinar_epoca(self.X, self.y, .1)
            b = loaded_net.treinar_epoca(self.X, self.y, .1)
            np.testing.assert_allclose(a, b, atol=0, rtol=0)

    def test_rejects_invalid_files_and_patterns(self):
        with self.assertRaises(ValueError):
            validar_amostra({"rotulo": "A", "pixels": [0] * 40})
        with self.assertRaises(ValueError):
            validar_amostra({"rotulo": "A", "pixels": [1] * 35})
        self.assertEqual(contradicoes([self.samples[0], {"rotulo": "B", "pixels": self.samples[0]["pixels"]}]), 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ruim.json"
            path.write_text(json.dumps({"formato": "rede-caracteres-5x8", "versao": 1, "amostras": [{"rotulo":"A", "pixels":[9]*40}]}))
            with self.assertRaises(ValueError):
                carregar_projeto(str(path))


if __name__ == "__main__":
    unittest.main()
