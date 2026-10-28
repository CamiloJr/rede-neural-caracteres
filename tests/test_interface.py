import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pygame
import numpy as np
from interface import App


class InterfaceTest(unittest.TestCase):
    def setUp(self):
        self.app = App(size=(1600, 900))
        self.app.render()

    def tearDown(self):
        pygame.quit()

    def click(self, name, button=1):
        self.app.render()
        rect, _, enabled = self.app.buttons[name]
        self.assertTrue(enabled, name)
        pos = rect.center
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": button, "pos": pos}))
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": button, "pos": pos}))

    def paint(self, pos, button=1):
        self.app.render()
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": button, "pos": pos}))
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": button, "pos": pos}))

    def type_label(self, text):
        self.click("label")
        self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": ord(text.lower()), "unicode": text}))

    def test_mouse_keyboard_training_inference_end_to_end(self):
        self.paint((56, 233))
        self.assertEqual(self.app.train_pixels[0], 1)
        self.paint((56, 233), button=3)
        self.assertEqual(self.app.train_pixels[0], 0)
        self.paint((56, 233))
        self.type_label("a")
        self.click("add")
        self.assertEqual(self.app.samples[0]["rotulo"], "A")
        self.click("step")
        self.assertIsNone(self.app.net)
        self.paint((93, 233))
        self.type_label("b")
        self.click("add")
        self.click("step")
        self.assertEqual(len(self.app.history), 1)
        self.app.advance_epochs(99)
        self.click("tab_infer")
        self.paint((118, 243))
        result = self.app.inference_values()
        self.assertIsNotNone(result)
        self.assertEqual(self.app.net.classes[int(np.argmax(result[1]))], "A")
        self.paint((118, 243), button=3)
        self.assertIsNone(self.app.inference_values())

    def test_demo_running_parameter_view_and_project(self):
        self.click("demo")
        self.click("start")
        self.assertTrue(self.app.running)
        self.app.advance_epochs(20)
        self.click("weights")
        self.app.render()
        self.assertEqual(len(self.app.parameter_entries()), 40*16 + 16 + 16*3 + 3)
        before = self.app.parameter_entries()[0][1]
        self.app.advance_epochs(5)
        after = self.app.parameter_entries()[0][1]
        self.assertNotEqual(before, after)
        self.click("param_next")
        self.assertEqual(self.app.parameter_offset, 36)
        self.click("param_close")
        self.click("pause")
        with tempfile.TemporaryDirectory() as d:
            file = str(Path(d) / "p.json")
            with patch.object(self.app, "file_dialog", return_value=file):
                self.app.save_file()
            expected = self.app.net.para_dict()
            self.app.clear_samples()
            with patch.object(self.app, "file_dialog", return_value=file):
                self.app.open_file()
            self.assertEqual(len(self.app.history), 25)
            self.assertEqual(self.app.net.para_dict(), expected)
            self.click("step")
            self.assertEqual(len(self.app.history), 26)

    def test_fast_drag_fills_intermediate_cells(self):
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (57, 234)}))
        self.app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, {"pos": (205, 234)}))
        self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (205, 234)}))
        self.assertEqual(self.app.train_pixels[:5], [1, 1, 1, 1, 1])

    def test_many_classes_paging_and_render(self):
        for i, c in enumerate("ABCDEFGHIJ"):
            self.app.train_pixels = [0] * 40
            self.app.train_pixels[i] = 1
            self.app.label = c
            self.app.add_sample()
        self.app.one_epoch()
        self.app.set_page(1)
        self.app.copy_sample()
        self.app.render()
        self.assertTrue(self.app.buttons["pred_next"][2])
        self.click("pred_next")
        self.assertEqual(self.app.inference_offset, 8)
        self.app.render()
        self.click("pred_prev")
        self.assertEqual(self.app.inference_offset, 0)

    def test_invalid_open_does_not_replace_current_data(self):
        self.app.add_demo()
        previous = list(self.app.samples)
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "broken.json"
            bad.write_text('{"formato": "incorreto"}', encoding="utf-8")
            with patch.object(self.app, "file_dialog", return_value=str(bad)):
                self.app.open_file()
        self.assertEqual(self.app.samples, previous)
        self.assertIn("Não foi possível", self.app.status)


if __name__ == "__main__":
    unittest.main()
