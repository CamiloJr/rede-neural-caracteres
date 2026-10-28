"""Interface gráfica em Pygame. Uma janela com duas etapas integradas."""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import numpy as np
import pygame

from dados import (CARACTERES, COLUNAS, LINHAS, LIMITE_AMOSTRAS, carregar_projeto,
                   contagens, contradicoes, exemplos_demonstracao, matrizes,
                   salvar_projeto, validar_amostra)
from modelo import ENTRADAS, RedeNeural

W, H = 1600, 900
BG = (10, 12, 17)
PANEL = (19, 23, 31)
PANEL_DARK = (14, 17, 24)
BORDER = (54, 63, 76)
TEXT = (234, 240, 247)
MUTED = (154, 169, 185)
FAINT = (101, 118, 136)
CYAN = (70, 211, 220)
BLUE = (68, 134, 219)
GREEN = (83, 206, 147)
RED = (243, 111, 111)
GOLD = (247, 190, 92)
BUTTON = (42, 50, 62)
BTN_HOVER = (60, 72, 86)


def dpi_awareness():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


class App:
    def __init__(self, size=None):
        dpi_awareness()
        pygame.init()
        pygame.display.set_caption("Aprendizado de Máquina | Caracteres 5 × 8")
        info = pygame.display.Info()
        if size is None:
            desktop_w = info.current_w if info.current_w >= 800 else W
            desktop_h = info.current_h if info.current_h >= 600 else H
            scale = min(1.0, desktop_w * .96 / W, desktop_h * .92 / H)
            size = (max(800, int(W * scale)), max(450, int(H * scale)))
        self.window = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.canvas = pygame.Surface((W, H))
        self.clock = pygame.time.Clock()
        self.fonts = {
            "title": pygame.font.SysFont("segoeui,arial", 29, bold=True),
            "head": pygame.font.SysFont("segoeui,arial", 20, bold=True),
            "body": pygame.font.SysFont("segoeui,arial", 17),
            "bold": pygame.font.SysFont("segoeui,arial", 17, bold=True),
            "small": pygame.font.SysFont("segoeui,arial", 14),
            "tiny": pygame.font.SysFont("segoeui,arial", 12),
            "large": pygame.font.SysFont("segoeui,arial", 58, bold=True),
            "digit": pygame.font.SysFont("consolas,monospace", 24, bold=True),
            "mono": pygame.font.SysFont("consolas,monospace", 15),
        }
        self.page = 0
        self.train_pixels = [0] * ENTRADAS
        self.infer_pixels = [0] * ENTRADAS
        self.label = ""
        self.typing = False
        self.samples: list[dict] = []
        self.selected = -1
        self.sample_offset = 0
        self.net: RedeNeural | None = None
        self.history: list[dict] = []
        self.history_offset = 0
        self.hidden_options = [8, 12, 16, 24]
        self.epoch_options = [100, 300, 600, 1000, 2000, 5000]
        self.rate_options = [.01, .03, .1, .2, .4]
        self.hidden = 16
        self.epochs = 600
        self.rate = .1
        self.speed_options = [1, 10, 50]
        self.speed = 1
        self.running = False
        self.last_tick = 0.0
        self.X = self.y = None
        self.inference_offset = 0
        self.parameter_offset = 0
        self.status = "Desenhe um caractere, informe seu rótulo e adicione exemplos."
        self.modal = None
        self.buttons = {}
        self.sample_rows = []
        self.mouse_mode = None
        self.last_cell = None
        self.alive = True
        self.viewport = (0, 0, 1)

    def font(self, name):
        return self.fonts[name]

    def text(self, value, x, y, font="body", color=TEXT, max_width=None):
        value = str(value)
        f = self.font(font)
        if max_width is not None and f.size(value)[0] > max_width:
            while value and f.size(value + "…")[0] > max_width:
                value = value[:-1]
            value += "…"
        self.canvas.blit(f.render(value, True, color), (int(x), int(y)))

    def centered(self, value, rect, font="body", color=TEXT):
        surf = self.font(font).render(str(value), True, color)
        self.canvas.blit(surf, surf.get_rect(center=pygame.Rect(rect).center))

    def panel(self, rect, color=PANEL, border=BORDER, radius=12):
        rect = pygame.Rect(rect)
        pygame.draw.rect(self.canvas, color, rect, border_radius=radius)
        pygame.draw.rect(self.canvas, border, rect, width=1, border_radius=radius)

    def button(self, key, text, rect, action, kind="normal", enabled=True, font="small"):
        rect = pygame.Rect(rect)
        mouse = self.logical_mouse(pygame.mouse.get_pos())
        hover = rect.collidepoint(mouse) and enabled
        if not enabled:
            bg, fg = (31, 36, 44), (96, 108, 121)
        elif kind == "primary":
            bg, fg = ((51, 172, 183) if hover else (34, 140, 153)), (255, 255, 255)
        elif kind == "active":
            bg, fg = (42, 99, 123), (237, 252, 255)
        elif kind == "danger":
            bg, fg = ((99, 56, 61) if hover else (71, 44, 49)), (250, 208, 208)
        else:
            bg, fg = (BTN_HOVER if hover else BUTTON), TEXT
        pygame.draw.rect(self.canvas, bg, rect, border_radius=7)
        pygame.draw.rect(self.canvas, BORDER, rect, 1, border_radius=7)
        self.centered(text, rect, font, fg)
        self.buttons[key] = (rect, action, enabled)

    def logical_mouse(self, pos):
        x0, y0, scale = self.viewport
        return int((pos[0] - x0) / scale), int((pos[1] - y0) / scale)

    def header(self):
        self.text("Aprendizado de Máquina", 26, 20, "title")
        self.text("Uma rede neural que aprende caracteres desenhados", 28, 59, "body", MUTED)
        self.text("VISUALIZADOR DIDÁTICO  •  5 × 8", 1285, 29, "small", CYAN)
        self.button("help", "Como usar", (1112, 75, 132, 39), self.show_help)
        self.button("open", "Abrir projeto", (1253, 75, 150, 39), self.open_request)
        self.button("save", "Salvar projeto", (1412, 75, 164, 39), self.save_file)
        self.button("tab_train", "1   Dados e aprendizado", (26, 88, 255, 43),
                    lambda: self.set_page(0), "active" if self.page == 0 else "normal", font="bold")
        self.button("tab_infer", "2   Inferência ao vivo", (289, 88, 246, 43),
                    lambda: self.set_page(1), "active" if self.page == 1 else "normal", font="bold")
        if self.running:
            self.text("● TREINANDO", 557, 101, "small", GREEN)
        elif self.net and self.history:
            self.text("● REDE TREINADA", 557, 101, "small", CYAN)
        else:
            self.text("● AGUARDANDO TREINAMENTO", 557, 101, "small", MUTED)

    def footer(self):
        pygame.draw.line(self.canvas, BORDER, (26, 850), (1575, 850))
        self.text("STATUS", 27, 866, "small", CYAN)
        self.text(self.status, 95, 864, "body", TEXT, max_width=1455)

    def title(self, num, title, x, y):
        self.text(num, x, y, "head", CYAN)
        self.text(title, x + 43, y, "head")

    def draw_grid(self, x, y, cell, values, tag):
        self.grid_rect = pygame.Rect(x, y, COLUNAS * cell, LINHAS * cell)
        self.grid_cell_size = cell
        self.grid_tag = tag
        pygame.draw.rect(self.canvas, (5, 8, 12), self.grid_rect, border_radius=5)
        for r in range(LINHAS):
            for c in range(COLUNAS):
                i = r * COLUNAS + c
                square = pygame.Rect(x + c * cell + 2, y + r * cell + 2, cell - 4, cell - 4)
                color = CYAN if values[i] else (34, 43, 54)
                pygame.draw.rect(self.canvas, color, square, border_radius=3)
                if not values[i]:
                    pygame.draw.rect(self.canvas, (58, 69, 81), square, width=1, border_radius=3)

    def miniature(self, values, x, y, cell=3):
        for r in range(LINHAS):
            for c in range(COLUNAS):
                color = CYAN if values[r * COLUNAS + c] else (49, 58, 69)
                pygame.draw.rect(self.canvas, color, (x + c * cell, y + r * cell, cell - 1, cell - 1))

    def train_page(self):
        self.panel((24, 145, 410, 694))
        self.panel((450, 145, 490, 694))
        self.panel((956, 145, 620, 694))
        self.title("01", "EXEMPLOS", 43, 162)
        self.text("Desenhe e atribua um rótulo ao exemplo.", 43, 197, "body", MUTED)
        self.draw_grid(53, 230, 37, self.train_pixels, "train")
        self.text("RÓTULO", 258, 235, "small", MUTED)
        label_rect = pygame.Rect(258, 260, 144, 58)
        self.panel(label_rect, PANEL_DARK, CYAN if self.typing else BORDER, 7)
        self.centered(self.label or "A–Z / 0–9", label_rect, "digit" if self.label else "small", TEXT if self.label else FAINT)
        self.buttons["label"] = (label_rect, self.focus_label, not self.running)
        self.button("add", "Adicionar exemplo", (252, 338, 160, 42), self.add_sample, "primary", not self.running)
        self.button("clear_grid", "Limpar desenho", (252, 388, 160, 40), self.clear_train, enabled=not self.running)
        self.button("demo", "+ Exemplos A / B / C", (252, 438, 160, 40), self.add_demo, enabled=not self.running)
        self.text("Esq.: pintar", 258, 492, "small", MUTED)
        self.text("Dir.: apagar", 258, 514, "small", MUTED)
        self.text("BASE DE EXEMPLOS", 44, 551, "bold")
        self.text(f"{len(self.samples)} amostras  •  {len(contagens(self.samples))} classes", 44, 577, "small", MUTED)
        self.button("sample_up", "▲", (343, 550, 30, 28), lambda: self.scroll_samples(-1), enabled=self.sample_offset > 0)
        self.button("sample_down", "▼", (378, 550, 30, 28), lambda: self.scroll_samples(1), enabled=self.sample_offset + 4 < len(self.samples))
        self.sample_rows = []
        for j, idx in enumerate(range(self.sample_offset, min(len(self.samples), self.sample_offset + 4))):
            s = self.samples[idx]
            rect = pygame.Rect(44, 603 + j * 37, 362, 33)
            pygame.draw.rect(self.canvas, (37, 70, 79) if idx == self.selected else (30, 36, 46), rect, border_radius=5)
            self.miniature(s["pixels"], rect.x + 8, rect.y + 4, 3)
            self.text(f"{s['rotulo']}", rect.x + 34, rect.y + 5, "bold", CYAN)
            n = sum(1 for z in self.samples[:idx + 1] if z["rotulo"] == s["rotulo"])
            self.text(f"Amostra {n} da classe {s['rotulo']}", rect.x + 69, rect.y + 7, "small", TEXT)
            self.text(f"#{idx + 1:03}", rect.right - 43, rect.y + 8, "tiny", MUTED)
            self.sample_rows.append((rect, idx))
        if not self.samples:
            self.text("Nenhum exemplo cadastrado.", 45, 624, "small", FAINT)
            self.text("Experimente o botão de exemplos A / B / C.", 45, 650, "small", FAINT)
        self.button("delete_sample", "Excluir selecionado", (44, 773, 174, 41), self.delete_sample,
                    enabled=bool(self.samples) and not self.running)
        self.button("clear_samples", "Limpar base", (228, 773, 178, 41), self.clear_samples_request,
                    "danger", bool(self.samples) and not self.running)
        self.training_controls()
        self.training_monitor()

    def training_controls(self):
        self.title("02", "TREINAMENTO", 471, 162)
        self.text("A rede ajusta pesos para reduzir o erro.", 471, 197, "body", MUTED)
        self.panel((471, 224, 448, 65), PANEL_DARK, BORDER, 8)
        n = len(contagens(self.samples))
        self.text("40 ENTRADAS", 490, 242, "bold", CYAN)
        self.text("→", 630, 241, "bold", MUTED)
        self.text(f"{self.hidden} OCULTOS", 658, 242, "bold", TEXT)
        self.text("→", 790, 241, "bold", MUTED)
        self.text(f"{n} SAÍDAS", 820, 242, "bold", GREEN)
        self.text("CONFIGURAÇÃO", 471, 306, "small", MUTED)
        rows = [
            ("Neurônios ocultos", str(self.hidden), "hidden", 333),
            ("Limite de épocas", str(self.epochs), "epochs", 380),
            ("Taxa de aprendizado", f"{self.rate:.2f}", "rate", 427),
        ]
        for name, value, key, y in rows:
            self.text(name, 471, y + 8, "body")
            self.button(key + "_minus", "−", (757, y, 40, 36), lambda k=key: self.change_config(k, -1), enabled=not self.running, font="bold")
            self.panel((803, y, 62, 36), PANEL_DARK, BORDER, 6)
            self.centered(value, (803, y, 62, 36), "bold", CYAN)
            self.button(key + "_plus", "+", (871, y, 40, 36), lambda k=key: self.change_config(k, 1), enabled=not self.running, font="bold")
        self.button("step", "Avançar 1 época", (471, 485, 214, 43), self.one_epoch, enabled=not self.running)
        self.button("start", "▶  Treinar", (695, 485, 224, 43), self.start, "primary", not self.running)
        self.button("pause", "Pausar", (471, 536, 214, 41), self.pause, enabled=self.running)
        self.button("reset", "Reiniciar rede", (695, 536, 224, 41), self.reset_net, enabled=bool(self.net) and not self.running)
        self.button("speed", f"Ritmo: {self.speed} época(s) por atualização", (471, 587, 292, 38), self.change_speed)
        self.button("weights", "Ver parâmetros", (773, 587, 146, 38), self.show_parameters, enabled=bool(self.net and self.history))
        self.panel((471, 642, 448, 174), PANEL_DARK, BORDER, 8)
        self.text("AMOSTRA EM FOCO", 487, 656, "small", MUTED)
        if 0 <= self.selected < len(self.samples):
            sample = self.samples[self.selected]
            self.miniature(sample["pixels"], 493, 686, 13)
            self.text(f"Rótulo esperado:  {sample['rotulo']}", 585, 690, "bold", TEXT)
            if self.net is not None and self.history:
                _, probs = self.net.propagar(np.array(sample["pixels"]))
                idx = int(np.argmax(probs[0]))
                self.text(f"Resposta da rede:  {self.net.classes[idx]}", 585, 724, "bold", GREEN if self.net.classes[idx] == sample["rotulo"] else RED)
                self.text(f"Saída relativa: {100 * probs[0, idx]:.1f}%", 585, 758, "small", MUTED)
                self.text("Clique em outra amostra para compará-la.", 585, 781, "tiny", FAINT)
            else:
                self.text("Inicie o treinamento para ver a resposta.", 585, 730, "small", MUTED)
        else:
            self.text("Selecione um desenho na lista à esquerda.", 487, 712, "body", MUTED)
            self.text("A previsão será atualizada durante o treino.", 487, 741, "small", FAINT)

    def training_monitor(self):
        self.title("03", "ACOMPANHAMENTO", 977, 162)
        self.text("Valores calculados na própria base de treinamento.", 977, 197, "body", MUTED)
        epoch = len(self.history)
        latest = self.history[-1] if self.history else None
        stats = [("ÉPOCA", f"{epoch} / {self.epochs}"),
                 ("ERRO", f"{latest['erro']:.4f}" if latest else "—"),
                 ("ACURÁCIA", f"{latest['acuracia'] * 100:.1f}%" if latest else "—"),
                 ("AMOSTRAS", str(len(self.samples)))]
        for j, (label, val) in enumerate(stats):
            x = 977 + 145 * j
            self.panel((x, 230, 133, 75), PANEL_DARK, BORDER, 8)
            self.text(label, x + 10, 241, "tiny", MUTED)
            self.text(val, x + 10, 264, "bold", GREEN if label == "ACURÁCIA" else TEXT, max_width=115)
        self.panel((977, 324, 577, 208), PANEL_DARK, BORDER, 8)
        self.text("EVOLUÇÃO POR ÉPOCA", 992, 336, "small", MUTED)
        self.draw_chart(pygame.Rect(1005, 368, 520, 128))
        self.text("erro (escala própria)", 1000, 504, "tiny", CYAN)
        self.text("acurácia: 0–100%", 1202, 504, "tiny", GREEN)
        self.text("HISTÓRICO DE ÉPOCAS", 978, 555, "bold")
        self.text("Erro e acurácia após cada atualização dos pesos", 978, 581, "small", MUTED)
        self.text("ÉPOCA", 990, 611, "small", FAINT)
        self.text("ERRO", 1145, 611, "small", FAINT)
        self.text("ACURÁCIA", 1320, 611, "small", FAINT)
        for row, entry in enumerate(self.history[self.history_offset:self.history_offset + 5]):
            y = 637 + 28 * row
            pygame.draw.line(self.canvas, (42, 49, 59), (986, y + 25), (1534, y + 25))
            self.text(str(entry["epoca"]), 990, y, "mono", TEXT)
            self.text(f"{entry['erro']:.5f}", 1145, y, "mono", CYAN)
            self.text(f"{entry['acuracia'] * 100:6.2f}%", 1320, y, "mono", GREEN)
        if not self.history:
            self.text("Execute uma época para iniciar o histórico.", 990, 650, "small", FAINT)
        self.button("hist_prev", "◀ Anteriores", (977, 781, 155, 36), lambda: self.scroll_history(-5), enabled=self.history_offset > 0)
        shown = f"{min(self.history_offset + 1, len(self.history))}–{min(self.history_offset + 5, len(self.history))} de {len(self.history)}"
        self.centered(shown, (1142, 781, 235, 36), "small", MUTED)
        self.button("hist_next", "Próximas ▶", (1390, 781, 164, 36), lambda: self.scroll_history(5), enabled=self.history_offset + 5 < len(self.history))

    def draw_chart(self, rect):
        for n in range(5):
            y = rect.top + int(rect.height * n / 4)
            pygame.draw.line(self.canvas, (42, 52, 64), (rect.left, y), (rect.right, y), 1)
        if len(self.history) < 2:
            self.centered("O gráfico aparecerá após duas épocas.", rect, "small", FAINT)
            return
        losses = np.asarray([e["erro"] for e in self.history], dtype=float)
        accs = np.asarray([e["acuracia"] for e in self.history], dtype=float)
        # Até uma amostra por pixel horizontal; preserva início e fim.
        indexes = np.unique(np.linspace(0, len(losses) - 1, min(len(losses), rect.width), dtype=int))
        max_loss = max(float(losses.max()), 1e-10)
        def coords(values, maximum):
            return [(rect.left + int(i / (len(values) - 1) * rect.width),
                     rect.bottom - int(values[i] / maximum * rect.height)) for i in indexes]
        pygame.draw.lines(self.canvas, CYAN, False, coords(losses, max_loss), 2)
        pygame.draw.lines(self.canvas, GREEN, False, coords(accs, 1.0), 2)

    def inference_page(self):
        self.panel((24, 145, 410, 694))
        self.panel((450, 145, 698, 694))
        self.panel((1164, 145, 412, 694))
        self.title("01", "DESENHO DE ENTRADA", 43, 162)
        self.text("A resposta muda enquanto você desenha.", 43, 197, "body", MUTED)
        self.draw_grid(115, 240, 43, self.infer_pixels, "infer")
        self.text("Botão esquerdo: pintar   •   Direito: apagar", 62, 606, "small", MUTED)
        self.button("infer_clear", "Limpar desenho", (57, 646, 342, 43), self.clear_inference)
        self.button("copy_sample", "Copiar amostra selecionada", (57, 698, 342, 43), self.copy_sample, enabled=0 <= self.selected < len(self.samples))
        self.text("A grade usa os mesmos 40 pixels do treinamento.", 55, 767, "small", FAINT)
        self.text("Teste também desenhos não cadastrados.", 55, 790, "small", FAINT)
        self.draw_network()
        self.draw_predictions()

    def inference_values(self):
        if self.net is None or not self.history or not any(self.infer_pixels):
            return None
        h, probs = self.net.propagar(np.asarray(self.infer_pixels, dtype=float))
        return h[0], probs[0]

    def draw_network(self):
        self.title("02", "REDE NEURAL EM AÇÃO", 471, 162)
        self.text("Contribuições locais das conexões para este desenho", 471, 197, "body", MUTED)
        if not self.net or not self.history:
            self.panel((473, 247, 651, 474), PANEL_DARK, BORDER, 8)
            self.centered("Cadastre exemplos de pelo menos duas classes", (473, 410, 651, 30), "body", MUTED)
            self.centered("e treine a rede na etapa 1.", (473, 445, 651, 30), "body", MUTED)
            return
        self.text("40 PIXELS", 494, 245, "small", MUTED)
        self.text(f"{self.net.ocultos} OCULTOS", 746, 245, "small", MUTED)
        self.text(f"{len(self.net.classes)} SAÍDAS", 1011, 245, "small", MUTED)
        values = self.inference_values()
        h = np.zeros(self.net.ocultos) if values is None else values[0]
        probs = np.zeros(len(self.net.classes)) if values is None else values[1]
        order = sorted(range(len(probs)), key=lambda i: (-probs[i], self.net.classes[i]))
        start = min(self.inference_offset, max(0, len(order) - 1))
        output_indices = order[start:start + 8]
        inputs = [(506 + (i % 5) * 15, 353 + (i // 5) * 25) for i in range(40)]
        hidden = [(790, int(y)) for y in np.linspace(301, 715, self.net.ocultos)]
        outputs = [(1043, int(y)) for y in np.linspace(319, 697, len(output_indices))] if output_indices else []
        if values is not None:
            # Each edge is an actual local term x_i*w_ij or h_j*w_jk.
            in_terms = np.asarray(self.infer_pixels, dtype=float)[:, None] * self.net.W1
            out_terms = h[:, None] * self.net.W2[:, output_indices]
            self.draw_strong_edges(inputs, hidden, in_terms, 38)
            self.draw_strong_edges(hidden, outputs, out_terms, 38)
        for i, pos in enumerate(inputs):
            active = self.infer_pixels[i] and values is not None
            pygame.draw.circle(self.canvas, CYAN if active else (55, 68, 81), pos, 5)
        max_h = max(float(np.max(h)), 1e-12)
        for j, pos in enumerate(hidden):
            amount = float(h[j] / max_h) if values is not None else 0
            color = tuple(int((1 - amount) * a + amount * b) for a, b in zip((52, 63, 76), (76, 211, 196)))
            pygame.draw.circle(self.canvas, color, pos, 7)
            pygame.draw.circle(self.canvas, (148, 178, 190), pos, 7, 1)
        for pos, idx in zip(outputs, output_indices):
            is_top = values is not None and idx == order[0]
            pygame.draw.circle(self.canvas, GREEN if is_top else (60, 89, 102), pos, 11)
            self.text(self.net.classes[idx], pos[0] + 18, pos[1] - 10, "bold", GREEN if is_top else TEXT)
        if values is None:
            self.centered("Desenhe para ativar a rede.", (500, 735, 560, 25), "small", MUTED)
        self.text("Ciano: contribuição positiva", 474, 755, "small", CYAN)
        self.text("Vermelho: contribuição negativa", 715, 755, "small", RED)
        self.text("São exibidas até 38 conexões de maior magnitude por camada; vieses não aparecem.", 474, 784, "tiny", MUTED)
        self.text("As linhas mostram a contribuição local, não uma explicação global do modelo.", 474, 804, "tiny", FAINT)

    def draw_strong_edges(self, starts, ends, terms, limit):
        if not terms.size:
            return
        flat = np.abs(terms).ravel()
        nonzero = np.flatnonzero(flat > 1e-10)
        if not len(nonzero):
            return
        selected = nonzero[np.argsort(flat[nonzero])[-limit:]]
        maxval = max(float(flat[selected].max()), 1e-12)
        for index in selected:
            i, j = np.unravel_index(index, terms.shape)
            term = float(terms[i, j])
            strength = abs(term) / maxval
            base, target = ((36, 72, 80), CYAN) if term > 0 else ((81, 46, 52), RED)
            factor = .28 + .65 * strength
            color = tuple(int(a + (b - a) * factor) for a, b in zip(base, target))
            pygame.draw.line(self.canvas, color, starts[i], ends[j], 1 if strength < .7 else 2)

    def draw_predictions(self):
        self.title("03", "RESULTADO", 1185, 162)
        self.text("Saídas para as classes aprendidas", 1185, 197, "body", MUTED)
        self.panel((1185, 231, 370, 124), PANEL_DARK, BORDER, 9)
        vals = self.inference_values()
        if vals is None:
            self.text("RECONHECIMENTO", 1203, 247, "small", MUTED)
            message = "Treine a rede primeiro" if self.net is None or not self.history else "Desenhe um caractere"
            self.text(message, 1203, 283, "body", FAINT, max_width=335)
        else:
            _, probs = vals
            best = int(np.argmax(probs))
            self.text("MAIOR SAÍDA", 1203, 247, "small", MUTED)
            self.text(self.net.classes[best], 1201, 273, "large", GREEN)
            self.text(f"{100 * probs[best]:.1f}%", 1285, 282, "head", TEXT)
            self.text("entre as classes treinadas", 1286, 313, "small", MUTED)
        self.text("SAÍDA DA REDE", 1185, 377, "bold")
        self.text("Valores relativos; não garantem acerto.", 1185, 405, "small", MUTED)
        if self.net and self.history and vals is not None:
            probs = vals[1]
            order = sorted(range(len(probs)), key=lambda i: (-probs[i], self.net.classes[i]))
            start = min(self.inference_offset, max(0, len(order) - 1))
            for row, idx in enumerate(order[start:start + 8]):
                y = 438 + 36 * row
                self.text(self.net.classes[idx], 1187, y, "bold", GREEN if idx == order[0] else TEXT)
                bar = pygame.Rect(1225, y + 5, 238, 13)
                pygame.draw.rect(self.canvas, (40, 49, 60), bar, border_radius=6)
                fill = int(bar.width * probs[idx])
                if fill > 0:
                    pygame.draw.rect(self.canvas, GREEN if idx == order[0] else CYAN, (bar.x, bar.y, fill, bar.h), border_radius=6)
                self.text(f"{100 * probs[idx]:.1f}%", 1473, y + 1, "small", TEXT)
            self.button("pred_prev", "◀", (1186, 747, 54, 36), lambda: self.scroll_predictions(-8), enabled=start > 0)
            self.centered(f"{start + 1}–{min(start + 8, len(order))} de {len(order)}", (1245, 747, 215, 36), "small", MUTED)
            self.button("pred_next", "▶", (1498, 747, 54, 36), lambda: self.scroll_predictions(8), enabled=start + 8 < len(order))
        else:
            self.text("As saídas aparecem durante o desenho.", 1185, 455, "small", FAINT)
        self.text("A maior saída é sempre uma das classes", 1185, 796, "tiny", FAINT)
        self.text("ensinadas, mesmo quando o desenho é estranho.", 1185, 813, "tiny", FAINT)

    def set_page(self, page):
        self.page = page
        self.typing = False
        self.mouse_mode = None
        if page == 1 and not self.history:
            self.status = "A inferência ficará disponível após pelo menos uma época de treinamento."
        elif page == 1:
            self.status = "Desenhe: as saídas e as conexões serão atualizadas imediatamente."
        else:
            self.status = "Cadastre exemplos; depois ajuste os parâmetros e treine a rede."

    def focus_label(self):
        self.typing = True
        self.status = "Digite uma letra de A a Z ou um dígito de 0 a 9."

    def invalidate(self):
        self.net = None
        self.history = []
        self.history_offset = 0
        self.X = self.y = None
        self.inference_offset = 0
        self.running = False

    def add_sample(self):
        self.typing = False
        if len(self.samples) >= LIMITE_AMOSTRAS:
            self.status = f"Limite de {LIMITE_AMOSTRAS} amostras atingido."
            return
        try:
            sample = validar_amostra({"rotulo": self.label, "pixels": list(self.train_pixels)})
        except ValueError as exc:
            self.status = str(exc)
            return
        model_was_present = self.net is not None
        self.samples.append(sample)
        self.selected = len(self.samples) - 1
        self.sample_offset = max(0, len(self.samples) - 4)
        self.train_pixels = [0] * ENTRADAS
        self.invalidate()
        n = contagens(self.samples)[sample["rotulo"]]
        self.status = f"Exemplo {n} da classe {sample['rotulo']} adicionado."
        if model_was_present:
            self.status += " A base mudou: treine novamente."
        if contradicoes(self.samples):
            self.status += " Atenção: há desenhos idênticos com rótulos diferentes."

    def clear_train(self):
        self.train_pixels = [0] * ENTRADAS
        self.status = "Desenho de treinamento limpo; a base foi preservada."

    def add_demo(self):
        demo = exemplos_demonstracao()
        if len(self.samples) + len(demo) > LIMITE_AMOSTRAS:
            self.status = "A base não comporta os exemplos de demonstração."
            return
        had_net = self.net is not None
        self.samples.extend(demo)
        self.selected = len(self.samples) - len(demo)
        self.sample_offset = self.selected
        self.invalidate()
        self.status = "Adicionadas 12 amostras de A, B e C. Ajuste os parâmetros e treine."
        if had_net:
            self.status += " A rede anterior foi reiniciada porque a base mudou."

    def scroll_samples(self, delta):
        self.sample_offset = max(0, min(max(0, len(self.samples) - 4), self.sample_offset + delta))

    def delete_sample(self):
        if not 0 <= self.selected < len(self.samples):
            return
        removed = self.samples.pop(self.selected)
        self.selected = min(self.selected, len(self.samples) - 1)
        self.sample_offset = min(self.sample_offset, max(0, len(self.samples) - 4))
        self.invalidate()
        self.status = f"Amostra da classe {removed['rotulo']} excluída. Treine novamente se necessário."

    def clear_samples_request(self):
        self.modal = ("Limpar base de exemplos?",
                      ["Todos os desenhos cadastrados e a rede atual serão removidos.",
                       "Salve o projeto antes, caso queira recuperá-los."], self.clear_samples)

    def clear_samples(self):
        self.samples = []
        self.selected = -1
        self.sample_offset = 0
        self.invalidate()
        self.status = "Base e rede limpas. Cadastre novos exemplos."

    def change_config(self, key, delta):
        if self.running:
            return
        attr, options = {"hidden": ("hidden", self.hidden_options),
                         "epochs": ("epochs", self.epoch_options),
                         "rate": ("rate", self.rate_options)}[key]
        old = getattr(self, attr)
        idx = options.index(old)
        new = options[max(0, min(len(options) - 1, idx + delta))]
        setattr(self, attr, new)
        if key == "hidden" and new != old and self.net is not None:
            self.invalidate()
            self.status = "Arquitetura alterada: a rede foi reiniciada e precisa ser treinada."
        elif new != old:
            self.status = "Configuração atualizada."

    def change_speed(self):
        idx = self.speed_options.index(self.speed)
        self.speed = self.speed_options[(idx + 1) % len(self.speed_options)]
        self.status = f"Ritmo: {self.speed} época(s) por atualização visual."

    def ensure_net(self):
        if len({s["rotulo"] for s in self.samples}) < 2:
            self.status = "Cadastre exemplos de pelo menos duas classes diferentes."
            return False
        if self.net is None:
            self.X, self.y, classes = matrizes(self.samples)
            self.net = RedeNeural(classes, self.hidden, seed=42)
            self.history = []
            self.history_offset = 0
            self.status = "Rede criada com pesos iniciais. Execute uma época."
        if self.X is None or self.y is None:
            self.X, self.y, _ = matrizes(self.samples, self.net.classes)
        return True

    def one_epoch(self):
        if self.running or not self.ensure_net():
            return
        if len(self.history) >= self.epochs:
            self.status = "Limite de épocas alcançado. Aumente o limite ou reinicie a rede."
            return
        self.advance_epochs(1)

    def start(self):
        if not self.ensure_net():
            return
        if len(self.history) >= self.epochs:
            self.status = "Limite de épocas alcançado. Aumente o limite ou reinicie a rede."
            return
        self.running = True
        self.last_tick = time.monotonic() - 1
        self.status = "Treinamento em andamento; os pesos são atualizados a cada época."

    def pause(self):
        self.running = False
        self.status = f"Treinamento pausado na época {len(self.history)}."

    def reset_net(self):
        self.invalidate()
        self.status = "Rede reiniciada; os exemplos foram preservados."

    def advance_epochs(self, count):
        for _ in range(count):
            if len(self.history) >= self.epochs:
                break
            loss, acc = self.net.treinar_epoca(self.X, self.y, self.rate)
            self.history.append({"epoca": len(self.history) + 1, "erro": loss, "acuracia": acc})
        self.history_offset = max(0, len(self.history) - 5)
        if len(self.history) >= self.epochs:
            self.running = False
            self.status = f"Treinamento concluído em {len(self.history)} épocas. Experimente a etapa 2."
        elif not self.running:
            self.status = f"Época {len(self.history)} concluída. Avance ou continue o treinamento."

    def scroll_history(self, delta):
        self.history_offset = max(0, min(max(0, len(self.history) - 5), self.history_offset + delta))

    def clear_inference(self):
        self.infer_pixels = [0] * ENTRADAS
        self.inference_offset = 0
        self.status = "Desenho de inferência limpo."

    def copy_sample(self):
        if 0 <= self.selected < len(self.samples):
            self.infer_pixels = self.samples[self.selected]["pixels"].copy()
            self.inference_offset = 0
            self.status = f"Amostra da classe {self.samples[self.selected]['rotulo']} copiada; experimente modificá-la."

    def scroll_predictions(self, amount):
        if self.net:
            self.inference_offset = max(0, min(max(0, ((len(self.net.classes) - 1) // 8) * 8), self.inference_offset + amount))

    def show_parameters(self):
        self.parameter_offset = 0
        self.modal = ("Parâmetros da rede", [], None)

    def parameter_entries(self):
        if self.net is None:
            return []
        entries = []
        for i in range(ENTRADAS):
            for j in range(self.net.ocultos):
                entries.append((f"E{i + 1:02} → H{j + 1:02}", float(self.net.W1[i, j])))
        for j in range(self.net.ocultos):
            entries.append((f"b(H{j + 1:02})", float(self.net.b1[j])))
        for j in range(self.net.ocultos):
            for k, label in enumerate(self.net.classes):
                entries.append((f"H{j + 1:02} → {label}", float(self.net.W2[j, k])))
        for k, label in enumerate(self.net.classes):
            entries.append((f"b({label})", float(self.net.b2[k])))
        return entries

    def scroll_parameters(self, amount):
        count = len(self.parameter_entries())
        maximum = max(0, ((count - 1) // 36) * 36)
        self.parameter_offset = max(0, min(maximum, self.parameter_offset + amount))

    def draw_parameters(self):
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 185))
        self.canvas.blit(shade, (0, 0))
        box = pygame.Rect(275, 115, 1050, 660)
        self.panel(box, (25, 30, 40), (87, 105, 121), 12)
        self.text("Parâmetros da rede", box.x + 28, box.y + 22, "head", CYAN)
        self.text("Valores atuais: pesos e vieses são atualizados a cada época durante o treinamento.", box.x + 28, box.y + 56, "body", MUTED)
        self.text("E01–E40 = pixels por linhas; H = neurônio oculto; b = viés.", box.x + 28, box.y + 84, "small", MUTED)
        entries = self.parameter_entries()
        page = entries[self.parameter_offset:self.parameter_offset + 36]
        for z, (name, value) in enumerate(page):
            col, row = divmod(z, 12)
            x, y = box.x + 30 + 330 * col, box.y + 125 + 31 * row
            if row % 2 == 0:
                pygame.draw.rect(self.canvas, (31, 38, 49), (x - 6, y - 3, 312, 27), border_radius=4)
            self.text(name, x, y, "mono", TEXT)
            self.text(f"{value:+.5f}", x + 170, y, "mono", CYAN if value >= 0 else RED)
        self.buttons = {}
        self.button("param_prev", "◀ Anteriores", (box.x + 29, box.bottom - 68, 165, 40), lambda: self.scroll_parameters(-36), enabled=self.parameter_offset > 0)
        self.centered(f"{min(self.parameter_offset + 1, len(entries))}–{min(self.parameter_offset + 36, len(entries))} de {len(entries)} parâmetros  •  época {len(self.history)}", (box.x + 214, box.bottom - 68, 570, 40), "small", MUTED)
        self.button("param_next", "Próximos ▶", (box.right - 224, box.bottom - 68, 165, 40), lambda: self.scroll_parameters(36), enabled=self.parameter_offset + 36 < len(entries))
        self.button("param_close", "Fechar", (box.right - 113, box.y + 21, 85, 34), self.close_modal)

    def show_help(self):
        self.modal = ("Como usar", [
            "1. Pinte uma grade, digite o rótulo e adicione vários desenhos por classe.",
            "2. Cadastre ao menos duas classes. Ou carregue os exemplos A / B / C.",
            "3. Ajuste a rede e avance uma época ou inicie o treinamento.",
            "4. Abra 'Inferência ao vivo' e desenhe para acompanhar as saídas.",
            "Clique com o botão esquerdo para pintar e com o direito para apagar.",
            "Acurácia exibida: apenas na base de treinamento, não em dados inéditos.",
        ], None)

    def open_request(self):
        if self.samples:
            self.modal = ("Abrir outro projeto?", ["O projeto aberto substituirá os dados atuais.",
                          "Salve o trabalho atual antes de continuar, se necessário."], self.open_file)
        else:
            self.open_file()

    def file_dialog(self, save=False):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                options = {"title": "Salvar projeto" if save else "Abrir projeto",
                           "filetypes": [("Projeto de caracteres", "*.json"), ("Todos os arquivos", "*.*")]}
                if save:
                    options.update({"defaultextension": ".json", "initialfile": "projeto_caracteres.json"})
                    return filedialog.asksaveasfilename(**options)
                return filedialog.askopenfilename(**options)
            finally:
                root.destroy()
        except Exception as exc:
            self.status = f"Não foi possível abrir o seletor de arquivos: {exc}"
            return ""

    def save_file(self):
        path = self.file_dialog(save=True)
        if not path:
            return
        try:
            config = {"ocultos": self.hidden, "epocas": self.epochs, "taxa": self.rate,
                      "ritmo": self.speed, "selecionada": self.selected}
            salvar_projeto(path, self.samples, self.net, self.history, config)
            self.status = f"Projeto salvo: {Path(path).name}"
        except (OSError, ValueError, TypeError) as exc:
            self.status = f"Não foi possível salvar: {exc}"

    def open_file(self):
        path = self.file_dialog(save=False)
        if not path:
            return
        try:
            samples, net, hist, config = carregar_projeto(path)
            # Validate everything before changing the active project.
            hidden = net.ocultos if net else int(config.get("ocultos", 16))
            epochs = int(config.get("epocas", 600))
            rate = float(config.get("taxa", .1))
            speed = int(config.get("ritmo", 1))
            selected = int(config.get("selecionada", 0 if samples else -1))
            if hidden not in self.hidden_options or epochs not in self.epoch_options or rate not in self.rate_options or speed not in self.speed_options:
                raise ValueError("Configuração fora das opções disponíveis.")
            if len(hist) > self.epoch_options[-1]:
                raise ValueError("Histórico excede o limite do aplicativo.")
            new_X, new_y = (None, None)
            if net:
                new_X, new_y, _ = matrizes(samples, net.classes)
            self.running = False
            self.samples, self.net, self.history = samples, net, hist
            self.hidden, self.epochs, self.rate, self.speed = hidden, epochs, rate, speed
            self.selected = max(-1, min(selected, len(samples) - 1))
            self.sample_offset = max(0, min(max(0, len(samples) - 4), self.selected))
            self.history_offset = max(0, len(hist) - 5)
            self.X, self.y = new_X, new_y
            self.inference_offset = 0
            self.train_pixels = [0] * ENTRADAS
            self.infer_pixels = [0] * ENTRADAS
            self.status = f"Projeto carregado: {Path(path).name}"
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
            self.status = f"Não foi possível carregar o projeto: {exc}"

    def draw_modal(self):
        if self.modal[0] == "Parâmetros da rede":
            self.draw_parameters()
            return
        title, lines, action = self.modal
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 175))
        self.canvas.blit(shade, (0, 0))
        box_h = 165 + len(lines) * 28
        box = pygame.Rect(350, (H - box_h) // 2, 900, box_h)
        self.panel(box, (25, 30, 40), (87, 105, 121), 12)
        self.text(title, box.x + 28, box.y + 25, "head", CYAN)
        for i, line in enumerate(lines):
            self.text(line, box.x + 28, box.y + 65 + i * 28, "body", TEXT, max_width=840)
        self.buttons = {}
        if action is None:
            self.button("modal_close", "Entendi", (box.right - 170, box.bottom - 62, 140, 40), self.close_modal, "primary")
        else:
            self.button("modal_no", "Cancelar", (box.right - 318, box.bottom - 62, 132, 40), self.close_modal)
            self.button("modal_yes", "Continuar", (box.right - 170, box.bottom - 62, 140, 40), self.confirm_modal, "primary")

    def close_modal(self):
        self.modal = None

    def confirm_modal(self):
        action = self.modal[2]
        self.modal = None
        if action:
            action()

    def render(self):
        self.buttons = {}
        self.canvas.fill(BG)
        self.header()
        if self.page == 0:
            self.train_page()
        else:
            self.inference_page()
        self.footer()
        if self.modal:
            self.draw_modal()
        ww, wh = self.window.get_size()
        scale = min(ww / W, wh / H)
        dest_w, dest_h = max(1, int(W * scale)), max(1, int(H * scale))
        x, y = (ww - dest_w) // 2, (wh - dest_h) // 2
        self.viewport = (x, y, scale)
        self.window.fill((5, 6, 9))
        if (dest_w, dest_h) == (W, H):
            self.window.blit(self.canvas, (x, y))
        else:
            self.window.blit(pygame.transform.smoothscale(self.canvas, (dest_w, dest_h)), (x, y))
        pygame.display.flip()

    def paint_at(self, pos):
        x, y = pos
        if not self.grid_rect.collidepoint(x, y):
            self.last_cell = None
            return
        c = (x - self.grid_rect.x) // self.grid_cell_size
        r = (y - self.grid_rect.y) // self.grid_cell_size
        idx = int(r * COLUNAS + c)
        if idx != self.last_cell:
            pixels = self.train_pixels if self.grid_tag == "train" else self.infer_pixels
            # Interpolate between successive mouse events so a fast drag does
            # not leave gaps when the system skips intermediate coordinates.
            if self.last_cell is None:
                cells = [(r, c)]
            else:
                last_r, last_c = divmod(self.last_cell, COLUNAS)
                steps = max(abs(r - last_r), abs(c - last_c))
                cells = [(round(last_r + (r - last_r) * t / steps),
                          round(last_c + (c - last_c) * t / steps))
                         for t in range(steps + 1)]
            for row, col in cells:
                pixels[row * COLUNAS + col] = 1 if self.mouse_mode == "paint" else 0
            self.last_cell = idx
            if self.grid_tag == "infer":
                self.inference_offset = 0

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.alive = False
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.modal:
                    self.close_modal()
                else:
                    self.typing = False
                return
            if self.modal:
                return
            if self.typing and not self.running:
                if event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                    self.label = ""
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.typing = False
                    self.add_sample()
                else:
                    char = event.unicode.upper()
                    if len(char) == 1 and char in CARACTERES:
                        self.label = char
                return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
            pos = self.logical_mouse(event.pos)
            if self.modal:
                if event.button == 1:
                    for rect, action, enabled in list(self.buttons.values()):
                        if enabled and rect.collidepoint(pos):
                            action()
                            break
                return
            if event.button == 1:
                clicked_button = False
                for key, (rect, action, enabled) in list(self.buttons.items()):
                    if enabled and rect.collidepoint(pos):
                        self.typing = (key == "label")
                        action()
                        clicked_button = True
                        break
                if clicked_button:
                    return
                self.typing = False
                if self.page == 0:
                    for rect, idx in self.sample_rows:
                        if rect.collidepoint(pos):
                            self.selected = idx
                            self.status = f"Amostra {idx + 1} selecionada; rótulo esperado: {self.samples[idx]['rotulo']}."
                            return
            if self.grid_rect.collidepoint(pos) and (self.page == 1 or not self.running):
                self.mouse_mode = "paint" if event.button == 1 else "erase"
                self.last_cell = None
                self.paint_at(pos)
        elif event.type == pygame.MOUSEMOTION and self.mouse_mode and not self.modal:
            self.paint_at(self.logical_mouse(event.pos))
        elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 3):
            self.mouse_mode = None
            self.last_cell = None
        elif event.type == pygame.MOUSEWHEEL and not self.modal and self.page == 0:
            mx, my = self.logical_mouse(pygame.mouse.get_pos())
            if 24 <= mx <= 434 and 548 <= my <= 770:
                self.scroll_samples(-event.y)
            elif 956 <= mx <= 1576 and 555 <= my <= 778:
                self.scroll_history(-event.y * 3)

    def run(self):
        while self.alive:
            # Draw once before consuming clicks so their hitboxes match the frame.
            self.render()
            for event in pygame.event.get():
                self.handle_event(event)
            if self.running and (self.modal is None or self.modal[0] == "Parâmetros da rede") and time.monotonic() - self.last_tick >= .06:
                self.advance_epochs(self.speed)
                self.last_tick = time.monotonic()
            self.clock.tick(60)
        pygame.quit()
