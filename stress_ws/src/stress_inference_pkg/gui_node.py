import sys, threading, time, math, numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QPlainTextEdit, QSizePolicy, QLineEdit)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QRadialGradient, QPen, QFont
import pyqtgraph as pg
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32

class Communicate(QObject):
    state_signal = pyqtSignal(str)
    eda_signal = pyqtSignal(float)
    ibi_signal = pyqtSignal(float) 
    chat_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)

# Box-breathing widget  (4-4-4-4, 4 cycles)
BOX_PHASES = [("INHALE",4,"#3498db"),("HOLD",4,"#9b59b6"),
              ("EXHALE",4,"#2ecc71"),("HOLD",4,"#e67e22")]
BREATHING_TOTAL_CYCLES = 4

class BreathingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._active=False; self._phase_idx=0; self._phase_elapsed=0.0
        self._radius_norm=0.35; self._cycles_done=0
        self._tick = QTimer(self)
        self._tick.setInterval(50)
        self._tick.timeout.connect(self._on_tick)

    def start(self):
        self._active=True; self._phase_idx=0; self._phase_elapsed=0.0
        self._radius_norm=0.35; self._cycles_done=0
        self._tick.start(); self.update()

    def stop(self):
        self._active=False; self._tick.stop(); self.update()

    def set_cycles_remaining(self, n):
        self._cycles_done = BREATHING_TOTAL_CYCLES - n; self.update()

    @property
    def is_active(self): return self._active

    def _on_tick(self):
        dt = self._tick.interval() / 1000.0
        phase_name, phase_dur, _ = BOX_PHASES[self._phase_idx]
        self._phase_elapsed += dt
        progress = min(self._phase_elapsed / phase_dur, 1.0)
        if phase_name == "INHALE":
            self._radius_norm = 0.35 + 0.45 * self._ease(progress)
        elif phase_name == "EXHALE":
            self._radius_norm = 0.80 - 0.45 * self._ease(progress)
        if self._phase_elapsed >= phase_dur:
            self._phase_elapsed = 0.0
            self._phase_idx = (self._phase_idx + 1) % len(BOX_PHASES)
            if self._phase_idx == 0:
                self._cycles_done += 1
        self.update()

    @staticmethod
    def _ease(t): return (1 - math.cos(math.pi * t)) / 2

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        max_r = min(w, h) / 2 - 4
        if not self._active:
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(40,40,40))
            p.drawEllipse(int(cx-max_r),int(cy-max_r),int(max_r*2),int(max_r*2))
            p.setPen(QColor(100,100,100)); p.setFont(QFont("Courier New",10))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "BREATHING\nINACTIVE")
            return
        phase_name, phase_dur, color_hex = BOX_PHASES[self._phase_idx]
        r = int(max_r * self._radius_norm); color = QColor(color_hex)
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(3):
            gc = QColor(color_hex); gc.setAlpha(max(4, 28-i*8)); p.setBrush(gc)
            gr = r+8+i*10; p.drawEllipse(int(cx-gr),int(cy-gr),int(gr*2),int(gr*2))
        grad = QRadialGradient(cx, cy, r)
        c1 = QColor(color_hex); c1.setAlpha(220)
        c2 = QColor(color_hex); c2.setAlpha(140)
        grad.setColorAt(0.0,c1); grad.setColorAt(1.0,c2)
        p.setBrush(grad); p.setPen(QPen(color,2))
        p.drawEllipse(int(cx-r),int(cy-r),int(r*2),int(r*2))
        remaining = int(phase_dur - self._phase_elapsed) + 1
        p.setPen(QColor(255,255,255))
        lf = QFont("Courier New", max(10,int(r/3)), QFont.Weight.Bold); p.setFont(lf)
        p.drawText(int(cx-r),int(cy-r*0.45),int(r*2),int(r*0.8),
                   Qt.AlignmentFlag.AlignCenter, phase_name)
        cf = QFont("Courier New", max(8,int(r/4))); p.setFont(cf)
        p.drawText(int(cx-r),int(cy+r*0.05),int(r*2),int(r*0.5),
                   Qt.AlignmentFlag.AlignCenter, str(remaining))
        sf = QFont("Courier New", 8); p.setFont(sf); p.setPen(QColor(180,180,180))
        p.drawText(0, h-22, w, 20, Qt.AlignmentFlag.AlignCenter,
                   f"Cycle {self._cycles_done+1}/{BREATHING_TOTAL_CYCLES}")


# 6-dot prediction indicator (yellow - uncertainty, green - calm, red - stress)
DOT_OFF = QColor(45,45,45)
DOT_GREEN = QColor(46,204,113)
DOT_YELLOW = QColor(241,196,15)
DOT_RED = QColor(231,76,60)
COLOUR_MAP = {"green":DOT_GREEN,"yellow":DOT_YELLOW,"red":DOT_RED}
N_DOTS  = 6

class DotIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._colours  = [DOT_OFF] * N_DOTS
        self._head = 0
        self._row_full = False
        self.setFixedHeight(28)

    def push(self, colour_name):
        c = COLOUR_MAP.get(colour_name, DOT_OFF)
        if self._row_full:
            self._colours  = [DOT_OFF] * N_DOTS
            self._head = 0
            self._row_full = False
        self._colours[self._head] = c
        self._head += 1
        if self._head >= N_DOTS:
            self._row_full = True
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r=10; gap=8
        total = N_DOTS*(2*r) + (N_DOTS-1)*gap
        x0 = (w-total)//2; cy = h//2
        for i, col in enumerate(self._colours):
            cx = x0 + i*(2*r+gap) + r
            ring = QColor(col); ring.setAlpha(60)
            p.setPen(QPen(ring,2)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(cx-r-2, cy-r-2, (r+2)*2, (r+2)*2)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(col)
            p.drawEllipse(cx-r, cy-r, r*2, r*2)


# Active learning status panel
class ALStatusPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self._calm=0; self._stress=0; self._w=0.0; self._ready=False

    def update_al(self, calm, stress, w, ready):
        self._calm=calm; self._stress=stress; self._w=w; self._ready=ready
        self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(20,20,20))
        p.drawRoundedRect(0,0,w,h,6,6)
        # Label counts
        p.setPen(QColor(46,204,113))
        p.setFont(QFont("Courier New",9,QFont.Weight.Bold))
        p.drawText(8,0,w//2,h//2+4,Qt.AlignmentFlag.AlignVCenter,
                   f"CALM labels: {self._calm}")
        p.setPen(QColor(231,76,60))
        p.drawText(8,h//2-4,w//2,h//2+4,Qt.AlignmentFlag.AlignVCenter,
                   f"STRESS labels: {self._stress}")
        # Blend weight bar
        bx=w//2+8; bw=w-bx-8; bh=8; by=h//2-bh//2
        p.setBrush(QColor(50,50,50)); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(bx,by,bw,bh,3,3)
        fw = int(bw*(self._w/0.5))
        p.setBrush(QColor(52,152,219))
        p.drawRoundedRect(bx,by,max(fw,0),bh,3,3)
        p.setPen(QColor(150,150,150)); p.setFont(QFont("Courier New",8))
        p.drawText(bx,by-14,bw,14,
                   Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,
                   f"Personal model influence: {int(self._w*100)}%")
        dc = QColor(46,204,113) if self._ready else QColor(80,80,80)
        p.setBrush(dc); p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(bx+bw-10, by+bh+4, 8, 8)
        p.setPen(QColor(150,150,150)); p.setFont(QFont("Courier New",7))
        p.drawText(bx,by+bh+2,bw-14,14,
                   Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,
                   "Model ready" if self._ready else "Collecting labels…")

# State styles
STATE_STYLES = {
    "buffering": {"text":"BUFFERING DATA...", "bg":"#34495e","fg":"white"},
    "monitoring": {"text":"MONITORING", "bg":"#1a6fa8","fg":"white"},
    "calm": {"text":"CALM", "bg":"#27ae60","fg":"white"},
    "stress": {"text":"STRESS DETECTED", "bg":"#c0392b","fg":"white"},
    "uncertain":  {"text":"UNCERTAIN", "bg":"#d4a017","fg":"#1a1a1a"},
}

# Main Dashboard
class StressDashboard(QMainWindow):
    def __init__(self, feedback_fn):
        super().__init__()
        self.feedback_fn = feedback_fn
        self.setWindowTitle("User Interface Active Learning Dashboard")
        self.resize(1200, 880) 
        self.setStyleSheet("background-color:#0d0d0d; color:#e0e0e0;")

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(12); root.setContentsMargins(12,12,12,12)

        # LEFT COLUMN 
        left = QVBoxLayout(); left.setSpacing(8)
        root.addLayout(left, stretch=1)

        # Status bar
        self.status_box = QLabel("WAITING FOR DATA")
        self.status_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_box.setFixedHeight(60)
        self.status_box.setStyleSheet(
            "font-size:17px;font-weight:bold;letter-spacing:2px;"
            "border-radius:8px;background-color:#222;color:#aaa;"
        )
        left.addWidget(self.status_box)

        # 6-dot prediction indicator
        dl = QLabel("PREDICTION HISTORY")
        dl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dl.setStyleSheet("color:#555;font-size:9px;letter-spacing:2px;")
        left.addWidget(dl)
        self.dot_indicator = DotIndicator()
        left.addWidget(self.dot_indicator)

        # initialised from al_status published at node startup
        al_lbl = QLabel("ACTIVE LEARNING")
        al_lbl.setStyleSheet("color:#555;font-size:9px;letter-spacing:2px;")
        left.addWidget(al_lbl)
        self.al_panel = ALStatusPanel()
        left.addWidget(self.al_panel)

        # Breathing widget
        bl = QLabel("BOX BREATHING EXERCISE")
        bl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.setStyleSheet("color:#888;font-size:10px;letter-spacing:1px;")
        left.addWidget(bl)
        self.breathing_widget = BreathingWidget()
        self.breathing_widget.setMinimumHeight(150)
        self.breathing_widget.setMaximumHeight(160)
        left.addWidget(self.breathing_widget)

        # Dialogue log 
        ll = QLabel("CURRENT DIALOGUE")
        ll.setStyleSheet("color:#555;font-size:13px;letter-spacing:2px;")
        left.addWidget(ll)
        
        self.chat_display = QPlainTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(60)   # Made dialogue box smaller
        self.chat_display.setMaximumHeight(100)  # Made dialogue box smaller
        self.chat_display.setStyleSheet(
            "background:#0a0a0a;color:#00e676;"
            "font-family:'Courier New';font-size:16px;font-weight:bold;" # Increased text size
            "border:1px solid #2a2a2a;border-radius:4px;padding:8px;"
        )
        left.addWidget(self.chat_display)

        # Free-text input
        il = QLabel("HOW DO YOU FEEL?  /  ASK A QUESTION")
        il.setStyleSheet("color:#555;font-size:9px;letter-spacing:2px;")
        left.addWidget(il)
        tr = QHBoxLayout(); tr.setSpacing(6)
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type here and press Send or Enter…")
        self.text_input.setFixedHeight(36)
        self.text_input.setStyleSheet(
            "background:#111;color:#e0e0e0;"
            "border:1px solid #2c3e50;border-radius:4px;"
            "padding-left:8px;font-size:12px;"
        )
        self.text_input.returnPressed.connect(self._send_text_input)
        self.btn_send = QPushButton("SEND")
        self.btn_send.setFixedSize(60,36)
        self.btn_send.setStyleSheet(
            "background:#1a5276;color:white;font-weight:bold;"
            "border-radius:4px;font-size:12px;"
        )
        self.btn_send.clicked.connect(self._send_text_input)
        tr.addWidget(self.text_input); tr.addWidget(self.btn_send)
        left.addLayout(tr)

        # Feedback buttons (Calm / Stressed)
        br = QHBoxLayout(); br.setSpacing(6)
        self.btn_calm = QPushButton("I'm  CALM")
        self.btn_stress = QPushButton("I'm STRESSED")
        for b in (self.btn_calm, self.btn_stress):
            b.setFixedHeight(40)
            b.setStyleSheet("font-weight:bold;border-radius:6px;font-size:12px;")
        self.btn_calm.setStyleSheet(
            self.btn_calm.styleSheet() + "background:#1e7e44;color:white;")
        self.btn_stress.setStyleSheet(
            self.btn_stress.styleSheet() + "background:#8c1a1a;color:white;")
        self.btn_calm.clicked.connect(  lambda: self.feedback_fn("calm"))
        self.btn_stress.clicked.connect(lambda: self.feedback_fn("stressed"))
        br.addWidget(self.btn_calm); br.addWidget(self.btn_stress)
        left.addLayout(br)

        ar = QHBoxLayout(); ar.setSpacing(6)
        self.btn_yes = QPushButton("YES")
        self.btn_no = QPushButton("NO")
        for b in (self.btn_yes, self.btn_no):
            b.setFixedHeight(40)
            b.setStyleSheet("font-weight:bold;border-radius:6px;font-size:12px;")
        self.btn_yes.setStyleSheet(
            self.btn_yes.styleSheet() + "background:#154360;color:white;")
        self.btn_no.setStyleSheet(
            self.btn_no.styleSheet()  + "background:#3d3d3d;color:white;")
        self.btn_yes.clicked.connect(lambda: self.feedback_fn("yes"))
        self.btn_no.clicked.connect( lambda: self.feedback_fn("no"))
        ar.addWidget(self.btn_yes); ar.addWidget(self.btn_no)
        left.addLayout(ar)

        # RIGHT COLUMN: PLOTS
        right = QVBoxLayout(); root.addLayout(right, stretch=2)
        pg.setConfigOption('background','#0d0d0d')
        pg.setConfigOption('foreground','#e0e0e0')
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setStyleSheet("border:none;")
        right.addWidget(self.graph_widget)

        self.eda_plot = self.graph_widget.addPlot(title="EDA  ( Electrodermal Activity )")
        self.eda_plot.setLabel('left','µS')
        self.eda_curve = self.eda_plot.plot(pen=pg.mkPen('#00e5ff', width=2))
        self.eda_data = np.zeros(500)

        self.graph_widget.nextRow()
        self.ibi_plot = self.graph_widget.addPlot(title="IBI  ( Inter-Beat Interval )")
        self.ibi_plot.setLabel('left', 'Seconds')
        self.ibi_plot.setLabel('bottom', 'Beats / Samples')

        #self.ibi_plot.setYRange(0.4, 1.5, padding=0.1)
        
        self.ibi_curve = self.ibi_plot.plot(pen=pg.mkPen('#ff4081', width=2))
        self.ibi_data  = np.zeros(100)

        self._current_state = "buffering"

    # Text input
    def _send_text_input(self):
        text = self.text_input.text().strip()
        if not text: return
        self.add_chat_msg(f"[YOU] {text}")
        self.feedback_fn(text)
        self.text_input.clear()

    # Status topic handler
    def handle_status(self, msg: str):
        msg = msg.strip()

        if msg.startswith("dot:"):
            self.dot_indicator.push(msg.split(":",1)[1])

        elif msg.startswith("breath_cycle:"):
            self.breathing_widget.set_cycles_remaining(int(msg.split(":",1)[1]))

        elif msg == "breath_done":
            self.breathing_widget.stop()

        elif msg.startswith("al_status:"):
            self._parse_al_status(msg[len("al_status:"):])


    def _parse_al_status(self, payload: str):
        try:
            parts = {k.strip(): v.strip() for k, v in (kv.split("=") for kv in payload.split(","))}
            
            calm = int(parts.get("calm", 0))
            stress = int(parts.get("stress", 0))
            w = float(parts.get("w", 0.0))
            n = int(parts.get("n", 0))
            
            self.al_panel.update_al(calm, stress, w, n >= 2)
        except Exception:
            pass

    # State update
    def update_status_ui(self, state: str):
        state = state.lower()
        self._current_state = state
        style = STATE_STYLES.get(state, STATE_STYLES["monitoring"])
        self.status_box.setText(style["text"])
        self.status_box.setStyleSheet(
            f"font-size:17px;font-weight:bold;letter-spacing:2px;border-radius:8px;"
            f"background-color:{style['bg']};color:{style['fg']};"
        )
        if state in ("calm", "buffering"):
            self.breathing_widget.stop()

    
    # Dialogue messages
    def add_chat_msg(self, text: str):
        if text == "breathing_start":
            self.breathing_widget.start()
            self.add_chat_msg("Breathing started, follow the animation")
            return
        ts = time.strftime('%H:%M:%S')
        self.chat_display.setPlainText(f"[{ts}]  {text}")

    # Plot updates
    def update_eda(self, val: float):
        self.eda_data = np.roll(self.eda_data, -1)
        self.eda_data[-1] = val
        self.eda_curve.setData(self.eda_data)

    def update_ibi(self, val: float):
        self.ibi_data = np.roll(self.ibi_data, -1)
        self.ibi_data[-1] = val
        self.ibi_curve.setData(self.ibi_data)

# ROS Node
class RosNode(Node):
    def __init__(self, comms: Communicate):
        super().__init__('gui_node'); self.comms = comms
        self.create_subscription(String,  '/stress_state',
            lambda m: self.comms.state_signal.emit(m.data), 10)
        self.create_subscription(Float32, '/eda_raw',
            lambda m: self.comms.eda_signal.emit(m.data), 10)
        self.create_subscription(Float32, '/ibi_raw', 
            lambda m: self.comms.ibi_signal.emit(m.data), 10)
        self.create_subscription(String,  'stress_dialogue_response',
            lambda m: self.comms.chat_signal.emit(m.data), 10)
        self.create_subscription(String,  'stress_status',
            lambda m: self.comms.status_signal.emit(m.data), 10)
        self.feedback_pub = self.create_publisher(String, '/user_feedback', 10)

    def send_feedback(self, label: str):
        self.feedback_pub.publish(String(data=label))

def main():
    rclpy.init()
    app  = QApplication(sys.argv)
    comms = Communicate()
    gui  = StressDashboard(lambda label: node.send_feedback(label))
    node = RosNode(comms)
    
    comms.state_signal.connect(gui.update_status_ui)
    comms.eda_signal.connect(gui.update_eda)
    comms.ibi_signal.connect(gui.update_ibi) 
    comms.chat_signal.connect(gui.add_chat_msg)
    comms.status_signal.connect(gui.handle_status)
    
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()