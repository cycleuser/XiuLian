"""PySide6图形界面"""

import sys
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QListWidget, QGroupBox, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal
from . import __version__, Engine
from .api import Result

class Worker(QThread):
    done = Signal(object)
    def __init__(self, engine, text):
        super().__init__()
        self.engine, self.text = engine, text
    def run(self):
        self.done.emit(self.engine.process(self.text))

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = Engine()
        self.worker = None
        self.setWindowTitle(f"修炼 v{__version__}")
        self.setMinimumSize(900, 600)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        left = QWidget()
        llayout = QVBoxLayout(left)
        
        input_group = QGroupBox("输入")
        ilayout = QVBoxLayout(input_group)
        self.input = QTextEdit()
        self.input.setPlaceholderText("输入查询...")
        self.input.setMaximumHeight(100)
        ilayout.addWidget(self.input)
        
        btns = QHBoxLayout()
        run_btn = QPushButton("执行")
        run_btn.clicked.connect(self.run)
        run_btn.setDefault(True)
        btns.addWidget(run_btn)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(lambda: (self.input.clear(), self.output.clear()))
        btns.addWidget(clear_btn)
        ilayout.addLayout(btns)
        llayout.addWidget(input_group)
        
        tools_group = QGroupBox("工具")
        tlayout = QVBoxLayout(tools_group)
        self.tools_list = QListWidget()
        for t in self.engine.tools.list():
            self.tools_list.addItem(f"{t['name']}: {t['desc']}")
        tlayout.addWidget(self.tools_list)
        llayout.addWidget(tools_group)
        
        layout.addWidget(left)
        
        right = QWidget()
        rlayout = QVBoxLayout(right)
        output_group = QGroupBox("输出")
        olayout = QVBoxLayout(output_group)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        olayout.addWidget(self.output)
        self.latency_label = QLabel("")
        olayout.addWidget(self.latency_label)
        rlayout.addWidget(output_group)
        
        stats_group = QGroupBox("统计")
        slayout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("查询: 0 | 成功率: 100%")
        slayout.addWidget(self.stats_label)
        rlayout.addWidget(stats_group)
        
        layout.addWidget(right, stretch=2)
        
        self.queries = 0
        self.successes = 0
    
    def run(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.worker = Worker(self.engine, text)
        self.worker.done.connect(self.show_result)
        self.worker.start()
    
    def show_result(self, r: Result):
        self.output.setPlainText(json.dumps(r.data, ensure_ascii=False, indent=2) if r.success else f"错误: {r.error}")
        self.latency_label.setText(f"延迟: {r.latency_ms:.2f}ms")
        self.queries += 1
        if r.success:
            self.successes += 1
        self.stats_label.setText(f"查询: {self.queries} | 成功率: {self.successes/self.queries:.0%}")

def run_gui():
    app = QApplication(sys.argv)
    app.setApplicationName("xiulian")
    Window().show()
    sys.exit(app.exec())