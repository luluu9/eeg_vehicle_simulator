import sys
from PyQt6.QtWidgets import QApplication
from src.collector.gui.main_window import CollectorWindow


def main():
    app = QApplication(sys.argv)
    window = CollectorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
