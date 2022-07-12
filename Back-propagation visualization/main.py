from PyQt5.QtWidgets import QApplication
from ShallowNN import ShallowNN
from MainWindow import Window
import sys

app = QApplication([])
Window = Window()
Window.show()

sys.exit(app.exec())
