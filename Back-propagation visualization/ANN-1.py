from PyQt5.QtWidgets import QApplication, QWidget
import sys

#create your application (first window)
app = QApplication(sys.argv)
window = QWidget()



window.show()
sys.exit(app.exec())




print("Hello")
