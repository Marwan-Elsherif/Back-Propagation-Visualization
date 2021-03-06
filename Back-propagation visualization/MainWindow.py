from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QComboBox
from PyQt5.QtGui import QIcon, QFont, QDoubleValidator, QValidator
from VisualizeWindow import VisualizeWindow
from ShallowNN import ShallowNN


class Window(QWidget):

    inNumInputValue = 0
    hidNumInputValue = 0
    outNumInputValue = 0
    iRateInputValue = 0
    wInitInputValue = 0
    hidActFnInputValue = 0
    outActFnInputValue = 0
    optimizerInputValue = 0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shallow NN Visualizer")
        # self.setWindowIcon(QIcon("myapp.png"))

        self.setFixedHeight(500)
        self.setFixedWidth(900)
        # self.setGeometry(500, 300, 400, 300)

        stylesheet = (
            'background-color:#D3D3D3'
        )
        self.setStyleSheet(stylesheet)

        self.create_widgets()

    def create_widgets(self):
        # Craeting Visualize button
        visBtn = QPushButton("Visualize", self)

        # Stylying Visualize button
        # visBtn.move(100,100)
        visBtn.setGeometry(1200, 500, 200, 100)
        visBtn.setStyleSheet('background-color:white')
        visBtn.setFixedSize(250, 50)
        visBtn.setFont(QFont("Times New Roman", 20))
        # visBtn.setIcon(QIcon("whatever.png"))

        # Putting Visualize button into a layout
        visBtnHbox = QHBoxLayout()
        visBtnHbox.addWidget(visBtn)

        # Connecting Visualize button to its driver function
        visBtn.clicked.connect(self.clicked_visBtn)

        # Creating needed labels
        mainLabel = QLabel("Shallow NN Visualizer", self)
        inNum = QLabel("# Inputs", self)
        hidNum = QLabel("# Hidden Neurons", self)
        outNum = QLabel("# Output Neurons", self)
        iRate = QLabel("Learning Rate", self)
        wInit = QLabel("Weight Init.", self)
        hidActFn = QLabel("Hidden Activation Fn.", self)
        outActFn = QLabel("Output Activation Fn", self)
        # optimizer = QLabel("Optimizer", self)

        # Styiling the labels
        mainLabel.setStyleSheet('color:black')
        mainLabel.setFont(QFont("Times New Roman", 20))

        # inNum.setGeometry(100,100,300,100)
        inNum.setStyleSheet('color:black')
        inNum.setFont(QFont("Times New Roman", 15))

        # hidNum.setGeometry(100,200,300,100)
        hidNum.setStyleSheet('color:black')
        hidNum.setFont(QFont("Times New Roman", 15))

        # outNum.setGeometry(100,300,300,100)
        outNum.setStyleSheet('color:black')
        outNum.setFont(QFont("Times New Roman", 15))

        # iRate.setGeometry(100,400,300,100)
        iRate.setStyleSheet('color:black')
        iRate.setFont(QFont("Times New Roman", 15))

        # wInit.setGeometry(100,500,300,100)
        wInit.setStyleSheet('color:black')
        wInit.setFont(QFont("Times New Roman", 15))

        # hidActFn.setGeometry(100,600,300,100)
        hidActFn.setStyleSheet('color:black')
        hidActFn.setFont(QFont("Times New Roman", 15))

        # outActFn.setGeometry(100,700,300,100)
        outActFn.setStyleSheet('color:black')
        outActFn.setFont(QFont("Times New Roman", 15))

        # optimizer.setGeometry(100,800,300,100)
        # optimizer.setStyleSheet('color:black')
        # optimizer.setFont(QFont("Times New Roman", 15))

        # Grouping labels into a Layout
        labelsVbox = QVBoxLayout()
        labelsVbox.addWidget(inNum)
        labelsVbox.addWidget(hidNum)
        labelsVbox.addWidget(outNum)
        labelsVbox.addWidget(iRate)
        labelsVbox.addWidget(wInit)
        labelsVbox.addWidget(hidActFn)
        labelsVbox.addWidget(outActFn)
        # labelsVbox.addWidget(optimizer)

        # Creating needed LineEdits
        self.inNumInput = QLineEdit(self)
        self.hidNumInput = QLineEdit(self)
        self.outNumInput = QLineEdit(self)
        self.iRateInput = QLineEdit(self)

        # Creating needed comboboxes for choosing a string value from many
        self.wInitInput = QComboBox(self)
        self.wInitInput.addItem("zeros")
        self.wInitInput.addItem("small random values")
        self.hidActFnInput = QComboBox(self)
        self.hidActFnInput.addItem("RELU")
        self.hidActFnInput.addItem("Sigmoid")
        self.hidActFnInput.addItem("Tanh")
        self.outActFnInput = QComboBox(self)
        self.outActFnInput.addItem("RELU")
        self.outActFnInput.addItem("Sigmoid")
        self.outActFnInput.addItem("Tanh")
        # self.optimizerInput = QComboBox(self)
        # self.optimizerInput.addItem("SGD")
        # self.optimizerInput.addItem("ADAM")

        # Styiling LineEdits and comboxes
        self.inNumInput.setStyleSheet("border: 1px solid black")
        self.inNumInput.setFixedSize(200, 25)

        self.hidNumInput.setStyleSheet("border: 1px solid black")
        self.hidNumInput.setFixedSize(200, 25)

        self.outNumInput.setStyleSheet("border: 1px solid black")
        self.outNumInput.setFixedSize(200, 25)

        self.iRateInput.setStyleSheet("border: 1px solid black")
        self.iRateInput.setFixedSize(200, 25)

        self.wInitInput.setStyleSheet('color:black')
        self.wInitInput.setFixedSize(200, 25)

        self.hidActFnInput.setStyleSheet('color:black')
        self.hidActFnInput.setFixedSize(200, 25)

        self.outActFnInput.setStyleSheet('color:black')
        self.outActFnInput.setFixedSize(200, 25)

        # self.optimizerInput.setStyleSheet('color:black')
        # self.optimizerInput.setFixedSize(200, 25)

        # Grouping LineEdits and scroll bars into a Layout
        inputsVbox = QVBoxLayout()
        inputsVbox.addWidget(self.inNumInput)
        inputsVbox.addWidget(self.hidNumInput)
        inputsVbox.addWidget(self.outNumInput)
        inputsVbox.addWidget(self.iRateInput)
        inputsVbox.addWidget(self.wInitInput)
        inputsVbox.addWidget(self.hidActFnInput)
        inputsVbox.addWidget(self.outActFnInput)
        # inputsVbox.addWidget(self.optimizerInput)

        # self.setLayout(inputsVbox)

        # Grouping Labels and LineEdits into a horizontal layout
        labelsInputsGrid = QGridLayout()
        labelsInputsGrid.addWidget(mainLabel, 0, 2)
        labelsInputsGrid.addLayout(labelsVbox, 1, 0)
        labelsInputsGrid.addLayout(inputsVbox, 1, 2)
        labelsInputsGrid.addLayout(visBtnHbox, 2, 2)
        emptyLayout = QHBoxLayout()
        labelsInputsGrid.addLayout(emptyLayout, 3, 2, 10, 10)

        self.setLayout(labelsInputsGrid)

        '''
        # If we want to use vertical box layout
        labelsVbox = QVBoxLayout()
        labelsVbox.addWidget(inNum)
        
        self.setLayout(labelsVbox)
        '''

        '''
        # If we want to use horizontal box layout
        labelsHbox = QHBoxLayout()
        labelsHbox.addWidget(inNum)

        self.setLayout(labelsHbox)
        '''
        '''
        # If we want to use grid layout
        labelsGrid = QGridLayout()
        labelsGrid.addWidget(inNum, 0, 0)

        self.setLayout(labelsGrid)
        '''

    def clicked_visBtn(self):
        # this function will be used to pass the parameters entered in the window to the NN
        self.inNumInputValue = self.inNumInput.text()
        self.hidNumInputValue = self.hidNumInput.text()
        self.outNumInputValue = self.outNumInput.text()
        self.iRateInputValue = self.iRateInput.text()
        self.wInitInputValue = self.wInitInput.currentText()
        self.hidActFnInputValue = self.hidActFnInput.currentText()
        self.outActFnInputValue = self.outActFnInput.currentText()
        # self.optimizerInputValue = self.optimizerInput.currentText()

        # Then move to the next window with the needed NN drawn

        # Tests
        print("The visualize button is clicked")
        print(self.inNumInputValue + ", " + self.hidNumInputValue +
              ", " + self.outNumInputValue + ", ")
        print(self.iRateInputValue + ", " + self.wInitInputValue +
              ", " + self.hidActFnInputValue + ", ")
        # print(self.outActFnInputValue + ", " + self.optimizerInputValue + ", ")

        shallow_network = ShallowNN(
            self.inNumInputValue, self.hidNumInputValue, self.outNumInputValue, self.iRateInputValue, self.wInitInputValue,
            self.hidActFnInputValue, self.outActFnInputValue, self.optimizerInputValue)
        # Open Second Window to Visualize the NN
        self.vis_window = VisualizeWindow(shallow_network)
        self.vis_window.showMaximized()
