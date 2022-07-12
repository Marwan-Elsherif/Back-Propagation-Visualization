from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit
from PyQt5.QtGui import QIcon, QFont, QDoubleValidator, QValidator
from VisualizeWindow import VisualizeWindow


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

        # self.setFixedHeight(1200)
        # self.setFixedWidth(1500)
        self.setGeometry(500, 300, 400, 300)

        # self.setStyleSheet('background-color:gray')
        stylesheet = (
            'background-color:gray'
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
        visBtn.setFixedSize(200, 25)
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
        optimizer = QLabel("Optimizer", self)

        # Styiling the labels
        mainLabel.setStyleSheet('color:white')
        mainLabel.setFont(QFont("Times New Roman", 20))

        # inNum.setGeometry(100,100,300,100)
        inNum.setStyleSheet('color:white')
        inNum.setFont(QFont("Times New Roman", 15))

        # hidNum.setGeometry(100,200,300,100)
        hidNum.setStyleSheet('color:white')
        hidNum.setFont(QFont("Times New Roman", 15))

        # outNum.setGeometry(100,300,300,100)
        outNum.setStyleSheet('color:white')
        outNum.setFont(QFont("Times New Roman", 15))

        # iRate.setGeometry(100,400,300,100)
        iRate.setStyleSheet('color:white')
        iRate.setFont(QFont("Times New Roman", 15))

        # wInit.setGeometry(100,500,300,100)
        wInit.setStyleSheet('color:white')
        wInit.setFont(QFont("Times New Roman", 15))

        # hidActFn.setGeometry(100,600,300,100)
        hidActFn.setStyleSheet('color:white')
        hidActFn.setFont(QFont("Times New Roman", 15))

        # outActFn.setGeometry(100,700,300,100)
        outActFn.setStyleSheet('color:white')
        outActFn.setFont(QFont("Times New Roman", 15))

        # optimizer.setGeometry(100,800,300,100)
        optimizer.setStyleSheet('color:white')
        optimizer.setFont(QFont("Times New Roman", 15))

        # Grouping labels into a Layout
        labelsVbox = QVBoxLayout()
        labelsVbox.addWidget(inNum)
        labelsVbox.addWidget(hidNum)
        labelsVbox.addWidget(outNum)
        labelsVbox.addWidget(iRate)
        labelsVbox.addWidget(wInit)
        labelsVbox.addWidget(hidActFn)
        labelsVbox.addWidget(outActFn)
        labelsVbox.addWidget(optimizer)

        # self.setLayout(labelsVbox)

        # Creating needed LineEdits and scroll bars
        self.inNumInput = QLineEdit(self)
        self.hidNumInput = QLineEdit(self)
        self.outNumInput = QLineEdit(self)
        self.iRateInput = QLineEdit(self)
        self.wInitInput = QLineEdit(self)
        self.hidActFnInput = QLineEdit(self)
        self.outActFnInput = QLineEdit(self)
        self.optimizerInput = QLineEdit(self)

        # Styiling LineEdits and scroll bars
        self.inNumInput.setStyleSheet('color:black')
        self.inNumInput.setFixedSize(200, 25)

        self.hidNumInput.setStyleSheet('color:black')
        self.hidNumInput.setFixedSize(200, 25)

        self.outNumInput.setStyleSheet('color:black')
        self.outNumInput.setFixedSize(200, 25)

        self.iRateInput.setStyleSheet('color:black')
        self.iRateInput.setFixedSize(200, 25)

        self.wInitInput.setStyleSheet('color:black')
        self.wInitInput.setFixedSize(200, 25)

        self.hidActFnInput.setStyleSheet('color:black')
        self.hidActFnInput.setFixedSize(200, 25)

        self.outActFnInput.setStyleSheet('color:black')
        self.outActFnInput.setFixedSize(200, 25)

        self.optimizerInput.setStyleSheet('color:black')
        self.optimizerInput.setFixedSize(200, 25)

        # Grouping LineEdits and scroll bars into a Layout
        inputsVbox = QVBoxLayout()
        inputsVbox.addWidget(self.inNumInput)
        inputsVbox.addWidget(self.hidNumInput)
        inputsVbox.addWidget(self.outNumInput)
        inputsVbox.addWidget(self.iRateInput)
        inputsVbox.addWidget(self.wInitInput)
        inputsVbox.addWidget(self.hidActFnInput)
        inputsVbox.addWidget(self.outActFnInput)
        inputsVbox.addWidget(self.optimizerInput)

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
        self.wInitInputValue = self.wInitInput.text()
        self.hidActFnInputValue = self.hidActFnInput.text()
        self.outActFnInputValue = self.outActFnInput.text()
        self.optimizerInputValue = self.optimizerInput.text()

        # Then move to the next window with the needed NN drawn

        # Tests
        print("The visualize button is clicked")
        print(self.inNumInputValue)
        print(self.hidNumInputValue)
        print(self.outNumInputValue)

        # Open Second Window to Visualize the NN
        self.vis_window = VisualizeWindow()
        self.vis_window.show()
