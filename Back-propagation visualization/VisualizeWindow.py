from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QIcon, QFont, QDoubleValidator, QValidator
from PyQt5.QtCore import Qt
import numpy as np
import sys


class VisualizeWindow(QWidget):

    def __init__(self, shallow_network):
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
        self.shallow_network = shallow_network
        self.create_widgets()
        self.cache = None
        self.grads = None

    def create_widgets(self):
        # create a number of QlineEdits equal to the inputs
        numinputs = self.shallow_network.inputLayerSize
        self.lineEdits = [None] * int(numinputs)
        lineEditsVbox = QVBoxLayout()
        x_header = QLabel("Inputs")
        x_header.setFixedSize(50, 50)
        lineEditsVbox.addWidget(x_header)
        for index in range(numinputs):
            # Make sure to set correct parent
            self.lineEdits[index] = QLineEdit(self)
            # Use the line edit you just added
            self.lineEdits[index].setFixedSize(200, 25)
            self.lineEdits[index].move(0, 50*index)
            # group the input QlineEdits in a vertical layout
            lineEditsVbox.addWidget(self.lineEdits[index])

        self.ylineEdits = [None] * int(numinputs)
        ylineEditsVbox = QVBoxLayout()
        y_header = QLabel("True Ys")
        y_header.setFixedSize(50, 50)
        ylineEditsVbox.addWidget(y_header)
        for index in range(1):
            # Make sure to set correct parent
            self.ylineEdits[index] = QLineEdit(self)
            # Use the line edit you just added
            self.ylineEdits[index].setFixedSize(200, 25)
            self.ylineEdits[index].move(0, 50*index)
            # group the input QlineEdits in a vertical layout
            ylineEditsVbox.addWidget(self.ylineEdits[index])

        # Craete the 4 buttons we need "Forward Prop.", "Backward Prop.", "Update Werights" and "Test Inputs"

        # Craeting Forward Prop. button
        fProp = QPushButton("Forward Prop.", self)
        fProp.clicked.connect(self.clicked_fProp)

        # Stylying Forward Prop. button
        fProp.setStyleSheet('background-color:white')
        fProp.setFixedSize(200, 25)

        # Craeting Backward Prop. button
        bProp = QPushButton("Backward Prop.", self)
        bProp.clicked.connect(self.clicked_bProp)

        # Stylying Backward Prop.Backward Prop. button
        bProp.setStyleSheet('background-color:white')
        bProp.setFixedSize(200, 25)

        # Craeting Update Werights button
        upWeights = QPushButton("Update Werights", self)
        upWeights.clicked.connect(self.clicked_upWeights)

        # Stylying Update Werights button
        upWeights.setStyleSheet('background-color:white')
        upWeights.setFixedSize(200, 25)

        # Craeting Test Inputs button
        testInputs = QPushButton("Test Inputs", self)
        testInputs.clicked.connect(self.clicked_testInputs)

        # Stylying Test Inputs button
        testInputs.setStyleSheet('background-color:white')
        testInputs.setFixedSize(200, 25)

        forBackVbox = QVBoxLayout()
        forBackVbox.addWidget(fProp)
        forBackVbox.addWidget(bProp)

        upTestVbox = QVBoxLayout()
        upTestVbox.addWidget(upWeights)
        upTestVbox.addWidget(testInputs)

        # Dynamic craetion of NN with changing text on labels of each circle
        # numinputs = 5
        self.inputNeuronLabel = [None] * int(numinputs)
        inputNeuronsVbox = QVBoxLayout()
        for index in range(numinputs):
            self.inputNeuronLabel[index] = QLabel('', self)
            self.inputNeuronLabel[index].setFixedSize(140, 140)
            self.inputNeuronLabel[index].setStyleSheet(
                "border: 3px solid black;border-radius: 40px;background-color:blue")
            self.inputNeuronLabel[index].setFont(QFont("Times New Roman", 12))
            # Change the label of the Nueron according to the stage (i.e: display A and Z or der or fired/notfired)
            # self.inputNeuronLabel[index].setText()
            inputNeuronsVbox.addWidget(self.inputNeuronLabel[index])

        numhidden = self.shallow_network.hiddenLayerSize
        self.hiddenNeuronLabel = [None] * int(numhidden)
        hiddenNeuronsVbox = QVBoxLayout()
        for index in range(numhidden):
            self.hiddenNeuronLabel[index] = QLabel('', self)
            self.hiddenNeuronLabel[index].setFixedSize(140, 140)
            self.hiddenNeuronLabel[index].setStyleSheet(
                "border: 3px solid black;border-radius: 40px;background-color:blue")
            self.hiddenNeuronLabel[index].setFont(QFont("Times New Roman", 12))
            # Change the label of the Nueron according to the stage (i.e: display A and Z or der or fired/notfired)
            # self.hiddenNeuronLabel[index].setText()
            hiddenNeuronsVbox.addWidget(self.hiddenNeuronLabel[index])

        numoutputs = self.shallow_network.outputLayerSize
        self.outputNeuronLabel = [None] * int(numoutputs)
        outputNeuronsVbox = QVBoxLayout()
        for index in range(numoutputs):
            self.outputNeuronLabel[index] = QLabel('', self)
            self.outputNeuronLabel[index].setFixedSize(140, 140)
            self.outputNeuronLabel[index].setStyleSheet(
                "border: 3px solid black;border-radius: 40px;background-color:blue")
            self.outputNeuronLabel[index].setFont(QFont("Times New Roman", 12))
            # Change the label of the Nueron according to the stage (i.e: display A and Z or der or fired/notfired)
            # self.outputNeuronLabel[index].setText()
            outputNeuronsVbox.addWidget(self.outputNeuronLabel[index])

        # Craeting tables to show the values of the weights and the gradients 

        # First empty table to show the weights
        weights_table = QTableWidget()
        weight_table_row_size = (numhidden * numinputs) + (numoutputs * numhidden)
        weights_table.setRowCount(int(weight_table_row_size))
        weights_table.setColumnCount(3)
        weights_table_label = QLabel("Weights (W1) and (W2)")
        weights_table_label.setFixedSize(500, 50)

        # Change the headers of columns in weights table
        weights_headers = ["From", "To", "W"]
        for col in range(3):
            hitem = QTableWidgetItem()
            hitem.setText(weights_headers[col])
            weights_table.setHorizontalHeaderItem(col, hitem)

        # Filling the "From" and "To" columns inside the weights table
        for col in range(2):
            for row in range(weight_table_row_size):
                item = QTableWidgetItem()
                # item.setText("N" + )
                # weights_table.setItem(col, row, item)
                pass 

        # Grouping the label and the weights table into a vertical layout
        weights_label_table_layout = QVBoxLayout()
        weights_label_table_layout.addWidget(weights_table_label)
        weights_label_table_layout.addWidget(weights_table)

        # Second table to show the gradients     
        gradients_table = QTableWidget()
        gradients_table.setRowCount(int(weight_table_row_size))
        gradients_table.setColumnCount(int(3))
        gradients_table_label = QLabel("Gradients (dW1) and (dw2)")
        gradients_table_label.setFixedSize(500, 50)

        # Change the headers of columns in gradients table
        gradients_headers = ["From", "To", "dW"]
        for col in range(3):
            hitem = QTableWidgetItem()
            hitem.setText(gradients_headers[col])
            gradients_table.setHorizontalHeaderItem(col, hitem)

        # Grouping the label and the gradients table into a vertical layout
        gradients_label_table_layout = QVBoxLayout()
        gradients_label_table_layout.addWidget(gradients_table_label)
        gradients_label_table_layout.addWidget(gradients_table)

        # Putting all the tables created in a horizontal layout
        tablesHLayout = QHBoxLayout()
        tablesHLayout.addLayout(weights_label_table_layout)
        tablesHLayout.addLayout(gradients_label_table_layout)


        # Making a horizontal layout for the upper half of the screen
        upperHalfLayout = QHBoxLayout()
        upperHalfLayout.addLayout(lineEditsVbox,)
        upperHalfLayout.addLayout(ylineEditsVbox)
        upperHalfLayout.addLayout(inputNeuronsVbox)
        upperHalfLayout.addLayout(hiddenNeuronsVbox)
        upperHalfLayout.addLayout(outputNeuronsVbox)

        # Making a horizontal layout for the lower half of the screen
        lowerHalfLayout = QHBoxLayout()
        lowerHalfLayout.addLayout(forBackVbox)
        lowerHalfLayout.addLayout(upTestVbox)
        lowerHalfLayout.addLayout(tablesHLayout)

        overallGrid = QVBoxLayout()
        # overallGrid.addLayout(lineEditsVbox, 0, 0)
        # overallGrid.addLayout(ylineEditsVbox, 0, 1)
        # overallGrid.addLayout(inputNeuronsVbox, 0, 2)
        # overallGrid.addLayout(hiddenNeuronsVbox, 0, 3)
        # overallGrid.addLayout(outputNeuronsVbox, 0, 4)
        # overallGrid.addLayout(forBackVbox, 2, 0)
        # overallGrid.addLayout(upTestVbox, 2, 1)

        #overallGrid.addWidget(in_hid_weights_table, 2, 2)

        overallGrid.addLayout(upperHalfLayout)
        overallGrid.addLayout(lowerHalfLayout)

        self.setLayout(overallGrid)

    

    def clicked_fProp(self):
        # This function shall call the forward propagation function using the inputs and the selected weight init
        # Also will show the value of a and z above each node in the NN
        X = self.get_X()
        yhat, cache = self.shallow_network.forward_propagation(
            X, self.shallow_network.params, self.shallow_network.hidden_act_type, self.shallow_network.op_act_type)

        self.cache = cache
        A1 = cache['A1']
        for index in range(self.shallow_network.inputLayerSize):
            self.inputNeuronLabel[index].setText(str(X[index][0]))

        for index in range(self.shallow_network.hiddenLayerSize):
            self.hiddenNeuronLabel[index].setText(str(A1[index][0]))

        for index in range(self.shallow_network.outputLayerSize):
            self.outputNeuronLabel[index].setText(str(yhat[index][0]))

    def clicked_bProp(self):
        # This function shall call the backward propagation function
        # Also will show the value of the derivative below each node in the NN
        X = self.get_X()
        Y = self.get_true_Y()
        print(Y)
        self.grads = self.shallow_network.backward_propagation(
            self.shallow_network.params, self.cache, X, Y)
        print(self.grads)
        # for index in range(self.shallow_network.inputLayerSize):
        #     self.inputNeuronLabel[index].setText(str(X[index]))

        # for index in range(self.shallow_network.hiddenLayerSize):
        #     self.hiddenNeuronLabel[index].setText(str(A1[0][index]))

        # for index in range(self.shallow_network.outputLayerSize):
        # self.outputNeuronLabel[index].setText(str(yhat[0][index]))

    def clicked_upWeights(self):
        # This function shall update the weigts shown above the arrows with the new weights after back Prop
        print(self.shallow_network.params)
        self.shallow_network.params = self.shallow_network.update_parameters(
            self.shallow_network.params, self.grads)
        print(self.shallow_network.params)

    def clicked_testInputs(self):
        # This function will just call the forward propagation function on the inputs using the new weights
        # and just show the output
        pass

    def get_X(self):
        ip_array = np.zeros(shape=(self.shallow_network.inputLayerSize, 1))
        for index in range(self.shallow_network.inputLayerSize):
            ip_array[index] = float(self.lineEdits[index].text())
        return ip_array

    def get_true_Y(self):
        ip_array = np.zeros(shape=(1, 1))
        for index in range(1):
            ip_array[index] = float(self.ylineEdits[index].text())
        print(ip_array)
        return ip_array
