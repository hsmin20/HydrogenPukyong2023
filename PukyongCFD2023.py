import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget, QAction, QFileDialog, \
    QVBoxLayout, QWidget, QPushButton, QGridLayout, QLabel, QInputDialog, \
    QLineEdit, QComboBox, QMessageBox, QCheckBox, QProgressBar, QHBoxLayout, QTableWidget, QTableWidgetItem, \
    QAbstractItemView, QHeaderView, QDialogButtonBox, QDialog, QGroupBox, QRadioButton, QButtonGroup
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def R_squared(y, y_pred):
    MAX_LINE = 4681
    total_len = len(y)
    no_of_loop = (int)(total_len / MAX_LINE)

    resi = 0.0
    tota = 0.0
    for i in range(no_of_loop):
        start_index = i * MAX_LINE
        y_sub = y[start_index:start_index+MAX_LINE]
        y_pred_sub = y_pred[start_index:start_index+MAX_LINE]

        residual = tf.reduce_sum(tf.square(tf.subtract(y_sub, y_pred_sub)))
        total = tf.reduce_sum(tf.square(tf.subtract(y_sub, tf.reduce_mean(y_sub))))
        # r2_sub = tf.subtract(1.0, tf.divide(residual, total))

        resi += residual.numpy()
        tota += total.numpy()

    if tota == 0:
        r2 = 0
    else:
        r2 = 1.0 - (resi / tota)

    return r2

class PukyongMachineLearner:
    def __init__(self):
        self.modelLoaded = False

    def set(self, nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb, callback):
        self.batchSize = batchSize
        self.epoch = epoch
        self.learningRate = learningRate
        self.splitPercentage = splitPercentage
        self.earlyStopping = earlyStopping
        self.verbose = verb
        self.callback = callback
        self.model = self.createModel(nnList)
        self.modelLoaded = True

    def isModelLoaded(self):
        return self.modelLoaded

    def fit(self, x_data, y_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=500)
            _callbacks.append(early_stopping)

        if self.splitPercentage > 0:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              validation_split=self.splitPercentage, verbose=self.verbose,
                                              callbacks=[self.callback])
        else:
            training_history = self.model.fit(x_data, y_data, batch_size=self.batchSize, epochs=self.epoch,
                                              verbose=self.verbose, callbacks=_callbacks)

        return training_history

    def fitWithValidation(self, x_train_data, y_train_data, x_valid_data, y_valid_data):
        _callbacks = [self.callback]
        if self.earlyStopping == True:
            early_stopping = tf.keras.callbacks.EarlyStopping()
            _callbacks.append(early_stopping)

        training_history = self.model.fit(x_train_data, y_train_data, batch_size=self.batchSize, epochs=self.epoch,
                                          verbose=self.verbose, validation_data=(x_valid_data, y_valid_data),
                                          callbacks=_callbacks)

        return training_history

    def predict(self, x_data):
        y_predicted = self.model.predict(x_data)

        return y_predicted

    def createModel(self, nnList):
        adamOpt = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        model = tf.keras.Sequential()

        firstLayer = True
        for nn in nnList:
            noOfNeuron = nn[0]
            activationFunc = nn[1]
            if firstLayer:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc, input_shape=[noOfNeuron]))
                firstLayer = False
            else:
                model.add(tf.keras.layers.Dense(units=noOfNeuron, activation=activationFunc))

        model.compile(loss='mse', optimizer=adamOpt)

        if self.verbose:
            model.summary()

        return model

    def saveModel(self, foldername):
        if self.modelLoaded == True:
            self.model.save(foldername)

    def saveModelJS(self, filename):
        if self.modelLoaded == True:
            tfjs.converters.save_keras_model(self.model, filename)

    def loadModel(self, foldername):
        self.model = keras.models.load_model(foldername)
        self.modelLoaded = True

    def showResult(self, y_data, training_history, y_predicted, sensor_name, height):
        # max_val = max(y_predicted)
        index_at_max = max(range(len(y_predicted)), key=y_predicted.__getitem__)
        index_at_zero = index_at_max
        for i in range(index_at_max, len(y_predicted)-1):
            val = y_predicted[i]
            val_next = y_predicted[i+1]
            if val >= 0 and val_next <=0:
                index_at_zero = i
                break;

        fig, axs = plt.subplots(2, 1, figsize=(12, 12))
        title = sensor_name + height + ' / zeroIndexTime=' + str(index_at_zero * 0.000002)
        fig.suptitle(title)

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        axs[0].scatter(x_display, y_data, color="red", s=1)
        axs[0].plot(x_display, y_predicted, color='blue')
        axs[0].grid()

        lossarray = training_history.history['loss']
        axs[1].plot(lossarray, label='Loss')
        axs[1].grid()

        plt.show()

    def showResultValid(self, y_data, training_history, y_predicted, y_train_data, y_train_pred,
                                          y_valid_data, y_valid_pred, sensor_name, height):
        r2All = R_squared(y_data, y_predicted)
        r2Train = R_squared(y_train_data, y_train_pred)
        r2Valid = R_squared(y_valid_data, y_valid_pred)

        r2AllValue = r2All  # .numpy()
        r2TrainValue = r2Train  # .numpy()
        r2ValidValue = r2Valid  # .numpy()

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        title = sensor_name + height
        fig.suptitle(title)

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        datasize = len(y_train_data)
        x_display2 = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display2[j][0] = j

        datasize = len(y_valid_data)
        x_display3 = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display3[j][0] = j

        axs[0, 0].scatter(x_display, y_data, color="red", s=1)
        axs[0, 0].plot(x_display, y_predicted, color='blue')
        title = f'All Data (R2 = {r2AllValue})'
        axs[0, 0].set_title(title)
        axs[0, 0].grid()

        lossarray = training_history.history['loss']
        axs[0, 1].plot(lossarray, label='Loss')
        axs[0, 1].set_title('Loss')
        axs[0, 1].grid()

        axs[1, 0].scatter(x_display2, y_train_data, color="red", s=1)
        axs[1, 0].scatter(x_display2, y_train_pred, color='blue', s=1)
        title = f'Train Data (R2 = {r2TrainValue})'
        axs[1, 0].set_title(title)
        axs[1, 0].grid()

        axs[1, 1].scatter(x_display3, y_valid_data, color="red", s=1)
        axs[1, 1].scatter(x_display3, y_valid_pred, color='blue', s=1)
        title = f'Validation Data (R2 = {r2ValidValue})'
        axs[1, 1].set_title(title)
        axs[1, 1].grid()

        plt.show()

class LayerDlg(QDialog):
    def __init__(self, unit='128', af='relu'):
        super().__init__()
        self.initUI(unit, af)

    def initUI(self, unit, af):
        self.setWindowTitle('Machine Learning Curve Fitting/Interpolation')

        label1 = QLabel('Units', self)
        self.tbUnits = QLineEdit(unit, self)
        self.tbUnits.resize(100, 40)

        label2 = QLabel('Activation f', self)
        self.cbActivation = QComboBox(self)
        self.cbActivation.addItem(af)
        self.cbActivation.addItem('swish')
        self.cbActivation.addItem('relu')
        self.cbActivation.addItem('selu')
        self.cbActivation.addItem('sigmoid')
        self.cbActivation.addItem('softmax')

        if af == 'linear':
            self.cbActivation.setEnabled(False)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QGridLayout()
        layout.addWidget(label1, 0, 0)
        layout.addWidget(self.tbUnits, 0, 1)

        layout.addWidget(label2, 1, 0)
        layout.addWidget(self.cbActivation, 1, 1)

        layout.addWidget(self.buttonBox, 2, 1)

        self.setLayout(layout)

class Pukyong2023MLWindow(QMainWindow):
    N_FEATURE = 4
    DATA_SET_SIZE = 111
    NUM_DATA = 4681

    def __init__(self):
        super().__init__()

        self.DEFAULT_LAYER_FILE = 'defaultNuCFD.nn'

        # self.distArrayName = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
        # self.distList = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]

        self.heightArrayName = ['H1.0m', 'H2.0m']
        self.heightList = [1.0, 2.0]

        self.barrierLocName = ['2m', '5m']
        self.barrierLocList = [2.0, 5.0]

        self.distArray2m = []
        self.heightArray2m = []
        self.cbArray2m = []

        self.distArray5m = []
        self.heightArray5m = []
        self.cbArray5m = []

        self.initUI()

        self.time_data = None
        self.time_data_n = None
        self.indexijs2m = []
        self.indexijs5m = []
        self.southSensors2m = []
        self.southSensors5m = []
        self.dataLoaded = False
        self.modelLearner = PukyongMachineLearner()

    def initMenu(self):
        # Menu
        openNN = QAction(QIcon('open.png'), 'Open NN', self)
        openNN.setStatusTip('Open Neural Network Structure from a File')
        openNN.triggered.connect(self.showNNFileDialog)

        saveNN = QAction(QIcon('save.png'), 'Save NN', self)
        saveNN.setStatusTip('Save Neural Network Structure in a File')
        saveNN.triggered.connect(self.saveNNFileDialog)

        exitMenu = QAction(QIcon('exit.png'), 'Exit', self)
        exitMenu.setStatusTip('Exit')
        exitMenu.triggered.connect(self.close)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openNN)
        fileMenu.addAction(saveNN)
        fileMenu.addSeparator()
        fileMenu.addAction(exitMenu)

        self.statusBar().showMessage('Welcome to Machine Learning for Pukyong CFD 2023')
        self.central_widget = QWidget()  # define central widget
        self.setCentralWidget(self.central_widget)  # set QMainWindow.centralWidget

    def initCSVFileReader(self):
        layout = QHBoxLayout()

        fileLabel = QLabel('csv file')
        self.editFile = QLineEdit('Please load /DataRepositoy/Pukyong2023_2m5m.csv')
        # self.editFile.setFixedWidth(700)
        openBtn = QPushButton('...')
        openBtn.clicked.connect(self.showFileDialog)

        layout.addWidget(fileLabel)
        layout.addWidget(self.editFile)
        layout.addWidget(openBtn)

        return layout

    def initSensor2m(self):
        layout = QGridLayout()

        rows = len(self.heightArrayName)

        for i in range(rows):
            cbHeight = QCheckBox(self.heightArrayName[i])
            self.heightArray2m.append(cbHeight)

        barrierPos = QLabel('Barrier 2m')
        layout.addWidget(barrierPos, 0, 0)

        for i in range(len(self.heightArray2m)):
            cbHeight = self.heightArray2m[i]
            layout.addWidget(cbHeight, i + 1, 0)

        return layout

    def initSensor5m(self):
        layout = QGridLayout()

        rows = len(self.heightArrayName)
        for i in range(rows):
            cbHeight = QCheckBox(self.heightArrayName[i])
            self.heightArray5m.append(cbHeight)

        barrierPos = QLabel('Barrier 5m')
        layout.addWidget(barrierPos, 0, 0)

        for i in range(len(self.distArray5m)):
            cbDist = self.distArray5m[i]
            layout.addWidget(cbDist, 0, i + 1)

        for i in range(len(self.heightArray5m)):
            cbHeight = self.heightArray5m[i]
            layout.addWidget(cbHeight, i + 1, 0)

        return layout

    def initReadOption(self):
        layout = QHBoxLayout()

        loadButton = QPushButton('Load Data')
        loadButton.clicked.connect(self.loadData)
        startLabel = QLabel('Start Distance')
        self.editStart = QLineEdit('1.0')
        self.editStart.setFixedWidth(100)
        endLabel = QLabel('End Distance')
        self.editEnd = QLineEdit('12.0')
        self.editEnd.setFixedWidth(100)
        showPressureButton = QPushButton('Show Pressure Graph')
        showPressureButton.clicked.connect(self.showPressureGraphs)
        showOverPressureButton = QPushButton('Show Overpressure Graph')
        showOverPressureButton.clicked.connect(self.showOverpressureGraphs)
        showImpulseButton = QPushButton('Show Impulse Graph')
        showImpulseButton.clicked.connect(self.showImpulseGraphs)

        layout.addWidget(startLabel)
        layout.addWidget(self.editStart)
        layout.addWidget(endLabel)
        layout.addWidget(self.editEnd)
        layout.addWidget(loadButton)
        layout.addWidget(showPressureButton)
        layout.addWidget(showOverPressureButton)
        layout.addWidget(showImpulseButton)

        return layout

    def initNNTable(self):
        layout = QGridLayout()

        # NN Table
        self.tableNNWidget = QTableWidget()
        self.tableNNWidget.setColumnCount(2)
        self.tableNNWidget.setHorizontalHeaderLabels(['Units', 'Activation'])

        # read default layers
        DEFAULT_LAYER_FILE = 'default4.nn'
        self.updateNNList(DEFAULT_LAYER_FILE)

        self.tableNNWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableNNWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableNNWidget.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Button of NN
        btnAdd = QPushButton('Add')
        btnAdd.setToolTip('Add a Hidden Layer')
        btnAdd.clicked.connect(self.addLayer)
        btnEdit = QPushButton('Edit')
        btnEdit.setToolTip('Edit a Hidden Layer')
        btnEdit.clicked.connect(self.editLayer)
        btnRemove = QPushButton('Remove')
        btnRemove.setToolTip('Remove a Hidden Layer')
        btnRemove.clicked.connect(self.removeLayer)
        btnLoad = QPushButton('Load')
        btnLoad.setToolTip('Load a NN File')
        btnLoad.clicked.connect(self.showNNFileDialog)
        btnSave = QPushButton('Save')
        btnSave.setToolTip('Save a NN File')
        btnSave.clicked.connect(self.saveNNFileDialog)
        btnMakeDefault = QPushButton('Make default')
        btnMakeDefault.setToolTip('Make this as a default NN layer')
        btnMakeDefault.clicked.connect(self.makeDefaultNN)

        layout.addWidget(self.tableNNWidget, 0, 0, 9, 6)
        layout.addWidget(btnAdd, 9, 0)
        layout.addWidget(btnEdit, 9, 1)
        layout.addWidget(btnRemove, 9, 2)
        layout.addWidget(btnLoad, 9, 3)
        layout.addWidget(btnSave, 9, 4)
        layout.addWidget(btnMakeDefault, 9, 5)

        return layout

    def initMLOption(self):
        layout = QGridLayout()

        batchLabel = QLabel('Batch Size')
        batchLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBatch = QLineEdit('64')
        self.editBatch.setFixedWidth(100)
        epochLabel = QLabel('Epoch')
        epochLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editEpoch = QLineEdit('50')
        self.editEpoch.setFixedWidth(100)
        lrLabel = QLabel('Learning Rate')
        lrLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editLR = QLineEdit('0.0003')
        self.editLR.setFixedWidth(100)
        self.cbVerbose = QCheckBox('Verbose')
        self.cbVerbose.setChecked(True)

        splitLabel = QLabel('Split for Validation (0 means no split-data for validation)')
        splitLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editSplit = QLineEdit('0.2')
        self.editSplit.setFixedWidth(100)
        self.cbSKLearn = QCheckBox('use sklearn split')
        self.cbSKLearn.setChecked(True)
        self.cbEarlyStop = QCheckBox('Use Early Stopping (validation data)')
        multiplierLabel = QLabel('Multiplier for Barrier Position (0 is Normalization)')
        multiplierLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editMultiplier = QLineEdit('15')
        self.editMultiplier.setFixedWidth(100)

        self.cbMinMax = QCheckBox('Use Min/Max of  All data')
        self.cbMinMax.setChecked(True)
        self.cbValidData = QCheckBox('Use Partially checked  sensors as validation')
        self.cbValidData.stateChanged.connect(self.validDataClicked)
        self.epochPbar = QProgressBar()

        layout.addWidget(batchLabel, 0, 0, 1, 1)
        layout.addWidget(self.editBatch, 0, 1, 1, 1)
        layout.addWidget(epochLabel, 0, 2, 1, 1)
        layout.addWidget(self.editEpoch, 0, 3, 1, 1)
        layout.addWidget(lrLabel, 0, 4, 1, 1)
        layout.addWidget(self.editLR, 0, 5, 1, 1)
        layout.addWidget(self.cbVerbose, 0, 6, 1, 1)

        layout.addWidget(splitLabel, 1, 0, 1, 2)
        layout.addWidget(self.editSplit, 1, 2, 1, 1)
        layout.addWidget(self.cbSKLearn, 1, 3, 1, 1)
        layout.addWidget(self.cbEarlyStop, 1, 4, 1, 1)
        layout.addWidget(multiplierLabel, 1, 5, 1, 1)
        layout.addWidget(self.editMultiplier, 1, 6, 1, 1)

        layout.addWidget(self.cbMinMax, 2, 0, 1, 2)
        layout.addWidget(self.cbValidData, 2, 2, 1, 2)
        layout.addWidget(self.epochPbar, 2, 4, 1, 4)

        return layout

    def initCommand(self):
        layout = QGridLayout()

        mlWithDataBtn = QPushButton('ML with Data')
        mlWithDataBtn.clicked.connect(self.doMachineLearningWithData)
        self.cbResume = QCheckBox('Resume Learning')
        # self.cbResume.setChecked(True)
        saveModelBtn = QPushButton('Save Model')
        saveModelBtn.clicked.connect(self.saveModel)
        loadModelBtn =  QPushButton('Load Model')
        loadModelBtn.clicked.connect(self.loadModel)
        self.cbH5Format = QCheckBox('H5')
        checkValBtn = QPushButton('Check Trained')
        checkValBtn.clicked.connect(self.checkVal)
        saveModelJSBtn = QPushButton('Save Model for JS')
        saveModelJSBtn.clicked.connect(self.saveModelJS)

        layout.addWidget(mlWithDataBtn, 0, 0, 1, 1)
        layout.addWidget(self.cbResume, 0, 1, 1, 1)
        layout.addWidget(saveModelBtn, 0, 2, 1, 1)
        layout.addWidget(loadModelBtn, 0, 3, 1, 1)
        layout.addWidget(self.cbH5Format, 0, 4, 1, 1)
        layout.addWidget(checkValBtn, 0, 5, 1, 1)
        layout.addWidget(saveModelJSBtn, 0, 6, 1, 1)

        return layout

    def initGridTable(self):
        layout = QGridLayout()

        self.tableGridWidget = QTableWidget()

        self.tableGridWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # self.tableGridWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.tableGridWidget.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.tableGridWidget.setColumnCount(1)
        item = QTableWidgetItem('')
        self.tableGridWidget.setHorizontalHeaderItem(0, item)

        # Buttons
        btnAddDist = QPushButton('Add Distance')
        btnAddDist.setToolTip('Add a Distance')
        btnAddDist.clicked.connect(self.addDistance)
        btnAddHeight = QPushButton('Add Height')
        btnAddHeight.setToolTip('Add a Height')
        btnAddHeight.clicked.connect(self.addHeight)
        btnRemoveDist = QPushButton('Remove Distance')
        btnRemoveDist.setToolTip('Remove a Distance')
        btnRemoveDist.clicked.connect(self.removeDistance)
        btnRemoveHeight = QPushButton('Remove Height')
        btnRemoveHeight.setToolTip('Remove a Height')
        btnRemoveHeight.clicked.connect(self.removeHeight)
        btnLoadDistHeight = QPushButton('Load')
        btnLoadDistHeight.setToolTip('Load predefined Distance/Height structure')
        btnLoadDistHeight.clicked.connect(self.loadDistHeight)
        barrierPos = QLabel('Barrier Position(m)')
        barrierPos.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.editBarrierPos = QLineEdit('3')
        self.editBarrierPos.setFixedWidth(100)
        predictBtn = QPushButton('Predict')
        predictBtn.clicked.connect(self.predict)

        layout.addWidget(self.tableGridWidget, 0, 0, 9, 8)
        layout.addWidget(btnAddDist, 9, 0)
        layout.addWidget(btnAddHeight, 9, 1)
        layout.addWidget(btnRemoveDist, 9, 2)
        layout.addWidget(btnRemoveHeight, 9, 3)
        layout.addWidget(btnLoadDistHeight, 9, 4)
        layout.addWidget(barrierPos, 9, 5)
        layout.addWidget(self.editBarrierPos, 9, 6)
        layout.addWidget(predictBtn, 9, 7)

        return layout

    def initUI(self):
        self.setWindowTitle('Machine Learning for Pukyong CFD')
        self.setWindowIcon(QIcon('web.png'))

        self.initMenu()

        layout = QVBoxLayout()

        sensorLayout2m = self.initSensor2m()
        sensorLayout5m = self.initSensor5m()
        readOptLayout = self.initReadOption()
        fileLayout = self.initCSVFileReader()
        cmdLayout = self.initCommand()
        nnLayout = self.initNNTable()
        mlOptLayout = self.initMLOption()
        tableLayout = self.initGridTable()

        layout.addLayout(fileLayout)
        layout.addLayout(sensorLayout2m)
        layout.addLayout(sensorLayout5m)
        layout.addLayout(readOptLayout)
        layout.addLayout(mlOptLayout)
        layout.addLayout(nnLayout)
        layout.addLayout(cmdLayout)
        layout.addLayout(tableLayout)

        self.centralWidget().setLayout(layout)

        self.resize(1200, 800)
        self.center()
        self.show()

    def showNNFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open NN file', './', filter="NN file (*.nn);;All files (*)")
        if fname[0] != '':
            self.updateNNList(fname[0])

    def updateNNList(self, filename):
        self.tableNNWidget.setRowCount(0)
        with open(filename, "r") as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                strAr = line.split(',')
                self.tableNNWidget.insertRow(count)
                self.tableNNWidget.setItem(count, 0, QTableWidgetItem(strAr[0]))
                self.tableNNWidget.setItem(count, 1, QTableWidgetItem(strAr[1].rstrip()))
                count += 1

    def saveNNFileDialog(self):
        fname = QFileDialog.getSaveFileName(self, 'Save NN file', './', filter="NN file (*.nn)")
        if fname[0] != '':
            self.saveNNFile(fname[0])

    def makeDefaultNN(self):
        filename = 'default.nn'
        self.saveNNFile(filename)
        QMessageBox.information(self, 'Saved', 'Neural Network Default Layers are set')

    def saveNNFile(self, filename):
        with open(filename, "w") as f:
            count = self.tableNNWidget.rowCount()
            for row in range(count):
                unit = self.tableNNWidget.item(row, 0).text()
                af = self.tableNNWidget.item(row, 1).text()
                f.write(unit + "," + af + "\n")

    def getNNLayer(self):
        nnList = []
        count = self.tableNNWidget.rowCount()
        for row in range(count):
            unit = int(self.tableNNWidget.item(row, 0).text())
            af = self.tableNNWidget.item(row, 1).text()
            nnList.append((unit, af))

        return nnList

    def addLayer(self):
        dlg = LayerDlg()
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            size = self.tableNNWidget.rowCount()
            self.tableNNWidget.insertRow(size-1)
            self.tableNNWidget.setItem(size-1, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(size-1, 1, QTableWidgetItem(af))

    def editLayer(self):
        row = self.tableNNWidget.currentRow()
        if row == -1 or row == (self.tableNNWidget.rowCount() - 1):
            return

        unit = self.tableNNWidget.item(row, 0).text()
        af = self.tableNNWidget.item(row, 1).text()
        dlg = LayerDlg(unit, af)
        rc = dlg.exec()
        if rc == 1: # ok
            unit = dlg.tbUnits.text()
            af = dlg.cbActivation.currentText()
            self.tableNNWidget.setItem(row, 0, QTableWidgetItem(unit))
            self.tableNNWidget.setItem(row, 1, QTableWidgetItem(af))

    def removeLayer(self):
        row = self.tableNNWidget.currentRow()
        if row > 0 and row < (self.tableNNWidget.rowCount() - 1):
            self.tableNNWidget.removeRow(row)

    def addDistance(self):
        sDist, ok = QInputDialog.getText(self, 'Input Distance', 'Distance to add:')
        if ok:
            cc = self.tableGridWidget.columnCount()
            self.tableGridWidget.setColumnCount(cc + 1)
            item = QTableWidgetItem('S' + sDist + 'm')
            self.tableGridWidget.setHorizontalHeaderItem(cc, item)

    def addHeight(self):
        sHeight, ok = QInputDialog.getText(self, 'Input Height', 'Height to add:')
        if ok:
            rc = self.tableGridWidget.rowCount()
            self.tableGridWidget.setRowCount(rc + 1)
            item = QTableWidgetItem('H' + sHeight + 'm')
            self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def removeDistance(self):
        col = self.tableGridWidget.currentColumn()
        if col == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        if col == 0:
            QMessageBox.warning(self, 'Warning', 'First column cannot be removed')
            return

        self.tableGridWidget.removeColumn(col)

    def removeHeight(self):
        row = self.tableGridWidget.currentRow()
        if row == -1:
            QMessageBox.warning(self, 'Warning', 'Select any cell')
            return
        self.tableGridWidget.removeRow(row)

    def loadDistHeight(self):
        fname = QFileDialog.getOpenFileName(self, 'Open distance/height data file', '/srv/MLData',
                                            filter="CSV file (*.csv);;All files (*)")

        if fname[0] != '':
            with open(fname[0], "r") as f:
                self.tableGridWidget.setRowCount(0)
                self.tableGridWidget.setColumnCount(0)

                lines = f.readlines()

                distAr = lines[0].split(',')
                for sDist in distAr:
                    cc = self.tableGridWidget.columnCount()
                    self.tableGridWidget.setColumnCount(cc + 1)
                    item = QTableWidgetItem('S' + sDist + 'm')
                    self.tableGridWidget.setHorizontalHeaderItem(cc, item)

                heightAr = lines[1].split(',')
                for sHeight in heightAr:
                    rc = self.tableGridWidget.rowCount()
                    self.tableGridWidget.setRowCount(rc + 1)
                    item = QTableWidgetItem('H' + sHeight + 'm')
                    self.tableGridWidget.setVerticalHeaderItem(rc, item)

    def validDataClicked(self, state):
        if state == Qt.Checked:
            self.editSplit.setEnabled(False)
            self.cbSKLearn.setEnabled(False)
        else:
            self.editSplit.setEnabled(True)
            self.cbSKLearn.setEnabled(True)

    def showFileDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open data file', '/srv/MLData', filter="CSV file (*.csv);;All files (*)")
        if fname[0]:
            self.editFile.setText(fname[0])
            self.df = pd.read_csv(fname[0], dtype=float)

    def loadData(self):
        filename = self.editFile.text()
        if filename == '' or filename.startswith('Please'):
            QMessageBox.about(self, 'Warining', 'No CSV Data')
            return

        rows = len(self.heightArrayName)
        # cols = len(self.distArrayName)

        listSelected2m = []
        for i in range(rows):
            qcheckbox = self.heightArray2m[i]
            if qcheckbox.checkState() != Qt.Unchecked:
                listSelected2m.append(i)

        listSelected5m = []
        for i in range(rows):
            qcheckbox = self.heightArray5m[i]
            if qcheckbox.checkState() != Qt.Unchecked:
                listSelected5m.append(i)

        if len(listSelected2m) == 0 and len(listSelected5m) == 0:
            QMessageBox.information(self, 'Warning', 'Select sensor(s) first..')
            return

        try:
            self.readCSV(listSelected2m, listSelected5m)

        except ValueError:
            QMessageBox.information(self, 'Error', 'There is some error...')
            return

        self.dataLoaded = True
        QMessageBox.information(self, 'Done', 'Data is Loaded')

    def readCSV(self, listSelected2m, listSelected5m):
        self.indexijs2m.clear()
        self.indexijs5m.clear()
        self.time_data = None
        self.time_data_n = []
        self.southSensors2m.clear()
        self.southSensors5m.clear()

        self.time_data = self.df.values[0:self.NUM_DATA, 0:1].flatten()
        max_time = max(self.time_data)
        min_time = min(self.time_data)

        for i in range(len(self.time_data)):
            t = self.time_data[i]
            self.time_data_n.append((t - min_time) / (max_time - min_time))

        startDist = float(self.editStart.text())
        endDist = float(self.editEnd.text())

        if startDist < 1.0:
            QMessageBox.warning(self, 'warning', 'start dist is less than 1.0')
            return
        if endDist > 12.0:
            QMessageBox.warning(self, 'warning', 'end dist is bigger than 12.0')
            return
        if startDist >= endDist:
            QMessageBox.warning(self, 'warning', 'start dist is bigger than end dist')

        startIndex = int((startDist - 1.0) * 10)
        endIndex = int((endDist - 1.0) * 10)

        for index in listSelected2m:
            for j in range(startIndex, endIndex+1, 1):
                sensorName, data = self.getPressureData(index, j, 0)

                self.indexijs2m.append((index, j))
                self.southSensors2m.append(data)

        for index in listSelected5m:
            for j in range(self.DATA_SET_SIZE):
                sensorName, data = self.getPressureData(index, j, self.DATA_SET_SIZE*2)

                self.indexijs5m.append((index, j))
                self.southSensors5m.append(data)

    def getPressureData(self, i, j, skipCol):
        col_no = i * self.DATA_SET_SIZE + j + 1 + skipCol

        distArrayName = 'H1mB'
        if i == 1:
            distArrayName = 'H2mB'
        if i == 2:
            distArrayName = 'H1mB'
        if i == 3:
            distArrayName = 'H2mB'
        sensorName = distArrayName + self.barrierLocName[i]

        data = self.df.values[0:self.NUM_DATA, col_no:col_no + 1].flatten()

        return sensorName, data


    def showPressureGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        plt.figure()

        numSensors = len(self.southSensors2m)
        for index in range(numSensors):
            t_data = self.time_data
            s_data = self.southSensors2m[index]

            index_ij = self.indexijs2m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + str(index/10.0 + 1) + 'm' + self.heightArrayName[i] + 'Barrier2m'
            plt.scatter(t_data, s_data, label=sensorName, s=1)

        numSensors = len(self.southSensors5m)
        for index in range(numSensors):
            t_data = self.time_data
            s_data = self.southSensors5m[index]

            index_ij = self.indexijs5m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + str(index) + 'm' + self.heightArrayName[i] + 'Barrier5m'
            plt.scatter(t_data, s_data, label=sensorName, s=1)

        plt.title('Pressure Graph')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

    def showOverpressureGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        plt.figure()

        xdata = []
        ydata = []

        numSensors = len(self.southSensors2m)
        for index in range(numSensors):
            s_data = self.southSensors2m[index]
            overpressure = max(s_data)

            index_ij = self.indexijs2m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + str(index/10.0 + 1) + 'm' + self.heightArrayName[i] + 'Barrier2m'

            xdata.append(sensorName)
            ydata.append(overpressure)

        numSensors = len(self.southSensors5m)
        for index in range(numSensors):
            s_data = self.southSensors5m[index]
            overpressure = max(s_data)

            index_ij = self.indexijs5m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + str(index) + 'm' + self.heightArrayName[i] + 'Barrier5m'

            xdata.append(sensorName)
            ydata.append(overpressure)

        plt.plot(xdata, ydata, '-o')

        plt.title('Overpressure Graph')
        plt.xlabel('Sensors')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.xticks(rotation=60)
        plt.grid()
        plt.tight_layout()

        plt.show()

    def showImpulseGraphs(self):
        if self.dataLoaded == False:
            QMessageBox.information(self, 'Warning', 'Load Data First')
            return

        plt.figure()

        xdata = []
        ydata = []

        numSensors = len(self.southSensors2m)
        for index in range(numSensors):
            s_data = self.southSensors2m[index]
            impulse = self.getImpulse(s_data)

            index_ij = self.indexijs2m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + str(index/10.0 + 1) + 'm' + self.heightArrayName[i] + 'Barrier2m'

            # print(sensorName + ' : ' + str(impulse))

            xdata.append(sensorName)
            ydata.append(impulse)

        numSensors = len(self.southSensors5m)
        for index in range(numSensors):
            s_data = self.southSensors5m[index]
            impulse = self.getImpulse(s_data)

            index_ij = self.indexijs5m[index]
            i = index_ij[0]
            j = index_ij[1]

            sensorName = 'S' + str(index/10.0 + 1) + 'm' + self.heightArrayName[i] + 'Barrier5m'

            # print(sensorName + ' : ' + str(impulse))

            xdata.append(sensorName)
            ydata.append(impulse)

        plt.bar(xdata, ydata)

        plt.title('Impulse Graph')
        plt.xlabel('Sensors')
        plt.ylabel('Impulse(kgm)/s')
        plt.legend(loc='upper right', markerscale=4.)
        plt.xticks(rotation=60)
        plt.grid()
        plt.tight_layout()

        plt.show()

    def _prepareForMachineLearning(self):
        heightList_n, distList_n = self._normalize()

        # separate data to train & validation
        barrierPositionList = [ 2, 5 ]
        # Normalized check
        multiplier = float(self.editMultiplier.text())
        for i in range(len(barrierPositionList)):
            if multiplier < 0.1:
                barrierPositionList[i] = (barrierPositionList[i] - 2) / 3
            else:
                barrierPositionList[i] *= multiplier

        trainArray2m, trainIndex2m = self.getArrayFromSensors2m()
        trainArray5m, trainIndex5m = self.getArrayFromSensors5m()

        trainArray = trainArray2m + trainArray5m
        trainIndex = trainIndex2m + trainIndex5m
        trainBarrierPos = []
        for i in range(len(trainArray2m)):
            trainBarrierPos.append(barrierPositionList[0])
        for i in range(len(trainArray5m)):
            trainBarrierPos.append(barrierPositionList[1])

        if len(trainArray) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_train_data = np.zeros(shape=(self.NUM_DATA * len(trainArray), self.N_FEATURE))
        y_train_data = np.zeros(shape=(self.NUM_DATA * len(trainArray) , 1))

        # get maxp/minp from all of the training data
        maxp = -1000000
        minp = 100000
        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(trainArray)

        for i in range(len(trainArray)):
            index_ij = trainIndex[i]
            index_i = index_ij[0]
            index_j = index_ij[1]
            s_data = trainArray[i]
            barrierPos = trainBarrierPos[i]
            x_data_1, y_data_1 = self.prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxp, minp, barrierPos)

            for j in range(self.NUM_DATA):
                x_train_data[i*self.NUM_DATA + j] = x_data_1[j]
                y_train_data[i*self.NUM_DATA + j] = y_data_1[j]

        return x_train_data, y_train_data

    def _prepareForMachineLearningSKLearn(self, splitPecentage):
        heightList_n, distList_n = self._normalize()

        allArray, allIndex, barrierPosList = self.getAllArray()

        if len(allArray) == 0:
            QMessageBox.warning(self, 'warning', 'No Training Data!')
            return

        x_all_data = np.zeros(shape=(self.NUM_DATA * len(allArray), self.N_FEATURE))
        y_all_data = np.zeros(shape=(self.NUM_DATA * len(allArray) , 1))

        # get maxp/minp from all of the training data
        if self.cbMinMax.isChecked() == True:
            maxp, minp = self.getMinMaxPresssureOfAllData()
        else:
            maxp, minp = self.getMinMaxPressureOfLoadedData(allArray)

        for i in range(len(allArray)):
            index_ij = allIndex[i]
            s_data = allArray[i]
            barrierPos = barrierPosList[i]
            x_data_1, y_data_1 = self.prepareOneSensorData(heightList_n[index_ij[0]], distList_n[index_ij[1]], s_data,
                                                           maxp, minp, barrierPos)

            for j in range(self.NUM_DATA):
                x_all_data[i*self.NUM_DATA + j] = x_data_1[j]
                y_all_data[i*self.NUM_DATA + j] = y_data_1[j]

        x_train_data, x_valid_data, y_train_data, y_valid_data = train_test_split(x_all_data, y_all_data,
                                                                                test_size=splitPecentage,
                                                                                random_state=42)

        return x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data

    def getMinMaxPresssureOfAllData(self):
        maxPressure = -1000000
        minPressure = 1000000
        for i in range(self.DATA_SET_SIZE*4 + 1):
            if i == 0:
                continue

            one_data = self.df.values[::, i:i+1].flatten()
            maxp_local = max(one_data)
            minp_local = min(one_data)
            if maxp_local > maxPressure:
                maxPressure = maxp_local
            if minp_local < minPressure:
                minPressure = minp_local

        return maxPressure, minPressure

    def getMinMaxPressureOfLoadedData(self, trainArray):
        maxPressure = -100000
        minPressure = 100000

        numSensors = len(trainArray)
        for i in range(numSensors):
            s_data = trainArray[i]
            maxp_local = max(s_data)
            minp_local = min(s_data)
            if maxp_local > maxPressure:
                maxPressure = maxp_local
            if minp_local < minPressure:
                minPressure = minp_local

        return maxPressure, minPressure

    def doMachineLearningWithData(self):
        if (not self.indexijs2m and not self.indexijs5m) or (not self.southSensors2m and not self.southSensors5m):
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        batchSize = int(self.editBatch.text())
        epoch = int(self.editEpoch.text())
        if epoch < 1:
            QMessageBox.warning(self, 'warning', 'Epoch shall be greater than 0')
            return

        learningRate = float(self.editLR.text())
        verbose = self.cbVerbose.isChecked()

        # useValidation = self.cbValidData.isChecked()
        splitPercentage = float(self.editSplit.text())
        useSKLearn = self.cbSKLearn.isChecked()
        earlyStopping = self.cbEarlyStop.isChecked()

        if splitPercentage > 0.0 and useSKLearn:
            x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data, y_valid_data = \
                self._prepareForMachineLearningSKLearn(splitPercentage)
        else:
            x_train_data, y_train_data = self._prepareForMachineLearning()

        if splitPercentage > 0.0 and useSKLearn:
            self.doMachineLearningWithValidation(x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data,
                                                 y_valid_data, batchSize, epoch, learningRate, splitPercentage,
                                                 earlyStopping, verbose)
        else:
            self.doMachineLearning(x_train_data, y_train_data, batchSize, epoch, learningRate, splitPercentage,
                                   earlyStopping, verbose)

        QApplication.restoreOverrideCursor()

    def saveData(self):
        if (not self.indexijs2m and not self.indexijs5m and not self.indexijs8m) or \
                (not self.southSensors2m and not self.southSensors5m and not self.southSensors8m):
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        numSensors = len(self.southSensors)
        for i in range(numSensors):
            index_ij = self.indexijs[i]
            t_data = self.time_data
            s_data = self.southSensors[i]

            suggestion = '/srv/MLData/' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                file = open(filename[0], 'w')

                column1 = 'Time, ' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '\n'
                file.write(column1)

                numData = len(s_data)
                for j in range(numData):
                    line = str(t_data[j]) + ',' + str(s_data[j])
                    file.write(line)
                    file.write('\n')

                file.close()

                QMessageBox.information(self, "Save", filename[0] + " is saved successfully")

    def saveDataAll(self):
        suggestion = '/srv/MLData/Changed.csv'
        # suggestion = '../MLData/' + self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]] + '.csv'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")
        if filename[0] != '':
            file = open(filename[0], 'w')

            column1 = 'Time, '
            numSensors = len(self.southSensors)
            for i in range(numSensors):
                index_ij = self.indexijs[i]
                column1 += self.distArrayName[index_ij[1]] + self.heightArrayName[index_ij[0]]
                if i != (numSensors - 1):
                    column1 += ','
            column1 += '\n'

            file.write(column1)

            t_data = self.time_data

            numData = len(t_data)
            for j in range(numData):
                line = str(t_data[j]) + ','

                for i in range(numSensors):
                    s_data = self.southSensors[i]
                    line += str(s_data[j])
                    if i != (numSensors - 1):
                        line += ','

                file.write(line)
                file.write('\n')

            file.close()

            QMessageBox.information(self, "Save", filename[0] + " is saved successfully")

    def saveModel(self):
        suggestion = ''
        filename = ''
        h5checked = self.cbH5Format.isChecked()
        if h5checked == True:
            suggestion = '/srv/MLData/newModel.h5'
            filename = QFileDialog.getSaveFileName(self, 'Save Model File', suggestion, filter="h5 file (*.h5)")
        else:
            suggestion = '/srv/MLData'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "Model")

        if filename[0] != '':
            self.modelLearner.saveModel(filename[0])

            QMessageBox.information(self, 'Saved', 'Model is saved.')

    def saveModelJS(self):
        suggestion = '/srv/MLData'
        filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "Model")
        if filename[0] != '':
            self.modelLearner.saveModelJS(filename[0])

            QMessageBox.information(self, 'Saved', 'Model is saved for Javascript.')

    def loadModel(self):
        if self.cbH5Format.isChecked():
            fname = QFileDialog.getOpenFileName(self, 'Open data file', '/srv/MLData',
                                                filter="CSV file (*.h5);;All files (*)")
            if fname[0] != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.modelLearner.loadModel(fname[0])
                QApplication.restoreOverrideCursor()

                QMessageBox.information(self, 'Loaded', 'Model is loaded.')

        else:
            fname = QFileDialog.getExistingDirectory(self, 'Select Folder', "/srv/MLData",
                                                      QFileDialog.ShowDirsOnly)

            if fname != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.modelLearner.loadModel(fname)
                QApplication.restoreOverrideCursor()

                QMessageBox.information(self, 'Loaded', 'Model is loaded.')

    def checkVal(self):
        if (not self.indexijs2m and not self.indexijs5m) or \
                (not self.southSensors2m and not self.southSensors5m ):
            QMessageBox.about(self, 'Warining', 'Load Data First')
            return

        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        start_time = time.time()

        QApplication.setOverrideCursor(Qt.WaitCursor)

        x_data, y_data = self._prepareForMachineLearning()
        y_predicted = self.modelLearner.predict(x_data)

        QApplication.restoreOverrideCursor()

        print("--- %s seconds ---" % (time.time() - start_time))

        datasize = len(y_data)
        x_display = np.zeros((datasize, 1))
        for j in range(datasize):
            x_display[j][0] = j

        plt.figure()
        plt.scatter(x_display, y_data, label='original data', color="red", s=1)
        plt.scatter(x_display, y_predicted, label='predicted', color="blue", s=1)
        plt.title('Machine Learning Prediction')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def predict(self):
        if self.time_data is None:
            QMessageBox.warning(self, 'Warning', 'For timedata, load at least one data')
            return

        if self.modelLearner.modelLoaded == False:
            QMessageBox.about(self, 'Warning', 'Model is not created/loaded')
            return

        distCount = self.tableGridWidget.columnCount()
        heightCount = self.tableGridWidget.rowCount()
        if distCount < 1 or heightCount < 1:
            QMessageBox.warning(self, 'Warning', 'You need to add distance or height to predict')
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        sDistHeightArray = []
        distHeightArray = []

        # put sensors in order of original Pukyong Excel file
        for j in range(heightCount):
            rownum = heightCount - (j + 1)
            height = self.tableGridWidget.verticalHeaderItem(rownum).text()
            heightOnly = height[1:len(height) - 1]

            for i in range(distCount):
                dist = self.tableGridWidget.horizontalHeaderItem(i).text()
                distOnly = dist[1:len(dist) - 1]

                distf = float(distOnly)
                heightf = float(heightOnly)

                sDistHeightArray.append(dist+height)
                distHeightArray.append((distf, heightf))

        # numSensors = len(distHeightArray)
        # if self.cbMinMax.isChecked() == True:
        #     maxp, minp = self.getMinMaxPresssureOfAllData()
        # else:
        #     maxp, minp = self.getMinMaxPressureOfLoadedData()

        maxp, minp = self.getMinMaxPresssureOfAllData()

        maxDist = 12.0
        minDist = 1.0

        heightList_n = np.copy(self.heightList)
        maxHeight = max(heightList_n)
        minHeight = min(heightList_n)

        sBarrierPosition = self.editBarrierPos.text()
        barrierList = sBarrierPosition.split(',')

        for sBarrier in barrierList:
            barrierPosition = float(sBarrier)

            y_array = []
            for distHeight in distHeightArray:
                x_data = self._prepareOneSensorForPredict(distHeight, maxDist, minDist, maxHeight, minHeight, barrierPosition)

                y_predicted = self.modelLearner.predict(x_data)
                self.unnormalize(y_predicted, maxp, minp)

                y_array.append(y_predicted)

            QApplication.restoreOverrideCursor()

            self.savePressureData(barrierPosition, distHeightArray, y_array)

            resultArray = self.showPredictionGraphs(sDistHeightArray, distHeightArray, y_array)

            # self.saveOPImpulse(resultArray, barrierPosition)

    def saveOPImpulse(self, resultArray, barrierPosition):
        # reply = QMessageBox.question(self, 'Message', 'Do you want to save overpressure and impulse to a file?',
        #                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        reply = QMessageBox.Yes
        if reply == QMessageBox.Yes:
            sMultiplier = self.editMultiplier.text()

            suggestion = '/srv/MLData/opAndImpulse_pukyong_B' + str(barrierPosition) + 'm_' + sMultiplier + 'xB.csv'
            # filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")
            #
            # if filename[0] != '':
            if suggestion != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)
                # file = open(filename[0], 'w')
                file = open(suggestion, 'w')

                column1 = 'distance,height,indexAtMax,overpressure,indexAtZero,impulse\n'
                file.write(column1)

                for col in resultArray:
                    file.write(col+'\n')

                QApplication.restoreOverrideCursor()

    def savePressureData(self, barrierPosition, distHeightArray, y_array):
        reply = QMessageBox.question(self, 'Message', 'Do you want to save Pressure data to files?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            suggestion = '/srv/MLData/predicted_barrier' + str(barrierPosition) + 'm.csv'
            filename = QFileDialog.getSaveFileName(self, 'Save File', suggestion, "CSV Files (*.csv)")

            if filename[0] != '':
                QApplication.setOverrideCursor(Qt.WaitCursor)

                file = open(filename[0], 'w')

                column1 = 'Time,Barrier_Dist,'
                for distHeight in distHeightArray:
                    distance = distHeight[0]
                    height = distHeight[1]
                    colname = 'S' + str(distance) + 'mH' + str(height) + 'm,'
                    column1 += colname

                column1 = column1[:-1]
                column1 += '\n'
                file.write(column1)

                dataSize = len(y_array[0])
                for index in range(dataSize):
                    time = self.time_data[index]
                    nextColumn = str(time) + ',' + self.editBarrierPos.text() + ','

                    for yindex in range(len(y_array)):
                        one_y = y_array[yindex]
                        onep = one_y[index][0]
                        nextColumn += str(onep) + ','

                    nextColumn = nextColumn[:-1]
                    nextColumn += '\n'
                    file.write(nextColumn)

                QApplication.restoreOverrideCursor()

    def unnormalize(self, data, max, min):
        for i in range(len(data)):
            data[i] = data[i] * (max - min) + min

    def _prepareOneSensorForPredict(self, distHeight, maxDist, minDist, maxHeight, minHeight, barrierPosition):
        distance = (distHeight[0] - minDist) / (maxDist - minDist)
        height = (distHeight[1] - minHeight) / (maxHeight - minHeight)
        # check for barrierPosition normalization
        multiplier = float(self.editMultiplier.text())
        if multiplier < 0.1:
            barrierPosition = (barrierPosition - 2) / 3
        else:
            barrierPosition *= multiplier

        feature_size = 4  # pressure and height and distance and barrier position
        x_data = np.zeros(shape=(self.NUM_DATA, feature_size))

        for i in range(self.NUM_DATA):
            x_data[i][0] = self.time_data_n[i]
            x_data[i][1] = height
            x_data[i][2] = distance
            x_data[i][3] = barrierPosition

        return x_data

    def showPredictionGraphs(self, sDistHeightArray, distHeightArray, y_array):
        # numSensors = len(y_array)
        resultArray = []

        plt.figure()
        for i in range(len(y_array)):
            t_data = self.time_data
            s_data = y_array[i]

            distHeight = distHeightArray[i]
            lab = sDistHeightArray[i]

            distance = distHeight[0]
            height = distHeight[1]

            index_at_max = max(range(len(s_data)), key=s_data.__getitem__)
            overpressure = max(s_data)
            impulse, index_at_zero = self.getImpulseAndIndexZero(s_data)

            dispLabel = lab + '/op=' + format(overpressure[0], '.2f') + '/impulse=' + format(impulse, '.2f')

            resultArray.append(str(distance) + ',' + str(height) + ',' + str(index_at_max) + ',' +
                               format(overpressure[0], '.6f') + ',' + str(index_at_zero) + ',' + format(impulse, '.6f'))

            plt.scatter(t_data, s_data, label=dispLabel, s=1)

        plt.title('Pressure Graph')
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right', markerscale=4.)
        plt.grid()

        plt.show()

        return resultArray

    def getImpulse(self, data):
        sumImpulse = 0
        impulseArray = []
        for i in range(len(data)):
            cur_p = data[i]
            sumImpulse += cur_p
            impulseArray.append(sumImpulse)

        impulse = max(impulseArray)

        return impulse

    def getImpulseAndIndexZero(self, data):
        index_at_max = max(range(len(data)), key=data.__getitem__)

        sumImpulse = 0
        impulseArray = []
        initP = data[0][0]
        for i in range(len(data)):
            cur_p = data[i][0]
            if cur_p > 0 and cur_p <= initP :
                cur_p = 0

            sumImpulse += cur_p * 0.000002
            impulseArray.append(sumImpulse)

        # index_at_zero = max(range(len(impulseArray)), key=impulseArray.__getitem__)
        impulse = max(impulseArray)
        index_at_zero = impulseArray.index(impulse)

        return impulse, index_at_zero

    def checkDataGraph(self, sensorName, time_data, rawdata, filetered_data, data_label, iterNum):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ' (iter=' + str(iterNum) + ', impulse=' + format(impulse_filtered, '.4f') + ')'
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def checkDataGraph2(self, sensorName, time_data, rawdata, filetered_data, data_label, index_at_max, overpressure, index_at_zero):
        impulse_original = self.getImpulse(rawdata)
        impulse_filtered = self.getImpulse(filetered_data)

        plt.figure()

        rawLabel = 'Raw-Normalized (impulse=' + format(impulse_original, '.4f') + ')'
        plt.scatter(time_data, rawdata, label=rawLabel, color="red", s=1)
        filterLabel = data_label + ', impulse=' + format(impulse_filtered, '.4f') + ',indexMax=' + str(index_at_max) \
                      + ',Overpressure=' + str(overpressure) + ',indexZero=' + str(index_at_zero)
        plt.scatter(time_data, filetered_data, label=filterLabel, color="blue", s=1)
        plt.title(sensorName)
        plt.xlabel('time (ms)')
        plt.ylabel('pressure (kPa)')
        plt.legend(loc='upper right')
        plt.grid()

        plt.show()

    def prepareOneSensorData(self, height, dist, data, maxp, minp, barrierPos):
        datasize = len(data)
        cur_data = list(data)

        # don't remember why I didn't normalize pressure....
        for j in range(datasize):
            cur_data[j] = (cur_data[j] - minp) / (maxp - minp)

        # time and height and distance and barrier location
        x_data = np.zeros((datasize, self.N_FEATURE))

        # fill the data
        for j in range(datasize):
            x_data[j][0] = self.time_data_n[j]
            x_data[j][1] = height
            x_data[j][2] = dist
            x_data[j][3] = barrierPos

        y_data = np.zeros((datasize, 1))
        for j in range(datasize):
            y_data[j][0] = cur_data[j]

        return x_data, y_data

    def getAllArray(self):
        barrierPositionList = [2, 5]
        # Normalized check
        multiplier = float(self.editMultiplier.text())
        for i in range(len(barrierPositionList)):
            if multiplier < 0.1:
                barrierPositionList[i] = (barrierPositionList[i] - 2) / 3
            else:
                barrierPositionList[i] *= multiplier

        allArray = []
        allIndex = []
        barrierPosList = []

        numSensors = len(self.southSensors2m)
        for i in range(numSensors):
            indexij = self.indexijs2m[i]
            one_data = self.southSensors2m[i]
            allArray.append(one_data)
            allIndex.append(indexij)
            barrierPosList.append(barrierPositionList[0])

        numSensors = len(self.southSensors5m)
        for i in range(numSensors):
            indexij = self.indexijs5m[i]
            one_data = self.southSensors5m[i]
            allArray.append(one_data)
            allIndex.append(indexij)
            barrierPosList.append(barrierPositionList[1])

        return allArray, allIndex, barrierPosList

    def getArrayFromSensors2m(self):
        numSensors = len(self.southSensors2m)

        # separate data to train & validation
        trainArray = []
        trainIndex = []

        for i in range(numSensors):
            indexij = self.indexijs2m[i]
            # row_num = indexij[0]
            one_data = self.southSensors2m[i]

            trainArray.append(one_data)
            trainIndex.append(indexij)

        return trainArray, trainIndex

    def getArrayFromSensors5m(self):
        numSensors = len(self.southSensors5m)

        # separate data to train & validation
        trainArray = []
        trainIndex = []

        for i in range(numSensors):
            indexij = self.indexijs5m[i]
            # row_num = indexij[0]
            one_data = self.southSensors5m[i]

            trainArray.append(one_data)
            trainIndex.append(indexij)

        return trainArray, trainIndex

    def _normalize(self):
        heightList_n = np.copy(self.heightList)
        maxHeight = max(heightList_n)
        minHeight = min(heightList_n)
        for j in range(len(heightList_n)):
            heightList_n[j] = (heightList_n[j] - minHeight) / (maxHeight - minHeight)

        distList = []
        for i in range(self.DATA_SET_SIZE):
            distList.append(i * 0.1 + 1)
        distList_n = np.copy(distList)
        maxDist = max(distList_n) #12
        minDist = min(distList_n) #1.0
        for j in range(len(distList_n)):
            distList_n[j] = (distList_n[j] - minDist) / (maxDist - minDist)

        return heightList_n, distList_n

    def doMachineLearning(self, x_data, y_data, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb):
        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
                                  TfCallback(self.epochPbar))

        training_history = self.modelLearner.fit(x_data, y_data)

        y_predicted = self.modelLearner.predict(x_data)
        self.modelLearner.showResult(y_data, training_history, y_predicted, 'Sensors', 'Height')

    def doMachineLearningWithValidation(self, x_all_data, y_all_data, x_train_data, y_train_data, x_valid_data,
                                        y_valid_data, batchSize, epoch, learningRate, splitPercentage, earlyStopping,
                                        verb):
        self.epochPbar.setMaximum(epoch)

        if self.cbResume.isChecked() == False or self.modelLearner.modelLoaded == False:
            nnList = self.getNNLayer()
            self.modelLearner.set(nnList, batchSize, epoch, learningRate, splitPercentage, earlyStopping, verb,
                                  TfCallback(self.epochPbar))

        training_history = self.modelLearner.fitWithValidation(x_train_data, y_train_data, x_valid_data, y_valid_data)

        y_predicted = self.modelLearner.predict(x_all_data)
        y_train_pred = self.modelLearner.predict(x_train_data)
        y_valid_pred = self.modelLearner.predict(x_valid_data)

        self.modelLearner.showResultValid(y_all_data, training_history, y_predicted, y_train_data, y_train_pred,
                                          y_valid_data, y_valid_pred, 'Sensors', 'Height')

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

class TfCallback(tf.keras.callbacks.Callback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.curStep = 1

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.setValue(self.curStep)
        self.curStep += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Pukyong2023MLWindow()
    sys.exit(app.exec_())