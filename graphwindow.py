import sys, os, random
import time

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from perceptron import *
from lms import *
from backpropagation import *
from datetime import datetime
from app import *
from graphwindow import *


class ShowOutput(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('Demo: PyQt with matplotlib')
        self.mw = matplotlibWidget()
        self.setupUi(self)
        self.statusbar.showMessage("Ready",5000)
        self.plotButton.clicked.connect(self.config_settings)
        self.snapButton.clicked.connect(self.take_shot)
        self.actionAbout.triggered.connect(self.about)
        self.actionSnapshot.triggered.connect(self.take_shot)
        self.actionExit.triggered.connect(self.exit)
        self.actionNew.triggered.connect(self.newProject)
        self.actionOpen.triggered.connect(self.openProject)

    def openProject(self):
        fname = QFileDialog.getOpenFileUrl(self,'Open File','c://','CFG files (*.cfg)')
        fname = fname[0].toString(QUrl.RemoveScheme)
        fname = fname.split('///')
        self.myapp = ShowOutput()
        self.myapp.set_settings(fname[-1].split('/')[-1])
        self.myapp.show()

    def newProject(self):
        self.myapp = NextOne()
        self.myapp.show()

    def exit(self):
        self.close()

    def about(self):
        msgBox = QMessageBox()
        horizontalSpacer = QSpacerItem(500, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        msgBox.setWindowTitle( "Artificial Neural Network Simulator" )
        msgBox.setText( "Developed by:\n\t\tAmalesh Jana\n\t\tAyan Adhikari\n\t\tBiswadip Mondal\n\t\tDipanjan Saha\nGuided by:\n\t\tDr.(Prof.) Sudarshan Nandi" )
        layout = msgBox.layout()
        layout.addItem(horizontalSpacer, layout.rowCount(), 0, 1, layout.columnCount())
        msgBox.exec()

    def take_shot(self):
        date = datetime.now()
        filename = date.strftime('%Y-%m-%d_%H-%M-%S')
        self.mw.canvas.print_figure(filename)
        self.show()

    def set_settings(self,fname):
        self.fname = fname

    def config_settings(self):
        #configure the network
        print(self.fname)
        with open(self.fname, "r") as file:
            cfg_resource = file.read().split('\n')
            proj_name = (cfg_resource[0].split(':'))[-1]
            root_dir = (cfg_resource[1].split(':'))[-1]
            file_name = (cfg_resource[2].split(':'))[-1]
            learning_type = (cfg_resource[3].split(':'))[-1]
            algorithm_type = (cfg_resource[4].split(':'))[-1]
            input_node_no = int((cfg_resource[5].split(':'))[-1])
            hidden_node_no = int((cfg_resource[6].split(':'))[-1])
            output_node_no = int((cfg_resource[7].split(':'))[-1])
            iteration_no = int((cfg_resource[8].split(':'))[-1])
            learning_rate = float((cfg_resource[9].split(':'))[-1])
            momentum_factor = float((cfg_resource[10].split(':'))[-1])

        #change path to project's root directory
        os.chdir(root_dir)

        #train the network
        if learning_type == "0":
            if algorithm_type == "0":
                #perceptron algorithm
                arr = train_using_perceptron(file_name,iteration_no,learning_rate)
            if algorithm_type == "1":
                #lms algorithm
                arr = train_using_lms(file_name,iteration_no,learning_rate)
            if algorithm_type  == "2":
                #backpropagation algorithm
                arr = train_using_backpropagation(file_name,iteration_no,learning_rate,
                    momentum_factor,input_node_no,hidden_node_no,output_node_no)

        self.plot_sse(arr)

    def plot_sse(self,arr):
        self.mw.canvas.ax.clear()
        self.mw.canvas.ax.set_title("SSE vs Iteration")
        self.mw.canvas.ax.set_xlabel("Iteration -->")
        self.mw.canvas.ax.set_ylabel("SSE -->")
        self.mw.canvas.ax.plot(arr)
        self.mw.canvas.draw()
        return

    def setupUi(self,MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setCentralWidget(self.mw)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew = QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSnapshot = QAction(MainWindow)
        self.actionSnapshot.setObjectName("actionSnapshot")
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionChange_Dataset = QAction(MainWindow)
        self.actionChange_Dataset.setObjectName("actionChange_Dataset")
        self.actionChange_Parameters = QAction(MainWindow)
        self.actionChange_Parameters.setObjectName("actionChange_Parameters")
        self.actionDocumentation = QAction(MainWindow)
        self.actionDocumentation.setObjectName("actionDocumentation")
        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSnapshot)
        self.menuFile.addAction(self.actionExit)
        self.menuEdit.addAction(self.actionChange_Dataset)
        self.menuEdit.addAction(self.actionChange_Parameters)
        self.menuHelp.addAction(self.actionDocumentation)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.plotButton = QPushButton('Plot SSE')
        self.plotButton.setGeometry(QRect(520, 460, 75, 23))
        self.snapButton = QPushButton('Snapshot')
        self.snapButton.setGeometry(QRect(520, 460, 75, 23))
        # hbox = QHBoxLayout()
        # hbox.addStretch(1)
        # hbox.addWidget(plotButton)
        # hbox.addWidget(snapButton)
        self.statusbar.addPermanentWidget(self.plotButton, 0)
        self.statusbar.addPermanentWidget(self.snapButton, 0)
        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionNew.setText(_translate("MainWindow", "New Project"))
        self.actionOpen.setText(_translate("MainWindow", "Open Project"))
        self.actionSnapshot.setText(_translate("MainWindow", "Snapshot"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionChange_Dataset.setText(_translate("MainWindow", "Change Dataset"))
        self.actionChange_Parameters.setText(_translate("MainWindow", "Change Parameters"))
        self.actionDocumentation.setText(_translate("MainWindow", "Documentation"))
        self.actionAbout.setText(_translate("MainWindow", "About"))


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class matplotlibWidget(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)