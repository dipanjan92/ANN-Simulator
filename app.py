from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import sys,os,shutil,time,webbrowser
import urllib.request, random
from welcome import *
from new_wizard import *
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from multiprocessing import Pool
from perceptron import *
from lms import *
from backpropagation import *
from som import *
from kmeans import *
from knn import *
from datetime import datetime

cfg_resource = []

class MyForm(QDialog):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_Welcome()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.next_one)
        self.ui.pushButton_2.clicked.connect(self.open_one)
        self.ui.pushButton_4.clicked.connect(self.open_about)
        self.ui.pushButton_3.clicked.connect(self.open_docs)

    def open_docs(self):
        str = os.getcwd()+"\docs.html"
        print(str)
        webbrowser.open(str)
        self.show()


    def open_about(self):
        msgBox = QMessageBox()
        horizontalSpacer = QSpacerItem(500, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        msgBox.setWindowTitle( "Artificial Neural Network Simulator" )
        msgBox.setText( "Developed by:\n\t\tAmalesh Jana\n\t\tAyan Adhikari\n\t\tBiswadip Mondal\n\t\tDipanjan Saha\nGuided by:\n\t\tDr.(Prof.) Sudarshan Nandy" )
        layout = msgBox.layout()
        layout.addItem(horizontalSpacer, layout.rowCount(), 0, 1, layout.columnCount())
        msgBox.exec()
        self.show()

    
    def next_one(self):
        self.myapp = NextOne()
        self.myapp.show()

    def open_one(self):
        try:
            fname = QFileDialog.getOpenFileUrl(self,'Open File','c://','CFG files (*.cfg)')
            fname = fname[0].toString(QUrl.RemoveScheme)
            fname = fname.split('///')
            self.myapp = ShowOutput()
            self.myapp.set_settings(fname[-1].split('/').pop())
            fname = fname[-1].split('/')
            fname.pop()
            path =""
            for x in fname:
                path = path+x+'/'
            os.chdir(path)
            self.myapp.show()
        except OSError:
            self.show()

class NextOne(QWizard):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_Wizard()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.open_dir)
        self.ui.pushButton_2.clicked.connect(self.open_file)
        self.ui.pushButton_3.clicked.connect(self.download)
        self.ui.lineEdit.textChanged.connect(self.check_state)
        self.ui.lineEdit.textChanged.emit(self.ui.lineEdit.text())
        regexp = QRegExp('^([a-zA-Z0-9\s\._-]+)$')
        validator = QRegExpValidator(regexp)
        self.ui.lineEdit.setValidator(validator)
        self.button(QWizard.NextButton).clicked.connect(self.check_project_name)
        self.button(QWizard.FinishButton).clicked.connect(self.output)
        self.ui.comboBox.currentIndexChanged.connect(self.handleComboBox)
        self.ui.radioButton.toggled.connect(self.toggle1)
        self.ui.radioButton_2.toggled.connect(self.toggle2)

    def handleComboBox(self):
        self.ui.comboBox_2.clear()
        if self.ui.comboBox.currentIndex()==0:
            self.ui.comboBox_2.addItem("Perceptron")
            self.ui.comboBox_2.addItem("LMS")
            self.ui.comboBox_2.addItem("Backpropaation")
            self.ui.comboBox_2.addItem("k-Nearest Neighbor")

        if self.ui.comboBox.currentIndex()==1:
            self.ui.comboBox_2.addItem("Self-organizing Maps")
            self.ui.comboBox_2.addItem("k-Means Clustering")

    def output(self):
        cfg_resource.insert(0,self.ui.lineEdit.text())
        cfg_resource.insert(1,self.ui.lineEdit_2.text())
        a = self.ui.comboBox.currentIndex()
        cfg_resource.insert(3,str(a))
        a = self.ui.comboBox_2.currentIndex()
        cfg_resource.insert(4,str(a))
        a = self.ui.spinBox.value()
        cfg_resource.insert(5,str(a))
        a = self.ui.spinBox_2.value()
        cfg_resource.insert(6,str(a))
        a = self.ui.spinBox_3.value()
        cfg_resource.insert(7,str(a))
        a = self.ui.spinBox_4.value()
        cfg_resource.insert(8,str(a))
        a = self.ui.doubleSpinBox.value()
        cfg_resource.insert(9,str(a))
        a = self.ui.doubleSpinBox_2.value()
        cfg_resource.insert(10,str(a))
        with open(cfg_resource[0]+'.cfg','w') as cfg_file:
            cfg_file.write("Project Name:"+cfg_resource[0]+"\n")
            cfg_file.write("Root Directory:"+cfg_resource[1]+"\n")
            cfg_file.write("File Name:"+cfg_resource[2]+"\n")
            cfg_file.write("Learning Type:"+cfg_resource[3]+"\n")
            cfg_file.write("Algorithm Type:"+cfg_resource[4]+"\n")
            cfg_file.write("No of Input Nodes:"+cfg_resource[5]+"\n")
            cfg_file.write("No of Hidden Nodes:"+cfg_resource[6]+"\n")
            cfg_file.write("No of Output Nodes:"+cfg_resource[7]+"\n")
            cfg_file.write("No of Iteration:"+cfg_resource[8]+"\n")
            cfg_file.write("Learning Rate:"+cfg_resource[9]+"\n")
            cfg_file.write("Momentum Factor:"+cfg_resource[10])
        self.myapp = ShowOutput()
        self.myapp.set_settings(cfg_resource[0]+".cfg")
        self.myapp.show()

    def check_project_name(self):
        QWizard.back(self)
        os.chdir(self.ui.lineEdit_2.text())
        proj_name = self.ui.lineEdit.text()+".cfg"
        if os.path.isfile(proj_name):
            str1 = "Project already exists"
            str2 = "Change the project name"
            show_warning(str1,str2)
        elif self.ui.lineEdit_3.isEnabled() and self.ui.lineEdit_3.text() == "":
            str1 = "Choose input file"
            str2 = "Choose input file"
            show_warning(str1,str2)
        elif self.ui.lineEdit_4.isEnabled() and self.ui.lineEdit_4.text() == "":
            str1 = "Download input file"
            str2 = "Enter the download link"
            show_warning(str1,str2)
        else:
            open(proj_name,'w')
            if self.ui.lineEdit_3.isEnabled()==True:
                src = self.ui.lineEdit_3.text()
                dst = self.ui.lineEdit_2.text()
                try:
                    shutil.copy(src,dst)
                except shutil.SameFileError:
                    pass
            QWizard.next(self)


    def check_state(self, *args, **kwargs):
        sender = self.sender()
        validator = sender.validator()
        state = validator.validate(sender.text(), 0)[0]
        if state == QValidator.Acceptable:
            color = '#c4df9b' # green
        elif state == QValidator.Intermediate:
            color = '#fff79a' # yellow
        else:
            color = '#f6989d' # red
        sender.setStyleSheet('QLineEdit { background-color: %s }' % color)

    def open_dir(self):
        fname = QFileDialog.getExistingDirectoryUrl(self,'Open Directory',QUrl(),QFileDialog.ShowDirsOnly)
        fname = fname.toString(QUrl.RemoveScheme)
        fname = fname.split('///')
        self.ui.lineEdit_2.setText(fname[-1])

    def open_file(self):
        fname = QFileDialog.getOpenFileUrl(self,'Open File','c://','CSV files (*.csv);;Text files (*.txt);;XML files (*.xml)')
        fname = fname[0].toString(QUrl.RemoveScheme)
        fname = fname.split('///')
        self.ui.lineEdit_3.setText(fname[-1])
        fname = fname[-1].split('/')[-1]
        cfg_resource.insert(2,fname)

    def download(self):
        url = self.ui.lineEdit_4.text()
        file_name = url.split('/')[-1] # Set file_name as the last part of the url, i.e the original file name
        file_name = file_name.split('.')[0]+'.csv'
        with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        cfg_resource.insert(2,file_name)

    def toggle1(self,enabled):
        if enabled:
            self.ui.lineEdit_4.setEnabled(False)
            self.ui.pushButton_3.setEnabled(False)
            self.ui.lineEdit_3.setEnabled(True)
            self.ui.pushButton_2.setEnabled(True)

    def toggle2(self,enabled):
        if enabled:
            self.ui.lineEdit_4.setEnabled(True)
            self.ui.pushButton_3.setEnabled(True)
            self.ui.lineEdit_3.setEnabled(False)
            self.ui.pushButton_2.setEnabled(False)


def show_warning(str1,str2):
    QMessageBox.warning(None, str1, str2, QMessageBox.Ok)
    return

class MySplashScreen(QSplashScreen):
    def __init__(self, animation, flags):
        # run event dispatching in another thread
        QSplashScreen.__init__(self, QPixmap(), flags)
        self.movie = QMovie(animation)
        self.movie.frameChanged.connect(self.onNextFrame)
        self.movie.start()

    def onNextFrame(self):
        pixmap = self.movie.currentPixmap()
        self.setPixmap(pixmap)
        self.setMask(pixmap.mask())

# Put your initialization code here
def longInitialization(arg):
  time.sleep(6)
  return 0





class ShowOutput(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('Demo: PyQt with matplotlib')
        self.mw = matplotlibWidget()
        self.setupUi(self)
        self.statusbar.showMessage("Ready",5000)
        self.plotButton.clicked.connect(self.get_sse)
        self.plotButton2.clicked.connect(self.get_accuracy)
        self.plotButton3.clicked.connect(self.get_fmeaure)
        self.snapButton.clicked.connect(self.take_shot)
        self.actionAbout.triggered.connect(self.about)
        self.actionSnapshot.triggered.connect(self.take_shot)
        self.actionExit.triggered.connect(self.exit)
        self.actionNew.triggered.connect(self.newProject)
        self.actionOpen.triggered.connect(self.openProject)
        self.actionChange_Dataset.triggered.connect(self.changeDataset)
        self.actionChange_Parameters.triggered.connect(self.changeDataset)
        self.actionDocumentation.triggered.connect(self.open_docs)

    def open_docs(self):
        str = "C:\\Users\\Dipanjan\\Desktop\\Project\\Prototype"+"\\docs.html"
        webbrowser.open(str)
        self.show()

    def changeDataset(self):
        fname = QFileDialog.getOpenFileUrl(self,'Open File','c://','CFG files (*.cfg)')
        fname = fname[0].toString(QUrl.RemoveScheme)
        fname = fname.split('///')
        proj_name = fname[-1].split('/')[-1]
        osCommandString = "notepad.exe "+proj_name
        os.system(osCommandString)

    def openProject(self):
        fname = QFileDialog.getOpenFileUrl(self,'Open File','c://','CFG files (*.cfg)')
        fname = fname[0].toString(QUrl.RemoveScheme)
        fname = fname.split('///')
        self.myapp = ShowOutput()
        self.myapp.set_settings(fname[-1].split('/')[-1])
        fname = fname[-1].split('/')
        fname.pop()
        path =""
        for x in fname:
            path = path+x+'/'
        os.chdir(path)
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
        msgBox.setText( "Developed by:\n\t\tAmalesh Jana\n\t\tAyan Adhikari\n\t\tBiswadip Mondal\n\t\tDipanjan Saha\nGuided by:\n\t\tDr.(Prof.) Sudarshan Nandy" )
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

    def get_sse(self):
        #configure the network
        self.statusbar.showMessage("Processing data....")
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
                arr = perceptron_sse(file_name,iteration_no,learning_rate)
            if algorithm_type == "1":
                #lms algorithm
                arr = lms_sse(file_name,iteration_no,learning_rate)
            if algorithm_type  == "2":
                #backpropagation algorithm
                arr = backpropagation_sse(file_name,iteration_no,learning_rate,
                    momentum_factor,input_node_no,hidden_node_no,output_node_no)
            if algorithm_type  == "3":
                #backpropagation algorithm
                arr = knn_sse(file_name,iteration_no,learning_rate)
        if learning_type == "1":
            if algorithm_type == "0":
                arr = som_sse(file_name,iteration_no,learning_rate)
            if algorithm_type == "1":
                arr = kmeans_sse(file_name,iteration_no,learning_rate)

        str1 = "SSE vs Iteration"
        str2 = "Iteration -->"
        str3 = "SSE -->"

        self.plot_graph(arr,str1,str2,str3)
        self.statusbar.showMessage("Done",5000)

    def get_accuracy(self):
        #configure the network
        self.statusbar.showMessage("Processing data....")
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
                arr = perceptron_accuracy(file_name,iteration_no,learning_rate)
            if algorithm_type == "1":
                #lms algorithm
                arr = lms_accuracy(file_name,iteration_no,learning_rate)
            if algorithm_type  == "2":
                #backpropagation algorithm
                arr = backpropagation_accuracy(file_name,iteration_no,learning_rate,
                    momentum_factor,input_node_no,hidden_node_no,output_node_no)
            if algorithm_type  == "3":
                #backpropagation algorithm
                arr = knn_accuracy(file_name,iteration_no,learning_rate)
        if learning_type == "1":
            if algorithm_type == "0":
                arr = som_accuracy(file_name,iteration_no,learning_rate)
            if algorithm_type == "1":
                arr = kmeans_accuracy(file_name,iteration_no,learning_rate)

        str1 = "Accuracy vs Iteration"
        str2 = "Iteration -->"
        str3 = "Accuracy -->"

        self.plot_graph(arr,str1,str2,str3)
        self.statusbar.showMessage("Done",5000)

    def get_fmeaure(self):
        #configure the network
        self.statusbar.showMessage("Processing data....")
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
                arr = perceptron_fmeasure(file_name,iteration_no,learning_rate)
            if algorithm_type == "1":
                #lms algorithm
                arr = lms_fmeasure(file_name,iteration_no,learning_rate)
            if algorithm_type  == "2":
                #backpropagation algorithm
                arr = backpropagation_fmeasure(file_name,iteration_no,learning_rate,
                    momentum_factor,input_node_no,hidden_node_no,output_node_no)
            if algorithm_type  == "3":
                #backpropagation algorithm
                arr = knn_fmeasure(file_name,iteration_no,learning_rate)
        if learning_type == "1":
            if algorithm_type == "0":
                arr = som_fmeasure(file_name,iteration_no,learning_rate)
            if learning_type == "1":
                arr = kmeans_fmeasure(file_name,iteration_no,learning_rate)

        str1 = "F-Measure vs Iteration"
        str2 = "Iteration -->"
        str3 = "F-Measure -->"

        self.plot_graph(arr,str1,str2,str3)
        self.statusbar.showMessage("Done",5000)

    def plot_graph(self,arr,str1,str2,str3):
        self.mw.canvas.ax.clear()
        self.mw.canvas.ax.set_title(str1)
        self.mw.canvas.ax.set_xlabel(str2)
        self.mw.canvas.ax.set_ylabel(str3)
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
        self.plotButton = QPushButton('SSE')
        self.plotButton.setGeometry(QRect(520, 460, 75, 23))
        self.plotButton2 = QPushButton('Accuracy')
        self.plotButton.setGeometry(QRect(520, 460, 75, 23))
        self.plotButton3 = QPushButton('F-Measure')
        self.plotButton.setGeometry(QRect(520, 460, 75, 23))
        self.snapButton = QPushButton('Snapshot')
        self.snapButton.setGeometry(QRect(520, 460, 75, 23))
        self.statusbar.addPermanentWidget(self.plotButton, 0)
        self.statusbar.addPermanentWidget(self.plotButton2, 0)
        self.statusbar.addPermanentWidget(self.plotButton3, 0)
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
        self.gl = QGridLayout()
        alignment = Qt.Alignment()
        self.gl.addWidget(self.canvas,0,0,-1,-1,alignment)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.hide()

        # Just some button 

        self.button1 = QPushButton('Zoom')
        self.button1.clicked.connect(self.zoom)

        self.button2 = QPushButton('Pan')
        self.button2.clicked.connect(self.pan)

        self.button3 = QPushButton('Home')
        self.button3.clicked.connect(self.home)

        self.gl.addWidget(self.toolbar,1,0,alignment)
        self.gl.addWidget(self.button1,1,1,alignment)
        self.gl.addWidget(self.button2,1,2,alignment)
        self.gl.addWidget(self.button3,1,3,alignment)

        self.setLayout(self.gl)

    def home(self):
        self.toolbar.home()
    def zoom(self):
        self.toolbar.zoom()
    def pan(self):
        self.toolbar.pan()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = MySplashScreen('welcome.gif', Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    initLoop = QEventLoop()
    pool = Pool(processes=1)
    pool.apply_async(longInitialization, [2], callback=lambda exitCode: initLoop.exit(exitCode))
    initLoop.exec_()

    myapp = MyForm()
    myapp.show()
    splash.finish(myapp)
    sys.exit(app.exec_())