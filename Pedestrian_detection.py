import platform

import cv2
import joblib
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QWidget, QHBoxLayout, QLabel, QDesktopWidget
from skimage.feature import hog as hog_image
from sklearn.metrics import accuracy_score, classification_report
from aboutme import About_Me
import svm
from svm import load_data, get_model


class Result(QWidget):
    def __init__(self, result):
        super().__init__()
        self.init_window(result)

    def init_window(self, result):
        hbox = QHBoxLayout(self)
        if result == 1:
            pixmap = QPixmap("true1.png")
        else:
            pixmap = QPixmap("false.png")
        lbl = QLabel(self)
        lbl.setPixmap(pixmap)
        hbox.addWidget(lbl)
        self.setLayout(hbox)

        self.setWindowTitle('Result (Is an pedestrian in this image ?) ')
        self.setMinimumSize(360, 360)
        self.setMaximumSize(360, 360)
        self.center()
        self.show()

    def center(self):
        fg = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        fg.moveCenter(cp)
        self.move(fg.topLeft())

    def compute_gradiant(img):
        img = cv2.imread(img, 0)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = hog_image(img, visualize=True, feature_vector=False)
        plt.imshow(img[1], cmap=plt.cm.gray)
        plt.show()


class Ui_MainWindow(object):
    classifier = None
    loaded = False

    def get_model(self, check, file, data):
        try:
            return svm.get_model(check, file, data)
        except FileNotFoundError:
            QMessageBox.critical(self.centralwidget, "Error", "Can't load model \nMybe it removed or renamed")

    def browse_button_clicked(self, input, isFile, filter):

        if isFile:
            text = QFileDialog.getOpenFileName(self.centralwidget, 'Open File', filter=filter)
            input.setText(str(text[0]))
            if 'Image' in filter:
                print(True)

                # img = mpimg.imread(input.text())
                # imgplot = plt.imshow(img)
                # plt.show()
            # input.setText("kkkkk")
        else:
            text = QFileDialog.getExistingDirectory(None, "Open a folder")
            input.setText(str(text))

        if self.load_model.isChecked() and not self.loaded:
            # print(self.model_file.text())
            try:
                self.classifier = self.get_model(True, self.model_file.text(), '')
                QMessageBox.information(None, 'Information ', 'Model loaded !')
                self.loaded = True
            except FileNotFoundError:
                pass

    def default_model_selected(self):
        self.classifier = None
        self.loaded = False
        if self.use_default_model.isChecked():
            self.browse_model.setEnabled(False)
            self.selct_folder_label.setEnabled(False)
            self.browse_folder_create.setEnabled(False)
            self.folder_image_train.setEnabled(False)
            self.selct_folder_label_2.setEnabled(False)
            self.browse_folder_create_2.setEnabled(False)
            self.folder_image_train_2.setEnabled(False)
            self.model_file.setEnabled(False)
            self.create_model_2.setEnabled(False)
            print(self.model_file.text())
            self.classifier = self.get_model(True, "pedestrian_detection_model.pkl", '')

    def create_model_selected(self):
        self.loaded = False
        self.classifier = None
        self.browse_model.setEnabled(False)
        self.selct_folder_label.setEnabled(True)
        self.browse_folder_create.setEnabled(True)
        self.folder_image_train.setEnabled(True)
        self.selct_folder_label_2.setEnabled(True)
        self.browse_folder_create_2.setEnabled(True)
        self.folder_image_train_2.setEnabled(True)
        self.create_model_2.setEnabled(True)
        self.model_file.setEnabled(False)

    def create_model_button(self):
        x = self.browse_folder_create_2.text().split('/')
        if (platform.system() == 'Windows'):
            sep = '\\'
        else:
            sep = '/'
        print(self.folder_image_train.text() + sep + '*.*')
        svm.labels.clear
        data = load_data(self.folder_image_train.text() + sep + '*.*', self.folder_image_train_2.text() + sep + '*.*',
                         'train')

        print(len(data))

        if len(data) < 1:
            QMessageBox.information(None, 'No Data', "there is no data ")
        else:
            self.classifier = get_model(False, '', data)
        result = QMessageBox.information(None, 'Save model', 'Do you want to save this model ',
                                         QMessageBox.Yes | QMessageBox.No)

        if result == QMessageBox.Yes:
            path = QFileDialog.getSaveFileName(None, filter="Model files (*.pkl)")
            if path[0].endswith('.pkl'):
                joblib.dump(self.classifier, path[0])
            else:
                joblib.dump(self.classifier, path[0] + '.pkl')

            QMessageBox.information(None, 'Saved File', " Saved ! ")

    def test_button_clicked(self):
        if self.classifier == None:
            QMessageBox.warning(None, "Wanrning", "Select a model please !!")
        else:
            try:
                if (platform.system() == 'Windows'):
                    sep = '\\'
                else:
                    sep = '/'
                test_data = load_data(self.folder_image_test.text() + sep + '*.*',
                                      self.folder_image_test_2.text() + sep + '*.*', 'test')
                predict = self.classifier.predict(test_data)
                message = "Accuracy: " + str(accuracy_score(svm.test_labels, predict))
                QMessageBox.information(None, "Result", message + "\nReport.txt file created has the full report!")
                text = classification_report(svm.test_labels, predict)
                f = open("Report.txt", "w")
                f.write(message + "\n" + text)
            except:
                QMessageBox.critical(None, 'Error', "Verify the data")

    def load_model_selected(self):
        self.classifier = None
        self.browse_model.setEnabled(True)
        self.selct_folder_label.setEnabled(False)
        self.browse_folder_create.setEnabled(False)
        self.folder_image_train.setEnabled(False)
        self.selct_folder_label_2.setEnabled(False)
        self.browse_folder_create_2.setEnabled(False)
        self.folder_image_train_2.setEnabled(False)
        self.model_file.setEnabled(True)
        self.create_model_2.setEnabled(False)
        # print(self.model_file.text())

    def predict(self):
        if self.classifier == None:
            QMessageBox.warning(None, "Wanrning", "Select a model please !!")
        else:
            img = svm.hog_image(self.image_file.text())
            predict = self.classifier.predict(img)
            self.result = Result(predict[0])
            self.result.show()
            Result.compute_gradiant(self.image_file.text())
            # self.result2 = test_center.Demo()
            # self.result2.show
    def open_about_me(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.Dialog = QtWidgets.QDialog()
        self.ui = About_Me()
        self.ui.setupUi(self.Dialog)
        self.Dialog.show()

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(800, 600)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        MainWindow.setMaximumSize(QtCore.QSize(800, 600))
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.browse_an_image = QtWidgets.QPushButton(self.centralwidget)
        self.browse_an_image.setGeometry(QtCore.QRect(550, 70, 80, 25))
        self.browse_an_image.setObjectName("browse_an_image")

        self.browse_an_image.clicked.connect(
            lambda: self.browse_button_clicked(self.image_file, True, "Image files (*.jpg *.png *.jpeg)"))

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 70, 161, 31))
        self.label.setObjectName("label")

        self.browse_model = QtWidgets.QPushButton(self.centralwidget)
        self.browse_model.setEnabled(False)
        self.browse_model.setGeometry(QtCore.QRect(540, 390, 80, 25))
        self.browse_model.setObjectName("browse_model")

        self.browse_model.clicked.connect(
            lambda: self.browse_button_clicked(self.model_file, True, "Model files (*.pkl)"))

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(50, 120, 161, 31))
        self.label_2.setObjectName("label_2")

        self.use_default_model = QtWidgets.QRadioButton(self.centralwidget)
        self.use_default_model.setGeometry(QtCore.QRect(80, 230, 151, 23))
        self.use_default_model.setObjectName("use_default_model")

        self.use_default_model.clicked.connect(self.default_model_selected)

        self.create_model = QtWidgets.QRadioButton(self.centralwidget)
        self.create_model.setGeometry(QtCore.QRect(80, 260, 141, 23))
        self.create_model.setObjectName("create_model")

        self.create_model.clicked.connect(self.create_model_selected)

        self.load_model = QtWidgets.QRadioButton(self.centralwidget)
        self.load_model.setGeometry(QtCore.QRect(80, 390, 141, 31))
        self.load_model.setObjectName("load_model")

        self.load_model.clicked.connect(self.load_model_selected)

        self.browse_folder_test = QtWidgets.QPushButton(self.centralwidget)
        self.browse_folder_test.setGeometry(QtCore.QRect(550, 120, 80, 25))
        self.browse_folder_test.setObjectName("brose_folder_test")
        self.browse_folder_test.clicked.connect(
            lambda: self.browse_button_clicked(self.folder_image_test, False, 'All Directory'))

        self.browse_folder_create = QtWidgets.QPushButton(self.centralwidget)
        self.browse_folder_create.setEnabled(False)
        self.browse_folder_create.setGeometry(QtCore.QRect(600, 300, 80, 25))
        self.browse_folder_create.setObjectName("browse_folder_create")

        self.browse_folder_create.clicked.connect(
            lambda: self.browse_button_clicked(self.folder_image_train, False, 'All Directory'))

        self.folder_image_test = QtWidgets.QLineEdit(self.centralwidget)
        self.folder_image_test.setGeometry(QtCore.QRect(232, 120, 301, 25))
        self.folder_image_test.setObjectName("folder_image_test")

        self.image_file = QtWidgets.QLineEdit(self.centralwidget)
        self.image_file.setGeometry(QtCore.QRect(230, 70, 301, 25))
        self.image_file.setObjectName("image_file")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(50, 190, 161, 31))
        self.label_3.setObjectName("label_3")

        self.folder_image_train = QtWidgets.QLineEdit(self.centralwidget)
        self.folder_image_train.setEnabled(False)
        self.folder_image_train.setGeometry(QtCore.QRect(280, 300, 301, 25))
        self.folder_image_train.setObjectName("folder_image_train")

        self.model_file = QtWidgets.QLineEdit(self.centralwidget)
        self.model_file.setEnabled(False)
        self.model_file.setGeometry(QtCore.QRect(220, 390, 301, 25))
        self.model_file.setObjectName("model_file")

        self.selct_folder_label = QtWidgets.QLabel(self.centralwidget)
        self.selct_folder_label.setEnabled(False)
        self.selct_folder_label.setGeometry(QtCore.QRect(90, 290, 161, 41))
        self.selct_folder_label.setObjectName("selct_folder_label")

        self.browse_folder_create_2 = QtWidgets.QPushButton(self.centralwidget)
        self.browse_folder_create_2.setEnabled(False)
        self.browse_folder_create_2.setGeometry(QtCore.QRect(600, 340, 80, 25))
        self.browse_folder_create_2.setObjectName("browse_folder_create_2")

        self.browse_folder_create_2.clicked.connect(
            lambda: self.browse_button_clicked(self.folder_image_train_2, False, 'All Directory'))

        self.selct_folder_label_2 = QtWidgets.QLabel(self.centralwidget)
        self.selct_folder_label_2.setEnabled(False)
        self.selct_folder_label_2.setGeometry(QtCore.QRect(90, 330, 181, 41))

        self.selct_folder_label_2.setObjectName("selct_folder_label_2")

        self.folder_image_train_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.folder_image_train_2.setEnabled(False)
        self.folder_image_train_2.setGeometry(QtCore.QRect(280, 340, 301, 25))
        self.folder_image_train_2.setObjectName("folder_image_train_2")

        self.create_model_2 = QtWidgets.QPushButton(self.centralwidget)
        self.create_model_2.setGeometry(QtCore.QRect(710, 340, 80, 25))
        self.create_model_2.setObjectName("create_model_2")
        self.create_model_2.setEnabled(False)

        self.create_model_2.clicked.connect(self.create_model_button)

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(50, 150, 171, 41))
        self.label_4.setObjectName("label_4")

        self.folder_image_test_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.folder_image_test_2.setGeometry(QtCore.QRect(230, 160, 301, 25))
        self.folder_image_test_2.setObjectName("folder_image_test_2")

        self.browse_folder_test_2 = QtWidgets.QPushButton(self.centralwidget)
        self.browse_folder_test_2.setGeometry(QtCore.QRect(550, 160, 80, 25))
        self.browse_folder_test_2.setObjectName("brose_folder_test_2")

        self.browse_folder_test_2.clicked.connect(
            lambda: self.browse_button_clicked(self.folder_image_test_2, False, ''))

        self.start_pred = QtWidgets.QPushButton(self.centralwidget)
        self.start_pred.setGeometry(QtCore.QRect(660, 70, 80, 25))
        self.start_pred.setObjectName("start_pred")

        self.start_pred.clicked.connect(self.predict)

        self.exit_button = QtWidgets.QPushButton(self.centralwidget)
        self.exit_button.setGeometry(QtCore.QRect(700, 510, 80, 25))
        self.exit_button.setObjectName("exit_button")
        self.exit_button.clicked.connect(exit)

        self.about_me = QtWidgets.QPushButton(self.centralwidget)
        self.about_me.setGeometry(QtCore.QRect(590, 510, 80, 25))
        self.about_me.setObjectName("aboutMe_button")
        self.about_me.clicked.connect(self.open_about_me)

        self.test_model = QtWidgets.QPushButton(self.centralwidget)
        self.test_model.setGeometry(QtCore.QRect(660, 160, 121, 25))
        self.test_model.setObjectName("test_model")

        self.test_model.clicked.connect(self.test_button_clicked)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)



        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):

        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pedestrian Detection"))
        self.browse_an_image.setText(_translate("MainWindow", "Browse"))
        self.label.setText(_translate("MainWindow", "Select an image to predict :"))
        self.browse_model.setText(_translate("MainWindow", "Browse"))
        self.label_2.setToolTip(_translate("MainWindow", "Select images to test the model \n"
                                                         "this will result a report"))
        self.label_2.setText(_translate("MainWindow", "Select folder of pedestrains :"))
        self.use_default_model.setText(_translate("MainWindow", "Use the default model"))
        self.create_model.setText(_translate("MainWindow", "Create a new model"))
        self.load_model.setText(_translate("MainWindow", "Load a model "))
        self.browse_folder_test.setText(_translate("MainWindow", "Browse"))
        self.browse_folder_create.setText(_translate("MainWindow", "Browse"))
        self.label_3.setText(_translate("MainWindow", "Select a model : "))
        self.selct_folder_label.setText(_translate("MainWindow", "Select folder of pedestrains :"))
        self.start_pred.setText(_translate("MainWindow", "Start"))
        self.exit_button.setText(_translate("MainWindow", "Quit"))
        self.about_me.setText(_translate("MainWindow", "About Me"))
        self.test_model.setText(_translate("MainWindow", "Test the model"))
        self.browse_folder_create_2.setText(_translate("MainWindow", "Browse"))
        self.selct_folder_label_2.setText(_translate("MainWindow", "Select folder of no pedestrain :"))
        self.create_model_2.setText(_translate("MainWindow", "Create"))
        self.label_4.setToolTip(_translate("MainWindow", "Select images to test the model \n"
                                                         "this will result a report"))
        self.label_4.setText(_translate("MainWindow", "Select folder of no pedestrains :"))
        self.browse_folder_test_2.setText(_translate("MainWindow", "Browse"))

    def center(self):
        fg = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        fg.moveCenter(cp)
        self.move(fg.topLeft())

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    fg = MainWindow.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    fg.moveCenter(cp)
    MainWindow.move(fg.topLeft())
    sys.exit(app.exec_())
