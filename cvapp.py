import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QGroupBox, QSlider, QComboBox, QHBoxLayout, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class WebcamThresholdApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Webcam Thresholding'
        self.left = 100
        self.top = 100
        self.width = 1600
        self.height = 800 
        self.mode = "Grayscale"
        self.camera=0
        self.is_first_frame = True
        self.is_stopped = False
        self.locked_frame = None
        self.thresholded_image = None
        #定义初始值
        self.threshold_values = {"Grayscale": [100,255], "RGB": [100, 255, 100, 255, 100, 255], "HSV": [100, 255, 100,255 ,100 ,255], "LAB": [100,255,100,255,100,255]}

        self.initUI()
        self.startWebcam()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.original_image_label = QLabel(self)
        self.original_image_label.setAlignment(Qt.AlignCenter)

        self.thresholded_image_label = QLabel(self)
        self.thresholded_image_label.setAlignment(Qt.AlignCenter)
        #切换摄像头组件
        self.camera_combo_box = QComboBox(self)
        self.camera_combo_box.addItem("camera-0")
        self.camera_combo_box.addItem("camera-1")
        self.camera_combo_box.addItem("camera-2")
        self.camera_combo_box.addItem("camera-3")
        self.camera_combo_box.currentTextChanged.connect(self.changeCamera)
        #切换图片格式组件
        self.mode_combo_box = QComboBox(self)
        self.mode_combo_box.addItem("Grayscale")
        self.mode_combo_box.addItem("RGB")
        self.mode_combo_box.addItem("HSV")
        self.mode_combo_box.addItem("LAB")
        self.mode_combo_box.currentTextChanged.connect(self.changeMode)
        #最小灰度
        self.min_threshold_slider = QSlider(Qt.Horizontal)
        self.min_threshold_slider.setMinimum(0)
        self.min_threshold_slider.setMaximum(255)
        self.min_threshold_slider.setSliderPosition(self.threshold_values["Grayscale"][0])
        self.min_threshold_slider.valueChanged.connect(self.onThresholdChanged)
        #最大灰度
        self.max_threshold_slider = QSlider(Qt.Horizontal)
        self.max_threshold_slider.setMinimum(0)
        self.max_threshold_slider.setMaximum(255)
        self.max_threshold_slider.setSliderPosition(self.threshold_values["Grayscale"][0])
        self.max_threshold_slider.valueChanged.connect(self.onThresholdChanged)

        #最小最大RGB
        self.min_red_slider = QSlider(Qt.Horizontal)
        self.min_red_slider.setMinimum(0)
        self.min_red_slider.setMaximum(255)
        self.min_red_slider.setSliderPosition(self.threshold_values["RGB"][0])
        self.min_red_slider.valueChanged.connect(self.onThresholdChanged)

        self.max_red_slider = QSlider(Qt.Horizontal)
        self.max_red_slider.setMinimum(0)
        self.max_red_slider.setMaximum(255)
        self.max_red_slider.setSliderPosition(self.threshold_values["RGB"][1])
        self.max_red_slider.valueChanged.connect(self.onThresholdChanged)

        self.min_green_slider = QSlider(Qt.Horizontal)
        self.min_green_slider.setMinimum(0)
        self.min_green_slider.setMaximum(255)
        self.min_green_slider.setSliderPosition(self.threshold_values["RGB"][2])
        self.min_green_slider.valueChanged.connect(self.onThresholdChanged)

        self.max_green_slider = QSlider(Qt.Horizontal)
        self.max_green_slider.setMinimum(0)
        self.max_green_slider.setMaximum(255)
        self.max_green_slider.setSliderPosition(self.threshold_values["RGB"][3])
        self.max_green_slider.valueChanged.connect(self.onThresholdChanged)

        self.min_blue_slider = QSlider(Qt.Horizontal)
        self.min_blue_slider.setMinimum(0)
        self.min_blue_slider.setMaximum(255)
        self.min_blue_slider.setSliderPosition(self.threshold_values["RGB"][4])
        self.min_blue_slider.valueChanged.connect(self.onThresholdChanged)

        self.max_blue_slider = QSlider(Qt.Horizontal)
        self.max_blue_slider.setMinimum(0)
        self.max_blue_slider.setMaximum(255)
        self.max_blue_slider.setSliderPosition(self.threshold_values["RGB"][5])
        self.max_blue_slider.valueChanged.connect(self.onThresholdChanged)

        #最小最大HSV
        self.min_hue_slider = QSlider(Qt.Horizontal)
        self.min_hue_slider.setMinimum(0)
        self.min_hue_slider.setMaximum(179)
        self.min_hue_slider.setSliderPosition(self.threshold_values["HSV"][0])
        self.min_hue_slider.valueChanged.connect(self.onThresholdChanged)
        self.max_hue_slider = QSlider(Qt.Horizontal)
        self.max_hue_slider.setMinimum(0)
        self.max_hue_slider.setMaximum(179)
        self.max_hue_slider.setSliderPosition(self.threshold_values["HSV"][1])
        self.max_hue_slider.valueChanged.connect(self.onThresholdChanged)

        self.min_saturation_slider = QSlider(Qt.Horizontal)
        self.min_saturation_slider.setMinimum(0)
        self.min_saturation_slider.setMaximum(255)
        self.min_saturation_slider.setSliderPosition(self.threshold_values["HSV"][2])
        self.min_saturation_slider.valueChanged.connect(self.onThresholdChanged)
        self.max_saturation_slider = QSlider(Qt.Horizontal)
        self.max_saturation_slider.setMinimum(0)
        self.max_saturation_slider.setMaximum(255)
        self.max_saturation_slider.setSliderPosition(self.threshold_values["HSV"][3])
        self.max_saturation_slider.valueChanged.connect(self.onThresholdChanged)

        self.min_value_slider = QSlider(Qt.Horizontal)
        self.min_value_slider.setMinimum(0)
        self.min_value_slider.setMaximum(255)
        self.min_value_slider.setSliderPosition(self.threshold_values["HSV"][4])
        self.min_value_slider.valueChanged.connect(self.onThresholdChanged)
        self.max_value_slider = QSlider(Qt.Horizontal)
        self.max_value_slider.setMinimum(0)
        self.max_value_slider.setMaximum(255)
        self.max_value_slider.setSliderPosition(self.threshold_values["HSV"][5])
        self.max_value_slider.valueChanged.connect(self.onThresholdChanged)

        #最小最大LAB
        self.min_l_slider = QSlider(Qt.Horizontal)
        self.min_l_slider.setMinimum(0)
        self.min_l_slider.setMaximum(255)
        self.min_l_slider.setSliderPosition(self.threshold_values["LAB"][0])
        self.min_l_slider.valueChanged.connect(self.onThresholdChanged)
        self.max_l_slider = QSlider(Qt.Horizontal)
        self.max_l_slider.setMinimum(0)
        self.max_l_slider.setMaximum(255)
        self.max_l_slider.setSliderPosition(self.threshold_values["LAB"][1])
        self.max_l_slider.valueChanged.connect(self.onThresholdChanged)
     
        self.min_a_slider = QSlider(Qt.Horizontal)
        self.min_a_slider.setMinimum(0)
        self.min_a_slider.setMaximum(255)
        self.min_a_slider.setSliderPosition(self.threshold_values["LAB"][2])
        self.min_a_slider.valueChanged.connect(self.onThresholdChanged)
        self.max_a_slider = QSlider(Qt.Horizontal)
        self.max_a_slider.setMinimum(0)
        self.max_a_slider.setMaximum(255)
        self.max_a_slider.setSliderPosition(self.threshold_values["LAB"][3])
        self.max_a_slider.valueChanged.connect(self.onThresholdChanged)

        self.min_b_slider = QSlider(Qt.Horizontal)
        self.min_b_slider.setMinimum(0)
        self.min_b_slider.setMaximum(255)
        self.min_b_slider.setSliderPosition(self.threshold_values["LAB"][4])
        self.min_b_slider.valueChanged.connect(self.onThresholdChanged)
        self.max_b_slider = QSlider(Qt.Horizontal)
        self.max_b_slider.setMinimum(0)
        self.max_b_slider.setMaximum(255)
        self.max_b_slider.setSliderPosition(self.threshold_values["LAB"][5])
        self.max_b_slider.valueChanged.connect(self.onThresholdChanged)

        #标签组件
        self.min_gray_label = QLabel("Min_Gray:")
        self.max_gray_label = QLabel("Max_Gray:")

        self.min_red_label = QLabel("Min_R:")
        self.max_red_label = QLabel("Max_R:")
        self.min_green_label = QLabel("Min_G:")
        self.max_green_label = QLabel("Max_G:")
        self.min_blue_label = QLabel("Min_B:")
        self.max_blue_label = QLabel("Max_B:")
        self.min_hue_label = QLabel("Min_H:")
        self.max_hue_label = QLabel("Max_H:")
        self.min_saturation_label = QLabel("Min_S:")
        self.max_saturation_label = QLabel("Max_S:")
        self.min_value_label = QLabel("Min_V:")
        self.max_value_label = QLabel("Max_V:")
        self.min_l_label = QLabel("Min_L:")
        self.max_l_label = QLabel("Max_L:")
        self.min_a_label = QLabel("Min_A:")
        self.max_a_label = QLabel("Max_A:")
        self.min_b_label = QLabel("Min_B:")
        self.max_b_label = QLabel("Max_B:")

        self.min_gray_label.show()
        self.max_gray_label.show()
        self.min_red_label.hide()
        self.max_red_label.hide()
        self.min_green_label.hide()
        self.max_green_label.hide()
        self.min_blue_label.hide()
        self.max_blue_label.hide()

        self.min_hue_label.hide()
        self.max_hue_label.hide()
        self.min_saturation_label.hide()
        self.max_saturation_label.hide()
        self.min_value_label.hide()
        self.max_value_label.hide()

        self.min_l_label.hide()
        self.max_l_label.hide()
        self.min_a_label.hide()
        self.max_a_label.hide()
        self.min_b_label.hide()
        self.max_b_label.hide()
        
        self.min_red_slider.hide()
        self.max_red_slider.hide()
        self.min_green_slider.hide()
        self.max_green_slider.hide()
        self.min_blue_slider.hide()
        self.max_blue_slider.hide()
        self.min_hue_slider.hide()
        self.max_hue_slider.hide()
        self.min_saturation_slider.hide()
        self.max_saturation_slider.hide()
        self.min_value_slider.hide()
        self.max_value_slider.hide()
        self.min_l_slider.hide()
        self.max_l_slider.hide()
        self.min_a_slider.hide()
        self.max_a_slider.hide()
        self.min_b_slider.hide()
        self.max_b_slider.hide()

        self.threshold_group_box = QGroupBox()
        #布置组件
        vbox = QVBoxLayout()
        vbox.addWidget(self.min_gray_label)
        vbox.addWidget(self.min_threshold_slider)
        vbox.addWidget(self.max_gray_label)
        vbox.addWidget(self.max_threshold_slider)
        
        vbox.addWidget(self.min_red_label)
        vbox.addWidget(self.min_red_slider)
        vbox.addWidget(self.max_red_label)
        vbox.addWidget(self.max_red_slider)
        vbox.addWidget(self.min_green_label)
        vbox.addWidget(self.min_green_slider)
        vbox.addWidget(self.max_green_label)
        vbox.addWidget(self.max_green_slider)
        vbox.addWidget(self.min_blue_label)
        vbox.addWidget(self.min_blue_slider)
        vbox.addWidget(self.max_blue_label)
        vbox.addWidget(self.max_blue_slider)
        vbox.addWidget(self.min_hue_label)
        vbox.addWidget(self.min_hue_slider)
        vbox.addWidget(self.max_hue_label)
        vbox.addWidget(self.max_hue_slider)
        vbox.addWidget(self.min_saturation_label)
        vbox.addWidget(self.min_saturation_slider)
        vbox.addWidget(self.max_saturation_label)
        vbox.addWidget(self.max_saturation_slider)
        vbox.addWidget(self.min_value_label)
        vbox.addWidget(self.min_value_slider)
        vbox.addWidget(self.max_value_label)
        vbox.addWidget(self.max_value_slider)
        vbox.addWidget(self.min_l_label)
        vbox.addWidget(self.min_l_slider)
        vbox.addWidget(self.max_l_label)
        vbox.addWidget(self.max_l_slider)
        vbox.addWidget(self.min_a_label)
        vbox.addWidget(self.min_a_slider)
        vbox.addWidget(self.max_a_label)
        vbox.addWidget(self.max_a_slider)
        vbox.addWidget(self.min_b_label)
        vbox.addWidget(self.min_b_slider)
        vbox.addWidget(self.max_b_label)
        vbox.addWidget(self.max_b_slider)

        self.threshold_group_box.setLayout(vbox)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.onStopButtonClicked)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.original_image_label)
        hbox_layout.addWidget(self.thresholded_image_label)

        vbox_layout = QVBoxLayout()
        vbox_layout.addLayout(hbox_layout)
        vbox_layout.addWidget(self.camera_combo_box)
        vbox_layout.addWidget(self.mode_combo_box)
        vbox_layout.addWidget(self.threshold_group_box)
        vbox_layout.addWidget(self.stop_button)

        self.setLayout(vbox_layout)

    def startWebcam(self):
        #打开摄像头
        self.cap = cv2.VideoCapture(self.camera)
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)

    def updateFrame(self):
        if self.is_stopped:
            frame=self.locked_frame.copy()
        
        if self.is_stopped == False:
            ret, frame = self.cap.read()

            if not ret:
                return
            #锁住静态图片
            self.locked_frame = frame.copy()

        if self.mode == "Grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thresholded=cv2.inRange(gray,self.min_threshold_slider.value(),self.max_threshold_slider.value())
            height, width = thresholded.shape
            q_img = QImage(thresholded.data, width, height, QImage.Format_Grayscale8)
        elif self.mode == "RGB":
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            low=np.array([self.min_red_slider.value(),self.min_green_slider.value(),self.min_blue_slider.value()])
            up=np.array([self.max_red_slider.value(),self.max_green_slider.value(),self.max_blue_slider.value()])
            thresholded=cv2.inRange(RGB,low,up)
            thresholded_gray = thresholded
            thresholded_gray = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

            height, width, _ = thresholded_gray.shape
            q_img = QImage(thresholded_gray.data, width, height, QImage.Format_RGB888).rgbSwapped()
        elif self.mode == "HSV":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            low=np.array([self.min_hue_slider.value(),self.min_saturation_slider.value(),self.min_value_slider.value()])
            up=np.array([self.max_hue_slider.value(),self.max_saturation_slider.value(),self.max_value_slider.value()])
            thresholded=cv2.inRange(hsv,low,up)
            thresholded_gray = thresholded

            height, width = thresholded_gray.shape
            q_img = QImage(thresholded_gray.data, width, height, QImage.Format_Grayscale8)
        elif self.mode == "LAB":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            low=np.array([self.min_l_slider.value(),self.min_a_slider.value(),self.min_b_slider.value()])
            up=np.array([self.max_l_slider.value(),self.max_a_slider.value(),self.max_b_slider.value()])
           
            thresholded=cv2.inRange(lab,low,up)
            thresholded_gray = thresholded

            height, width = thresholded_gray.shape
            q_img = QImage(thresholded_gray.data, width, height, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img)
        resized_pixmap = pixmap.scaled(self.width // 2, self.height, Qt.KeepAspectRatio)

        self.thresholded_image_label.setPixmap(resized_pixmap)

        orig_pixmap = QPixmap.fromImage(self.cvMatToQImage(frame))
        orig_resized_pixmap = orig_pixmap.scaled(self.width // 2, self.height, Qt.KeepAspectRatio)
        self.original_image_label.setPixmap(orig_resized_pixmap)

    #获取滑动组件的阈值
    def onThresholdChanged(self):
        if self.mode == "Grayscale":
            self.threshold_group_box.setTitle(f"Threshold: Gray: {self.min_threshold_slider.value(),self.max_threshold_slider.value()}")
        elif self.mode == "RGB":
            self.threshold_group_box.setTitle(f"Threshold: R:{self.min_red_slider.value(),self.max_red_slider.value()}, G:{self.min_green_slider.value(),self.max_green_slider.value()}, B:{self.min_blue_slider.value(),self.max_blue_slider.value()}")
        elif self.mode == "HSV":
            self.threshold_group_box.setTitle(f"Threshold: H:{self.min_hue_slider.value(),self.max_hue_slider.value()}, S:{self.min_saturation_slider.value(),self.max_saturation_slider.value()}, V:{self.min_value_slider.value(),self.max_value_slider.value()}")
        elif self.mode == "LAB":
            self.threshold_group_box.setTitle(f"Threshold: L:{self.min_l_slider.value(),self.max_l_slider.value()}, a:{self.min_a_slider.value(),self.max_a_slider.value()}, b:{self.min_b_slider.value(),self.max_b_slider.value()}")

    #暂停按钮
    def onStopButtonClicked(self):
        self.is_stopped = not self.is_stopped
        if self.is_stopped:
            self.saveThresholdValues()
        else:
            self.loadThresholdValues()

    #保存阈值
    def saveThresholdValues(self):
        if self.mode == "Grayscale":
            self.threshold_values[self.mode] = [self.min_threshold_slider.value(),self.max_threshold_slider.value()]
        elif self.mode == "RGB":
            self.threshold_values[self.mode] = [self.min_red_slider.value(), self.max_red_slider.value(),self.min_green_slider.value(), self.max_green_slider.value(),self.min_blue_slider.value(),self.max_blue_slider.value()]
        elif self.mode == "HSV":
            self.threshold_values[self.mode] = [self.min_hue_slider.value(), self.max_hue_slider.value(),self.min_saturation_slider.value(), self.max_saturation_slider.value(),self.min_value_slider.value(),self.max_value_slider.value()]
        elif self.mode == "LAB":
            self.threshold_values[self.mode] = [self.min_l_slider.value(), self.max_l_slider.value(), self.min_a_slider.value(), self.max_a_slider.value(),self.min_b_slider.value(),self.max_b_slider.value()]
    #加载阈值
    def loadThresholdValues(self):
        if self.mode == "Grayscale":
            self.min_threshold_slider.setValue(self.threshold_values[self.mode][0])
            self.max_threshold_slider.setValue(self.threshold_values[self.mode][1])
        elif self.mode == "RGB":
            self.min_red_slider.setValue(self.threshold_values[self.mode][0])
            self.max_red_slider.setValue(self.threshold_values[self.mode][1])
            self.min_green_slider.setValue(self.threshold_values[self.mode][2])
            self.max_green_slider.setValue(self.threshold_values[self.mode][3])
            self.min_blue_slider.setValue(self.threshold_values[self.mode][4])
            self.max_blue_slider.setValue(self.threshold_values[self.mode][5])
        elif self.mode == "HSV":
            self.min_hue_slider.setValue(self.threshold_values[self.mode][0])
            self.max_hue_slider.setValue(self.threshold_values[self.mode][1])
            self.min_saturation_slider.setValue(self.threshold_values[self.mode][2])
            self.max_saturation_slider.setValue(self.threshold_values[self.mode][3])
            self.min_value_slider.setValue(self.threshold_values[self.mode][4])
            self.max_value_slider.setValue(self.threshold_values[self.mode][5])
        elif self.mode == "LAB":
            self.min_l_slider.setValue(self.threshold_values[self.mode][0])
            self.max_l_slider.setValue(self.threshold_values[self.mode][1])
            self.min_a_slider.setValue(self.threshold_values[self.mode][2])
            self.max_a_slider.setValue(self.threshold_values[self.mode][3])
            self.min_b_slider.setValue(self.threshold_values[self.mode][4])
            self.max_b_slider.setValue(self.threshold_values[self.mode][5])

    #更换摄像头
    def changeCamera(self,cam):
        self.camera=int(cam[-1])
        self.cap.release()
        try:
            self.cap = cv2.VideoCapture(self.camera)
        except:
            pass

    #更换模式
    def changeMode(self, mode):
        self.min_threshold_slider.hide()
        self.max_threshold_slider.hide()
        self.min_red_slider.hide()
        self.max_red_slider.hide()
        self.min_green_slider.hide()
        self.max_green_slider.hide()
        self.min_blue_slider.hide()
        self.max_blue_slider.hide()
        self.min_hue_slider.hide()
        self.max_hue_slider.hide()
        self.min_saturation_slider.hide()
        self.max_saturation_slider.hide()
        self.min_value_slider.hide()
        self.max_value_slider.hide()
        self.min_l_slider.hide()
        self.max_l_slider.hide()
        self.min_a_slider.hide()
        self.max_a_slider.hide()
        self.min_b_slider.hide()
        self.max_b_slider.hide()
        self.min_gray_label.hide()
        self.max_gray_label.hide()
        self.min_red_label.hide()
        self.max_red_label.hide()
        self.min_green_label.hide()
        self.max_green_label.hide()
        self.min_blue_label.hide()
        self.max_blue_label.hide()
        self.min_hue_label.hide()
        self.max_hue_label.hide()
        self.min_saturation_label.hide()
        self.max_saturation_label.hide()
        self.min_value_label.hide()
        self.max_value_label.hide()
        self.min_l_label.hide()
        self.max_l_label.hide()
        self.min_a_label.hide()
        self.max_a_label.hide()
        self.min_b_label.hide()
        self.max_b_label.hide()
        
        self.mode = mode
        if self.mode == "Grayscale":
            self.threshold_group_box.setTitle("Threshold: Gray:")
            self.min_gray_label.show()
            self.max_gray_label.show()
            self.min_threshold_slider.show()
            self.max_threshold_slider.show()          
            

        elif self.mode == "RGB":
            self.threshold_group_box.setTitle("Threshold: R, G, B:")
            self.min_red_slider.show()
            self.max_red_slider.show()
            self.min_green_slider.show()
            self.max_green_slider.show()
            self.min_blue_slider.show()
            self.max_blue_slider.show()
            self.min_red_label.show()
            self.max_red_label.show()
            self.min_green_label.show()
            self.max_green_label.show()
            self.min_blue_label.show()
            self.max_blue_label.show()
            

        elif self.mode == "HSV":
            self.threshold_group_box.setTitle("Threshold: H, S, V:")
            self.min_hue_slider.show()
            self.max_hue_slider.show()
            self.min_saturation_slider.show()
            self.max_saturation_slider.show()
            self.min_value_slider.show() 
            self.max_value_slider.show() 
            self.min_hue_label.show()
            self.max_hue_label.show()
            self.min_saturation_label.show()
            self.max_saturation_label.show()
            self.min_value_label.show()
            self.max_value_label.show()

        elif self.mode == "LAB":
            self.threshold_group_box.setTitle("Threshold: L, A, B:")      
            self.min_l_slider.show()
            self.max_l_slider.show()
            self.min_a_slider.show()
            self.max_a_slider.show()
            self.min_b_slider.show()
            self.max_b_slider.show()
            self.min_l_label.show()
            self.max_l_label.show()
            self.min_a_label.show()
            self.max_a_label.show()
            self.min_b_label.show()
            self.max_b_label.show()

        self.loadThresholdValues()

    def cvMatToQImage(self, mat):
        if mat is None:
            return QImage()
        height, width, channels = mat.shape
        bytesPerLine = channels * width
        return QImage(mat.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = WebcamThresholdApp()
    ex.show()
    sys.exit(app.exec_())