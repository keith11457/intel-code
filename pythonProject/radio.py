# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'radio.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(748, 443)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_7.setContentsMargins(10, 10, 10, 10)
        self.horizontalLayout_7.setSpacing(2)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.frame = QtWidgets.QFrame(self.frame_4)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.recommend_toolButton = QtWidgets.QToolButton(self.frame)
        self.recommend_toolButton.setObjectName("recommend_toolButton")
        self.verticalLayout_2.addWidget(self.recommend_toolButton, 0, QtCore.Qt.AlignHCenter)
        self.city_cate_toolButton = QtWidgets.QToolButton(self.frame)
        self.city_cate_toolButton.setObjectName("city_cate_toolButton")
        self.verticalLayout_2.addWidget(self.city_cate_toolButton, 0, QtCore.Qt.AlignHCenter)
        self.country_cate_toolButton = QtWidgets.QToolButton(self.frame)
        self.country_cate_toolButton.setObjectName("country_cate_toolButton")
        self.verticalLayout_2.addWidget(self.country_cate_toolButton, 0, QtCore.Qt.AlignHCenter)
        self.network_cate_toolButton = QtWidgets.QToolButton(self.frame)
        self.network_cate_toolButton.setObjectName("network_cate_toolButton")
        self.verticalLayout_2.addWidget(self.network_cate_toolButton, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(2, 2)
        self.verticalLayout_2.setStretch(3, 2)
        self.verticalLayout_2.setStretch(4, 2)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_4.addWidget(self.label_4)
        self.collection_toolButton = QtWidgets.QToolButton(self.frame)
        self.collection_toolButton.setObjectName("collection_toolButton")
        self.verticalLayout_4.addWidget(self.collection_toolButton, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_3.addLayout(self.verticalLayout_4)
        self.add_toolButton = QtWidgets.QToolButton(self.frame)
        self.add_toolButton.setObjectName("add_toolButton")
        self.verticalLayout_3.addWidget(self.add_toolButton, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_7.addWidget(self.label_5)
        self.toolButton_7 = QtWidgets.QToolButton(self.frame)
        self.toolButton_7.setObjectName("toolButton_7")
        self.verticalLayout_7.addWidget(self.toolButton_7, 0, QtCore.Qt.AlignHCenter)
        self.connect_pushButton = QtWidgets.QPushButton(self.frame)
        self.connect_pushButton.setObjectName("connect_pushButton")
        self.verticalLayout_7.addWidget(self.connect_pushButton)
        self.abou_qt_pushButton = QtWidgets.QPushButton(self.frame)
        self.abou_qt_pushButton.setObjectName("abou_qt_pushButton")
        self.verticalLayout_7.addWidget(self.abou_qt_pushButton)
        self.verticalLayout_3.addLayout(self.verticalLayout_7)
        self.horizontalLayout_7.addWidget(self.frame)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_3 = QtWidgets.QFrame(self.frame_4)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.scrollArea = QtWidgets.QScrollArea(self.frame_3)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 578, 153))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.toolButton_111 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_111.setStyleSheet("border:none")
        self.toolButton_111.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_111.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_111.setObjectName("toolButton_111")
        self.horizontalLayout_10.addWidget(self.toolButton_111)
        self.toolButton_112 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_112.setStyleSheet("border:none")
        self.toolButton_112.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_112.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_112.setObjectName("toolButton_112")
        self.horizontalLayout_10.addWidget(self.toolButton_112)
        self.toolButton_113 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_113.setStyleSheet("border:none")
        self.toolButton_113.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_113.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_113.setObjectName("toolButton_113")
        self.horizontalLayout_10.addWidget(self.toolButton_113)
        self.verticalLayout_8.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.toolButton_114 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_114.setStyleSheet("border:none")
        self.toolButton_114.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_114.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_114.setObjectName("toolButton_114")
        self.horizontalLayout_9.addWidget(self.toolButton_114)
        self.toolButton_115 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_115.setStyleSheet("border:none")
        self.toolButton_115.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_115.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_115.setObjectName("toolButton_115")
        self.horizontalLayout_9.addWidget(self.toolButton_115)
        self.toolButton_116 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_116.setStyleSheet("border:none")
        self.toolButton_116.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_116.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_116.setObjectName("toolButton_116")
        self.horizontalLayout_9.addWidget(self.toolButton_116)
        self.verticalLayout_8.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.toolButton_117 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_117.setStyleSheet("border:none")
        self.toolButton_117.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_117.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_117.setObjectName("toolButton_117")
        self.horizontalLayout_8.addWidget(self.toolButton_117)
        self.toolButton_118 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_118.setStyleSheet("border:none")
        self.toolButton_118.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_118.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_118.setObjectName("toolButton_118")
        self.horizontalLayout_8.addWidget(self.toolButton_118)
        self.toolButton_119 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_119.setStyleSheet("border:none")
        self.toolButton_119.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_119.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_119.setObjectName("toolButton_119")
        self.horizontalLayout_8.addWidget(self.toolButton_119)
        self.verticalLayout_8.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.toolButton_1110 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_1110.setStyleSheet("border:none")
        self.toolButton_1110.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_1110.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_1110.setObjectName("toolButton_1110")
        self.horizontalLayout_6.addWidget(self.toolButton_1110)
        self.toolButton_1111 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_1111.setStyleSheet("border:none")
        self.toolButton_1111.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_1111.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_1111.setObjectName("toolButton_1111")
        self.horizontalLayout_6.addWidget(self.toolButton_1111)
        self.toolButton_1112 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_1112.setStyleSheet("border:none")
        self.toolButton_1112.setIconSize(QtCore.QSize(50, 50))
        self.toolButton_1112.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolButton_1112.setObjectName("toolButton_1112")
        self.horizontalLayout_6.addWidget(self.toolButton_1112)
        self.verticalLayout_8.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_22.addItem(spacerItem)
        self.toolButton_21 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_21.setEnabled(False)
        self.toolButton_21.setObjectName("toolButton_21")
        self.horizontalLayout_22.addWidget(self.toolButton_21)
        self.toolButton_19 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_19.setEnabled(False)
        self.toolButton_19.setObjectName("toolButton_19")
        self.horizontalLayout_22.addWidget(self.toolButton_19)
        self.page_label = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.page_label.sizePolicy().hasHeightForWidth())
        self.page_label.setSizePolicy(sizePolicy)
        self.page_label.setObjectName("page_label")
        self.horizontalLayout_22.addWidget(self.page_label)
        self.toolButton_20 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_20.setEnabled(True)
        self.toolButton_20.setObjectName("toolButton_20")
        self.horizontalLayout_22.addWidget(self.toolButton_20)
        self.toolButton_22 = QtWidgets.QToolButton(self.scrollAreaWidgetContents)
        self.toolButton_22.setEnabled(True)
        self.toolButton_22.setObjectName("toolButton_22")
        self.horizontalLayout_22.addWidget(self.toolButton_22)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_22.addItem(spacerItem1)
        self.verticalLayout_8.addLayout(self.horizontalLayout_22)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_9.addWidget(self.scrollArea)
        self.frame_5 = QtWidgets.QFrame(self.frame_3)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.collection_listWidget = QtWidgets.QListWidget(self.frame_5)
        self.collection_listWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.collection_listWidget.setObjectName("collection_listWidget")
        self.verticalLayout_5.addWidget(self.collection_listWidget)
        self.verticalLayout_9.addWidget(self.frame_5)
        self.verticalLayout_6.addWidget(self.frame_3)
        self.frame_2 = QtWidgets.QFrame(self.frame_4)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.control_toolButton = QtWidgets.QToolButton(self.frame_2)
        self.control_toolButton.setObjectName("control_toolButton")
        self.horizontalLayout_4.addWidget(self.control_toolButton)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.raiod_name_label = QtWidgets.QLabel(self.frame_2)
        self.raiod_name_label.setObjectName("raiod_name_label")
        self.horizontalLayout_2.addWidget(self.raiod_name_label)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.time_label = QtWidgets.QLabel(self.frame_2)
        self.time_label.setObjectName("time_label")
        self.horizontalLayout_2.addWidget(self.time_label)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.progress_slider = QtWidgets.QSlider(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progress_slider.sizePolicy().hasHeightForWidth())
        self.progress_slider.setSizePolicy(sizePolicy)
        self.progress_slider.setMouseTracking(False)
        self.progress_slider.setOrientation(QtCore.Qt.Horizontal)
        self.progress_slider.setInvertedControls(False)
        self.progress_slider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.progress_slider.setObjectName("progress_slider")
        self.horizontalLayout_3.addWidget(self.progress_slider)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.line = QtWidgets.QFrame(self.frame_2)
        self.line.setFrameShadow(QtWidgets.QFrame.Raised)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setObjectName("line")
        self.horizontalLayout_4.addWidget(self.line)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.volum_Slider = QtWidgets.QSlider(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.volum_Slider.sizePolicy().hasHeightForWidth())
        self.volum_Slider.setSizePolicy(sizePolicy)
        self.volum_Slider.setMaximum(100)
        self.volum_Slider.setProperty("value", 100)
        self.volum_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.volum_Slider.setObjectName("volum_Slider")
        self.horizontalLayout.addWidget(self.volum_Slider)
        self.volume_label = QtWidgets.QLabel(self.frame_2)
        self.volume_label.setObjectName("volume_label")
        self.horizontalLayout.addWidget(self.volume_label)
        self.horizontalLayout_4.addLayout(self.horizontalLayout)
        self.do_collection_toolButton = QtWidgets.QToolButton(self.frame_2)
        self.do_collection_toolButton.setObjectName("do_collection_toolButton")
        self.horizontalLayout_4.addWidget(self.do_collection_toolButton)
        self.share_toolButton = QtWidgets.QToolButton(self.frame_2)
        self.share_toolButton.setObjectName("share_toolButton")
        self.horizontalLayout_4.addWidget(self.share_toolButton)
        self.verticalLayout_6.addWidget(self.frame_2)
        self.horizontalLayout_7.addLayout(self.verticalLayout_6)
        self.horizontalLayout_5.addWidget(self.frame_4)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; text-decoration: underline;\">??????</span></p></body></html>"))
        self.recommend_toolButton.setText(_translate("MainWindow", "??????"))
        self.city_cate_toolButton.setText(_translate("MainWindow", "??????"))
        self.country_cate_toolButton.setText(_translate("MainWindow", "??????"))
        self.network_cate_toolButton.setText(_translate("MainWindow", "??????"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; text-decoration: underline;\">??????</span></p></body></html>"))
        self.collection_toolButton.setText(_translate("MainWindow", "????????????"))
        self.add_toolButton.setText(_translate("MainWindow", "????????????"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; text-decoration: underline;\">??????</span></p></body></html>"))
        self.toolButton_7.setText(_translate("MainWindow", "????????????"))
        self.connect_pushButton.setText(_translate("MainWindow", "????????????"))
        self.abou_qt_pushButton.setText(_translate("MainWindow", "??????QT"))
        self.toolButton_111.setText(_translate("MainWindow", "..."))
        self.toolButton_112.setText(_translate("MainWindow", "..."))
        self.toolButton_113.setText(_translate("MainWindow", "..."))
        self.toolButton_114.setText(_translate("MainWindow", "..."))
        self.toolButton_115.setText(_translate("MainWindow", "..."))
        self.toolButton_116.setText(_translate("MainWindow", "..."))
        self.toolButton_117.setText(_translate("MainWindow", "..."))
        self.toolButton_118.setText(_translate("MainWindow", "..."))
        self.toolButton_119.setText(_translate("MainWindow", "..."))
        self.toolButton_1110.setText(_translate("MainWindow", "..."))
        self.toolButton_1111.setText(_translate("MainWindow", "..."))
        self.toolButton_1112.setText(_translate("MainWindow", "..."))
        self.toolButton_21.setText(_translate("MainWindow", "???"))
        self.toolButton_19.setText(_translate("MainWindow", "<"))
        self.page_label.setText(_translate("MainWindow", "<html><head/><body><p>???<span style=\" color:#55aaff;\">1</span>??????<span style=\" color:#55aaff;\">1</span>???</p></body></html>"))
        self.toolButton_20.setText(_translate("MainWindow", ">"))
        self.toolButton_22.setText(_translate("MainWindow", "???"))
        self.control_toolButton.setText(_translate("MainWindow", "??????"))
        self.raiod_name_label.setText(_translate("MainWindow", "????????????"))
        self.time_label.setText(_translate("MainWindow", "?????????"))
        self.label_2.setText(_translate("MainWindow", "?????????"))
        self.volume_label.setText(_translate("MainWindow", "100%"))
        self.do_collection_toolButton.setText(_translate("MainWindow", "??????"))
        self.share_toolButton.setText(_translate("MainWindow", "??????"))

