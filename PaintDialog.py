import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import os
import numpy as np
import copy

class PaintDialog(QDialog):
    def __init__(self, parent=None):
        super(PaintDialog, self).__init__(parent)
        self.setWindowTitle('직접 그리기')
        self.setWindowIcon(QIcon('icon/cctv.png'))
        self.setGeometry(0, 0, 1200, 650)
        self.videoname=None


        # 전체 폼 박스
        formbox = QHBoxLayout()
        self.setLayout(formbox)

        # 좌, 우 레이아웃박스
        left = QVBoxLayout()
        right = QVBoxLayout()

        gb = QGroupBox('모자이크 넓이')
        left.addWidget(gb)

        box = QVBoxLayout()
        gb.setLayout(box)

        # groub1
        text = ['30', '50', '100', '150']
        self.radiobtns = []

        for i in range(len(text)):

            self.radiobtns.append(QRadioButton(text[i], self))
            self.radiobtns[i].clicked.connect(self.radioClicked)
            if text[i]=='100':
                self.radiobtns[i].setChecked(True)
            box.addWidget(self.radiobtns[i])


        # groub 2
        gb = QGroupBox('비디오 입력')
        left.addWidget(gb)

        hbox = QVBoxLayout()
        gb.setLayout(hbox)

        videoBtn=QPushButton('비디오 찾기')
        videoBtn.clicked.connect(self.inputFileName)
        self.fileLabel=QLabel('')


        hbox.addWidget(videoBtn)
        hbox.addWidget(self.fileLabel)

        #group3
        gb = QGroupBox('비디오')
        left.addWidget(gb)

        hbox = QVBoxLayout()
        gb.setLayout(hbox)

        self.idx_string='0/0'
        self.idxLabel=QLabel(self.idx_string)
        backBtn=QPushButton('이미지 되돌리기')
        backBtn.setShortcut('Ctrl+z')

        self.slider=QSlider(Qt.Horizontal,self)
        self.slider.valueChanged.connect(self.slidechange)


        btnbox = QHBoxLayout()
        prevBtn=QPushButton('이전(P)')

        nextBtn = QPushButton('다음(N)')
        prevBtn.clicked.connect(self.previdx)
        nextBtn.clicked.connect(self.nextidx)
        prevBtn.setShortcut('p')
        nextBtn.setShortcut('n')

        btnbox.addWidget(prevBtn)
        btnbox.addWidget(nextBtn)
        hbox.addWidget(backBtn)
        hbox.addWidget(self.idxLabel)
        hbox.addWidget(self.slider)
        hbox.addLayout(btnbox)
        # hbox.addWidget(prevBtn)
        # hbox.addWidget(nextBtn)


        #group4
        gb = QGroupBox('저장 디렉토리')
        left.addWidget(gb)

        hbox = QVBoxLayout()
        gb.setLayout(hbox)

        ouputfileBtn = QPushButton('파일 찾기')
        self.outputLabel = QLabel('D:/GUI/result/')
        hbox.addWidget(ouputfileBtn)
        hbox.addWidget(self.outputLabel)
        ouputfileBtn.clicked.connect(self.outputFileName)

        storeBtn = QPushButton('저장')
        hbox.addWidget(storeBtn)
        storeBtn.clicked.connect(self.saveVideo)



        self.paintView = PaintView()
        right.addWidget(self.paintView)
        backBtn.clicked.connect(self.paintView.backImg)

        # 전체 폼박스에 좌우 박스 배치
        formbox.addLayout(left)
        formbox.addLayout(right)

        formbox.setStretchFactor(left, 0)
        formbox.setStretchFactor(right, 1)


    def radioClicked(self):
        for i in range(len(self.radiobtns)):
            if self.radiobtns[i].isChecked():
                self.paintView.setThickness(i)
                break

    def inputFileName(self):
        self.frames=[]
        self.num_frames=0
        self.idx=0
        self.videoname,_ = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.outputpath=os.path.abspath('result')

        if self.videoname!='':
            path, ext = os.path.splitext(self.videoname)

            if ext in ('.mp4', '.avi','.AVi'):
                cap=cv2.VideoCapture(self.videoname)
                while (cap.isOpened()):
                    ret, frame = cap.read()

                    if ret:
                        frame = np.array(frame)
                        b, g, r = cv2.split(frame)
                        frame = cv2.merge((r, g, b))
                        self.frames.append(frame)
                        self.num_frames+=1
                    else:
                        break

            self.fileLabel.setText(self.videoname)
            self.idx_string='{0}/{1}'.format(self.idx,self.num_frames-1)
            self.idxLabel.setText(self.idx_string)
            self.paintView.setImage(self.frames[self.idx])
            self.num_frames-=1
            self.slider.setValue(0)
            self.slider.setRange(0,self.num_frames)


    def outputFileName(self):
        self.outputpath= QFileDialog.getExistingDirectory(self)
        self.outputpath=os.path.abspath(self.outputpath)
        if self.outputpath!='':
            self.paintView.setoutputPath(self.outputpath)
            self.outputLabel.setText(self.outputpath)

    def saveVideo(self):
        if self.videoname!=None:
            filename=self.videoname
            filename=filename.split('/')
            filename=filename[-1].split('.')

            fourcc=cv2.VideoWriter_fourcc(*'XVID')
            writer=cv2.VideoWriter("{0}/masked_{1}.avi".format(self.outputpath,filename[0]),fourcc,30.0,(self.paintView.width,self.paintView.height))


            for i,frame in enumerate(self.frames):
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                writer.write(frame)
            writer.release()
            print('end')


    def nextidx(self):
        if self.videoname!=None:

            filename = self.videoname
            filename = filename.split('/')
            filename = filename[-1].split('.')

            self.idx+=1
            if self.num_frames<= self.idx:
                self.idx=0
            self.idxLabel.setText('{0}/{1}'.format(self.idx, self.num_frames))
            self.paintView.setImage(self.frames[self.idx])
            self.update()

    def previdx(self):
        if self.videoname!=None:
            filename = self.videoname
            filename = filename.split('/')
            filename = filename[-1].split('.')


            self.idx-=1
            if self.idx<0:
                self.idx=self.num_frames
            self.idxLabel.setText('{0}/{1}'.format(self.idx,self.num_frames))
            self.paintView.setImage(self.frames[self.idx])
            self.update()

    def slidechange(self):
        if self.videoname!=None:
            self.idx=self.slider.value()
            self.idxLabel.setText('{0}/{1}'.format(self.idx, self.num_frames))
            self.paintView.setImage(self.frames[self.idx])
            self.paintView.update()

class PaintView(QWidget):
    def __init__(self):
        super().__init__()
        file_path = 'icon/file_need.PNG'
        self.cvImage = cv2.imread(file_path)

        self.height, self.width, self.byteValue = self.cvImage.shape


        self.byteValue = self.byteValue * self.width

        cv2.cvtColor(self.cvImage, cv2.COLOR_BGR2RGB, self.cvImage)

        self.mQImage = QImage(self.cvImage, self.width, self.height, self.byteValue, QImage.Format_RGB888)
        self.x = 0
        self.y = 0
        self.w = 100
        self.h = 100
        self.mosaic_rate = 20
        self.output_path='result/'
        self.setWindowIcon(QIcon('icon/cctv.png'))
        self.image_stack=[]

    def paintEvent(self, QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self.mQImage)
        painter.end()

    def mousePressEvent(self, e):
        self.image_stack.append(copy.deepcopy(self.cvImage))
        point = e.pos()
        self.x = point.x()
        self.y = point.y()

        print(self.x, self.y)
        startx = self.x - self.w
        starty = self.y - self.h
        endx = self.x + self.w
        endy = self.y + self.h

        if startx<0:
            startx=0
        if starty<0:
            starty=0
        if endx>self.width:
            endx=self.width
        if endy>self.height:
            endy=self.height

        crop_img = self.cvImage[starty:endy, startx:endx]
        # 자른 이미지를 지정한 배율로 확대/축소하기
        face_img = cv2.resize(crop_img, (self.w // self.mosaic_rate, self.h // self.mosaic_rate))
        # 확대/축소한 그림을 원래 크기로 돌리기
        face_img = cv2.resize(face_img, (2 * self.w, 2 * self.h), interpolation=cv2.INTER_AREA)

        self.cvImage[starty:endy, startx:endx] = face_img[:endy-starty,:endx-startx]
        self.update()

    def backImg(self):
        if len(self.image_stack) !=0:
            img=self.image_stack.pop()
            self.setImage(img)
            self.update()

    def saveImg(self,idx,filename):
        path=self.output_path+filename+'_{0}.jpg'.format(idx)
        img=cv2.cvtColor(self.cvImage,cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, img)

    def setThickness(self, i):
        thickness = [30, 50, 100, 150]
        mosaic = [7, 10, 20, 30]
        self.w = thickness[i]
        self.h = thickness[i]
        self.mosaic_rate = mosaic[i]


    def setImage(self,img):
        self.cvImage=img
        self.height, self.width, byteValue = self.cvImage.shape

       # self.setGeometry(0, 0, self.width, self.height)
        byteValue = byteValue * self.width

        self.mQImage = QImage(self.cvImage, self.width, self.height, byteValue, QImage.Format_RGB888)

    def setoutputPath(self,path):
        self.output_path=path