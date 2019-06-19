import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import cv2
import os
import PaintDialog
import threading
from faced import FaceDetector
from faced.utils import annotate_image
from math import ceil
from demo_faster_rcnn import *

def XXplot_bbox(car_or_face, img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, ax=None,
              reverse_rgb=False, absolute_coordinates=True, ):
    """Visualize bounding boxes.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).

    Returns
    -------
    matplotlib axes
        The ploted axes.

    """

    img = img.copy()
    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if len(bboxes) < 1:
        return ax

    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(labels, mx.nd.NDArray):
        labels = labels.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    if not absolute_coordinates:
        # convert to absolute coordinates using image shape
        height = img.shape[0]
        width = img.shape[1]
        bboxes[:, (0, 2)] *= width
        bboxes[:, (1, 3)] *= height

    # use random colors if None is provided
    if colors is None:
        colors = dict()
    for i, bbox in enumerate(bboxes):
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        if cls_id not in colors:
            if class_names is not None:
                colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
            else:
                colors[cls_id] = (random.random(), random.random(), random.random())

        xmin, ymin, xmax, ymax = [int(x) for x in bbox]

        if xmin<0 :
            xmin = 0
        if ymin<0:
            ymin = 0


        if car_or_face == 'car':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        #color = None
        # if cls_id ==6:
        #     color=(0,255,0)
        # else:
        #     color=(0,0,255)

        ##for boundingbox
        #img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)

        ##for mosaic
        tomosaic_img = img[ymin:ymax, xmin:xmax]
        face_img_shape = tomosaic_img.shape

        tomosaic_img = cv2.resize(tomosaic_img, (3, 3))
        tomosaic_img = cv2.resize(tomosaic_img, (xmax-xmin, ymax-ymin), interpolation=cv2.INTER_AREA)
        tomosaic_img = tomosaic_img[:face_img_shape[0], :face_img_shape[1], :face_img_shape[2]]

        img[ymin:ymax, xmin:xmax] = tomosaic_img

        ##for text(confidence)
        # if class_names is not None and cls_id < len(class_names):
        #     class_name = class_names[cls_id]
        # else:
        #     class_name = str(cls_id) if cls_id >= 0 else ''
        # score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        # if class_name or score:
        #     img = cv2.putText(img, '{:s} {:s}'.format(class_name, score),\
        #                       (xmin, ymin-2),cv2.FONT_HERSHEY_SIMPLEX, 0.8,\
        #                       color,1, cv2.LINE_AA)
    return img


class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("PyQt Video Player Widget Example - pythonprogramminglanguage.com")

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Maximum)

        # Create new action
        openAction = QAction( QIcon('icon/folder.png'),'&Open', self)
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open movie')
        openAction.triggered.connect(self.openFile)


        # Create menu bar and add action
        menuBar = self.addToolBar('open')
        #fileMenu.addAction(newAction)
        menuBar.addAction(openAction)

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)

        # Set widget to contain window contents
        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

    def openFile(self,filename):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())

        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())

class Masking(QObject):
    percentage = pyqtSignal(float)

    def __init__(self, videoname, outputpath):
        QObject.__init__(self)
        self.videoname = videoname
        self.outputpath = outputpath

    def run(self):
        t = threading.Thread(target=self.masking)
        t.start()

    def masking(self):
        # face_detector = FaceDetector()
        self.percentage.emit(0)
        if self.videoname != None:
            cap = cv2.VideoCapture(self.videoname)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            filename = self.videoname
            filename = filename.split('/')
            filename = filename[-1].split('.')

            ret = True
            i = 0
            num_frame = 0
            frames = []
            img_list = []
            per = 0
            # make folder that will contain numpy images read via opencv
            if not os.path.exists('./imgsave_{}'.format(filename[0])):
                os.mkdir('./imgsave_{}'.format(filename[0]))
            save_imgdir = './imgsave_{}'.format(filename[0])

            while ret:
                '''
                read images frame from video.
                #1.frames is python list that will have numpy images (H,W,C)
                2.inside the save_imgdir, each read images will be located
                3.img_list is python list that contains images path    
                '''
                ret, frame = cap.read()
                if ret:
                    # frames.append(frame)
                    cv2.imwrite('./{}/{}.jpg'.format(save_imgdir, num_frame), frame)
                    img_list.append('./{}/{}.jpg'.format(save_imgdir, num_frame))
                    num_frame += 1
                    self.percentage.emit(num_frame/150)
                    #print('self.percentage.emit : ',num_frame/300)
            per = num_frame/150
            # else :
            #     saved_imgdir = './imgsave_{}'.format(filename[0])
            #     files = os.listdir(saved_imgdir)
            #     for file in files:
            #         temp = file.split('.')[0]
            #         img_list.append(temp)
            #     img_list.sort(reverse=False)
            #
            #     for file in img_list:
            #         file = file + '.txt'
            #
            #     for file in files:
            #         full_path = os.path.join(saved_imgdir, file)
            #         print('full_path is : ', full_path)
            # return

            """
            Todo:
                anotate each frame and make it to video

            """

            x, orig_img, _, per = data.transforms.presets.rcnn.load_test(self, per, img_list)

            frame_height = orig_img[0].shape[0]
            frame_width = orig_img[0].shape[1]
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter('{0}/masked_{1}.avi'.format(self.outputpath, filename[0]), fourcc, 30.0,
                                  (frame_width, frame_height))

            for i in range(len(img_list)):
                # (1.6000,1),(1,6000,1),(1,6000,4)
                self.percentage.emit(per+ (i / len(img_list)) * (100-per))
                #print('i/len(image_list) has been emitted! :', i / len(img_list))

                box_ids, scores, bboxes = net1(x[i].copyto(ctx[0]))
                box_ids2, scores2, bboxes2 = net2(x[i].copyto(ctx[1]))

                img_ = XXplot_bbox('face', orig_img[i], bboxes[0], scores[0], box_ids[0], class_names=net1.classes)
                img_ = XXplot_bbox('car', img_, bboxes2[0], scores2[0], box_ids2[0], class_names=net2.classes)

                out.write(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
                # cv2.imwrite('detected_image/{}.jpg'.format(i), cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))
            out.release()

class VideoInfo(QWidget):
    def __init__(self):
        QWidget.__init__(self, flags=Qt.Widget)
        self.setWindowTitle("QTreeWidget")
        self.videoname = None
        self.outputpath=os.path.abspath('result')

        # 전체 폼 박스
        formbox = QHBoxLayout()
        self.setLayout(formbox)

        # 좌, 우 레이아웃박스
        left = QVBoxLayout()
        right = QVBoxLayout()

        #group1
        gb = QGroupBox('비디오 찾기')
        left.addWidget(gb)

        box = QVBoxLayout()
        gb.setLayout(box)

        fileBtn=QPushButton('비디오 찾기')
        fileBtn.clicked.connect(self.openFileName)
        self.fileLabel=QLabel('')

        box.addWidget(fileBtn)
        box.addWidget(self.fileLabel)

        # group2
        gb = QGroupBox('저장 디렉토리')
        left.addWidget(gb)

        hbox = QVBoxLayout()
        gb.setLayout(hbox)

        ouputfileBtn = QPushButton('파일 찾기')
        self.outputLabel = QLabel('D:/GUI/result/')
        hbox.addWidget(ouputfileBtn)
        hbox.addWidget(self.outputLabel)
        ouputfileBtn.clicked.connect(self.outputFileName)



        #group3
        gb = QGroupBox('마스킹 툴')
        left.addWidget(gb)

        box = QVBoxLayout()
        gb.setLayout(box)

        paintBtn = QPushButton('실 행')
        paintBtn.clicked.connect(self.paintDialog)
        box.addWidget(paintBtn)


        #group4
        gb = QGroupBox('마스킹')
        left.addWidget(gb)

        box = QVBoxLayout()
        gb.setLayout(box)
        faceBtn = QPushButton('얼굴 + 차번호판 마스킹 시작')
        carBtn = QPushButton('차번호판 마스킹 시작')\



        faceBtn.clicked.connect(self.makefaceThread)
        carBtn.clicked.connect(self.carnumberdetect)

        self.progressbar = QProgressBar(self)
        self.progressbar.setValue(0)
        box.addWidget(self.progressbar)
        box.addWidget(faceBtn)
        #box.addWidget(carBtn)

        self.videoView = VideoWindow()
        right.addWidget(self.videoView)

        # 전체 폼박스에 좌우 박스 배치
        formbox.addLayout(left)
        formbox.addLayout(right)

        formbox.setStretchFactor(left, 1)
        formbox.setStretchFactor(right, 3)

    def openFileName(self):
        self.videoname,_ = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.fileLabel.setText(self.videoname)
        self.progressbar.setValue(0)

    def paintDialog(self):
        dlg = PaintDialog.PaintDialog()
        dlg.exec_()

    def makefaceThread(self):
        if self.videoname!=None:
            toMask = Masking(self.videoname, self.outputpath)
            toMask.percentage.connect(self.update_percentage)
            toMask.run()

    def facedect(self):

        face_detector = FaceDetector()
        self.progressbar.setValue(0)


        if self.videoname!=None:
            cap = cv2.VideoCapture(self.videoname)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            filename = self.videoname
            filename = filename.split('/')
            filename = filename[-1].split('.')

            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter('{0}/masked_{1}.avi'.format(self.outputpath,filename[0]), fourcc, 25.0, (frame_width, frame_height))

            ret = True
            i = 0
            frames=[]
            num_frame=0
            while ret:
                ret, frame = cap.read()

                if ret:
                    frames.append(frame)
                    num_frame+=1
            step = 1

            for i ,frame in enumerate(frames):

                rgb_img = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

                # Receives RGB numpy image (HxWxC) and
                # returns (x_center, y_center, width, height, prob) tuples.
                bboxes = face_detector.predict(rgb_img, 0.8)

                # Use this utils function to annotate the image.
                ann_img = annotate_image(frame, bboxes)
                out.write(ann_img)


                if (i/num_frame*100)-step>0:
                    step+=1
                    print(step)
                    self.progressbar.setValue(step)


            cap.release()
            out.release()


    def carnumberdetect(self):
        pass

    def outputFileName(self):
        self.outputpath = QFileDialog.getExistingDirectory(self)
        self.outputpath = os.path.abspath(self.outputpath)
        if self.outputpath != '':
            self.outputLabel.setText(self.outputpath)

    def update_percentage(self, percent):
        self.progressbar.setValue(int(ceil(percent)))

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.date = QDate.currentDate()
        self.initUI()

    def initUI(self):
        # 윈도우 아이콘
        self.setWindowIcon(QIcon('icon/cctv.png'))

        # 상태바 만들기
        self.statusBar().showMessage(self.date.toString(Qt.DefaultLocaleLongDate))

        # 레이아웃을 이용
        wg = VideoInfo()
        self.setCentralWidget(wg)


        #self.createToolbar()
        # 윈도우 위치
        self.setWindowTitle(' CCTV 모자이크')
        self.move(300, 300)
        self.setFixedSize(1100, 800)

        self.show()


    def createToolbar(self):
        faceAction = QAction(QIcon('icon/face.png'), 'Play', self)
        faceAction.setStatusTip('Face Masking')


        carAction = QAction(QIcon('icon/car.png'), 'Car', self)
        carAction.setStatusTip('Face Masking')


        fileAction = QAction(QIcon('icon/folder.png'), 'Open', self)
        fileAction.setShortcut('Ctrl+O')
        fileAction.setStatusTip('Open New File')
        fileAction.triggered.connect(self.openFileName)

        paintAction = QAction(QIcon('icon/paintbrush.png'), 'Paint', self)
        paintAction.setShortcut('Ctrl+P')
        paintAction.setStatusTip('Paint')
        paintAction.triggered.connect(self.paintDialog)

        # 툴바 만들기
        self.toolbar = self.addToolBar('open')
        self.toolbar.addAction(faceAction)
        self.toolbar.addAction(carAction)
        self.toolbar.addAction(fileAction)
        self.toolbar.addAction(paintAction)

    def openFileName(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')

        print(fname)

    def paintDialog(self):
        dlg = PaintDialog.PaintDialog()
        dlg.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
