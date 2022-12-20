
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import imutils
from imutils.perspective import four_point_transform


class AnswerSheetGrader():
  def __init__(self, img_path, height, width, max_qstn,prspctv):
    self.main_img = cv2.imread(img_path)
    self.height = height 
    self.width = width  
    self.mnplt_img = self.main_img.copy()
    self.max_qstn = max_qstn
    self.perspective = prspctv
    self.big_circle = None
    self.small_circles = None
    self.bar_code_roi= None
    self.answers_roi= None
    self.rects_roi= None
    self.rects_cnt= None
    self.vertical_cnt = None
    self.qstn = None
    self.answer= None

  def pre_processing(self):
    if self.perspective :
      self.remove_perspective()
    self.find_big_circle()
    self.rotate_img()
    self.find_small_circles()
    self.split_main_img()
    self.find_left_rects()
    self.find_vertical_lines()
    self.find_main_bubbles_pos()

  def remove_perspective(self) :

    gray_img = cv2.cvtColor(self.mnplt_img, cv2.COLOR_BGR2GRAY)    
    edge_img = cv2.Canny(gray_img,5,7)

    contours, h = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    alpha = 0.02
    rect_cnts = self.get_rect_cnts(contours ,alpha)
    
    if (len(rect_cnts) != 0):
      self.mnplt_img = four_point_transform(self.mnplt_img, rect_cnts[0].reshape(4, 2))
      self.mnplt_img = cv2.resize(self.mnplt_img, (self.width, self.height))
      # cv2.drawContours(self.p_img, rect_cnts, -1, (0, 0, 255), 3)
      # cv2_imshow(self.mnplt_img)   
      

  def get_rect_cnts(self, contours, alpha):
    
    rect_cnts = []
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, alpha * peri, True)
        
        if len(approx) == 4:
            rect_cnts.append(approx)
    
    rect_cnts = sorted(rect_cnts, key=cv2.contourArea, reverse=True)
    
    return rect_cnts

  def find_big_circle(self) :
    
    gray = cv2.cvtColor(self.mnplt_img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 5)

    bigcircle = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, self.mnplt_img.shape[0]/64, param1=200, param2=22, minRadius=7, maxRadius=10)
    if bigcircle is not None:
        bigcircle = np.uint16(np.around(bigcircle))
        for i in bigcircle[0, :]:
            cv2.circle(self.mnplt_img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    big_circle_x=bigcircle[0][0][0]
    big_circle_y=bigcircle[0][0][1]
    self.big_circle = (big_circle_x, big_circle_y)

  def rotate_img(self):

    if (abs(self.width-self.big_circle[0])<(0.1 * self.width) and abs(self.height-self.big_circle[1])>(0.9 * self.height)):
      self.mnplt_img = cv2.rotate(self.mnplt_img, cv2.ROTATE_90_CLOCKWISE)

    elif (abs(self.width-self.big_circle[0])>(0.9 * self.width) and abs(self.height-self.big_circle[1])>(0.9 * self.height)):
      self.mnplt_img = cv2.rotate(self.mnplt_img, cv2.ROTATE_180)

    elif (abs(self.width-self.big_circle[0])>(0.9 * self.width) and abs(self.height-self.big_circle[1])<(0.1 * self.height)):
      self.mnplt_img = cv2.rotate(self.mnplt_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    if not self.perspective:
       self.mnplt_img = cv2.resize(self.main_img, (self.width, self.height))


  def find_small_circles(self) :

    gray = cv2.cvtColor(self.mnplt_img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, self.mnplt_img.shape[0]/64, param1=200, param2=22, minRadius=5, maxRadius=7)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(self.mnplt_img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    self.small_circles = circles

  def split_main_img(self):

    max_XY = np.amax(self.small_circles[0], axis=0)[:2]
    min_XY = np.amin(self.small_circles[0], axis=0)[:2]

    self.bar_code_roi = self.mnplt_img[:min_XY[1],:]
    self.answers_roi = self.mnplt_img[min_XY[1]+14:,:]

    sec_w = int(self.answers_roi.shape[0]*1.3)
    sec_h = int(self.answers_roi.shape[1]*1.3)
    self.answers_roi = cv2.resize(self.answers_roi, (sec_h, sec_w))

    self.rects_roi = self.answers_roi[6:,0:45]
    # cv2_imshow(self.rects_roi)
    # cv2_imshow(self.answers_roi)
    # cv2_imshow(self.bar_code_roi)


  def find_barcode(self):
    barcode_roi = self.bar_code_roi.copy()
    gray = cv2.cvtColor(barcode_roi, cv2.COLOR_BGR2GRAY)

    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F

    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 20))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)

    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    rect = cv2.minAreaRect(c)

    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(barcode_roi, [box], -1, (0, 255, 0), 3)
    # cv2_imshow(barcode_roi)

    max_XY = np.amax(box, axis=0)
    min_XY = np.amin(box, axis=0)


    barcode = self.bar_code_roi[min_XY[1]:max_XY[1],min_XY[0]:max_XY[0]]

    bardet = cv2.barcode_BarcodeDetector()
    ok, decoded_info, decoded_type, corners = bardet.detectAndDecode(barcode)
    return (ok, decoded_info, decoded_type, corners)


  def find_left_rects(self):

    rects=self.rects_roi.copy()

    gray_img = cv2.cvtColor(rects, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(gray_img, 50, 70)
    contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    self.rects_cnt=[]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            if y not in self.rects_cnt: 
              self.rects_cnt.append(y)
            cv2.rectangle(rects,(x,y),(x+w,y+h),(0, 0, 255),2)
    # cv2_imshow(rects)
    self.rects_cnt.sort()
    i=1
    prv_rct = -100
    for rct in self.rects_cnt:
      if abs(rct - prv_rct) < 5 :
        self.rects_cnt.remove(rct)
      else:
        prv_rct = rct

  def find_vertical_lines(self):

    image = self.answers_roi.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray,5,100,apertureSize = 3)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
    detect_vertical = cv2.morphologyEx(edge, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    self.vertical_cnt = []

    for c in cnts:
        c_mean=int(c.mean(0)[0][0])
        x=[]
        x=[i for i in range(c_mean-10,c_mean+10) if i in self.vertical_cnt]
        if not x:
          self.vertical_cnt.append(c_mean)
          pnt=c_mean
        else:
          self.vertical_cnt.remove(x[0])
          pnt=int((x[0]+c_mean)/2)
          self.vertical_cnt.append(pnt)
          
        image = cv2.circle(self.answers_roi, (pnt,20), radius=2, color=(0, 0, 255), thickness=-1)
    # cv2_imshow(self.answers_roi)

  def find_main_bubbles_pos(self):
    # answers_roi_img = self.answers_roi.copy()

    # self.rects_cnt.sort()
    # self.vertical_cnt.sort()

    # i=5
    # j=7

    # sec1 = answers_roi_img[self.rects_cnt[i]+20:self.rects_cnt[i]+35,self.vertical_cnt[j]+18:self.vertical_cnt[j+1]]

    # gray = cv2.cvtColor(sec1, cv2.COLOR_BGR2GRAY)
    # edge_img = cv2.Canny(gray,10,150)

    # contours_Canny, h = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sg=sec1.copy()
    # cv2.drawContours(sg, contours_Canny, -1, (0, 0, 255), 1)
    # cv2_imshow(sg)

    # rects_qstn=[]

    # q_num=1
    # self.qstn={}
    # for c in contours_Canny:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.08 * peri, True)
    #     if len(approx) == 4:
    #         x,y,w,h = cv2.boundingRect(approx)
    #         w=15
    #         h=15
    #         if [x,y,w,h] not in rects_qstn: 
    #           rects_qstn.append([x,y,w,h])
    #           self.qstn[q_num]=x
    #           q_num+=1
    #           cv2.rectangle(sg,(x,y),(x+w,y+h),(0, 255, 0),1)
    # # cv2_imshow(sg)
    # if len(self.qstn) != 4 :
    self.qstn = { 4:58, 3:38, 2:20, 1:2}


  def get_answrs(self):
    answers_roi_image = self.answers_roi.copy()
    self.rects_cnt.sort()
    self.vertical_cnt.sort()
    qstn_ind=0
    self.answer={}

    for i in range(1,len(self.vertical_cnt)-2):

      sec1 = answers_roi_image[self.rects_cnt[0]+3:,self.vertical_cnt[i]+19:self.vertical_cnt[i+1]]

      gray = cv2.cvtColor(sec1, cv2.COLOR_BGR2GRAY)
      blurred = cv2.GaussianBlur(gray, (7,7), 0)

      _,thresh_OTSU = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      edge_img = cv2.Canny(thresh_OTSU,50,100,apertureSize = 3)
      # cv2_imshow(edge_img)

      contours_O, h = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

      so=sec1.copy()

      rects_answer=[]
      for c in contours_O:
        if c.shape[0] > 5:
          ellipse = cv2.fitEllipse(c)
          cv2.ellipse(edge_img,ellipse,(255,255,255),1)

      contours_O=[]
      contours_O, h = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
      # cv2_imshow(edge_img)
      contours_O = sorted(contours_O, key=cv2.contourArea, reverse=True)

      for c in contours_O:
          rect = cv2.minAreaRect(c)
          box = cv2.boxPoints(rect)
          if len(box) == 4:
              x,y,w,h = cv2.boundingRect(box)
              if w > 10 and h > 10 :
                w=15
                h=15
                if x < 0: x = 1
                if y < 0: y = 1
                answr=-1
                for q in self.qstn:
                  if (x > self.qstn[q]-5 and x < self.qstn[q]+5):
                    x = self.qstn[q]
                    answr=q

                if [x,y,w,h] not in rects_answer and answr != -1 :
                  for rect in self.rects_cnt:
                    if (y > rect-10 and y < rect+10):
                        y = rect
                        qstn_ind = self.rects_cnt.index(rect) + 1
                        qstn_ind = (40 * (i-1)) + qstn_ind
                  if qstn_ind > self.max_qstn: break
                  self.answer[qstn_ind] = answr
                  cv2.rectangle(so,(x,y),(x+w,y+h),(255, 0, 0),1)
                  so = cv2.putText(img = so,text = str(qstn_ind),org = (x,y+8),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 0.4,color = (0, 0, 255),thickness = 1)
      cv2_imshow(so)
    return self.answer 

def main():
  asg = AnswerSheetGrader('Images\4.jpg', 800, 700, 235, 0)
  asg.pre_processing()
  barcode_inf = asg.find_barcode()
  answers = asg.get_answrs()
  print(len(answers),answers)


main()