from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from final_project import Ui_MainWindow
import sys

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from tensorflow import keras
import cv2
import numpy as np

import os

class test_predict(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""
 
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
 
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size
 
    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths
        batch_target_img_paths = self.target_img_paths
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        y=(y/255)
        
        return x, y

    

class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.load_picture)#當load data的按鈕按下去的時候會發生甚麼事
        self.ui.pushButton_2.clicked.connect(self.load_directory)#當load directory的按鈕按下去的時候會發生甚麼事
    
    def load_picture(self):
        def dice(true_mask, pred_mask, non_seg_score=1.0):
            """
                Computes the Dice coefficient.
                Args:
                    true_mask : Array of arbitrary shape.
                    pred_mask : Array with the same shape than true_mask.  
                
                Returns:
                    A scalar representing the Dice coefficient between the two segmentations. 
                
            """
            #assert true_mask.shape == pred_mask.shape

            true_mask = np.array(true_mask).astype(np.bool)
            pred_mask = np.array(pred_mask).astype(np.bool)

            # If both segmentations are all zero, the dice will be 1. (Developer decision)
            im_sum = true_mask.sum() + pred_mask.sum()
            if im_sum == 0:
                return non_seg_score

            # Compute Dice coefficient
            intersection = np.logical_and(true_mask, pred_mask)
            return 2. * intersection.sum() / im_sum
        def return_mask(pred):
            mask = (pred>0.5).astype('int')*255
            mask=mask.squeeze()
            mask = np.expand_dims(mask, axis=-1)
            img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
            return img
        fileName, filetype = QFileDialog.getOpenFileName(self,"選取檔案","./","All Files (*)")  #設定副檔名過濾,注意用雙分號間隔

        
        img_size = (512, 512)
        test1= test_predict(1,img_size,[fileName],['test/target.jpg'])
        file_path=fileName.split('/')[:-2]
        model1 =keras.models.load_model('model/ct1.h5')
        predct=model1.predict(test1)
        
        model2= keras.models.load_model('model/mn.h5')
        predmn=model2.predict(test1)

        model3= keras.models.load_model('model/ft1.h5')
        predft=model3.predict(test1)
        
        CT_path=""
        for i in file_path:
            CT_path=CT_path+i+'/'
        CT_path=CT_path+"CT/"+fileName.split('/')[-1]
        true_mask_CT = PIL.Image.open(CT_path)
        print(CT_path)

        MN_path=""
        for i in file_path:
            MN_path=MN_path+i+'/'
        MN_path=MN_path+"MN/"+fileName.split('/')[-1]
        true_mask_MN = PIL.Image.open(MN_path)
        print(MN_path)

        FT_path=""
        for i in file_path:
            FT_path=FT_path+i+'/'
        FT_path=FT_path+"FT/"+fileName.split('/')[-1]
        true_mask_FT= PIL.Image.open(FT_path)
        print(FT_path)

        ctimage= cv2.imread(CT_path, cv2.IMREAD_GRAYSCALE) #cv_ground_truth
        ftimage= cv2.imread(FT_path, cv2.IMREAD_GRAYSCALE) #ft_ground_truth
        mnimage= cv2.imread(MN_path, cv2.IMREAD_GRAYSCALE) #mn_ground_truth
         
        self.ui.label_14.setText(str(dice(true_mask_CT,return_mask(predct))))
        self.ui.label_13.setText(str(dice(true_mask_FT,return_mask(predft))))
        self.ui.label_12.setText(str(dice(true_mask_MN,return_mask(predmn))))


        label1=return_mask(predmn)
        label1.save('test1.jpg')
        label2=return_mask(predct)
        label2.save('test2.jpg')
        label3=return_mask(predft)
        label3.save('test3.jpg')
        timage= cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #background
        timage=cv2.cvtColor(timage, cv2.COLOR_GRAY2BGR)
        for i in range(3):
            file_Name='test'+str(i+1)+'.jpg'
            image = cv2.imread(file_Name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)
            canny = cv2.Canny(blurred, 30, 150)
            contours, hierarchy= cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if i==0 :
                cv2.drawContours(timage, contours, -1, (0,255,0), 2)#mn
            elif i==1:
                cv2.drawContours(timage, contours, -1, (0,0,255), 2)#ct
            else:
                cv2.drawContours(timage, contours, -1, (255,0,0), 2)#ft

        pixmap = QPixmap(MN_path)
        pixmap = pixmap.scaled(256, 256)
        self.ui.label_15.setPixmap(pixmap)
        pixmap = QPixmap(FT_path)
        pixmap = pixmap.scaled(256, 256)
        self.ui.label_16.setPixmap(pixmap)
        pixmap = QPixmap(CT_path)
        pixmap = pixmap.scaled(256, 256)
        self.ui.label_17.setPixmap(pixmap)
        pixmap = QPixmap('test1.jpg')
        pixmap = pixmap.scaled(256, 256)
        self.ui.label_18.setPixmap(pixmap)
        pixmap = QPixmap('test3.jpg')
        pixmap = pixmap.scaled(256, 256)
        self.ui.label_19.setPixmap(pixmap)
        pixmap = QPixmap('test2.jpg')
        pixmap = pixmap.scaled(256, 256)
        self.ui.label_21.setPixmap(pixmap)

        cv2.imshow('predict result',timage)
        cv2.imwrite('output.png', timage)
        pixmap = QPixmap('output.png')
        pixmap = pixmap.scaled(256, 256)
        self.ui.label_27.setPixmap(pixmap)

       
            


    def load_directory(self):
        def dice(true_mask, pred_mask, non_seg_score=1.0):
            """
                Computes the Dice coefficient.
                Args:
                    true_mask : Array of arbitrary shape.
                    pred_mask : Array with the same shape than true_mask.  
                
                Returns:
                    A scalar representing the Dice coefficient between the two segmentations. 
                
            """
            #assert true_mask.shape == pred_mask.shape

            true_mask = np.array(true_mask).astype(np.bool)
            pred_mask = np.array(pred_mask).astype(np.bool)

            # If both segmentations are all zero, the dice will be 1. (Developer decision)
            im_sum = true_mask.sum() + pred_mask.sum()
            if im_sum == 0:
                return non_seg_score

            # Compute Dice coefficient
            intersection = np.logical_and(true_mask, pred_mask)
            return 2. * intersection.sum() / im_sum
        def return_mask(pred):
            mask = (pred>0.5).astype('int')*255
            mask=mask.squeeze()
            mask = np.expand_dims(mask, axis=-1)
            img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
            return img
        
        dir_input_path = "image/"+self.ui.lineEdit.text()+"/"+"T1/"
        allFileList = os.listdir(dir_input_path)
        input_files=[]
        output_file_ct=[]
        output_file_ft=[]
        output_file_mn=[]

        for file in allFileList:
            input_files.append("image/"+self.ui.lineEdit.text()+"/"+"T1/"+file)
            output_file_ct.append("image/"+self.ui.lineEdit.text()+"/"+"CT/"+file)
            output_file_ft.append("image/"+self.ui.lineEdit.text()+"/"+"FT/"+file)
            output_file_mn.append("image/"+self.ui.lineEdit.text()+"/"+"MN/"+file)
        
        img_size = (512, 512)
        test1= test_predict(20,img_size,input_files,output_file_ct)
        model1 =keras.models.load_model('model/ct1.h5')
        predct=model1.predict(test1)
        
        model2= keras.models.load_model('model/mn.h5')
        predmn=model2.predict(test1)

        model3= keras.models.load_model('model/ft1.h5')
        predft=model3.predict(test1)

        avg_ct=0
        avg_ft=0
        avg_mn=0
        for i in range(20):
            true_mask_ct = PIL.Image.open(output_file_ct[i])
            true_mask_ft = PIL.Image.open(output_file_ft[i])
            true_mask_mn = PIL.Image.open(output_file_mn[i])
            avg_ct=avg_ct+dice(true_mask_ct,return_mask(predct[i]))
            avg_ft=avg_ft+dice(true_mask_ft,return_mask(predft[i]))
            avg_mn=avg_mn+dice(true_mask_mn,return_mask(predmn[i]))
        print(avg_ct/20)
        print(avg_ft/20)
        print(avg_mn/20)

        self.ui.label_9.setText(str(avg_mn/20))
        self.ui.label_10.setText(str(avg_ft/20))
        self.ui.label_11.setText(str(avg_ct/20))

        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())