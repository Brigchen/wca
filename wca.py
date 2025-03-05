''' copyright@brigchen'''
# update: 2025-3-4

import sys, os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QScrollArea, QGroupBox, QFrame, QComboBox, QListWidget, QTableWidget, QTableWidgetItem,QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QFont, QCursor, QMouseEvent, QIcon
from PyQt5.QtCore import pyqtSignal, Qt, QSize
import pandas as pd
# import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#%%
# import glob
import re

def check_chinese(path):
    # 使用正则表达式匹配中文符号
    chinese_regex = r'[\u4e00-\u9fff\u3000-\u303f\ufe30-\ufe4f\ufe10-\ufe1f\uff00-\uffef]'
    return re.search(chinese_regex, path) is not None

#%%
class ClickableLabel(QLabel):
    # 自定义点击信号
    clicked = pyqtSignal(QMouseEvent)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_clickable = True  # 初始化为可点击状态
        self.setCursor(QCursor(Qt.PointingHandCursor))  # 设置鼠标指针样式为手型

    def set_clickable(self, clickable):
        # 设置是否可点击
        self._is_clickable = clickable
        if clickable:
            self.setCursor(QCursor(Qt.PointingHandCursor))  # 可点击时鼠标指针为手型
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))  # 不可点击时鼠标指针为箭头

    def mousePressEvent(self, event):
        # 鼠标点击事件处理
        if self._is_clickable:
            self.clicked.emit(event)  # 发出点击信号
        
#%%

class ImageColorAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_file = None
        self.image = None
        self.original_image = None
        self.points = []
        self.color_list = []
        self.circle_center = None
        self.current_folder = None
        self.image_files = []
        self.current_index = 0
        self.folder_paths = []

    def initUI(self):
        self.setObjectName("Water Color Analyzer")
        self.setWindowTitle('WCA: GUI for Water Color Analyzer')
        # self.showMaximized()

        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font_20 = QFont()
        font_20.setPointSize(20)
        font_15 = QFont()
        font_15.setPointSize(15)

        self.setFont(font)
        # 创建主布局
        main_layout = QVBoxLayout()

        # 上部分布局
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        # 上部分左边的文件管理区
        self.file_area = QFrame(self)
        file_layout = QVBoxLayout()

        self.groupBox_file = QGroupBox(self.file_area)
        self.groupBox_file.setTitle("Select Image")
        self.groupBox_file.setObjectName("file")

        vlayout_file = QHBoxLayout(self.groupBox_file)

        self.file_button = QPushButton('Open Images', self)
        self.file_button.clicked.connect(self.open_images)
        vlayout_file.addWidget(self.file_button, 1)

        self.comboBox_file = QComboBox(self.groupBox_file)
        self.comboBox_file.setObjectName("file_cb")
        # self.comboBox_file.currentIndexChanged.connect(self.renew_folder)
        self.comboBox_file.activated.connect(self.renew_folder)
        vlayout_file.addWidget(self.comboBox_file, 5)

        self.image_list = QListWidget(self.file_area)
        self.image_list.itemClicked.connect(self.get_image_path)

        self.exe_button = QPushButton('Analyze', self)
        self.exe_button.clicked.connect(self.analyze_image)

        file_layout.addWidget(self.groupBox_file, 1)
        file_layout.addWidget(self.image_list, 3)
        file_layout.addWidget(self.exe_button, 1)

        self.file_area.setLayout(file_layout)

        # 上部分右边的文字显示区
        self.text_area = QFrame(self)
        text_layout = QVBoxLayout()
        self.text_area.setLayout(text_layout)

        self.groupBox_text = QGroupBox(self)
        self.groupBox_text.setTitle("Hints")
        self.groupBox_text.setObjectName("text")
        text_layout.addWidget(self.groupBox_text, 1)
        vlayout_text = QVBoxLayout(self.groupBox_text)

        self.groupBox_result = QGroupBox(self.text_area)
        self.groupBox_result.setTitle("")
        self.groupBox_result.setObjectName("result")
        text_layout.addWidget(self.groupBox_result, 1)
        vlayout_result = QVBoxLayout(self.groupBox_result)

        self.logger_label = QLabel(self)
        self.logger_label.setAlignment(Qt.AlignLeft)
        self.logger_label.setText('Welcome use WCA for watercolor measurement! \n1. Load directory with images of waters with watercolor standard meter by “Open Images”. \n2.Select an image, and run Analyze. \n3.Select points of watercolor meter and center of ROI. \n4. Export measured values to excel.')
        vlayout_text.addWidget(self.logger_label, 1)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setText('')
        self.result_label.setFont(font_15)
        vlayout_result.addWidget(self.result_label, 1)

        top_layout.addWidget(self.file_area, 1)
        top_layout.addWidget(self.text_area, 3)

        # 下部分布局
        bottom_layout = QHBoxLayout()

        # 下部分左边的图像显示区
        self.groupBox_image = QGroupBox(self)
        self.groupBox_image.setTitle("Image Visualize")
        self.groupBox_image.setObjectName("image_box")
        vlayout_image = QVBoxLayout(self.groupBox_image)

        self.image_label = ClickableLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.clicked.connect(self.on_select_points)
        self.image_label.set_clickable(False)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        
        # 导航按钮
        self.frame_nav = QFrame(self.groupBox_image)
        nav_layout = QHBoxLayout()
        self.frame_nav.setLayout(nav_layout)
        self.prev_button = QPushButton('Last Image', self)
        self.prev_button.clicked.connect(self.show_prev_image)
        nav_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton('Next Image', self)
        self.next_button.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.next_button)
        # main_layout.addLayout(nav_layout)
                
        vlayout_image.addWidget(scroll_area)
        vlayout_image.addWidget(self.frame_nav)

        # 下部分右边的布局
        self.right_frame = QFrame(self)
        right_layout = QVBoxLayout()
        self.right_frame.setLayout(right_layout)

        # 下部分右边上面的图像显示区
        self.groupBox_color = QGroupBox(self)
        self.groupBox_color.setTitle("Color Visualize")
        self.groupBox_color.setObjectName("color_box")
        vlayout_color = QVBoxLayout(self.groupBox_color)

        self.color_image_label = QLabel(self.right_frame)
        self.color_image_label.setAlignment(Qt.AlignCenter)
        color_image_scroll = QScrollArea()
        color_image_scroll.setWidget(self.color_image_label)
        color_image_scroll.setWidgetResizable(True)
        vlayout_color.addWidget(color_image_scroll)

        # 列表显示区
        self.groupBox_list = QGroupBox(self)
        self.groupBox_list.setTitle("Data List")
        self.groupBox_list.setObjectName("list_box")
        vlayout_list = QVBoxLayout(self.groupBox_list)

        self.table_widget = QTableWidget(self.right_frame)
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(['Sample_Name', 'Watercolor Value'])
        self.adjust_table_column_widths()
        
        # 列表数据输出
        self.export_button = QPushButton('Export Data', self.right_frame)
        self.export_button.clicked.connect(self.export_to_excel)

        vlayout_list.addWidget(self.table_widget)
        vlayout_list.addWidget(self.export_button)

        right_layout.addWidget(self.groupBox_color, 1)
        right_layout.addWidget(self.groupBox_list, 1)

        bottom_layout.addWidget(self.groupBox_image, 3)
        bottom_layout.addWidget(self.right_frame, 1)

        # 将上部分和下部分布局添加到主布局
        main_layout.addLayout(top_layout, 1)
        main_layout.addLayout(bottom_layout, 3)

        self.setLayout(main_layout)
        # self.setWindowTitle('水色分析')
        self.showMaximized()

    def resizeEvent(self, event):
        # 窗口大小改变时调整列宽
        self.adjust_table_column_widths()
        super().resizeEvent(event)


    def adjust_table_column_widths(self):
        table_width = self.table_widget.width()
        column_width = table_width // 2
        for col in range(self.table_widget.columnCount()):
            self.table_widget.setColumnWidth(col, column_width)

    def open_images(self):
        if self.table_widget.rowCount() > 0:
            reply = QMessageBox.question(self, '提示', '表格中已有数据，打开新文件夹数据将丢失，是否继续打开新文件列表？',
                                 QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return  # 用户选择取消，不做任何操作
   
        folder_path = QFileDialog.getExistingDirectory(self, '选择待分析图像所在文件夹')
        if folder_path:
            # 新增：记录当前文件夹路径
            self.current_folder = folder_path
            # 获取文件夹名
            folder_name = os.path.basename(folder_path)
            
            if folder_path not in self.folder_paths:
                self.folder_paths.append(folder_path)
                # 将文件夹名添加到 QComboBox 中
                self.comboBox_file.addItem(folder_name)
            else:
                index = self.comboBox_file.findText(folder_name)
                # 更新文件列表
                self.comboBox_file.setCurrentIndex(index)
            self.update_image_list()
    
    def update_image_list(self):
        self.image_list.clear()
        # 获取文件夹下的所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.tif', '.bmp']
        self.table_widget.setRowCount(0)
        for root, dirs, files in os.walk(self.current_folder):
            print()
            for i, file in enumerate(files):
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension in image_extensions:
                    self.image_list.addItem(file)
                    self.image_files.append(os.path.join(self.current_folder, file))
                    file_name_without_ext = os.path.splitext(file)[0]
                    # 在表格中添加新行
                    row_position = self.table_widget.rowCount()
                    print(256, 'i=', i, 'n=', row_position)
                    self.table_widget.insertRow(row_position)
                    self.table_widget.setItem(row_position, 0, QTableWidgetItem(file_name_without_ext))     
        self.adjust_table_column_widths()
        self.logger_label.setText(f'{len(self.image_files)} images in {self.current_folder}')

    def get_image_path(self, item):
        # file_name = item.text()
        index = self.image_list.row(item)
        self.current_index = index
        # if self.current_folder:
        self.image_file = self.image_files[index]
        self.logger_label.setText(f"image path: {self.image_file}")
        if not check_chinese(self.image_file):
            self.show_image(self.image_file)
        else:
            self.logger_label.setText(f'novel character or chinese in path: {self.image_file}, please rename it.')

    def analyze_image(self):
        if self.image_file:
            print('--------------analyzing image-------------')
            self.logger_label.setText(r'依次选择水色标准板的四个顶角，按左上角、右上角、右下角、左下角的顺序。')   
            self.points=[]
            self.circle_center = None
            self.image_label.set_clickable(True)

    def show_image(self, image_path):
        self.image = cv2.imread(self.image_file)
        self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
        self.original_image = self.image.copy()
        self.update_image()
        
    def update_image(self):
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))#, Qt.KeepAspectRatio))
    
    def show_prev_image(self):
        if self.image_files:

            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.logger_label.setText(f'{self.current_index}')
            self.image_file = self.image_files[self.current_index]
            self.logger_label.setText(f"image path: {self.image_file}")
            if not check_chinese(self.image_file):
                self.show_image(self.image_file)
                self.image_list.setCurrentRow(self.current_index)
            else:
                self.logger_label.setText(f'novel character or chinese in path: {self.image_file}, please rename it.')
                
    def show_next_image(self):
        if self.image_files:
            # self.logger_label.setText(f'next image')
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.logger_label.setText(f'{self.current_index}')
            self.image_file = self.image_files[self.current_index]
            self.logger_label.setText(f"image path: {self.image_file}")
            if not check_chinese(self.image_file):
                self.show_image(self.image_file)
                self.image_list.setCurrentRow(self.current_index)
            else:
                self.logger_label.setText(f'novel character or chinese in path: {self.image_file}, please rename it.')

    def on_select_points(self, event):
        # if event.button() == Qt.LeftButton:
        x = event.pos().x()
        y = event.pos().y()
        print(f'选点：{x}，{y}')
        scale_x = self.image.shape[1] / self.image_label.pixmap().width()
        scale_y = self.image.shape[0] / self.image_label.pixmap().height()
        real_x = int(x * scale_x)
        real_y = int(y * scale_y)
        self.points.append((real_x, real_y))
        
        if len(self.points) < 5:
            # 将绘制圆点的颜色改为红色
            cv2.circle(self.image, (real_x, real_y), 20, (255, 0, 0), -1)
            self.update_image()
            
        if len(self.points) == 4:
            self.transform_image()
            print('--------------select water point to measure-------------')
            self.logger_label.setText(r'依次选择待测量的位置中心点（透明度盘）。')
            
        if len(self.points) == 5:
            self.circle_center = self.points[4]
            cv2.circle(self.image, self.circle_center, 50, (255, 0, 0), 5)
            self.update_image()
            self.measure_circle_color()
            self.image_label.set_clickable(False)


    def transform_image(self):
        pts1 = np.float32(self.points)
        pts2 = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_image = cv2.warpPerspective(self.original_image, matrix, (400, 400))

        # 预定的 20 个位置
        positions = [
            (52, 100), (84, 100), (57*2, 100), (73*2, 100), (89*2, 100), (105*2, 100), 
            (241, 100), (136*2, 100), (304, 100), (335, 100),(365, 100),
            (17*2, 300), (33*2, 300), (49*2, 300), (129, 300), (161, 300), (96*2, 300), 
            (112*2, 300), (127*2, 300), (285, 300), (317, 300),(174*2, 300),
        ]

        self.color_list = []
        for pos in positions:
            color = self.get_median_color(pos[0], pos[1],transformed_image)
            self.color_list.append(color)

        self.show_color_image(transformed_image, positions)
        # for i, color in enumerate(self.color_list):
        #     print(i,':', color)
        # self.image_label.mousePressEvent = 

    def show_color_image(self, img, positions):
        # 绘制空心方块和编号
        for i, pos in enumerate(positions):
            x, y = pos
            length = 6
            width = 100
            x1 = int(x - length // 2)
            y1 = int(y - width // 2)
            x2 = int(x + length // 2)
            y2 = int(y + width // 2)
            cv2.rectangle(img, (x1,y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(img, str(i + 1), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888) 
        pixmap = QPixmap.fromImage(q_img)
        self.color_image_label.setPixmap(pixmap.scaled(self.color_image_label.size()))#, Qt.KeepAspectRatio))

    def get_median_color(self, x, y, img):
        
        # radius = 3
        length = 6
        width = 50
        x1 = int(x - length // 2)
        y1 = int(y - width // 2)
        x2 = int(x + length // 2)
        y2 = int(y + width // 2)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        # print(x1,y1, x2, y2)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        pixels = img_lab[np.where(mask == 255)]
        median_color = np.median(pixels, axis=0)
        return median_color

    #%% RGB-->LAB, E
    def measure_circle_color(self):
        x, y = self.circle_center
        radius = 10
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        pixels = img[np.where(mask == 255)]
        median_color = np.median(pixels, axis=0)

        min_distance = float('inf')
        selected_index = -1
        for i, color in enumerate(self.color_list):
            distance = np.linalg.norm(np.array(color) - np.array(median_color))
            if distance < min_distance:
                min_distance = distance
                selected_index = i
        
        self.result_label.setText(f"最接近的颜色编号是: {selected_index + 1}\n测量值{median_color}\n{selected_index + 1}号标准值:{self.color_list[selected_index]}")
        row_position = self.image_list.currentRow()
        self.table_widget.setItem(row_position, 1, QTableWidgetItem(str(selected_index + 1))) 
        self.logger_label.setText('watercolor value measured.')

    def renew_folder(self):
        print(f'cb_file: {self.comboBox_file.count()}')
        idx = self.comboBox_file.currentIndex()
        if self.folder_paths[idx] == self.current_folder or self.comboBox_file.count() < 2:
            return
        
        self.current_folder = self.folder_paths[idx]
        self.logger_label.setText("Change image folder to: %s"%self.current_folder)
        self.update_image_list()
        
        
    def export_to_excel(self):
        if self.current_folder:
            data = []
            for row in range(self.table_widget.rowCount()):
                file_name = self.table_widget.item(row, 0).text()
                wc_value = int(self.table_widget.item(row, 1).text())
                data.append([file_name, wc_value])

            df = pd.DataFrame(data, columns=['Sample_Name', 'Watercolor Value'])

            file_path, _ = QFileDialog.getSaveFileName(self, '保存 Excel 文件', '', 'Excel 文件 (*.xlsx)')
            if file_path:
                df.to_excel(file_path, index=False)
                self.logger_label.setText(f'{file_path} saved.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    analyzer = ImageColorAnalyzer()
    analyzer.setWindowIcon(QIcon('./wca_l.ico'))
    sys.exit(app.exec_())