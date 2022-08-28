import os
import sys
import json
import random
import threading
import qtawesome
import webbrowser
from PyQt5.uic import loadUi
from PyQt5.Qt import *
from PyQt5.QtGui import *
from radio_engine1 import Radio_Spider
from radio import Ui_MainWindow


class My_Radio(QMainWindow):

    def __init__(self):
        super().__init__()
        # self.ui=loadUi("radio.ui",self)
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        # self.setFixedSize(self.width(),self.height())
        self.setWindowTitle("My_Radio")
        self.init()
        self.setWindowIcon(QIcon(qtawesome.icon('mdi.radio', color=self.theme_color)))

    def init(self):
        self.theme_color="brown"
        self.__isonline_flag=True
        self.__provence_item={'北京': '3', '天津': '5', '河北': '7', '上海': '83', '山西': '19', '内蒙古': '31', '辽宁': '44', '吉林': '59', '黑龙江': '69', '江苏': '85', '浙江': '99', '安徽': '111', '福建': '129', '江西': '139', '山东': '151', '河南': '169', '湖北': '187', '湖南': '202', '广东': '217', '广西': '239', '海南': '254', '重庆': '257', '四川': '259', '贵州': '281', '云南': '291', '陕西': '316', '甘肃': '327', '宁夏': '351', '新疆': '357', '西藏': '308', '青海': '342'}
        self.__current_radio_id=""
        self.__current_radio_name=""
        self.__collection_item={}
        self.__current_radio_station_items={}
        self.__current_radio_page_no=1
        self.__current_cate_id=0
        self.__max_page_num=0
        self.__random_recommend_list=[]
        # self.__author_qq = "tencent://message/?uin=1397852386&Site=Senlon.Net&Menu=yes"
        self.e=Radio_Spider()
        self.ui_init()
        self.player_init()
        self.slot_init()
        self.checknetwork()

    def ui_init(self):
        # self.ui.tableWidget.setColumnHidden(3, True)  # 隐藏第3列，电台ID列
        # self.about_ui=about_author_window()
        self.ui.collection_listWidget.hide()

    def slot_init(self):
        self.ui.country_cate_toolButton.clicked.connect(lambda :self.get_radios("country"))
        self.ui.network_cate_toolButton.clicked.connect(lambda :self.get_radios("network"))
        self.ui.city_cate_toolButton.clicked.connect(self.do_select_provence)
        self.ui.recommend_toolButton.clicked.connect(lambda :self.show_random_recommend_list(change=False))
        self.ui.about_sw_toolButton.clicked.connect(lambda :QMessageBox.information(self,"关于软件","本软件为一个在线收音机，依赖互联网。"))
        self.ui.toolButton_7.clicked.connect(self.about_ui.show)
        self.ui.connect_pushButton.clicked.connect(self.connect_to_author)
        self.ui.add_toolButton.clicked.connect(self.add_a_audio)
        self.ui.share_toolButton.clicked.connect(self.do_share_station)
        self.ui.toolButton_21.clicked.connect(lambda :self.do_turn_page("pre",max=True))
        self.ui.toolButton_19.clicked.connect(lambda :self.do_turn_page("pre"))
        self.ui.toolButton_20.clicked.connect(lambda :self.do_turn_page("next"))
        self.ui.toolButton_22.clicked.connect(lambda :self.do_turn_page("next",max=True))
        self.ui.abou_qt_pushButton.clicked.connect(lambda :QMessageBox.aboutQt(self,))
        self.ui.collection_toolButton.clicked.connect(self.show_collections)
        self.ui.do_collection_toolButton.clicked.connect(self.do_collect_radio)
        self.ui.control_toolButton.clicked.connect(self.do_process_pause_and_play)
        self.ui.change_toolButton.clicked.connect(lambda :self.show_random_recommend_list(change=True))
        self.ui.volum_Slider.valueChanged[int].connect(lambda v :self.do_change_player_volume(v))
        self.ui.collection_listWidget.itemDoubleClicked.connect(lambda i:self.do_play_collection(i))

    def player_init(self):
        self.__curPos = ''
        self.__dur = ''
        self.__isonline_flag=True
        self.player = QMediaPlayer(self)
        self.playlist = QMediaPlaylist(self)
        self.player.setPlaylist(self.playlist)
        self.player.durationChanged.connect(self.get_duration_func)#播放进度改变
        self.player.positionChanged.connect(self.get_position_func)#播放位置改版
        self.player.mediaStatusChanged.connect(self.on_mediaStatusChanged)
        self.player.bufferStatusChanged.connect(self.on_bufferStatusChanged)
        self.ui.progress_slider.sliderMoved.connect(self.update_position)

    def do_select_provence(self):
        """
        选择省份
        :return:
        """
        provences=self.__provence_item.keys()
        select,_=QInputDialog.getItem(self, "请选择","请选择一个省份", provences, 0, False)
        if _:
            selected_radio_id=self.__provence_item[select]
            radio_items=self.e.get_city_radio_item(int(selected_radio_id),self.__current_radio_page_no)
            self.__current_radio_page_no=1
            self.__current_cate_id=selected_radio_id
            mid_num=int(radio_items['count']/12)
            self.__max_page_num=mid_num
            self.ui.page_label.setText("""<html><head/><body><p>第<span style=" color:#55aaff;">{current_num}</span>页共<span style=" color:#55aaff;">{max_num}</span>页</p></body></html>""".replace("{current_num}",str(self.__current_radio_page_no)).replace("{max_num}",str(self.__max_page_num)))
            self.show_radio_stations(radio_items['items'])

    def do_collect_radio(self):
        """
        收藏音频
        :return:
        """
        if self.playlist.isEmpty():
            return
        else:
            if not self.new_id_is_in_all(self.__current_radio_id):
                #加入收藏
                self.__collection_item[self.__current_radio_id]=self.__current_radio_name
                if self.__current_radio_id==0:
                    return
                QMessageBox.information(self,"提示",'电台收藏成功！')
            else:
                QMessageBox.warning(self,"警告","此电台已收藏，请勿重复收藏！")
        if self.ui.frame_5.isVisible():
            self.show_collections()

    def show_collections(self):
        """
        我的收藏
        :return:
        """
        self.ui.scrollArea.hide()
        self.ui.collection_listWidget.clear()
        for i in self.__collection_item.values():
            self.ui.collection_listWidget.addItem(QListWidgetItem(i))
        self.ui.frame_5.show()
        self.ui.collection_listWidget.show()

    def new_id_is_in_all(self,new_id):
        """
        判断新ID是否在所有id中
        :param new_id:
        :return:
        """
        ids = self.__collection_item.keys()
        if new_id  in ids:
            return True
        else:
            return False

    @pyqtSlot()
    def do_play_collection(self,item):
        """
        播放收藏的电台
        :param item:
        :return:
        """
        radio_name=item.text()
        #列表反转
        item_reverse = {k: v for v, k in self.__collection_item.items()}
        radio_id=item_reverse[radio_name]
        self.ui.radio_name_label.setText(radio_name)
        self.play_online_audio(self.e.get_radio_play_mp3_link(radio_id))

    def add_a_audio(self):
        """
        添加一个mp3或者m4a格式音频到收藏
        :return:
        """
        audio_link,_=QInputDialog.getText(self,"请输入","请输入音频地址，最好为mp3或m4a格式")
        if _:
            if audio_link.startswith("http:") or audio_link.startswith("https:"):
                self.play_online_audio(audio_link)
                self.ui.radio_name_label.setText("手动添加音频")
            else:
                QMessageBox.critical(self,"错误","输入不合法，请检查！")

    def get_radios(self,type):
        """
        获取电台列表
        :return:
        """
        if type=="country":
            self.__current_cate_id=409
            item_=self.e.get_country_radio_item(self.__current_radio_page_no)
        elif type=="network":
            self.__current_cate_id=407
            item_=self.e.get_network_radio_item(self.__current_radio_page_no)
        if item_:
            items=item_['items']
            self.__current_radio_page_no=1
            self.ui.toolButton_20.setEnabled(True)
            self.ui.toolButton_19.setEnabled(False)
            mid_num = int(item_['count'] / 12)
            self.__max_page_num = mid_num
            self.ui.page_label.setText("""<html><head/><body><p>第<span style=" color:#55aaff;">{current_num}</span>页共<span style=" color:#55aaff;">{max_num}</span>页</p></body></html>""".replace("{current_num}",str(self.__current_radio_page_no)).replace("{max_num}",str(self.__max_page_num)))
            self.show_radio_stations(items)
        else:
            QMessageBox.critical(self,"错误","获取电台失败！")

    def show_radio_stations(self,items):
        """
        将电台展示到toolbutton中
        :param items:
        :return:
        """
        self.__current_radio_station_items=items
        item_num=len(items)
        for index, item in enumerate(items):
            radio_id = item.get("id")
            loca_file_path = self.e.get_radio_local_img(radio_id)
            if loca_file_path:
                pix = QPixmap(loca_file_path)
                exec("self.ui.toolButton_11{}.setIcon(QIcon(pix))".format(index + 1))
            else:
                # 加载网络上的图片，拉取到本地
                imgUrl = "https:"+item.get("imgUrl")
                self.show_radio_station_imgs(imgUrl, index)
                self.thread_it(self.e.download_imgs,radio_id,imgUrl)
            radio_title = item.get('title')
            radio_now = item.get('desc')
            exec("self.ui.toolButton_11{}.setText(radio_title)".format(index + 1))
            exec("self.ui.toolButton_11{}.setToolTip(radio_now)".format(index + 1))
            try:
                exec("self.ui.toolButton_11{}.clicked.disconnect()")
            except:
                pass
            symbols = {"self": self}
            exec("self.ui.toolButton_11{}.clicked.connect(lambda:self.on_toolButton_clicked({}))".format(index + 1,item),symbols)
            exec("self.ui.toolButton_11{}.show()".format(index + 1))#按钮显示
        if item_num!=12:
            #对于电台数量少于12个的进行隐藏
            for i in range(item_num,12):
                exec("self.ui.toolButton_11{}.hide()".format(i + 1))#按钮隐藏
        self.ui.scrollArea.show()
        self.ui.frame_5.hide()
        self.do_control_turn_page(hide=False)


    def show_radio_station_imgs(self,url,index):
        """
        加载网络上的图片到toolbutton
        :param url:
        :param index:
        :return:
        """
        img_bytes = self.e.do_get_img_bytes(url)
        # 这里最好设置一个默认值
        if img_bytes:
            pix = QPixmap()
            pix.loadFromData(img_bytes)
            exec("self.ui.toolButton_11{}.setIcon(QIcon(pix))".format(index + 1))
        else:
            #加载网络上的图片失败，则使用默认图片
            pass

    def do_turn_page(self,direction,max=False):
        """
        处理翻页
        :param direction:
        :param max:
        :return:
        """
        if max:
            if direction=="pre":
                self.__current_radio_page_no=1
                self.ui.toolButton_19.setEnabled(False)
                self.ui.toolButton_21.setEnabled(False)
            elif direction=="next":
                self.__current_radio_page_no=self.__max_page_num
                self.ui.toolButton_20.setEnabled(False)
                self.ui.toolButton_19.setEnabled(True)
                self.ui.toolButton_21.setEnabled(True)
        else:
            if direction=="pre":
                if self.__current_radio_page_no-1<=1:
                    self.ui.toolButton_19.setEnabled(False)
                self.__current_radio_page_no-=1
                self.ui.toolButton_19.setEnabled(True)
                self.ui.toolButton_21.setEnabled(True)
            elif direction=="next":
                if self.__current_radio_page_no+1>=self.__max_page_num:
                    self.ui.toolButton_20.setEnabled(False)
                    self.ui.toolButton_19.setEnabled(True)
                    self.ui.toolButton_21.setEnabled(True)
                self.__current_radio_page_no+=1
        radio_item=self.e.get_city_radio_item(self.__current_cate_id,self.__current_radio_page_no)
        if radio_item:
            self.ui.page_label.setText(
                """<html><head/><body><p>第<span style=" color:#55aaff;">{current_num}</span>页共<span style=" color:#55aaff;">{max_num}</span>页</p></body></html>""".replace(
                    "{current_num}", str(self.__current_radio_page_no)).replace("{max_num}",str(self.__max_page_num)))
            self.show_radio_stations(radio_item['items'])

    def get_duration_func(self, d):
        """
        :param d: 播放进度改变
        :return:
        """
        secs = d / 1000
        mins = secs / 60
        secs = secs % 60
        self.__dur = "%d:%d" % (mins, secs)
        self.ui.time_label.setText("{}/{}".format(self.__curPos, self.__dur))
        self.ui.progress_slider.setRange(0, d)
        self.ui.progress_slider.setEnabled(True)

    def do_share_station(self):
        """
        分享电台
        :return:
        """
        real_link=self.e.get_radio_play_mp3_link(self.__current_radio_id)
        clipboard = QApplication.clipboard()
        clipboard.setText(real_link)
        QMessageBox.information(self, '提示', '电台链接已复制到剪切板！')

    def connect_to_author(self):
        ret = QMessageBox.question(self, "联系作者", '是否要与作者进行QQ交谈？', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            webbrowser.open(self.__author_qq)

    def get_position_func(self, p):
        """
        播放位置改变，用此函数控制歌词显示
        :param p:
        :return:
        """
        if (self.ui.progress_slider.isSliderDown()):
            return
        self.ui.progress_slider.setValue(p)
        secs = p / 1000
        mins = secs / 60
        secs = secs % 60
        self.__curPos = "%d:%d" % (mins, secs)
        self.ui.time_label.setText("{}/{}".format(self.__curPos, self.__dur))

    def on_mediaStatusChanged(self,s):
        """
        媒体状态改变触发
        :param s:
        :return:
        """
        if s==QMediaPlayer.BufferingMedia:
            self.ui.radio_name_label.setText("正在缓冲~~~")
        elif s==QMediaPlayer.EndOfMedia:
            self.ui.radio_name_label.setText("播放完成~~~")
        elif s==QMediaPlayer.InvalidMedia:
            QMessageBox.warning(self,"警告","非法音频！")

    @pyqtSlot(int)
    def on_bufferStatusChanged(self,percent):
        """
        音频缓冲事件
        :param percent:
        :return:
        """
        self.ui.radio_name_label.setText(f"正在缓冲({percent}%)")

    def update_position(self, v):
        """
        播放进度条被改变
        :param v:
        :return:
        """
        self.player.setPosition(v)
        secs = v / 1000
        mins = secs / 60
        secs = secs % 60
        self.__curPos = "%d:%d" % (mins, secs)
        self.ui.time_label.setText("{}/{}".format(self.__curPos, self.__dur))

    def on_toolButton_clicked(self,item):
        """
        点击toolButton触发
        :return:
        """
        self.ui.control_toolButton.setText("暂停")
        radio_name=item.get('title')
        radio_play_now=item.get('desc').replace("正在直播： ","")
        radio_id=item.get('id')
        self.ui.radio_name_label.setText(radio_name+"-"+radio_play_now)
        self.ui.radio_name_label.setToolTip(radio_name+"-"+radio_play_now)
        real_mp3_link=self.e.get_radio_play_mp3_link(radio_id)
        self.__current_radio_id=int(radio_id)
        self.__current_radio_name=radio_name
        self.play_online_audio(real_mp3_link)

    def play_online_audio(self,link):
        """
        播放在线音频 不限于电台
        :param link:
        :return:
        """
        self.playlist.clear()
        song = QMediaContent(QUrl(link))
        self.playlist.addMedia(song)
        self.player.play()

    def get_recommend_radio_stations(self):
        """
        随即推荐电台
        :return:
        """
        rando_city_item=self.e.get_city_radio_item(int(random.choice(list(self.__provence_item.values()))),1)
        all_list=self.e.get_country_and_network_radio_statios()
        if rando_city_item:
            all_list.append(rando_city_item.get("count"))
        all_combine_list=[]
        try:
            for i in all_list:
                for j in i:
                    all_combine_list.append(j)
        except:
            pass
        self.all_combine_list=all_combine_list
        random_list=random.sample(self.all_combine_list,12)
        self.__random_recommend_list=random_list
        self.show_radio_stations(self.__random_recommend_list)
        self.do_control_turn_page()

    def show_random_recommend_list(self,change=False):
        """
        此函数加上get_recommend_radio_stations地返回值，可以刷新推荐
        :return:
        """
        if change:
            random_list=random.sample(self.all_combine_list,12)
            try:
                exec("self.ui.toolButton_11{}.clicked.disconnect()")
            except:
                pass
        else:
            random_list=self.__random_recommend_list
        self.__random_recommend_list=random_list
        self.show_radio_stations(self.__random_recommend_list)
        self.do_control_turn_page()

    def do_control_turn_page(self,hide=True):
        """
        控制是否显示分页
        :param hide:
        :return:
        """
        if hide:
            self.ui.toolButton_19.hide()
            self.ui.toolButton_20.hide()
            self.ui.toolButton_21.hide()
            self.ui.toolButton_22.hide()
            self.ui.page_label.hide()
            self.ui.label_3.show()
            self.ui.change_toolButton.show()
        else:
            self.ui.toolButton_19.show()
            self.ui.toolButton_20.show()
            self.ui.toolButton_21.show()
            self.ui.toolButton_22.show()
            self.ui.page_label.show()
            self.ui.label_3.hide()
            self.ui.toolButton_21.setEnabled(False)
            self.ui.toolButton_19.setEnabled(False)
            self.ui.toolButton_20.setEnabled(True)
            self.ui.toolButton_22.setEnabled(True)
            self.ui.change_toolButton.hide()

    def do_process_pause_and_play(self):
        """
        处理播放/暂停
        :return:
        """
        state=self.player.state()
        if state==1:
            self.player.pause()
            self.ui.control_toolButton.setText("播放")
        elif state==2:
            self.player.play()
            self.ui.control_toolButton.setText("暂停")

    def do_change_player_volume(self,v):
        """
        设置当前音量
        :param v:
        :return:
        """
        self.player.setVolume(v)
        self.ui.volume_label.setText(f"{v}%")

    def checknetwork(self):
        """
        检测当前网络状态
        :return:
        """
        if self.e.test_net_condition():
            self.__isonline_flag=True
            self.check_is_first_start()
            return
        else:
            box=QMessageBox(QMessageBox.Warning,"警告","您未接入互联网")
            box.setStandardButtons(QMessageBox.Retry|QMessageBox.Cancel)
            btnr=box.button(QMessageBox.Retry)
            btnr.setText("再次尝试一下")
            btnn=box.button(QMessageBox.Cancel)
            btnn.setText("使用离线版")
            box.exec_()
            if box.clickedButton()==btnr:
                self.checknetwork()
            else:
                self.__isonline_flag=False
                return

    def check_is_first_start(self):
        """
        如果第一次启动
        加载电台图片到本地
        :return:
        """
        aim_path="./radio_cache"
        if os.path.exists(aim_path):
            if len(os.listdir(aim_path))>0:
                #加载推荐
                self.do_control_turn_page()
                self.get_recommend_radio_stations()
                return
        all_list=self.e.get_country_and_network_radio_statios()
        all_item=self.e.parse_all_list(all_list)
        self.thread_it(self.e.pull_radio_station_imgs,all_item)

    def thread_it(self, func, *args):
        t = threading.Thread(target=func, args=args)
        t.setDaemon(True)
        t.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = My_Radio()
    ui.show()
    sys.exit(app.exec_())

