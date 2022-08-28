import json
import os
import random
from lxml import etree
import  requests

class Radio_Spider(object):

    def test_net_condition(self):
        """
        判断本机是否连接到互联网
        :return:
        """
        try:
            requests.get("http://www.baidu.com")
            return True
        except:
            return False

    def get_random_ua(self):
        """
        获取随机UA
        :return:
        """
        first_num = random.randint(55, 62)
        third_num = random.randint(0, 3200)
        fourth_num = random.randint(0, 140)
        os_type = [
            '(Windows NT 6.1; WOW64)', '(Windows NT 10.0; WOW64)', '(X11; Linux x86_64)',
            '(Macintosh; Intel Mac OS X 10_12_6)'
        ]
        chrome_version = 'Chrome/{}.0.{}.{}'.format(first_num, third_num, fourth_num)

        ua = ' '.join(['Mozilla/5.0', random.choice(os_type), 'AppleWebKit/537.36',
                       '(KHTML, like Gecko)', chrome_version, 'Safari/537.36']
                      )
        return ua

    def get_radio_local_img(self,radio_id):
        """
        根据id获取本地缓存电台图片
        提高程序运行速度
        :param radio_id:
        :return:
        """
        aim_path="./radio_cache"
        if os.path.exists(aim_path):
            for file in os.listdir(aim_path) :
                file_abs=os.path.abspath(os.path.join(aim_path,file))
                if os.path.basename(file_abs).split(".")[0]==str(radio_id):
                    return file_abs
        return False

    def do_get_request(self,url,):
        """
        发送Get请求
        :param url:
        :return:
        """
        headers={
            "host":"www.qingting.fm",
            "User-Agent":self.get_random_ua()}
        try:
            r=requests.get(url,headers=headers)
            if r.status_code==200:
                r.encoding="utf-8"
                return r.text

            else:
                return False
        except:
            return False

    def do_get_img_bytes(self,url):
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
                 }
        try:
            r=requests.get(url,headers=headers)
            if r.status_code==200:
                return r.content
            else:
                return  False
        except:
            return False

    def do_post_request(self,url,data):
        """
        发送Post请求
        :param url:
        :param data:
        :return:
        """
        headers={
            "host":"webbff.qingting.fm",
            "User-Agent":self.get_random_ua(),
        "Referer": "https://www.qingting.fm/",
            "Origin": "https://www.qingting.fm"
        }
        try:
            r=requests.post(url,data=data,headers=headers)
            if r.status_code==200:
                r.encoding="utf-8"
                return r.text
            else:
                return False
        except:
            return False

    def get_radio_big_cates(self):
        """
        根据城市获取电台大分类信息
        :return:
        """
        base_url="https://www.qingting.fm/radiopage"
        text=self.do_get_request(base_url)
        res=etree.HTML(text)
        if res is not None:
            selector=res.xpath('//div[@class="catSec"]//div[@class="regionsSec regionsSecHide"]/a')
            radio_city_item={}
            for i in selector:
                radio_id=i.xpath('./@id')
                radio_name=i.xpath('./text()')
                radio_city_item[radio_name[0]]=radio_id[0]
            return radio_city_item
        else:
            return False

    def get_city_radio_item(self,cid,page):
        """
        根据城市Id获取指定页数的电台信息
        :param cid:
        :param page:
        :return:
        """
        cid=int(cid)
        base_url="https://webbff.qingting.fm/www"
        data={"query":"{\n    radioPage(cid:%d, page:%d){\n      contents\n    }\n  }"%(cid,page)}
        res=self.do_post_request(base_url,(data))
        if res:
            radio_item={}
            _json=json.loads(res)
            contents=_json.get("data").get("radioPage").get("contents")
            count=contents.get("count")
            items=contents.get("items")
            radio_item["count"]=count
            radio_item["items"]=items
            return radio_item
        else:
            return False

    def get_country_radio_item(self,page):
        """
        获取国家电台节目。
        一共两页
        :return:
        """
        radio_item=self.get_city_radio_item(409,page)
        return radio_item

    def get_network_radio_item(self,page):
        """
        获取网络电台节目。
        一共两页
        :return:
        """
        radio_item = self.get_city_radio_item(407, page)
        return radio_item

    def get_radio_play_mp3_link(self,radio_id):
        """
        根据电台id获取电台mp3播放地址
        :param radio_id:
        :return:
        """
        full_url="https://lhttp.qtfm.cn/live/%d/64k.mp3"%int(radio_id)
        return full_url

    def get_country_and_network_radio_statios(self):
        """
        获取所有国家、网络电台的id和图片
        :return:
        """
        all_list=[]
        country_radios=self.get_country_radio_item(1)
        if country_radios:
            count=int(country_radios.get("count"))
            all_list.append(country_radios['items'])
            if count>12:
                max_num=int(count/12)+1
                for i in range(2,max_num+1):
                    radio_item = self.get_country_radio_item(i)
                    all_list.append(radio_item['items'])
        network_radios=self.get_network_radio_item(1)
        if network_radios:
            count2=int(network_radios.get("count"))
            all_list.append(network_radios['items'])
            if count2>12:
                max_num2=int(count2/12)+1
                for j in range(2,max_num2+1):
                    radio_item = self.get_network_radio_item(j)
                    all_list.append(radio_item['items'])
        return all_list

    def parse_all_list(self,all_list):
        """
        处理all_list中的图片
        :return:
        """
        aim_item={}
        if len(all_list)!=0:
            for data in all_list:
                for j in data:
                    aim_item[j["id"]]=j["imgUrl"]
        return aim_item

    def pull_radio_station_imgs(self,aim_item):
        """
        将网络电台图片拉取到本地
        :param aim_item:
        :return:
        """
        for k,v in aim_item.items():
            self.download_imgs(k,"https:"+v)

    def download_imgs(self,radio_id,img_url):
        """
        下载图片到本地
        :param file_name:
        :param img_url:
        :return:
        """
        img_bin=self.do_get_img_bytes(img_url)
        if img_bin:
            file_type=img_url.split(".")[-1]
            os.makedirs("./radio_cache",exist_ok=True)
            full_path=f'./radio_cache/{str(radio_id)}.{file_type}'
            with open(full_path,'wb')as f:
                f.write(img_bin)
