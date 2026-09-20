#!/usr/bin/env python
# coding: UTF-8 
# Created by Cmoon

import rospy
from std_msgs.msg import String
from soundplayer import Soundplayer
# from pdfmaker import Pdfmaker
Name = {
    "Jack":["jack","Jack"],
    "Grace":["grace","Grace"],
    "Linda":["Linda","linda"],
    "John":["John","john"],
    "Mary":["Marry","marry","Mary","mary"],
    "Allen":["Allen","allen","Ellen","ellen","Eden","eden","leven"],
    "Richard":["Richard","richard","reach","Reach"],
    "Mike":["Mike","mike","mark","Mark"],
    "Lily":["lily","Lily"],
    "Lucy":["Lucy","lucy"],
    "Alex":["alex","Alex"],
    "charlie":["Charlie","charlie"],
    "elizabeth":["Elizabeth","elizabeth"],
    "francis":["Francis","francis"],
    "jennifer":["Jennifer","jennifer"],
    "Linda":["Linda","linda"],
    "Mary":["Mary","mary"],
    "patricia":["patricia","patricia"],
    "Robin":["Robin","robin"],
    "skyler":["Skyler","skyler"],
    "james":["james"],
    "john":["john"],
    "micheal":["micheal"],
    "robert":["robert"],
    "william":["william"]
}
    
Drink = {
    "Orange juice":['Orange','orange'],
    "Cola":['Cola','cola'],
    "Water":['Water','water','what','What'],
    "Sprite":['Sprite','sprite'],
    "Black tea":["Black","black"],
    "Milk":["milk","Milk"],
    "Coffee":["Coffee","coffee"],
    "Wine":["wine","Wine","win","Win","mine","Mine"],
    "Beer":["Beer","beer","Bear","bear"],
    "Soda":["Soda","soda","So","so"],
    "chip":["chip"],
    "biscuit":["biscuit"],
    "bread":["bread"],
    "dishsoap":["dishsoap"],
    "shampoo":["shampoo"],
    "cookie":["cookie"],
    "lays":["lays"],
    "handwash":["handwash"],
    "chocolate drink":["chocolate"],
    "coke":["coke"],
    "grape juice":["grape"]
}

LOCATION = {  # 模糊词
    'reception point': ['reception point', 'receiption point', 'leception point','reception','research'],
    'host point1': ['host point1', 'hos point1', 'host poin1', 'hos poin1'],
    'host point2': ['host point2', 'hos point2', 'host poin2', 'hos poin2'],
    'guest1 point1': ['guest1 point1', 'gues1 point1', 'gues1 poin1'],
    'guest1 point2': ['guest1 point2', 'gues1 point2', 'gues1 poin2'],
    'guest2 point': ['guest2 point', 'gues point', 'gues poin'],
}

class Recognizer:
    def __init__(self):
        rospy.Subscriber('/xfspeech', String, self.talkback)
        self.wakeup = rospy.Publisher('/xfwakeup', String, queue_size=10)
        self.start_signal = rospy.Publisher('/start_signal', String, queue_size=10)
        self.cmd = None
        self.location = LOCATION
        self.goal = ''
        self._soundplayer = Soundplayer()
        # self._pdfmaker = Pdfmaker()
        self.status = 0
        self.key = 1
        self.nadrflg = 0
        self.name = Name
        self.drink = Drink

    def talkback(self, msg):
        if self.key == 1:
            print("\n讯飞读入的信息为: " + msg.data)
            self.cmd = self.processed_cmd(msg.data)
            self.judge()

    def judge(self):
        if(self.nadrflg):
            self.analyze()
            return
        if self.status == 0:

            response = self.analyze()

            if response == 'Do you need me':
                self._soundplayer.say("Please say the command again. ")
                self.get_cmd()
            else:
                self.status = 1
                print(response)
                self._soundplayer.say(response, 3)
                self._soundplayer.say("please say yes or no.", 1)
                print('Please say yes or no.')
                self.get_cmd()

        elif ('Yes.' in self.cmd) or ('yes' in self.cmd) or ('Yeah' in self.cmd) or ('yeah' in self.cmd) and (
                self.status == 1):

            self._soundplayer.say('Ok, I will.')
            # self._pdfmaker.write('Cmd: Do you need me go to the ' + self.goal + '?')
            # self._pdfmaker.write('Respond: Ok,I will.')
            print('Ok, I will.')
            self.start_signal.publish(self.goal)
            self.key = 0
            self.status = 0
            self.goal = ''


        elif ('No.' in self.cmd) or ('no' in self.cmd) or ('oh' in self.cmd) or ('know' in self.cmd) and (
                self.status == 1):
            self._soundplayer.say("Please say the command again. ")
            print("Please say the command again. ")
            self.status = 0
            self.goal = ''
            self.get_cmd()

        else:
            self._soundplayer.say("please say yes or no.")
            print('Please say yes or no.')
            self.get_cmd()

    def processed_cmd(self, cmd):
        cmd = cmd.lower()
        for i in " ,.;?":
            cmd = cmd.replace(i, ' ')
        return cmd

    def get_cmd(self):
        """获取一次命令"""
        self._soundplayer.play('Speak.')
        self.wakeup.publish('ok')

    def get_cmd2(self):
        """获取一次命令"""
        self.nameee = None
        self.drinkkk = None
        self.nadrflg = 1
        self._soundplayer.play('Speak.')
        self.wakeup.publish('ok')
        while(self.nadrflg):
            pass
        return self.nameee,self.drinkkk


    def analyze(self):
        if(self.nadrflg):
            for (key, val) in self.name.items():
                for word in val:
                    if word in self.cmd:
                        self.nameee = key
                        break
            for (key, val) in self.drink.items():
                for word in val:
                    if word in self.cmd:
                        self.drinkkk = key
                        break

            self.nadrflg = 0
            return

        response = 'Do you need me'
        for (key, val) in self.location.items():
            for word in val:
                if word in self.cmd:
                    self.goal = key
                    response = response + ' go to the ' + key + '?'
                    break
        return response


if __name__ == '__main__':
    try:
        rospy.init_node('voice_recognition')
        Recognizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
