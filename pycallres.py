#!/usr/bin/env python3
#_∗_coding: utf-8 _∗_


"""
"""
import os
import io
import string
import re
import sys
import datetime
import copy
import json
import operator #数学计算操作符
import random
import matplotlib.pyplot as plt
import pylab as mpl     #import matplotlib as mpl

#设置汉字格式
# sans-serif就是无衬线字体，是一种通用字体族。
# 常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica,SimHei 中文的幼圆、隶书等等
#mpl.rcParams['font.sans-serif'] = ['SimSun']  # 指定默认字体 FangSong,SimHei
#mpl.rcParams['font.serif'] = ['SimSun']
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
#sys.path.append("D:\\TexasHoldem\\LuaDemo\\agent") 

import numpy as np

from Ccardev.evaluatordll import *
from Ccardev.opmodels import *

try:
    import cPickle as pickle
except ImportError:
    import pickle

print('sys.path=',sys.path)

colorline=['-','--','+-','-.','y','m1']
markline=['o','v','<','>','1','2']



#用于处理服务器内部输出文件的结果信息
#主要内容是用正则表达式处理的字符串
def dataprocess(filename,usernames):
    try:
        fIn = open(filename, 'r', encoding="utf8")
        resdata=fIn.readlines()
        fIn.close()
        
        #1.初步将文件中最后一次游戏的信息筛选出来
        ##1.1 根据connect信息判断最后一次游戏的开始
        ndlinest=0 #需要信息开始的行：need-line-start
        linenum=0
        for r in resdata:
            linenum=linenum+1
            if r.find(usernames[0])>0 :
                ndlinest=linenum
                print(r)
                break
        print("ndlinest=",ndlinest)
        ##1.2将赛选信息存到文件
        fout = open('reslatest.md', 'w', encoding="utf8")
        print("INFO: writing play data to '" + 'reslatest.md' + "'")
        linenum=0
        for r in resdata:
            linenum=linenum+1
            if(linenum>=ndlinest):
                fout.write(r)
        fout.close()

        #2.对重要信息进行处理
        ##2.1直接从list中将重要信息转存并处理
        print('-----important info------')
        datalatest=resdata[ndlinest-1:] #//--特别注意切片取值是取到后面一个索引的前一个值
        #print('datalatest=',datalatest)
        for r in datalatest:
            if (r.find('result')>0 and r.find(usernames[0])>0 ):
                print('find r=',r)

                #private card
                p = re.compile(r'player_card:(.*?)}, },')
                stra=r.strip()
                m = p.search(stra)
                res1=m.group(1)
                print('res=',res1)
                stra=str.split(res1,'},')
                print('stra=',stra)
                privatecard=[]
                print('len(usernames)=',len(usernames))
                for i in range(len(usernames)):
                    p1=re.compile('\d\:\{(.*)')
                    m1 = p1.search(stra[i])
                    res2=m1.group(1)
                    if(len(res2)>0):
                        res2=re.sub('\d:','',res2)
                        strc=str.split(res2,', ')
                        strc.pop()
                        privatecard.append(strc)
                        print('i=',i," ",strc)
                    else:
                        privatecard.append(res2)
                        print('i=',i," ",res2)
                print('privatecard=',privatecard)

                #public card
                p = re.compile(r'public_card:{(.*?)}')
                stra=r.strip()
                m = p.search(stra)
                res1=m.group(1)
                print('res=',res1)
                stra=str.split(res1,',')
                stra.pop()
                print('stra=',stra)
                publiccard=[]
                for i in range(len(stra)):
                    res2=stra[i]
                    if(len(res2)>0):
                        res2=re.sub('\d:','',res2)
                        res2=re.sub(',\s','',res2)
                    print('i=',i," ",res2)
                    publiccard.append(res2)
                print('publiccard=',publiccard)


                #players
                p = re.compile(r'players:{(.*?)}, },')
                stra=r.strip()
                m = p.search(stra)
                res1=m.group(1)
                print('res=',res1)
                stra=str.split(res1,'},')
                print('stra=',stra)
                players=[]
                for i in range(len(stra)):
                    player={}
                    p1=re.compile('name:(.*?),')
                    m1 = p1.search(stra[i])
                    player["name"]=m1.group(1)
                    p1=re.compile('win_money:(.*?),')
                    m1 = p1.search(stra[i])
                    player["win_money"]=float(m1.group(1))
                    player["private_card"]=privatecard[i]
                    players.append(player)
                print('players=',players)

                handres={}
                handres["players"]=players
                handres['public_card']=publiccard
                datausers.append(handres)

    except IOError:
        print("ERROR: Input bib file '" + 'resserver.md' +
                "' doesn't exist or is not readable")
        sys.exit(-1)
    return None



def moneystatistic():
    global moneyusers,nameusers
    moneyusers={}
    nameusers=[]

    #将最后的能用的结果存储起来
    fout = open('resfinal.md', 'w', encoding="utf8")
    print("INFO: writing cited references to '" + 'resfinal.md' + "'")
    for r in datausers:
        fout.write(str(r)+"\n")
    fout.close()

    #要特别列表是从0开始索引的
    for v in datausers[0]['players']:#先获取姓名列表
        name=v['name']
        if name in nameusers:
            pass
        else:
            nameusers.append(name)
    print('usernames=',nameusers)

    for name in nameusers:#先设置数组用于记录信息
        moneyusers[name]=[]
        moneyusers[name+"hand"]=[]
        moneyusers[name+"pub"]=[]

    #记录钱的信息
    for v in datausers:
        print("v=",v)
        for p in v['players']:
            print("p=",p)
            moneyusers[p['name']].append(p['win_money'])
            moneyusers[p['name']+"hand"].append(p['private_card'])
            moneyusers[p['name']+"pub"].append(v['public_card'])

    print('nameusers=',nameusers)
    print('moneyusers=',moneyusers)


    fout = open('stat'+'.md', 'w', encoding="utf8")
    for i in range(len(datausers)):
        fout.write("\nhand:"+str(i)+'\n')
        for name in nameusers:
            strout=name+" : "
            strout+="private: "+str(moneyusers[name+"hand"][i])+" win: "+str(moneyusers[name][i])+'\n'
            fout.write(strout)
        fout.write("public: "+str(moneyusers[nameusers[0]+"pub"][i])+'\n')
    fout.close()

    fout = open('stat'+'.csv', 'w', encoding="utf8")
    summoney=[]
    strout=''
    for name in nameusers:
        strout+=name+"win, "+name+"wt, "+name+"w/h, "
        summoney.append(0.0)
    fout.write(strout+'\n')
    
    for i in range(len(datausers)):
        strout=''
        for j in range(len(nameusers)):
            summoney[j]=summoney[j]+moneyusers[nameusers[j]][i]
            strout=strout+str(+moneyusers[nameusers[j]][i])+', '+str(summoney[j])+', '+str(summoney[j]/(i+1))+', '
        fout.write(strout+'\n')
    fout.close()
    return None



#用于我们自己的ai打印输出文件的结果信息
#主要方法：不是用正则表达式来直接处理字符串，而是将字符串读成json格式方便处理
#因为要多次读取，所以用一个参数来dealgamenumber来控制
datausers=[]
def datadealing3(filename,dealgamenumber=0):
    global recAllagentsname
    global datausers
    global recAllgameHistory
    global recAllgameHistoryS
    global recAllgameHistoryF
    global recAllgameHistorySF #一个赢率只记录两个决策点各一次
    global recAllagentsstat
    global rechandPredicted

    #记录从文件中读取的数据
    rechandPredicted={}

    # 记录全部局数的整个信息集历史完整
    #每条记录的格式为：playername,streetflag,public_card,recOpLastbet,action,player_card
    recAllgameHistory=[]   #用betto来表示的历史
    recAllgameHistoryS=[]  #用check，call，r+number表示的历史
    recAllgameHistoryF=[]  #F表示first 一个赢率只记录两个决策点各一次
    recAllgameHistorySF=[] #一个赢率只记录两个决策点各一次

    # 记录并统计VPIP等特征数据
    recAllagentsstat={}
    #recAllagentsstat[name]={"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
    # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
    # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}

    if not datausers:
        #打开文件，读取信息
        try:
            fIn = open(filename, 'r', encoding="utf8")
            resdata=fIn.readlines()
            fIn.close()
        except IOError:
            print("ERROR: Input file '" + filename +
                    "' doesn't exist or is not readable")
            sys.exit(-1)

        lino=0
        for r in resdata:
            lino+=1
            if (r.count("result")>1):
                #print("r=",r)
                r1=r.replace("result info:","").strip()
                r2=r1.replace("'","\"")
                #print(lino,":",r2)
                r3=re.search("(.*\})",r2).group(0)
                #print(lino,":",r3)
                r4=r3.replace("False","\"False\"")
                r5=r4.replace("True","\"True\"")
                data=json.loads(r5)
                #print('data=',data)
                datausers.append(data)


    #记录所有玩家的名称信息
    recAllagentsname=[]
    for player in datausers[0]['players']:
        if player['name'] not in recAllagentsname:
            recAllagentsname.append(player['name'])
    print('recAllagentsname=',recAllagentsname)

    for name in recAllagentsname:
        #统计相关的次数，并计算特征量
        #{"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
        # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
        # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}
        recAllagentsstat[name]={"chand":0,"cfhandpre":0,"cpfvplay":0,"cpfvraise":0,"cafraise":0,"cafcall":0,
        'cpfccl':0,'cpfcclfd':0,'cpfcclcl':0,'cpfcclrs':0,'cpfcck':0,'cpfcckck':0,'cpfcckrs':0,
        'cafccl':0,'cafcclfd':0,'cafcclcl':0,'cafcclrs':0,'cafcck':0,'cafcckck':0,'cafcckrs':0,
        'cpfall':0,'cpfallfd':0,'cpfallcl':0,'cafall':0,'cafallfd':0,'cafallcl':0,
        "vpip":0.0,'pfr':0.0,'af':0.0,'pfrdvpip':0.0,'rvpip':0.0,"rpfr":0.0,
        'br':0.0,'fr':0.0,'rr':0.0,'pfbr':0.0,'pffr':0.0,'pfrr':0.0}

    #根据姓名确定玩家数量
    nplayer=len(recAllagentsname) 

    if dealgamenumber==0:
        numberbreak=len(datausers)
    else:
        numberbreak=dealgamenumber
    i=0
    for data in datausers:
        #处理用于统计的动作和其它信息
        #记录到recAllgameHistory和recAllgameHistoryS中
        get_actioninfosetall(data,nplayer)
        i+=1
        if i>= numberbreak:
            break

    
    allbots="-".join(recAllagentsname)
    handid=len(datausers)

    for name in recAllagentsname:
        recAllagentsstat[name]["chand"]=handid
        if recAllagentsstat[name]["cafcck"]!=0:
            recAllagentsstat[name]["br"]=int(recAllagentsstat[name]["cafcckrs"]/recAllagentsstat[name]["cafcck"]*100)
        else:
            recAllagentsstat[name]["br"]=0
        if recAllagentsstat[name]["cafccl"]!=0:
            recAllagentsstat[name]["fr"]=int(recAllagentsstat[name]["cafcclfd"]/recAllagentsstat[name]["cafccl"]*100)
            recAllagentsstat[name]["rr"]=int(recAllagentsstat[name]["cafcclrs"]/recAllagentsstat[name]["cafccl"]*100)
        else:
            recAllagentsstat[name]["fr"]=0
            recAllagentsstat[name]["rr"]=0

        if recAllagentsstat[name]["cpfcck"]!=0:
            recAllagentsstat[name]['pfbr']=int(recAllagentsstat[name]["cpfcckrs"]/recAllagentsstat[name]["cpfcck"]*100)
        else:
            recAllagentsstat[name]['pfbr']=0
        if recAllagentsstat[name]["cpfccl"]!=0:
            recAllagentsstat[name]['pffr']=int(recAllagentsstat[name]["cpfcclfd"]/recAllagentsstat[name]["cpfccl"]*100)
        else:
            recAllagentsstat[name]['pffr']=0
        if recAllagentsstat[name]["cpfccl"]!=0:
            recAllagentsstat[name]['pfrr']=int(recAllagentsstat[name]["cpfcclrs"]/recAllagentsstat[name]["cpfccl"]*100)
        else:
            recAllagentsstat[name]['pfrr']=0

    #保存打牌对抗数据到文件
    #print('recAllgameHistory=',recAllgameHistory)
    #print('recAllgameHistoryS=',recAllgameHistoryS)
    filename='rec'+"-"+allbots+"-"+str(handid)+'-history.md'
    f=open(filename, 'w')
    for x in recAllgameHistory:
        f.write(str(x)+"\n")
    f.close()
    
    filename='rec'+"-"+allbots+"-"+str(handid)+'-historyS.md'
    f=open(filename, 'w')
    for x in recAllgameHistoryS:
        f.write(str(x)+"\n")
    f.close()

    filename='rec'+"-"+allbots+"-"+str(handid)+'-historyF.md'
    f=open(filename, 'w')
    for x in recAllgameHistoryF:
        f.write(str(x)+"\n")
    f.close()
    
    filename='rec'+"-"+allbots+"-"+str(handid)+'-historySF.md'
    f=open(filename, 'w')
    for x in recAllgameHistorySF:
        f.write(str(x)+"\n")
    f.close()

    print('recAllagentsstat=',recAllagentsstat)
    
    return None
    
    



# 将acpc的信息转换为data数据
# 注意这是两人情况下用的，因为其中很多处理都是针对两人做的。
room_number=2
def ACPCmsgTodata(msg,myname,opname):

    name=myname
    data={} #字典

    #print('in msg=',msg)
    m1=re.search("^MATCHSTATE:(\d):(\d*):([^:]*):(.*)",msg.strip())
    #print('m1=',m1)
    position=int(m1.group(1))
    opp_post=1-position        #对手位置
    hand_id=int(m1.group(2))
    actions=m1.group(3)
    cards=m1.group(4)
    #print('position=',position)

    m2=re.search("([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)",actions.strip())
    #print('m2=',m2,m2.group(0))
    preflop_actions = m2.group(1)
    flop_actions = m2.group(2)
    turn_actions = m2.group(3)
    river_actions = m2.group(4)
    lastaction=''
    if actions:
        lastaction=actions[-1]
    if preflop_actions=='':
        flagFirstAction=True
    else:
        flagFirstAction=False

    #print("cards",cards)
    m3 = re.search("([^\|]*)\|([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)",cards.strip())
    #print('m3=',m3,m3.group(0))
    hand_p1=m3.group(1)
    hand_p2=m3.group(2)
    flopcds=m3.group(3)
    turncds=m3.group(4)
    rivercds=m3.group(5)
    #print("flopcds=",flopcds)
    #print("turncds=",turncds)
    #print("rivercds=",rivercds)

    #位置和局序号
    data['hand_id']=hand_id

    #手牌
    if position==0 :
        data['private_card']=[hand_p1[:2],hand_p1[2:]]
    else:
        data['private_card']=[hand_p2[:2],hand_p2[2:]]

    #print('hand_p1=',hand_p1)
    #print('hand_p2=',hand_p2)
    if hand_p1 and hand_p2 :
        data['player_card']=[[hand_p1[:2],hand_p1[2:]],[hand_p2[:2],hand_p2[2:]]]
    else:
        if position==0 :
            data['player_card']=[[hand_p1[:2],hand_p1[2:]],[]]
        else:
            data['player_card']=[[],[hand_p2[:2],hand_p2[2:]]]


    #公共牌
    street=1
    actionstrings=[preflop_actions]
    data['public_card']=[]
    if flopcds :
        data['public_card']=[flopcds[:2],flopcds[2:4],flopcds[4:]]
        street=2
        actionstrings.append(flop_actions)
    if turncds :
        data['public_card'].append(turncds)
        street=3
        actionstrings.append(turn_actions)
    if rivercds :
        data['public_card'].append(rivercds)
        street=4
        actionstrings.append(river_actions)
    data['street']=street
    #print('actionstrings=',actionstrings)

    #动作和下注额
    #要特别注意：下面的代码是很多地方都是基于两人对局做了特殊处理的
    #A raise is valid if a) it raises by at least one chip, b) the player has sufficient money in their stack to
    #raise to the value, and c) the raise would put the player all-in (they have spent all their chips) or the
    #amount they are raising by is at least as large as the big blind and the raise-by amount for any other
    #raise in the round. 
    #注意acpc协议中r****实际是raiseto，就是全局加注到多少(而不是一个轮次的)
    #一次raise实际下注额是call上的+对手raise增加的额度+大盲注
    #那么raiseto的最小值为：原来已经下注的额度+前一次raise增加的额度+大盲注
    money_beted_now=[params_sb,params_bb] #两个人对局：位置0的下注额，位置1的下注额
    money_beted_rnd=[0,0,0,0] #四个轮次 
    actions=[]
    actions_all=[]
    tmp_lastAction=[] #最后一个动作，倒数第一个动作
    tmp_lastSecAction=[] #最后第二个动作，倒数第二个动作
    tmp_lastThiAction=[] #倒数第三个动作
    tmp_roundactstr=''   #描述当前轮已经做的动作
    tmp_roundactnum=0    #描述当前要做的动作是第几个动作
    tmp_op_tideway=''    #描述前面各轮对手做的最强动作，主要是c和r
    tmp_my_tideway=''    #描述前面各轮自己做的最强动作
    raisetimes=0
    raisemprev=0
    raise_amount=0
    actpos=0    #各动作的所对应的玩家的位置
    for i  in range(len(actionstrings)):
        #tmp_lastRoundAction='' #当前轮的动作记录
        #tmp_lastMyRoundAction='' #当前轮的动作记录

        actions_remainder=actionstrings[i]
        actions_round=[]
        raisetimes=0 #一轮中raise的次数，各轮需要分别统计
        raisemprev=0 #一轮中一次raise增加的下注额度，各轮需要分别统计
        actsn=0      #一轮中动作的序号
        actstr=''    #一轮中的动作历史记录：用每个动作的首字母表示的
        acttide=''
        actmytide=''
        while actions_remainder != '':
            actsn+=1 #每一轮第一个动作序号为1，第二个动作序号为2，后续为3,4,5...
            if i==0: #preflop轮小盲先行
                actpos=1-actsn%2
            else: #flop/turn/river轮大盲先行
                actpos=actsn%2
            #print('actsn=',actsn)
            #print('actpos=',actpos)
            parsed_chunk = ''
            if actions_remainder.startswith("c"):
                #call 改成check
                if money_beted_now[actpos]==money_beted_now[1-actpos]:
                    actionstr="check"
                else:
                    actionstr="call"
                actions_round.append({"action":actionstr,"position": actpos,'mynowbet':money_beted_now[1-actpos],'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                actions_all.append({"action":actionstr,"position": actpos,'mynowbet':money_beted_now[1-actpos],'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                parsed_chunk = "c"
                if opp_post==actpos:
                    acttide='c'
                else:
                    actmytide='c'
                money_beted_now[actpos]=money_beted_now[1-actpos]
            elif actions_remainder.startswith("r"):
                #print('actions_remainder=',actions_remainder)
                raise_amount = int(re.search("^r(\d*).*",actions_remainder).group(1))
                bet_now=raise_amount
                parsed_chunk = "r" + str(raise_amount)
                #print('raise_amount=',raise_amount)
                if opp_post==actpos:
                    acttide='r'
                else:
                    actmytide='r'
                raisetimes+=1
                #raise的额度就是在call平基础上增加的额度
                #raisemprev=raise_amount-max(money_beted_now)
                #raise_amount就是raise后的下注额，即raiseto的额度
                #注意acpc协议是全局的raiseto，而cisia则是当前轮的raiseto
                #acpc
                #actions_round.append({"action":"r" + str(raise_amount),"position": actpos}) #,"raise_amount":raise_amount
                #actions_all.append({"action":"r" + str(raise_amount),"position": actpos})
                #parsed_chunk = "r" + str(raise_amount)
                #cisia
                if i==0:
                    actions_round.append({"action":"r" + str(raise_amount),"position": actpos,'mynowbet':bet_now,'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]}) #,"raise_amount":raise_amount
                    actions_all.append({"action":"r" + str(raise_amount),"position": actpos,'mynowbet':bet_now,'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                else:
                    raise_amount=raise_amount-money_beted_rnd[i-1]
                    actions_round.append({"action":"r" + str(raise_amount),"position": actpos,'mynowbet':bet_now,'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]}) #,"raise_amount":raise_amount
                    actions_all.append({"action":"r" + str(raise_amount),"position": actpos,'mynowbet':bet_now,'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                money_beted_now[actpos]=bet_now
            elif actions_remainder.startswith("f"):
                actions_round.append({"action":"fold","position": actpos,'mynowbet':money_beted_now[actpos],'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                actions_all.append({"action":"fold","position": actpos,'mynowbet':money_beted_now[actpos],'myprebet':money_beted_now[actpos],'opprebet':money_beted_now[1-actpos]})
                parsed_chunk = "f"
            else:
                print("wrong action string")
            actstr+=parsed_chunk[0]
            tmp_lastThiAction=tmp_lastSecAction
            tmp_lastSecAction=tmp_lastAction
            tmp_lastAction=[actions_round[-1]["action"],actpos,i+1,actions_round[-1]] #记录三个参数：动作名，行动者，轮次
            #tmp_lastMyRoundAction=tmp_lastRoundAction
            #tmp_lastRoundAction=actions_round[-1]["action"] #当前轮的动作记录
            
            actions_remainder = actions_remainder.replace(parsed_chunk,"",1)

        if money_beted_now[0]==money_beted_now[1]:
            money_beted_rnd[i]=money_beted_now[0]
        else:
            #print("not equal bet in round:",i)
            pass

        actions.append(actions_round)
        if acttide != '':
            tmp_op_tideway+=acttide
        if actmytide != '':
            tmp_my_tideway+=actmytide
        tmp_roundactstr=actstr
        tmp_roundactnum=actsn+1
    
    data['action_history']=actions
    data['act_next_sn']=tmp_roundactnum
    data['act_his_rnd']=tmp_roundactstr
    data['act_op_tide']=tmp_op_tideway
    tmp_mytd_plus=''
    for x in tmp_my_tideway:
        if x=='r':
            tmp_mytd_plus+='+'
        else:
            tmp_mytd_plus+='-'
    data['act_my_tide']=tmp_mytd_plus


    #判断决策点信息
    #这里虽然可能轮次变化后最后一个动作可能是前一轮的，但很巧的是
    #下面的逻辑也没有问题，因为前一个动作若是r的画，那么轮次必定没有结束
    #所以面临的决策点是pcl是没有问题的。
    if position==0 :
        flagDecisionPt=1
    else:
        flagDecisionPt=0

    if tmp_lastAction:
        if tmp_lastAction[0][0]=="r":
            flagDecisionPt=1 #pcl point
        else:
            flagDecisionPt=0 #pck point
    
    #记录对手的动作类型
    flag_opaction=None
    
    if tmp_lastAction:
        if tmp_lastAction[1]==position:#当最后一个动作是自己的时就一定在轮次交界处
            #我方的动作必然是c或者f
            #因此对手的动作必然是call或者raise
            #若最后一个动作在第一轮，则对手必然是在pcl
            if street==2: # cc|flop，rc|flop
                flagOpDecisionPt=1 #pcl point,前一次对手面对的决策点类型
            else: # 比如：  rrc|turn
                if tmp_lastThiAction:
                    if tmp_lastThiAction[0][0]=='r':
                        flagOpDecisionPt=1
                    else: # 比如：crc|turn
                        flagOpDecisionPt=0 #pck point,前一次对手面对的决策点类型
                else:
                    flagOpDecisionPt=1
        else:
            if tmp_lastSecAction:
                if tmp_lastSecAction[0][0]=='r':
                    flagOpDecisionPt=1 
                else:
                    flagOpDecisionPt=0

                #对手不是轮次开始时的动作
                #注意这种情况：MATCHSTATE:0:5:cc/:KhTh|/8dAs8s，下，已经进入到下一轮，但对手的动作是上一轮的，这时我方不决策，所以判断不判断不影响
                if tmp_lastAction[0][0]=='c':
                    flag_opaction='call'
                else:
                    invest=tmp_lastAction[-1]['mynowbet']-tmp_lastAction[-1]['myprebet']
                    prepot=tmp_lastAction[-1]['myprebet']+tmp_lastAction[-1]['opprebet']
                    print('prepot=',prepot,'raise_amount=',invest)
                    if invest < prepot:
                        flag_opaction='raise0'
                    elif invest < 2*prepot:
                        flag_opaction='raise1'
                    else:
                        flag_opaction='raise2'

            else:# 若倒数第二个动作不存在，且不是我做的倒数第一个动作，必然是对手做的，那么就是每局开始的时候
                flagOpDecisionPt=1

                #对手在轮次开始时的动作
                if tmp_lastAction[0][0]=='c':
                    if street==1:
                        flag_opaction='call'
                    else:
                        flag_opaction='check'
                else:
                    invest=tmp_lastAction[-1]['mynowbet']-tmp_lastAction[-1]['myprebet']
                    prepot=tmp_lastAction[-1]['myprebet']+tmp_lastAction[-1]['opprebet']
                    print('prepot=',prepot,'raise_amount=',invest)
                    if invest < prepot:
                        flag_opaction='bet0'
                    elif invest < 2*prepot:
                        flag_opaction='bet1'
                    else:
                        flag_opaction='bet2'

    else:#没有动作则无需考虑
        flagOpDecisionPt=-1

    data['LastAction']=tmp_lastAction
    data['LastSecAction']=tmp_lastSecAction
    data['LastTrdAction']=tmp_lastThiAction
    data['flagFirstAction']=flagFirstAction
    data['flagDecisionPt']=flagDecisionPt
    data['flagOpDecisionPt']=flagOpDecisionPt
    data['flagOpActtype']=flag_opaction

    #玩家信息
    data['players']=[]
    for i in range(room_number):
        data['players'].append({"position":i,"money_bet":money_beted_now[i],"money_left":params_stack-money_beted_now[i],'total_money':params_stack})
    data['roundbet']=money_beted_rnd
    data['pots']=money_beted_now

    #legal_action
    legalact=[]
    #当前最小的加注额应等于大盲注+对手的加注额度(对手在平基础上增加的额度)
    # 当money_beted_now[actpos]，money_beted_now[1-actpos]不相等时，必然还在一个轮次内
    # 那么对手的加注额必然是：abs(money_beted_now[actpos]-money_beted_now[1-actpos])
    # 当money_beted_now[actpos]，money_beted_now[1-actpos]相等时，要么新一轮开始或者一轮前面对手c
    # 那么对手的加注额为0也等于abs(money_beted_now[actpos]-money_beted_now[1-actpos])
    # 所以最小加注额如下是对的：
    raisetorangemin=abs(money_beted_now[actpos]-money_beted_now[1-actpos])+params_bb
    
    if money_beted_now[actpos]==money_beted_now[1-actpos]:#这里是典型的基于2人的考虑的处理
        legalact.append("check")
        #通常相等的时候raise是没有多次的
        legalact.append("raise")
    else:
        legalact.append("call")
        legalact.append("fold")
        if raisetimes<params_maxr:
            legalact.append("raise")
    data["legal_actions"]=legalact
    if "raise" in legalact:
        raisetorangemin+=max(money_beted_now)
        if raisetorangemin>params_stack:
            raisetorangemin=params_stack

    #注意acpc的服务器的raise的额度可以自动调整的，默认是raiseto，若这个值过小，则会自动调整调整到call后加上该值 
    #所以提示的额度最好使用加注至(即raiseto)，分两种，一种是而且是全局的"raise_to_range"，
    #另一种是像自动化所那样使用一轮的raiseto，使用"raise_range"。
    if "raise" in legalact:#raise的范围是一轮中的值
        if street ==1:
            #acpc的全局raiseto
            data["raise_to_range"]=[raisetorangemin,params_stack]
            #cisia中的当前轮的raiseto
            data["raise_range"]=[raisetorangemin,params_stack]
        else:
            #acpc的全局raiseto
            data["raise_to_range"]=[raisetorangemin,params_stack]
            #cisia中的当前轮的raiseto。因为从第二轮开始street=2，要减去第一轮的下注，第一轮在money_beted_rnd列表中位置是0。
            data["raise_range"]=[raisetorangemin-money_beted_rnd[street-2],params_stack-money_beted_rnd[street-2]]
    
    #info,action_position
    data['info']='state'
    if lastaction=='f' or (hand_p1 and hand_p2) :
        actpos=2 #为了给出action_position=-1，所以设置为2
        data["info"]="result"
    data['position']=position

    #print("actions_all=",actions_all,len(actions_all))
    #print("actions=",actions,len(actions))
    #print("street=",street)
    #if street==2: print("str=",street,actions[street-1],len(actions[street-1]))
    #if street==3: print("str=",street,actions[street-1],len(actions[street-1]))
    #if street==4: print("str=",street,actions[street-1],len(actions[street-1]))
    if len(actions_all)==0:#当全局没有任何动作时是小盲
        data["action_position"]=0
    elif street==2 and len(actions[1])==0:#当flop没有任何动作时是大盲
        data["action_position"]=1
    elif street==3 and len(actions[2])==0:#当turn没有任何动作时是大盲
        data["action_position"]=1
    elif street==4 and len(actions[3])==0:#当river全局没有任何动作时是大盲
        data["action_position"]=1
    else:#其它情况下，当前需要动作的玩家是最后一个动作的玩家的对手
        data["action_position"]=1-actpos 


    #一局的结果可以直接算出来
    win_money=[0,0]
    if hand_p1 and hand_p2:
        rk1=gethandrank(data['player_card'][0],data['public_card'])
        rk2=gethandrank(data['player_card'][1],data['public_card'])
        if rk1>rk2:
            win_money=[money_beted_now[1],-money_beted_now[1]]
        elif rk1==rk2:
            win_money=[0,0]
        else:
            win_money=[-money_beted_now[0],money_beted_now[0]]
    if lastaction=='f':
        win_money=[0,0]
        foldpos=actions_all[-1]["position"]
        win_money[foldpos]=-money_beted_now[foldpos]
        win_money[1-foldpos]=money_beted_now[foldpos]

    if data["info"]=="result":
        for i in range(room_number):
            data['players'][i]["win_money"]=win_money[i]
    
    data['players'][position]["name"]=name
    data['players'][1-position]["name"]=opname
    data['message']=msg

    return data





def datadealingnojsonres(filename,dealgamenumber=0):
    global recAllagentsname
    global datausers
    global recAllgameHistory
    global recAllgameHistoryS
    global recAllgameHistoryF
    global recAllgameHistorySF #一个赢率只记录两个决策点各一次
    global recAllagentsstat
    global rechandPredicted

    #记录从文件中读取的数据
    datausers=[]
    rechandPredicted={}

    # 记录全部局数的整个信息集历史完整
    #每条记录的格式为：playername,streetflag,public_card,recOpLastbet,action,player_card
    recAllgameHistory=[]   #用betto来表示的历史
    recAllgameHistoryS=[]  #用check，call，r+number表示的历史
    recAllgameHistoryF=[]  #F表示first 一个赢率只记录两个决策点各一次
    recAllgameHistorySF=[] #一个赢率只记录两个决策点各一次

    # 记录并统计VPIP等特征数据
    recAllagentsstat={}
    #recAllagentsstat[name]={"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
    # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
    # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}


    #打开文件，读取信息
    try:
        fIn = open(filename, 'r', encoding="utf8")
        resdata=fIn.readlines()
        fIn.close()
    except IOError:
        print("ERROR: Input file '" + filename +
                "' doesn't exist or is not readable")
        sys.exit(-1)


    myname=''
    opname=''
    lino=0
    for r in resdata:
        lino+=1
        if (r.count("connect:")>=1):
            r1=r.replace("connect:","").strip()
            r2=r1.replace("'","\"")
            data=json.loads(r2)
            myname=data['name']
            opname=data['bots'][0]
            print('myname=',myname)
            break
    

    lino=0
    for r in resdata:
        lino+=1
        #print('r=',r)
        if (r.count("MATCHSTATE")>=1):
            print('r=',r)
            msg=re.search("recv state msg=(.*)--endmsg",r).group(1)
            print('msg=',msg)
            data=ACPCmsgTodata(msg,myname,opname)
            if data['info']=='result':
                print('msg=',msg)
                print('data=',data)
                datausers.append(data)



    #记录所有玩家的名称信息
    recAllagentsname=[]
    for player in datausers[0]['players']:
        if player['name'] not in recAllagentsname:
            recAllagentsname.append(player['name'])
    print('recAllagentsname=',recAllagentsname)

    for name in recAllagentsname:
        #统计相关的次数，并计算特征量
        #{"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
        # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
        # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}
        recAllagentsstat[name]={"chand":0,"cfhandpre":0,"cpfvplay":0,"cpfvraise":0,"cafraise":0,"cafcall":0,
        'cpfccl':0,'cpfcclfd':0,'cpfcclcl':0,'cpfcclrs':0,'cpfcck':0,'cpfcckck':0,'cpfcckrs':0,
        'cafccl':0,'cafcclfd':0,'cafcclcl':0,'cafcclrs':0,'cafcck':0,'cafcckck':0,'cafcckrs':0,
        'cpfall':0,'cpfallfd':0,'cpfallcl':0,'cafall':0,'cafallfd':0,'cafallcl':0,
        'cpfspr':0,'cpfsprrs':0,
        "vpip":0.0,'pfr':0.0,'af':0.0,'pfrdvpip':0.0,'rvpip':0.0,"rpfr":0.0,
        'br':0.0,'fr':0.0,'rr':0.0,'pfbr':0.0,'pffr':0.0,'pfrr':0.0,'sprr':0.0}

    #根据姓名确定玩家数量
    nplayer=len(recAllagentsname) 


    if dealgamenumber==0:
        numberbreak=len(datausers)
    else:
        numberbreak=dealgamenumber
    i=0
    for data in datausers:
        #处理用于统计的动作和其它信息
        #记录到recAllgameHistory和recAllgameHistoryS中
        get_actioninfosetall(data,nplayer)
        i+=1
        if i>= numberbreak:
            break


    
    allbots="-".join(recAllagentsname)
    handid=len(datausers)

    #保存打牌对抗数据到文件
    #print('recAllgameHistory=',recAllgameHistory)
    #print('recAllgameHistoryS=',recAllgameHistoryS)
    filename='rec'+"-"+allbots+"-"+str(handid)+'-history.md'
    f=open(filename, 'w')
    for x in recAllgameHistory:
        f.write(str(x)+"\n")
    f.close()
    
    filename='rec'+"-"+allbots+"-"+str(handid)+'-historyS.md'
    f=open(filename, 'w')
    for x in recAllgameHistoryS:
        f.write(str(x)+"\n")
    f.close()

    filename='rec'+"-"+allbots+"-"+str(handid)+'-historyF.md'
    f=open(filename, 'w')
    for x in recAllgameHistoryF:
        f.write(str(x)+"\n")
    f.close()
    
    filename='rec'+"-"+allbots+"-"+str(handid)+'-historySF.md'
    f=open(filename, 'w')
    for x in recAllgameHistorySF:
        f.write(str(x)+"\n")
    f.close()
    
    return None





#从服务器数据获得的统计
#因为要多次读取，所以用一个参数来dealgamenumber来控制
def datadealing4(dealgamenumber=0):
    global recAllagentsname
    global datausers
    global recAllgameHistory
    global recAllgameHistoryS
    global recAllgameHistoryF
    global recAllgameHistorySF #一个赢率只记录两个决策点各一次
    global recAllagentsstat
    global rechandPredicted

    datausers=recAllresults

    #记录从文件中读取的数据
    rechandPredicted={}

    # 记录全部局数的整个信息集历史完整
    #每条记录的格式为：playername,streetflag,public_card,recOpLastbet,action,player_card
    recAllgameHistory=[]   #用betto来表示的历史
    recAllgameHistoryS=[]  #用check，call，r+number表示的历史
    recAllgameHistoryF=[]  #F表示first 一个赢率只记录两个决策点各一次
    recAllgameHistorySF=[] #一个赢率只记录两个决策点各一次

    # 记录并统计VPIP等特征数据
    recAllagentsstat={}
    #recAllagentsstat[name]={"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
    # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
    # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}

    #记录所有玩家的名称信息
    recAllagentsname=[]
    for player in datausers[0]['players']:
        if player['name'] not in recAllagentsname:
            recAllagentsname.append(player['name'])
    print('recAllagentsname=',recAllagentsname)

    for name in recAllagentsname:
        #统计相关的次数，并计算特征量
        #{"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
        # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
        # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}
        recAllagentsstat[name]={"chand":0,"cfhandpre":0,"cpfvplay":0,"cpfvraise":0,"cafraise":0,"cafcall":0,
        'cpfccl':0,'cpfcclfd':0,'cpfcclcl':0,'cpfcclrs':0,'cpfcck':0,'cpfcckck':0,'cpfcckrs':0,
        'cafccl':0,'cafcclfd':0,'cafcclcl':0,'cafcclrs':0,'cafcck':0,'cafcckck':0,'cafcckrs':0,
        'cpfall':0,'cpfallfd':0,'cpfallcl':0,'cafall':0,'cafallfd':0,'cafallcl':0,
        "vpip":0.0,'pfr':0.0,'af':0.0,'pfrdvpip':0.0,'rvpip':0.0,"rpfr":0.0,
        'br':0.0,'fr':0.0,'rr':0.0,'pfbr':0.0,'pffr':0.0,'pfrr':0.0}

    #根据姓名确定玩家数量
    nplayer=len(recAllagentsname) 

    #datausers=recAllresults
    if dealgamenumber==0:
        numberbreak=len(datausers)
    else:
        numberbreak=dealgamenumber
    i=0
    for data in datausers:
        #处理用于统计的动作和其它信息
        #记录到recAllgameHistory和recAllgameHistoryS中
        get_actioninfosetall(data,nplayer)
        i+=1
        if i>= numberbreak:
            break

    
    allbots="-".join(recAllagentsname)
    handid=len(datausers)

    for name in recAllagentsname:
        recAllagentsstat[name]["chand"]=handid
        if recAllagentsstat[name]["cafcck"]!=0:
            recAllagentsstat[name]["br"]=int(recAllagentsstat[name]["cafcckrs"]/recAllagentsstat[name]["cafcck"]*100)
        else:
            recAllagentsstat[name]["br"]=0
        if recAllagentsstat[name]["cafccl"]!=0:
            recAllagentsstat[name]["fr"]=int(recAllagentsstat[name]["cafcclfd"]/recAllagentsstat[name]["cafccl"]*100)
            recAllagentsstat[name]["rr"]=int(recAllagentsstat[name]["cafcclrs"]/recAllagentsstat[name]["cafccl"]*100)
        else:
            recAllagentsstat[name]["fr"]=0
            recAllagentsstat[name]["rr"]=0

        if recAllagentsstat[name]["cpfcck"]!=0:
            recAllagentsstat[name]['pfbr']=int(recAllagentsstat[name]["cpfcckrs"]/recAllagentsstat[name]["cpfcck"]*100)
        else:
            recAllagentsstat[name]['pfbr']=0
        if recAllagentsstat[name]["cpfccl"]!=0:
            recAllagentsstat[name]['pffr']=int(recAllagentsstat[name]["cpfcclfd"]/recAllagentsstat[name]["cpfccl"]*100)
        else:
            recAllagentsstat[name]['pffr']=0
        if recAllagentsstat[name]["cpfccl"]!=0:
            recAllagentsstat[name]['pfrr']=int(recAllagentsstat[name]["cpfcclrs"]/recAllagentsstat[name]["cpfccl"]*100)
        else:
            recAllagentsstat[name]['pfrr']=0

    print('recAllagentsstat=',recAllagentsstat)

    filename='rec'+"-"+allbots+"-"+str(handid)+'-historySF1.md'
    f=open(filename, 'w')
    for x in recAllgameHistorySF:
        f.write(str(x)+"\n")
    f.close()
    
    return None



#用于我们自己的ai打印输出文件的结果信息
#主要方法：不是用正则表达式来直接处理字符串，而是将字符串读成json格式方便处理
def datadealing(filename):
    global recAllagentsname
    global datausers
    global recAllgameHistory
    global recAllgameHistoryS
    global recAllgameHistoryF
    global recAllgameHistorySF #一个赢率只记录两个决策点各一次
    global recAllagentsstat
    global rechandPredicted

    #记录从文件中读取的数据
    datausers=[]
    rechandPredicted={}

    # 记录全部局数的整个信息集历史完整
    #每条记录的格式为：playername,streetflag,public_card,recOpLastbet,action,player_card
    recAllgameHistory=[]   #用betto来表示的历史
    recAllgameHistoryS=[]  #用check，call，r+number表示的历史
    recAllgameHistoryF=[]  #F表示first 一个赢率只记录两个决策点各一次
    recAllgameHistorySF=[] #一个赢率只记录两个决策点各一次

    # 记录并统计VPIP等特征数据
    recAllagentsstat={}
    #recAllagentsstat[name]={"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
    # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
    # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}


    #打开文件，读取信息
    try:
        fIn = open(filename, 'r', encoding="utf8")
        resdata=fIn.readlines()
        fIn.close()
    except IOError:
        print("ERROR: Input file '" + filename +
                "' doesn't exist or is not readable")
        sys.exit(-1)

    lino=0
    for r in resdata:
        lino+=1
        if (r.count("result")>1):
            #print("r=",r)
            r1=r.replace("result info:","").strip()
            r2=r1.replace("'","\"")
            #print(lino,":",r2)
            r3=re.search("(.*\})",r2).group(0)
            #print(lino,":",r3)
            r4=r3.replace("False","\"False\"")
            r5=r4.replace("True","\"True\"")
            data=json.loads(r5)
            #print('data=',data)
            datausers.append(data)


    #记录所有玩家的名称信息
    recAllagentsname=[]
    for player in datausers[0]['players']:
        if player['name'] not in recAllagentsname:
            recAllagentsname.append(player['name'])
    print('recAllagentsname=',recAllagentsname)

    for name in recAllagentsname:
        #统计相关的次数，并计算特征量
        #{"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
        # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
        # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}
        recAllagentsstat[name]={"chand":0,"cfhandpre":0,"cpfvplay":0,"cpfvraise":0,"cafraise":0,"cafcall":0,
        'cpfccl':0,'cpfcclfd':0,'cpfcclcl':0,'cpfcclrs':0,'cpfcck':0,'cpfcckck':0,'cpfcckrs':0,
        'cafccl':0,'cafcclfd':0,'cafcclcl':0,'cafcclrs':0,'cafcck':0,'cafcckck':0,'cafcckrs':0,
        'cpfall':0,'cpfallfd':0,'cpfallcl':0,'cafall':0,'cafallfd':0,'cafallcl':0,
        "vpip":0.0,'pfr':0.0,'af':0.0,'pfrdvpip':0.0,'rvpip':0.0,"rpfr":0.0,
        'br':0.0,'fr':0.0,'rr':0.0,'pfbr':0.0,'pffr':0.0,'pfrr':0.0}

    #根据姓名确定玩家数量
    nplayer=len(recAllagentsname) 

    for data in datausers:
        #处理用于统计的动作和其它信息
        #记录到recAllgameHistory和recAllgameHistoryS中
        get_actioninfosetall(data,nplayer)

    
    allbots="-".join(recAllagentsname)
    handid=len(datausers)

    for name in recAllagentsname:
        recAllagentsstat[name]["chand"]=handid
        recAllagentsstat[name]["br"]=(recAllagentsstat[name]["cafcckrs"]/recAllagentsstat[name]["cafcck"])
        recAllagentsstat[name]["fr"]=(recAllagentsstat[name]["cafcclfd"]/recAllagentsstat[name]["cafccl"])
        recAllagentsstat[name]["rr"]=(recAllagentsstat[name]["cafcclrs"]/recAllagentsstat[name]["cafccl"])
        recAllagentsstat[name]['pfbr']=(recAllagentsstat[name]["cpfcckrs"]/recAllagentsstat[name]["cpfcck"])
        recAllagentsstat[name]['pffr']=(recAllagentsstat[name]["cpfcclfd"]/recAllagentsstat[name]["cpfccl"])
        recAllagentsstat[name]['pfrr']=(recAllagentsstat[name]["cpfcclrs"]/recAllagentsstat[name]["cpfccl"])


    #保存打牌对抗数据到文件
    #print('recAllgameHistory=',recAllgameHistory)
    #print('recAllgameHistoryS=',recAllgameHistoryS)
    filename='rec'+"-"+allbots+"-"+str(handid)+'-history.md'
    f=open(filename, 'w')
    for x in recAllgameHistory:
        f.write(str(x)+"\n")
    f.close()
    
    filename='rec'+"-"+allbots+"-"+str(handid)+'-historyS.md'
    f=open(filename, 'w')
    for x in recAllgameHistoryS:
        f.write(str(x)+"\n")
    f.close()

    filename='rec'+"-"+allbots+"-"+str(handid)+'-historyF.md'
    f=open(filename, 'w')
    for x in recAllgameHistoryF:
        f.write(str(x)+"\n")
    f.close()
    
    filename='rec'+"-"+allbots+"-"+str(handid)+'-historySF.md'
    f=open(filename, 'w')
    for x in recAllgameHistorySF:
        f.write(str(x)+"\n")
    f.close()

    print('recAllagentsstat=',recAllagentsstat)
    
    return None
    
    

#------------------------------------------------------------
#获取动作信息集
#不要用call/raise这样的动作来描述，用bet来描述，就是投入的金额数来描述。
#记录到recAllgameHistory中的信息为：
#playername,streetflag,public_card,recOpLastbet,ourbet,player_card
#记录到recAllgameHistoryS中的信息为：
#playername,streetflag,public_card,recOpLastact,ouract,player_card
def get_actioninfosetall(data,nplayer):
    global recAllagentsname
    global recAllgameHistory
    global recAllgameHistoryS
    global recAllgameHistoryF
    global recAllgameHistorySF #一个赢率只记录两个决策点各一次

    #--计算每一轮的commit出来，方便加上raise值的转换
    commitchipres=[] #--记录每一轮结束的commit
    chipsnres=[]     #记录每个人的commit情况
    foldflagres=[]   #记录每个人的fold情况

    for i in range(nplayer):
        chipsnres.append(0)
        foldflagres.append(0)

    chipsnres[0]=50
    chipsnres[1]=100

    streetflag=0
    recordsn=[0,0,0,0] #--用于记录各个轮次玩家的行动数量
    recOpLastbet="none"
    recOpLastbetS="none"

    #recAllagentsstat[name]={"chand":0,"cpfvplay":0,"cpfvraise":0,"cafraise":0,"cafcall":0,
    #   "vpip":0.0,'pfr':0.0,'af':0.0,'pfrdvpip':0.0} 
    cpfvplay={}
    cpfvraise={}
    cafraise={}
    cafcall={}
    cfhandpre={}

    cafcck={}
    cafcckck={}
    cafcckrs={}
    cafccl={}
    cafcclfd={}
    cafcclcl={}
    cafcclrs={}

    cpfccl={}
    cpfcclfd={}
    cpfcclrs={}
    cpfcclcl={}
    cpfcck={}
    cpfcckrs={}
    cpfcckck={}

    cpfall={}
    cpfallfd={}
    cpfallcl={}
    cafall={}
    cafallfd={}
    cafallcl={}

    for name in recAllagentsname:
        cpfvplay[name]=False
        cpfvraise[name]=False
        cafraise[name]=0
        cafcall[name]=0
        cfhandpre[name]=0

        cafcck[name]=0
        cafcckrs[name]=0
        cafcckck[name]=0
        cafccl[name]=0
        cafcclfd[name]=0
        cafcclcl[name]=0
        cafcclrs[name]=0

        cpfccl[name]=0
        cpfcclfd[name]=0
        cpfcclrs[name]=0
        cpfcclcl[name]=0
        cpfcck[name]=0
        cpfcckrs[name]=0
        cpfcckck[name]=0

        cpfall[name]=0
        cpfallfd[name]=0
        cpfallcl[name]=0
        cafall[name]=0
        cafallfd[name]=0
        cafallcl[name]=0

    for lstaction in data["action_history"]:#--#一局的所有动作列表

        #print('len(lstaction)',len(lstaction))
        tmp_preAct='none'
        flg_CKcaseCounted={}
        flg_CLcaseCounted={}
        flg_CKcaseidx={}  #标记每一轮玩家的面临决策点的第几次
        flg_CLcaseidx={}
        flg_CKcaseCur=False
        flg_CLcaseCur=False
        for name in recAllagentsname:
            flg_CKcaseCounted[name]=False
            flg_CLcaseCounted[name]=False
            flg_CKcaseidx[name]=0
            flg_CLcaseidx[name]=0

        if (len(lstaction)>0):
            #注意street=1表示preflop
            streetflag=streetflag+1  #根据data["action_history"]中的列表数量确定street即轮次
            tmp_i_cAct=0 #用于统计当前动作在这一轮中是第几个动作
            for dictactround in lstaction:#--#每一轮的动作字典
                playername=data["players"][dictactround['position']]['name']
                playerpos=data["players"][dictactround['position']]['position']
                tmp_i_cAct+=1
                if tmp_i_cAct==1:
                    if streetflag==1:
                        recOpLastbetS="bigbet"
                    else:
                        recOpLastbetS="none"

                #--start注意：这一段是针对两人做的处理
                #当处于大盲位且是第一轮时，若对手直接fold，那么我方就会失去决策的机会，这种情况要去除掉
                #处理时根据第一轮第一个动作进行判断，若第一个动作为fold，那么对手就失去了决策的机会要去掉
                #由于第一个动作必然是小盲注位(0号位)的行动，那么必然是1号位玩家失去机会
                if streetflag==1 and dictactround["action"]=="fold" and tmp_i_cAct==1:  
                        cfhandpre[data["players"][1]['name']]=1
                #--end注意----
                flg_outAction=False
                action=""
                recordsn[streetflag-1]=recordsn[streetflag-1]+1 #--记录这一轮的玩家的动作总数
                #--无论处于哪个位置，动作的意义的一样的
                #--check意味着不加钱，raise意味着本轮从0开始加到多少，call意味着把钱加到其他最大的值
                #--因为动作历史有顺序在里头，因此不会出现问题
                #--但是要记录fold的玩家，计算投注平衡时需要避开fold的玩家
                action1=dictactround["action"]
                if (dictactround["action"]== "call"):
                    if streetflag==1:
                        cpfvplay[playername]=True

                        #--start注意：这一段是针对两人做的处理
                        if (tmp_preAct=='none' or tmp_preAct[0]=='r'):
                            if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                                cpfall[playername]+=1
                                cpfallcl[playername]+=1
                            else:
                                if (not flg_CLcaseCounted[playername]):
                                    cpfccl[playername]+=1 #既可以call，fold，raise
                                    cpfcclcl[playername]+=1
                                    flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                        #if cpfccl[playername]>1:
                        #   print('now call cpfccl=',cpfccl[playername])
                        #   anykey=input()
                        #--end注意----

                    else:
                        cafcall[playername]+=1

                        #--start注意：这一段是针对两人做的处理
                        if tmp_preAct[0]=='r':
                            if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                                cafall[playername]+=1
                                cafallcl[playername]+=1
                            else:
                                if (not flg_CLcaseCounted[playername]):
                                    cafccl[playername]+=1
                                    cafcclcl[playername]+=1
                                    flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                            '''
                            if playername=='TARD':
                                print('street=',streetflag,' preact=',tmp_preAct)
                                print('flg_CLcaseidx',playername,flg_CLcaseidx[playername])
                                print('flg_CLcaseCur',flg_CLcaseCur,flg_CKcaseCur)
                                print('call at data',data)
                                flg_outAction=True
                                anykey=input()
                            '''
                        #--end注意----

                    #--获取记录的chipsnres数组中的最大值
                    chipmax=max(chipsnres)
                    #--设置当前position为该最大值
                    chipsnres[dictactround["position"]]=chipmax
                    action="betto"+str(chipmax)
                elif (dictactround["action"]== "check"):
                    action="betto"+str(chipsnres[dictactround["position"]])

                    #--start注意：这一段是针对两人做的处理
                    if streetflag==1 and (tmp_preAct=='call' or tmp_preAct=='check'):
                        if (not flg_CKcaseCounted[playername]):
                            cpfcck[playername]+=1
                            cpfcckck[playername]+=1
                            flg_CKcaseCounted[playername]=True
                        flg_CKcaseidx[playername]+=1
                        flg_CKcaseCur=True
                        flg_CLcaseCur=False

                    if streetflag>1 and (tmp_preAct=='none' or tmp_preAct=='call' or tmp_preAct=='check'):
                        if (not flg_CKcaseCounted[playername]):
                            cafcck[playername]+=1
                            cafcckck[playername]+=1
                            flg_CKcaseCounted[playername]=True
                        flg_CKcaseidx[playername]+=1
                        flg_CKcaseCur=True
                        flg_CLcaseCur=False
                    #--end注意----

                elif (dictactround["action"]== "fold"):
                    #--将fold信息记录到foldflagres数组中
                    foldflagres[dictactround["position"]]=1
                    action="fold"
                    
                    #--start注意：这一段是针对两人做的处理
                    if streetflag==1 and (tmp_preAct=='none' or tmp_preAct[0]=='r') :
                        if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                            cpfall[playername]+=1
                            cpfallfd[playername]+=1
                        else:
                            if (not flg_CLcaseCounted[playername]):
                                cpfccl[playername]+=1    #既可以call，fold，raise
                                cpfcclfd[playername]+=1
                                flg_CLcaseCounted[playername]=True
                        flg_CLcaseidx[playername]+=1
                        flg_CKcaseCur=False
                        flg_CLcaseCur=True
                        #if cpfccl[playername]>1:
                        #   print('fold cpfccl=',cpfccl[playername])
                        #   anykey=input()
                    if streetflag>1 and tmp_preAct[0]=='r':
                        if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                            cafall[playername]+=1
                            cafallfd[playername]+=1
                        else:
                            if (not flg_CLcaseCounted[playername]):
                                cafccl[playername]+=1
                                cafcclfd[playername]+=1
                                flg_CLcaseCounted[playername]=True
                        flg_CLcaseidx[playername]+=1
                        flg_CKcaseCur=False
                        flg_CLcaseCur=True
                    #--end注意----

                else: #-- raise,从第二个字符开始的字符串转换成数字
                    if streetflag==1:
                        cpfvplay[playername]=True
                        cpfvraise[playername]=True

                        #--start注意：这一段是针对两人做的处理
                        if (tmp_preAct=='none' or tmp_preAct[0]=='r'):
                            if (not flg_CLcaseCounted[playername]):
                                cpfccl[playername]+=1 #既可以call，fold，raise
                                cpfcclrs[playername]+=1
                                flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                        elif (tmp_preAct=='call' or tmp_preAct=='check'):
                            if (not flg_CKcaseCounted[playername]):
                                cpfcck[playername]+=1
                                cpfcckrs[playername]+=1
                                flg_CKcaseCounted[playername]=True
                            flg_CKcaseidx[playername]+=1
                            flg_CKcaseCur=True
                            flg_CLcaseCur=False
                        #--end注意----

                    else:
                        cafraise[playername]+=1

                        #--start注意：这一段是针对两人做的处理
                        if tmp_preAct=='none' or tmp_preAct=='call' or tmp_preAct=='check':
                            if (not flg_CKcaseCounted[playername]):
                                cafcck[playername]+=1
                                cafcckrs[playername]+=1
                                flg_CKcaseCounted[playername]=True
                            flg_CKcaseidx[playername]+=1
                            flg_CKcaseCur=True
                            flg_CLcaseCur=False
                        if tmp_preAct[0]=='r':
                            if (not flg_CLcaseCounted[playername]):
                                cafccl[playername]+=1
                                cafcclrs[playername]+=1
                                flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                        #--end注意----

                    
                    raiseamount=int(dictactround["action"][1:])
                    betval=0
                    if(streetflag==1):
                        betval= raiseamount
                        pot=150
                    elif(streetflag==2):
                        betval=commitchipres[0]+raiseamount
                        pot=commitchipres[0]*nplayer
                    elif(streetflag==3):
                        betval=commitchipres[1]+raiseamount
                        pot=commitchipres[1]*nplayer
                    else:
                        betval=commitchipres[2]+raiseamount
                        pot=commitchipres[2]*nplayer
                    chipsnres[dictactround["position"]]=betval
                    action="betto"+str(betval)
                    
                    if raiseamount<=2*pot:
                        action1="r1"
                    else:
                        action1="r2"
                    if betval>=20000:
                        action1="allin"
                    
                    '''
                    if raiseamount<=pot:
                        action1="r1"
                    elif raiseamount<=2*pot:
                        action1="r2"
                    elif raiseamount<=4*pot:
                        action1="r3"
                    else:
                        action1="r4"
                    if betval>=20000:
                        action1="allin"
                    if betval<1000:
                        action1='r1'
                    elif betval<10000:
                        action1='r2'
                    elif betval<20000:
                        action1='r3'
                    '''

                reconeact=[playername,streetflag,data['public_card'],recOpLastbet,action,data['player_card'][playerpos],data['player_card'][1-playerpos]]
                reconeactS=[playername,streetflag,data['public_card'],recOpLastbetS,action1,data['player_card'][playerpos],data['player_card'][1-playerpos]]
                recAllgameHistory.append(reconeact)
                recAllgameHistoryS.append(reconeactS)

                #if flg_outAction:
                #   print('reconeactS',reconeactS)
                #   flg_outAction=False

                #通过分析表明当对手raise变成allin时，我方只能是call，无论决策是raise还是call，所以会造成模糊
                #因此我们去掉对手是allin动作的情况再进行统计。
                if flg_CLcaseidx[playername]==1 and flg_CLcaseCur and recOpLastbetS != "allin":
                    recAllgameHistoryF.append(reconeact)
                    recAllgameHistorySF.append(reconeactS)
                
                if flg_CKcaseidx[playername]==1 and flg_CKcaseCur:
                    recAllgameHistoryF.append(reconeact)
                    recAllgameHistorySF.append(reconeactS)
                
                recOpLastbet=action     #bet to 的形式 +fold
                recOpLastbetS=action1   #check，call，fold，分级的r1，r2，等
                tmp_preAct=dictactround["action"] #原始形式的记录


            #--判断非fold玩家是否投注已经平衡，如果平衡了说明当前轮已经结束了，否则是未结束的
            #--投注平衡的标记，也是当前轮是否结束的标志
            equiflag=True
            equichip=0
            for i in range(nplayer):
                if (foldflagres[i]!=1): #--随便找一个非fold的玩家来记录一个当前的投注额
                    equichip=chipsnres[i]
                    break

            mnotfold=0 #--统计一下没有fold玩家的数量
            for v in foldflagres:
                if(v==0):
                    mnotfold=mnotfold+1

            #--只有当当前轮的动作记录的数量不小于非fold玩家数量时，才有可能达到平衡
            if(recordsn[streetflag-1]>=mnotfold):
                for i in range(nplayer):
                    if (foldflagres[i]!=1 and chipsnres[i]!=equichip):
                        equiflag=False
                        break
            else:
                equiflag=False

            if(equiflag):
                commitchipres.append(equichip)
                #print("street",streetflag, "is finished")
            else:
                #print("street",streetflag, "is not finished")
                pass
    
    #统计到全局变量中去
    for name in recAllagentsname:
        if cpfvplay[name]:
            recAllagentsstat[name]["cpfvplay"]+=1
        if cpfvraise[name]:
            recAllagentsstat[name]["cpfvraise"]+=1
        recAllagentsstat[name]["cfhandpre"]+=cfhandpre[name]
        recAllagentsstat[name]["cafraise"]+=cafraise[name]
        recAllagentsstat[name]["cafcall"]+=cafcall[name]

        recAllagentsstat[name]["cafcck"]+=cafcck[name]
        recAllagentsstat[name]["cafcckrs"]+=cafcckrs[name]
        recAllagentsstat[name]["cafcckck"]+=cafcckck[name]
        recAllagentsstat[name]["cafccl"]+=cafccl[name]
        recAllagentsstat[name]["cafcclfd"]+=cafcclfd[name]
        recAllagentsstat[name]["cafcclcl"]+=cafcclcl[name]
        recAllagentsstat[name]["cafcclrs"]+=cafcclrs[name]

        recAllagentsstat[name]["cpfcck"]+=cpfcck[name]
        recAllagentsstat[name]["cpfcckrs"]+=cpfcckrs[name]
        recAllagentsstat[name]["cpfcckck"]+=cpfcckck[name]
        recAllagentsstat[name]["cpfccl"]+=cpfccl[name]
        recAllagentsstat[name]["cpfcclfd"]+=cpfcclfd[name]
        recAllagentsstat[name]["cpfcclcl"]+=cpfcclcl[name]
        recAllagentsstat[name]["cpfcclrs"]+=cpfcclrs[name]

        recAllagentsstat[name]["cpfall"]+=cpfall[name]
        recAllagentsstat[name]["cpfallcl"]+=cpfallcl[name]
        recAllagentsstat[name]["cpfallfd"]+=cpfallfd[name]
        recAllagentsstat[name]["cafall"]+=cafall[name]
        recAllagentsstat[name]["cafallcl"]+=cafallcl[name]
        recAllagentsstat[name]["cafallfd"]+=cafallfd[name]


    return None





# 对手模型的重建
# 不同属性表征的信息集的模型
# preflop轮用，因为preflop的特殊性
def statmodeldiffattrinojson(filename,opname1,decisionpt,gamenumbreak=0,wdin=0.02,cpf=20):


    datadealingnojsonres(filename,gamenumbreak)

    # 对手fps决策点的可观测行动的数据
    opname=opname1 #'TARD'
    round='preflop'

    raisehdsetrec=[]
    raisetpsetrec=np.zeros(169)
    raisetimesrec=0
    callhdsetrec=[]
    calltpsetrec=np.zeros(169)
    calltimesrec=0
    foldtimesrec=0

    wroftprec=np.zeros(169)
    for i in range(1326):
        handrk=get_handrank(IdxtoHand[i][0],IdxtoHand[i][1],0,0,0,0,0)
        wroftprec[handrk-1]=get_winrate(2,0,IdxtoHand[i][0],IdxtoHand[i][1],0,0,0,0,0)
    #print('wroftprec=',wroftprec)

    recAllgameHisTMP=recAllgameHistorySF
    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]

        if decisionpt=='fps':
            if playername==opname and streetnumb==1 and opponentac=='bigbet':
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                        raisetpsetrec[gethandrank(playercard,[])-1]+=1
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                        handrk=gethandrank(playercard,[])
                        #print('handrk=',handrk)
                        calltpsetrec[handrk-1]+=1
                elif playeractn[0]=='f':
                    foldtimesrec+=1
        elif decisionpt=='prp':
            if playername==opname and streetnumb==1 and (opponentac=='bigbet' or opponentac[0]=='r'):
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                        raisetpsetrec[gethandrank(playercard,[])-1]+=1
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                        handrk=gethandrank(playercard,[])
                        #print('handrk=',handrk)
                        calltpsetrec[handrk-1]+=1
                elif playeractn[0]=='f':
                    foldtimesrec+=1

    print('raisetimesrec=',raisetimesrec)
    print('calltimesrec=',calltimesrec)
    print('foldtimesrec=',foldtimesrec)
    #print('raisehdsetrec',raisehdsetrec)
    #print('raisetpsetrec',raisetpsetrec.tolist())

    
    npdfpts=101
    xstt=0.323032
    xend=0.852037
    minWridx=int(np.ceil(xstt*(npdfpts-1)))
    maxWridx=int(np.floor(xend*(npdfpts-1)))
    actionwrrange=np.linspace(0,1,npdfpts)

    widthdens=wdin
    #使用核密度估计计算pdf
    actiondensity=np.array(evalActionDensity(actionwrrange,raisehdsetrec,widthdens)) #核密度估计分布
    actiondensity[:minWridx]=0.0
    actiondensity[maxWridx:]=0.0
    
    #各信息集直接统计统计
    actiontpstat=[]
    for i in range(169):
        actiontpstat.append([i,wroftprec[i]])
    #按赢率把169种手牌牌型进行排序
    actiontpstatsorted=sorted(actiontpstat,key=lambda x: x[1])
    actiondensity1=[]
    for i in range(169):
        actiondensity1.append(raisetpsetrec[actiontpstatsorted[i][0]])
    actiondensity2=np.array(actiondensity1)
    #print('actiondensity1=',actiondensity1,sum(actiondensity1))

    #转换成0-1范围的核密度估计
    actiontps=[]
    for i in range(169):
        if int(actiondensity1[i])>0:
            actiontps+=[i/169.0]*(int(actiondensity1[i]))
    actiontprange=np.linspace(0,1,169)
    actiondensity3=np.array(evalActionDensity(actiontprange,actiontps,widthdens))
    #print('actiondensity3=',actiondensity3)

    #转成赢率的核密度估计
    actiontps=[]
    for i in range(169):
        if int(actiondensity1[i])>0:
            actiontps+=[actiontpstatsorted[i][1]]*(int(actiondensity1[i]))
    actiontprange=np.array(actiontpstatsorted)[:,1]
    actiondensity4=np.array(evalActionDensity(actiontprange,actiontps,widthdens))
    #print('actiondensity4=',actiondensity4)

    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='density of 1326 hands')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity2,marker='o',label='probability on 169 types')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity3,marker='>',label='density1 on 169 types')
    #plt.scatter(np.array(actiontpstatsorted)[:,1],actiondensity4,color='green',marker='<',label='density transfered from 169 types')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.legend(frameon=False,loc='upper right')
    plt.xlim(xstt,xend)
    plt.grid() 
    plt.title('Pdf of raise at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-raise.pdf'.format(decisionpt,round,opname))

    plt.figure()
    plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity2,marker='o',label='Frequency on 169 types')
    plt.xlabel('Wr')
    plt.ylabel('Frequency')
    plt.legend(frameon=False)
    plt.xlim(xstt,xend)
    plt.grid() 
    plt.title('Frequency of raise at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-prob-ker-{}-{}-{}-raise.pdf'.format(decisionpt,round,opname))


    #使用核密度估计计算pdf
    actiondensity5=np.array(evalActionDensity(actionwrrange,callhdsetrec,widthdens)) #核密度估计分布
    actiondensity5[:minWridx]=0.0
    actiondensity5[maxWridx:]=0.0
    
    #各信息集直接统计统计
    actiondensity6=[]
    for i in range(169):
        actiondensity6.append(calltpsetrec[actiontpstatsorted[i][0]])
    actiondensity6=np.array(actiondensity6)

    plt.figure()
    plt.plot(actionwrrange,actiondensity5,label='density of 1326 hands')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.xlim(xstt,xend)
    plt.grid() 
    plt.legend(frameon=False)
    plt.title('Pdf of call at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-call.pdf'.format(decisionpt,round,opname))

    plt.figure()
    plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity6,marker='o',label='Frequency on 169 types')
    plt.xlabel('Wr')
    plt.ylabel('Frequency')
    plt.xlim(xstt,xend)
    plt.grid() 
    plt.legend(frameon=False)
    plt.title('Frequency of call at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-prob-ker-{}-{}-{}-call.pdf'.format(decisionpt,round,opname))


    #fold情况下的信息进行估计
    roffold=foldtimesrec/(raisetimesrec+foldtimesrec+calltimesrec)
    if roffold>0.01:
        rofraise=raisetimesrec/(raisetimesrec+foldtimesrec+calltimesrec)
        print('ration of fold',roffold)
        cmprnew=cpf
        eshift=ActModeldistSe(1-roffold,cmprnew)
        Cv=[ActModeldistScv(handratiofromWR(wr),eshift,cmprnew) for wr in actionwrrange]

        #print('handratiofromWR=',[handratiofromWR(wr) for wr in actionwrrange])

        '''
        eshift1=ActModeldistSe(rofraise,cmprnew)
        Cv1=[ActModeldistScv(handratiofromWR(wr),eshift1,cmprnew) for wr in actionwrrange]
        plt.figure()
        plt.plot(actionwrrange,Cv)
        plt.plot(actionwrrange,Cv1)
        plt.figure()
        plt.plot(actionwrrange,1-np.array(Cv))
        plt.plot(actionwrrange,np.array(Cv)-np.array(Cv1))
        plt.plot(actionwrrange,np.array(Cv1))
        '''
        #print('Cv=',Cv)
        #anykey=input()

        freqc=np.array(actiondensity)*raisetimesrec
        freqr=np.array(actiondensity5)*calltimesrec
        #print('actionwrrange=',actionwrrange)

        freqfCv=(1-np.array(Cv))/roffold*foldtimesrec

        #flambdc=lambda x:1-x if x>0.0000001 else 0.0
        denfCvIwr=(1-np.array(Cv))/roffold*np.array([handderivwithWR(wr) for wr in actionwrrange])
        denfCvIwr[:minWridx]=0.0
        denfCvIwr[maxWridx:]=0.0
        #plt.figure()
        #plt.plot(actionwrrange,freqfCv)

        flambda=lambda i:(1-Cv[i])*(freqc[i]+freqr[i])/(Cv[i]) if i >= minWridx and i<=maxWridx else 0.0
        flambdb=lambda i: freqfCv[i] if flambda(i)> freqfCv[i] else flambda(i)
        freqf=[flambdb(i) for i in range(npdfpts)]
        #print('freqc=',freqc,np.sum(freqc))
        #print('freqr=',freqr,np.sum(freqr))
        #print('freqf=',freqf,np.sum(freqf))
        #folddensity=[freqf[i]/foldtimesrec for i in range(npdfpts)]
        folddensity=denfCvIwr
    else:
        freqf=np.zeros(npdfpts)

    #概率密度绘图
    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='raise eval')
    plt.plot(actionwrrange,actiondensity5,label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,folddensity,label='fold eval')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    #plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action pdf at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-pdf-{}-{}-{}.pdf'.format(decisionpt,round,opname))


    #行动概率计算
    #使用密度估计
    probactions1=np.zeros((npdfpts,3))
    probreal1=np.zeros((npdfpts,3))
    if roffold>0.01:
        for i in range(npdfpts):
            if freqf[i]>0 or actiondensity[i]*raisetimesrec>0 or actiondensity5[i]*calltimesrec>0:
                probactions1[i,0]=actiondensity[i]*raisetimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,1]=actiondensity5[i]*calltimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,2]=freqf[i]/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
    else:
        for i in range(npdfpts):
            if actiondensity[i]*raisetimesrec>0 or actiondensity5[i]*calltimesrec>0:
                probactions1[i,0]=actiondensity[i]*raisetimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,1]=actiondensity5[i]*calltimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)



    #使用直接统计
    probactions2=np.zeros((169,2))
    probreal2=np.zeros((169,3))
    for i in range(169):
        if actiondensity2[i]>0 or actiondensity6[i]>0:
            probactions2[i,0]=actiondensity2[i]/(actiondensity2[i]+actiondensity6[i])
            probactions2[i,1]=actiondensity6[i]/(actiondensity2[i]+actiondensity6[i])
        else:
            probactions2[i,:]=0


    if opname=='LA':
        alpha=0.3
        beta=0.45
        for i in range(minWridx,maxWridx+1):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=0.0
                probreal1[i,1]=1.0
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1.0
                probreal1[i,1]=0.0
                probreal1[i,2]=0.0
        for i in range(169):
            if np.array(actiontpstatsorted)[i,1]<alpha:
                probreal2[i,0]=0.0  #raise
                probreal2[i,1]=0.0  #call
                probreal2[i,2]=1.0  #fold
            elif np.array(actiontpstatsorted)[i,1]<beta:
                probreal2[i,0]=0.0
                probreal2[i,1]=1.0
                probreal2[i,2]=0.0
            else:
                probreal2[i,0]=1.0
                probreal2[i,1]=0.0
                probreal2[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(maxWridx-minWridx+1)
        Eap2=(np.sum(np.abs(probreal2[:,0]-probactions2[:,0]))+np.sum(np.abs(probreal2[:,1]-probactions2[:,1])))/2/169
        print('Eap1={}, Eap2={}'.format(Eap1,Eap2))
    elif opname=='LARD':
        alpha=0.3
        beta=0.45
        zeta1=0.7
        zeta2=0.2
        for i in range(minWridx,maxWridx+1):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        for i in range(169):
            if np.array(actiontpstatsorted)[i,1]<alpha:
                probreal2[i,0]=0.0  #raise
                probreal2[i,1]=0.0  #call
                probreal2[i,2]=1.0  #fold
            elif np.array(actiontpstatsorted)[i,1]<beta:
                probreal2[i,0]=1-zeta1
                probreal2[i,1]=zeta1
                probreal2[i,2]=0.0
            else:
                probreal2[i,0]=1-zeta2
                probreal2[i,1]=zeta2
                probreal2[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(maxWridx-minWridx+1)
        Eap2=(np.sum(np.abs(probreal2[:,0]-probactions2[:,0]))+np.sum(np.abs(probreal2[:,1]-probactions2[:,1])))/2/169
        print('Eap1={}, Eap2={}'.format(Eap1,Eap2))
    elif opname=='TARD':
        alpha=0.55
        beta=0.65
        zeta1=0.7
        zeta2=0.2
        for i in range(minWridx,maxWridx+1):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/3/(maxWridx-minWridx+1)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))

        #使用积分计算不同行动的比例
        #这是一个积分问题而不是一个密度转换问题，所以变量转换时不需要乘上导数
        #int_0^1 f(I)dI=int_0^1 g(Wr(I))dI
        #用数值方法求：
        nsects=400
        Isetstemp=np.linspace(0,1,nsects+1) #分为nsects段
        Isets=np.zeros(nsects)
        wrsets=np.zeros(nsects)
        dI=1/nsects
        racts=np.zeros((nsects,3))
        for i in range(nsects):
            Isets[i]=(Isetstemp[i+1]+Isetstemp[i])/2
            wrsets[i]=WRfromhandratio(Isets[i])
            if wrsets[i]<alpha:
                racts[i,0]=0.0*dI  #raise
                racts[i,1]=0.0*dI  #call
                racts[i,2]=1.0*dI  #fold
            elif wrsets[i]<beta:
                racts[i,0]=(1-zeta1)*dI
                racts[i,1]=zeta1*dI
                racts[i,2]=0.0*dI
            else:
                racts[i,0]=(1-zeta2)*dI
                racts[i,1]=zeta2*dI
                racts[i,2]=0.0*dI
        #print('racts=',racts)
        
        sacts=np.sum(racts,axis=0)
        print('sacts',sacts)
        print('ratio of acts raise={},call={},fold={}'.format(sacts[0]/np.sum(sacts),sacts[1]/np.sum(sacts),sacts[2]/np.sum(sacts)))
            
    elif opname=='MS':
        domains=[[0.0,0.4],[0.4,0.65],[0.65,1]]
        probreal1a=np.array(outDmodel(domains,npdfpts,xstt,xend)).T
        probreal1[:,0]=probreal1a[:,2]
        probreal1[:,1]=probreal1a[:,1]
        probreal1[:,2]=probreal1a[:,0]
        probreal1[:minWridx,:]=0.0
        probreal1[maxWridx:,:]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/3/(maxWridx-minWridx+1)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))


    plt.figure()
    plt.plot(actionwrrange,probactions1[:,0],label='raise eval')
    plt.plot(actionwrrange,probactions1[:,1],label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,probactions1[:,2],label='fold eval')
    plt.plot(actionwrrange,probreal1[:,0],'--',label='raise real')
    plt.plot(actionwrrange,probreal1[:,1],'--',label='call real')
    if roffold>0.01:
        plt.plot(actionwrrange,probreal1[:,2],'--',label='fold real ')

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}.pdf'.format(decisionpt,round,opname))


    plt.figure()
    plt.plot(np.array(actiontpstatsorted)[:,1],probactions2[:,0],label='raise')
    plt.plot(np.array(actiontpstatsorted)[:,1],probactions2[:,1],label='call')
    plt.plot(actionwrrange,probreal1[:,0],'--',label='raise real')
    plt.plot(actionwrrange,probreal1[:,1],'--',label='call real')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}-stat.pdf'.format(decisionpt,round,opname))


    plt.show()
    return Eap1







# 对手模型的重建
# 不同属性表征的信息集的模型
# preflop轮用，因为preflop的特殊性
def statmodeldiffattri(filename,opname1,decisionpt,gamenumbreak=0,wdin=0.02,cpf=20):


    datadealing3(filename,gamenumbreak)

    # 对手fps决策点的可观测行动的数据
    opname=opname1 #'TARD'
    round='preflop'

    raisehdsetrec=[]
    raisetpsetrec=np.zeros(169)
    raisetimesrec=0
    callhdsetrec=[]
    calltpsetrec=np.zeros(169)
    calltimesrec=0
    foldtimesrec=0

    wroftprec=np.zeros(169)
    for i in range(1326):
        handrk=get_handrank(IdxtoHand[i][0],IdxtoHand[i][1],0,0,0,0,0)
        wroftprec[handrk-1]=get_winrate(2,0,IdxtoHand[i][0],IdxtoHand[i][1],0,0,0,0,0)
    #print('wroftprec=',wroftprec)

    recAllgameHisTMP=recAllgameHistorySF
    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]

        if decisionpt=='fps':
            if playername==opname and streetnumb==1 and opponentac=='bigbet':
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                        raisetpsetrec[gethandrank(playercard,[])-1]+=1
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                        handrk=gethandrank(playercard,[])
                        #print('handrk=',handrk)
                        calltpsetrec[handrk-1]+=1
                elif playeractn[0]=='f':
                    foldtimesrec+=1
        elif decisionpt=='prp':
            if playername==opname and streetnumb==1 and (opponentac=='bigbet' or opponentac[0]=='r'):
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                        raisetpsetrec[gethandrank(playercard,[])-1]+=1
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                        handrk=gethandrank(playercard,[])
                        #print('handrk=',handrk)
                        calltpsetrec[handrk-1]+=1
                elif playeractn[0]=='f':
                    foldtimesrec+=1

    print('raisetimesrec=',raisetimesrec)
    print('calltimesrec=',calltimesrec)
    print('foldtimesrec=',foldtimesrec)
    #print('raisehdsetrec',raisehdsetrec)
    #print('raisetpsetrec',raisetpsetrec.tolist())

    
    npdfpts=101
    xstt=0.323032
    xend=0.852037
    minWridx=int(np.ceil(xstt*(npdfpts-1)))
    maxWridx=int(np.floor(xend*(npdfpts-1)))
    actionwrrange=np.linspace(0,1,npdfpts)

    widthdens=wdin
    #使用核密度估计计算pdf
    actiondensity=np.array(evalActionDensity(actionwrrange,raisehdsetrec,widthdens)) #核密度估计分布
    actiondensity[:minWridx]=0.0
    actiondensity[maxWridx:]=0.0
    
    #各信息集直接统计统计
    actiontpstat=[]
    for i in range(169):
        actiontpstat.append([i,wroftprec[i]])
    #按赢率把169种手牌牌型进行排序
    actiontpstatsorted=sorted(actiontpstat,key=lambda x: x[1])
    actiondensity1=[]
    for i in range(169):
        actiondensity1.append(raisetpsetrec[actiontpstatsorted[i][0]])
    actiondensity2=np.array(actiondensity1)
    #print('actiondensity1=',actiondensity1,sum(actiondensity1))

    #转换成0-1范围的核密度估计
    actiontps=[]
    for i in range(169):
        if int(actiondensity1[i])>0:
            actiontps+=[i/169.0]*(int(actiondensity1[i]))
    actiontprange=np.linspace(0,1,169)
    actiondensity3=np.array(evalActionDensity(actiontprange,actiontps,widthdens))
    #print('actiondensity3=',actiondensity3)

    #转成赢率的核密度估计
    actiontps=[]
    for i in range(169):
        if int(actiondensity1[i])>0:
            actiontps+=[actiontpstatsorted[i][1]]*(int(actiondensity1[i]))
    actiontprange=np.array(actiontpstatsorted)[:,1]
    actiondensity4=np.array(evalActionDensity(actiontprange,actiontps,widthdens))
    #print('actiondensity4=',actiondensity4)

    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='density of 1326 hands')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity2,marker='o',label='probability on 169 types')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity3,marker='>',label='density1 on 169 types')
    #plt.scatter(np.array(actiontpstatsorted)[:,1],actiondensity4,color='green',marker='<',label='density transfered from 169 types')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.legend(frameon=False,loc='upper right')
    plt.xlim(xstt,xend)
    plt.grid() 
    plt.title('Pdf of raise at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-raise.pdf'.format(decisionpt,round,opname))

    plt.figure()
    plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity2,marker='o',label='Frequency on 169 types')
    plt.xlabel('Wr')
    plt.ylabel('Frequency')
    plt.legend(frameon=False)
    plt.xlim(xstt,xend)
    plt.grid() 
    plt.title('Frequency of raise at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-prob-ker-{}-{}-{}-raise.pdf'.format(decisionpt,round,opname))


    #使用核密度估计计算pdf
    actiondensity5=np.array(evalActionDensity(actionwrrange,callhdsetrec,widthdens)) #核密度估计分布
    actiondensity5[:minWridx]=0.0
    actiondensity5[maxWridx:]=0.0
    
    #各信息集直接统计统计
    actiondensity6=[]
    for i in range(169):
        actiondensity6.append(calltpsetrec[actiontpstatsorted[i][0]])
    actiondensity6=np.array(actiondensity6)

    plt.figure()
    plt.plot(actionwrrange,actiondensity5,label='density of 1326 hands')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.xlim(xstt,xend)
    plt.grid() 
    plt.legend(frameon=False)
    plt.title('Pdf of call at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-call.pdf'.format(decisionpt,round,opname))

    plt.figure()
    plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity6,marker='o',label='Frequency on 169 types')
    plt.xlabel('Wr')
    plt.ylabel('Frequency')
    plt.xlim(xstt,xend)
    plt.grid() 
    plt.legend(frameon=False)
    plt.title('Frequency of call at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-prob-ker-{}-{}-{}-call.pdf'.format(decisionpt,round,opname))


    #fold情况下的信息进行估计
    roffold=foldtimesrec/(raisetimesrec+foldtimesrec+calltimesrec)
    if roffold>0.01:
        rofraise=raisetimesrec/(raisetimesrec+foldtimesrec+calltimesrec)
        print('ration of fold',roffold)
        cmprnew=cpf
        eshift=ActModeldistSe(1-roffold,cmprnew)
        Cv=[ActModeldistScv(handratiofromWR(wr),eshift,cmprnew) for wr in actionwrrange]

        #print('handratiofromWR=',[handratiofromWR(wr) for wr in actionwrrange])

        '''
        eshift1=ActModeldistSe(rofraise,cmprnew)
        Cv1=[ActModeldistScv(handratiofromWR(wr),eshift1,cmprnew) for wr in actionwrrange]
        plt.figure()
        plt.plot(actionwrrange,Cv)
        plt.plot(actionwrrange,Cv1)
        plt.figure()
        plt.plot(actionwrrange,1-np.array(Cv))
        plt.plot(actionwrrange,np.array(Cv)-np.array(Cv1))
        plt.plot(actionwrrange,np.array(Cv1))
        '''
        #print('Cv=',Cv)
        #anykey=input()

        freqc=np.array(actiondensity)*raisetimesrec
        freqr=np.array(actiondensity5)*calltimesrec
        #print('actionwrrange=',actionwrrange)

        freqfCv=(1-np.array(Cv))/roffold*foldtimesrec

        #flambdc=lambda x:1-x if x>0.0000001 else 0.0
        denfCvIwr=(1-np.array(Cv))/roffold*np.array([handderivwithWR(wr) for wr in actionwrrange])
        denfCvIwr[:minWridx]=0.0
        denfCvIwr[maxWridx:]=0.0
        #plt.figure()
        #plt.plot(actionwrrange,freqfCv)

        flambda=lambda i:(1-Cv[i])*(freqc[i]+freqr[i])/(Cv[i]) if i >= minWridx and i<=maxWridx else 0.0
        flambdb=lambda i: freqfCv[i] if flambda(i)> freqfCv[i] else flambda(i)
        freqf=[flambdb(i) for i in range(npdfpts)]
        #print('freqc=',freqc,np.sum(freqc))
        #print('freqr=',freqr,np.sum(freqr))
        #print('freqf=',freqf,np.sum(freqf))
        #folddensity=[freqf[i]/foldtimesrec for i in range(npdfpts)]
        folddensity=denfCvIwr
    else:
        freqf=np.zeros(npdfpts)

    #概率密度绘图
    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='raise eval')
    plt.plot(actionwrrange,actiondensity5,label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,folddensity,label='fold eval')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    #plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action pdf at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-pdf-{}-{}-{}.pdf'.format(decisionpt,round,opname))


    #行动概率计算
    #使用密度估计
    probactions1=np.zeros((npdfpts,3))
    probreal1=np.zeros((npdfpts,3))
    if roffold>0.01:
        for i in range(npdfpts):
            if freqf[i]>0 or actiondensity[i]*raisetimesrec>0 or actiondensity5[i]*calltimesrec>0:
                probactions1[i,0]=actiondensity[i]*raisetimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,1]=actiondensity5[i]*calltimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,2]=freqf[i]/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
    else:
        for i in range(npdfpts):
            if actiondensity[i]*raisetimesrec>0 or actiondensity5[i]*calltimesrec>0:
                probactions1[i,0]=actiondensity[i]*raisetimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,1]=actiondensity5[i]*calltimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)



    #使用直接统计
    probactions2=np.zeros((169,2))
    probreal2=np.zeros((169,3))
    for i in range(169):
        if actiondensity2[i]>0 or actiondensity6[i]>0:
            probactions2[i,0]=actiondensity2[i]/(actiondensity2[i]+actiondensity6[i])
            probactions2[i,1]=actiondensity6[i]/(actiondensity2[i]+actiondensity6[i])
        else:
            probactions2[i,:]=0


    if opname=='LA':
        alpha=0.3
        beta=0.45
        for i in range(minWridx,maxWridx+1):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=0.0
                probreal1[i,1]=1.0
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1.0
                probreal1[i,1]=0.0
                probreal1[i,2]=0.0
        for i in range(169):
            if np.array(actiontpstatsorted)[i,1]<alpha:
                probreal2[i,0]=0.0  #raise
                probreal2[i,1]=0.0  #call
                probreal2[i,2]=1.0  #fold
            elif np.array(actiontpstatsorted)[i,1]<beta:
                probreal2[i,0]=0.0
                probreal2[i,1]=1.0
                probreal2[i,2]=0.0
            else:
                probreal2[i,0]=1.0
                probreal2[i,1]=0.0
                probreal2[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(maxWridx-minWridx+1)
        Eap2=(np.sum(np.abs(probreal2[:,0]-probactions2[:,0]))+np.sum(np.abs(probreal2[:,1]-probactions2[:,1])))/2/169
        print('Eap1={}, Eap2={}'.format(Eap1,Eap2))
    elif opname=='LARD':
        alpha=0.3
        beta=0.45
        zeta1=0.7
        zeta2=0.2
        for i in range(minWridx,maxWridx+1):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        for i in range(169):
            if np.array(actiontpstatsorted)[i,1]<alpha:
                probreal2[i,0]=0.0  #raise
                probreal2[i,1]=0.0  #call
                probreal2[i,2]=1.0  #fold
            elif np.array(actiontpstatsorted)[i,1]<beta:
                probreal2[i,0]=1-zeta1
                probreal2[i,1]=zeta1
                probreal2[i,2]=0.0
            else:
                probreal2[i,0]=1-zeta2
                probreal2[i,1]=zeta2
                probreal2[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(maxWridx-minWridx+1)
        Eap2=(np.sum(np.abs(probreal2[:,0]-probactions2[:,0]))+np.sum(np.abs(probreal2[:,1]-probactions2[:,1])))/2/169
        print('Eap1={}, Eap2={}'.format(Eap1,Eap2))
    elif opname=='TARD':
        alpha=0.55
        beta=0.65
        zeta1=0.7
        zeta2=0.2
        for i in range(minWridx,maxWridx+1):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/3/(maxWridx-minWridx+1)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))

        #使用积分计算不同行动的比例
        #这是一个积分问题而不是一个密度转换问题，所以变量转换时不需要乘上导数
        #int_0^1 f(I)dI=int_0^1 g(Wr(I))dI
        #用数值方法求：
        nsects=400
        Isetstemp=np.linspace(0,1,nsects+1) #分为nsects段
        Isets=np.zeros(nsects)
        wrsets=np.zeros(nsects)
        dI=1/nsects
        racts=np.zeros((nsects,3))
        for i in range(nsects):
            Isets[i]=(Isetstemp[i+1]+Isetstemp[i])/2
            wrsets[i]=WRfromhandratio(Isets[i])
            if wrsets[i]<alpha:
                racts[i,0]=0.0*dI  #raise
                racts[i,1]=0.0*dI  #call
                racts[i,2]=1.0*dI  #fold
            elif wrsets[i]<beta:
                racts[i,0]=(1-zeta1)*dI
                racts[i,1]=zeta1*dI
                racts[i,2]=0.0*dI
            else:
                racts[i,0]=(1-zeta2)*dI
                racts[i,1]=zeta2*dI
                racts[i,2]=0.0*dI
        #print('racts=',racts)
        
        sacts=np.sum(racts,axis=0)
        print('sacts',sacts)
        print('ratio of acts raise={},call={},fold={}'.format(sacts[0]/np.sum(sacts),sacts[1]/np.sum(sacts),sacts[2]/np.sum(sacts)))
            
    elif opname=='MS':
        domains=[[0.0,0.4],[0.4,0.65],[0.65,1]]
        probreal1a=np.array(outDmodel(domains,npdfpts,xstt,xend)).T
        probreal1[:,0]=probreal1a[:,2]
        probreal1[:,1]=probreal1a[:,1]
        probreal1[:,2]=probreal1a[:,0]
        probreal1[:minWridx,:]=0.0
        probreal1[maxWridx:,:]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/3/(maxWridx-minWridx+1)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))


    plt.figure()
    plt.plot(actionwrrange,probactions1[:,0],label='raise eval')
    plt.plot(actionwrrange,probactions1[:,1],label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,probactions1[:,2],label='fold eval')
    plt.plot(actionwrrange,probreal1[:,0],'--',label='raise real')
    plt.plot(actionwrrange,probreal1[:,1],'--',label='call real')
    if roffold>0.01:
        plt.plot(actionwrrange,probreal1[:,2],'--',label='fold real ')

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}.pdf'.format(decisionpt,round,opname))


    plt.figure()
    plt.plot(np.array(actiontpstatsorted)[:,1],probactions2[:,0],label='raise')
    plt.plot(np.array(actiontpstatsorted)[:,1],probactions2[:,1],label='call')
    plt.plot(actionwrrange,probreal1[:,0],'--',label='raise real')
    plt.plot(actionwrrange,probreal1[:,1],'--',label='call real')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}-stat.pdf'.format(decisionpt,round,opname))


    plt.show()
    return Eap1





# 对手模型的重建
# 不同属性表征的信息集的模型
# afterflop轮用
def statmodeldiffattriAF(filename,opname1,decisionpt,gamenumbreak=0,wdin=0.015,cpf=50):


    datadealing3(filename,gamenumbreak)

    # 对手fps决策点的可观测行动的数据
    opname=opname1 #'TARD'
    round='afterflop'

    raisehdsetrec=[]
    raisetimesrec=0
    callhdsetrec=[]
    calltimesrec=0
    foldtimesrec=0

    recAllgameHisTMP=recAllgameHistorySF
    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]

        if decisionpt=='pra':
            if playername==opname and streetnumb==2 and opponentac[0]=='r':
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        if streetnumb==2:
                            playerwinrate=getwinrate(2,playercard,boardcards[:3])
                        elif streetnumb==3:
                            playerwinrate=getwinrate(2,playercard,boardcards[:4])
                        elif streetnumb==4:
                            playerwinrate=getwinrate(2,playercard,boardcards)
                        raisehdsetrec.append(playerwinrate)
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        if streetnumb==2:
                            playerwinrate=getwinrate(2,playercard,boardcards[:3])
                        elif streetnumb==3:
                            playerwinrate=getwinrate(2,playercard,boardcards[:4])
                        elif streetnumb==4:
                            playerwinrate=getwinrate(2,playercard,boardcards)
                        callhdsetrec.append(playerwinrate)
                elif playeractn[0]=='f':
                    foldtimesrec+=1

    print('raisetimesrec=',raisetimesrec)
    print('calltimesrec=',calltimesrec)
    print('foldtimesrec=',foldtimesrec)
    #print('raisehdsetrec',raisehdsetrec)
    #print('raisetpsetrec',raisetpsetrec.tolist())

    npdfpts=101
    actionwrrange=np.linspace(0,1,npdfpts)

    widthdens=wdin
    #使用核密度估计计算pdf-raise
    actiondensity=np.array(evalActionDensity(actionwrrange,raisehdsetrec,widthdens)) #核密度估计分布
    print('minwr-raise=',np.min(raisehdsetrec))
    #minWridx=int(np.round(np.min(raisehdsetrec)*(npdfpts-1)))
    #actiondensity[:minWridx]=0.0


    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='density of raise')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity2,marker='o',label='probability on 169 types')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity3,marker='>',label='density1 on 169 types')
    #plt.scatter(np.array(actiontpstatsorted)[:,1],actiondensity4,color='green',marker='<',label='density transfered from 169 types')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.legend(frameon=False,loc='upper right')
    plt.xlim(0,1)
    plt.grid() 
    plt.title('Pdf of raise at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-raise.pdf'.format(decisionpt,round,opname))

    #使用核密度估计计算pdf-call
    actiondensity5=np.array(evalActionDensity(actionwrrange,callhdsetrec,widthdens)) #核密度估计分布
    print('minwr-call=',np.min(callhdsetrec))
    #minWridx=int(np.round(np.min(callhdsetrec)*(npdfpts-1)))
    #actiondensity5[:minWridx]=0.0
    
    plt.figure()
    plt.plot(actionwrrange,actiondensity5,label='density of call')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.xlim(0,1)
    plt.grid() 
    plt.legend(frameon=False)
    plt.title('Pdf of call at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-call.pdf'.format(decisionpt,round,opname))

    #fold情况下的信息进行估计
    freqr=actiondensity*raisetimesrec
    freqc=actiondensity5*calltimesrec
    roffold=foldtimesrec/(foldtimesrec+calltimesrec+raisetimesrec)
    if roffold>0.01:
        print('ration of fold',roffold)
        cmprnew=cpf
        eshift=ActModeldistSe(1-roffold,cmprnew)
        print('e=',eshift)
        #flop轮以后不用再将wr转换为0-1实数范围表示的信息集上了
        #Cv=[ActModeldistScv(wr,eshift,cmprnew) for wr in actionwrrange]
        Iofwr=[handratiofromWR(wr,g_WrwithIflop) for wr in actionwrrange]
        Cv=[ActModeldistScv(I,eshift,cmprnew) for I in Iofwr]

        '''
        Cv1=[ActModeldistScv(wr,eshift,cmprnew) for wr in actionwrrange]
        plt.figure()
        plt.plot(actionwrrange,Iofwr)

        plt.figure()
        plt.plot(Iofwr,Cv,label='Cv')
        plt.plot(Iofwr,Cv1,label='Cv1')
        plt.legend()
        '''

        freqfCv=(1-np.array(Cv))/roffold*foldtimesrec

        #denfCvIwr=(1-np.array(Cv))/roffold
        denfCvIwr=(1-np.array(Cv))/roffold*np.array([handderivwithWR(wr,g_derivIvsWrflop) for wr in actionwrrange])

        flambda=lambda i:(1-Cv[i])*(freqc[i]+freqr[i])/(Cv[i]) #if freqc[i]+freqr[i]>1 else freqfCv[i]
        flambdb=lambda i: freqfCv[i] if flambda(i)> freqfCv[i] else flambda(i)
        freqf=[flambdb(i) for i in range(npdfpts)]
        freqf=freqfCv
        #print('cv=',Cv)
        #print('freqc=',freqc,np.sum(freqc))
        #print('freqr=',freqr,np.sum(freqr))
        #print('freqf=',freqf,np.sum(freqf))
        #print('freqfCv=',freqfCv,np.sum(freqfCv))
        #folddensity=[freqf[i]/foldtimesrec for i in range(npdfpts)]
        folddensity=denfCvIwr
    else:
        freqf=np.zeros(npdfpts)

    #概率密度绘图
    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='raise eval')
    plt.plot(actionwrrange,actiondensity5,label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,folddensity,label='fold eval')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    #plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action pdf at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-pdf-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    plt.figure()
    plt.plot(freqr,label='freq raise')
    plt.plot(freqc,label='freq call')
    plt.plot(freqf,label='freq fold')
    plt.plot(freqfCv,label='freq fold')


    #行动概率计算
    #使用密度估计
    probactions1=np.zeros((npdfpts,3))
    probreal1=np.zeros((npdfpts,3))
    if roffold>0.01:
        for i in range(npdfpts):
            probactions1[i,0]=actiondensity[i]*raisetimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
            probactions1[i,1]=actiondensity5[i]*calltimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
            probactions1[i,2]=freqf[i]/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
    else:
        for i in range(npdfpts):
            probactions1[i,0]=freqr[i]/(freqf[i]+freqc[i]+freqr[i])
            probactions1[i,1]=freqc[i]/(freqf[i]+freqc[i]+freqr[i])
            #probactions1[i,2]=freqf[i]/(freqf[i]+freqc[i]+freqr[i])
            #probactions1[i,0]=actiondensity[i]*raisetimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
            #probactions1[i,1]=actiondensity5[i]*calltimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)



    if opname=='LA':
        alpha=0.3
        beta=0.45
        for i in range(npdfpts):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=0.0
                probreal1[i,1]=1.0
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1.0
                probreal1[i,1]=0.0
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(maxWridx-minWridx+1)
        print('Eap1={}, gamebreak={}'.format(Eap1,gamenumbreak))
    elif opname=='LARD':
        alpha=0.3
        beta=0.45
        zeta1=0.7
        zeta2=0.2
        for i in range(npdfpts):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(npdfpts)
        print('Eap1={}, gamebreak={}'.format(Eap1,gamenumbreak))
    elif opname=='TARD' or opname=='TARDAF':
        alpha=0.55
        beta=0.65
        zeta1=0.7
        zeta2=0.2
        for i in range(npdfpts):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/3/(npdfpts)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))

        #使用积分计算不同行动的比例
        #这是一个积分问题而不是一个密度转换问题，所以变量转换时不需要乘上导数
        #int_0^1 f(I)dI=int_0^1 g(Wr(I))dI
        #用数值方法求：
        nsects=400
        Isetstemp=np.linspace(0,1,nsects+1) #分为nsects段
        Isets=np.zeros(nsects)
        wrsets=np.zeros(nsects)
        dI=1/nsects
        racts=np.zeros((nsects,3))
        for i in range(nsects):
            Isets[i]=(Isetstemp[i+1]+Isetstemp[i])/2
            wrsets[i]=WRfromhandratio(Isets[i],g_WrwithIflop)
            if wrsets[i]<alpha:
                racts[i,0]=0.0*dI  #raise
                racts[i,1]=0.0*dI  #call
                racts[i,2]=1.0*dI  #fold
            elif wrsets[i]<beta:
                racts[i,0]=(1-zeta1)*dI
                racts[i,1]=zeta1*dI
                racts[i,2]=0.0*dI
            else:
                racts[i,0]=(1-zeta2)*dI
                racts[i,1]=zeta2*dI
                racts[i,2]=0.0*dI
        #print('racts=',racts)
        
        sacts=np.sum(racts,axis=0)
        print('sacts',sacts)
        print('ratio of acts raise={},call={},fold={}'.format(sacts[0]/np.sum(sacts),sacts[1]/np.sum(sacts),sacts[2]/np.sum(sacts)))
            
    elif opname=='MS' or opname=='MSAF':
        domains=[[0.0,0.4],[0.4,0.65],[0.65,1]]
        probreal1a=np.array(outDmodel(domains,npdfpts)).T
        probreal1[:,0]=probreal1a[:,2]
        probreal1[:,1]=probreal1a[:,1]
        probreal1[:,2]=probreal1a[:,0]
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/3/(npdfpts)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))


    plt.figure()
    plt.plot(actionwrrange,probactions1[:,0],label='raise eval')
    plt.plot(actionwrrange,probactions1[:,1],label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,probactions1[:,2],label='fold eval')
    plt.plot(actionwrrange,probreal1[:,0],'--',label='raise real')
    plt.plot(actionwrrange,probreal1[:,1],'--',label='call real')
    if roffold>0.01:
        plt.plot(actionwrrange,probreal1[:,2],'--',label='fold real ')

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    plt.show()
    return None





# 对手模型的重建
# 从服务器数据重建，所有的都能用
# log文件指定后，这里不用再指定文件
def statmodeldiffattriALL(opname1,decisionpt,gamenumbreak=0):

    datadealing4(gamenumbreak)

    # 对手fps决策点的可观测行动的数据
    opname=opname1 #'TARD'
    round='afterflop'

    raisehdsetrec=[]
    raisetimesrec=0
    callhdsetrec=[]
    calltimesrec=0
    foldhdsetrec=[]
    foldtimesrec=0

    actshdsetrec=[]

    recAllgameHisTMP=recAllgameHistorySF
    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]

        if decisionpt=='pra':
            if playername==opname and streetnumb==2 and opponentac[0]=='r':
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        if streetnumb==2:
                            playerwinrate=getwinrate(2,playercard,boardcards[:3])
                        elif streetnumb==3:
                            playerwinrate=getwinrate(2,playercard,boardcards[:4])
                        elif streetnumb==4:
                            playerwinrate=getwinrate(2,playercard,boardcards)
                        raisehdsetrec.append(playerwinrate)
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        if streetnumb==2:
                            playerwinrate=getwinrate(2,playercard,boardcards[:3])
                        elif streetnumb==3:
                            playerwinrate=getwinrate(2,playercard,boardcards[:4])
                        elif streetnumb==4:
                            playerwinrate=getwinrate(2,playercard,boardcards)
                        callhdsetrec.append(playerwinrate)
                elif playeractn[0]=='f':
                    foldtimesrec+=1
                    if playercard:
                        if streetnumb==2:
                            playerwinrate=getwinrate(2,playercard,boardcards[:3])
                        elif streetnumb==3:
                            playerwinrate=getwinrate(2,playercard,boardcards[:4])
                        elif streetnumb==4:
                            playerwinrate=getwinrate(2,playercard,boardcards)
                        foldhdsetrec.append(playerwinrate)
        elif decisionpt=='pca':
            if playername==opname and streetnumb==3 and (opponentac[0]=='c' or opponentac=='none'):
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        if streetnumb==2:
                            playerwinrate=getwinrate(2,playercard,boardcards[:3])
                        elif streetnumb==3:
                            playerwinrate=getwinrate(2,playercard,boardcards[:4])
                        elif streetnumb==4:
                            playerwinrate=getwinrate(2,playercard,boardcards)
                        raisehdsetrec.append(playerwinrate)
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        if streetnumb==2:
                            playerwinrate=getwinrate(2,playercard,boardcards[:3])
                        elif streetnumb==3:
                            playerwinrate=getwinrate(2,playercard,boardcards[:4])
                        elif streetnumb==4:
                            playerwinrate=getwinrate(2,playercard,boardcards)
                        callhdsetrec.append(playerwinrate)
                elif playeractn[0]=='f':
                    foldtimesrec+=1
                    if playercard:
                        if streetnumb==2:
                            playerwinrate=getwinrate(2,playercard,boardcards[:3])
                        elif streetnumb==3:
                            playerwinrate=getwinrate(2,playercard,boardcards[:4])
                        elif streetnumb==4:
                            playerwinrate=getwinrate(2,playercard,boardcards)
                        foldhdsetrec.append(playerwinrate)

    print('raisetimesrec=',raisetimesrec)
    print('calltimesrec=',calltimesrec)
    print('foldtimesrec=',foldtimesrec)
    #print('raisehdsetrec',raisehdsetrec)
    #print('raisetpsetrec',raisetpsetrec.tolist())

    actshdsetrec=sorted(raisehdsetrec+callhdsetrec+foldhdsetrec)

    npdfpts=101
    actionwrrange=np.linspace(0,1,npdfpts)

    widthdens=0.015

    #估计所有的
    wrdensity=np.array(evalActionDensity(actionwrrange,actshdsetrec,widthdens)) #核密度估计分布
    plt.figure()
    plt.plot(actionwrrange,wrdensity)
    plt.ylim(0,5.2)
    plt.title('all actions wr dist')


    #使用核密度估计计算pdf-raise
    actiondensity=np.zeros(npdfpts)
    if raisetimesrec>0:
        actiondensity=np.array(evalActionDensity(actionwrrange,raisehdsetrec,widthdens)) #核密度估计分布
        print('minwr-raise=',np.min(raisehdsetrec))
        #minWridx=int(np.round(np.min(raisehdsetrec)*(npdfpts-1)))
        #actiondensity[:minWridx]=0.0


    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='density of 1326 hands')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity2,marker='o',label='probability on 169 types')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity3,marker='>',label='density1 on 169 types')
    #plt.scatter(np.array(actiontpstatsorted)[:,1],actiondensity4,color='green',marker='<',label='density transfered from 169 types')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.legend(frameon=False,loc='upper right')
    plt.xlim(0,1)
    plt.grid() 
    plt.title('Pdf of raise at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-raise.pdf'.format(decisionpt,round,opname))

    #使用核密度估计计算pdf-call
    actiondensity5=np.zeros(npdfpts)
    if calltimesrec>0:
        actiondensity5=np.array(evalActionDensity(actionwrrange,callhdsetrec,widthdens)) #核密度估计分布
        print('minwr-call=',np.min(callhdsetrec))
        #minWridx=int(np.round(np.min(callhdsetrec)*(npdfpts-1)))
        #actiondensity5[:minWridx]=0.0
    
    plt.figure()
    plt.plot(actionwrrange,actiondensity5,label='density of 1326 hands')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.xlim(0,1)
    plt.grid() 
    plt.legend(frameon=False)
    plt.title('Pdf of call at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-call.pdf'.format(decisionpt,round,opname))

    #使用核密度估计计算pdf-fold
    actiondensity7=np.zeros(npdfpts)
    if foldtimesrec>0:
        actiondensity7=np.array(evalActionDensity(actionwrrange,foldhdsetrec,widthdens)) #核密度估计分布
        print('minwr-fold=',np.min(foldhdsetrec))



    #概率密度绘图
    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='raise eval')
    plt.plot(actionwrrange,actiondensity5,label='call eval')
    plt.plot(actionwrrange,actiondensity7,label='fold eval')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    #plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action pdf at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-pdf-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    sumacts=raisetimesrec+calltimesrec+foldtimesrec
    ratioacts=[raisetimesrec/sumacts,calltimesrec/sumacts,foldtimesrec/sumacts]
    print('ratioacts',ratioacts)
    freqr=actiondensity*raisetimesrec
    freqc=actiondensity5*calltimesrec
    freqf=actiondensity7*foldtimesrec

    plt.figure()
    plt.plot(freqr,label='freq raise')
    plt.plot(freqc,label='freq call')
    plt.plot(freqf,label='freq fold')

    print('freqr',freqr)
    print('freqc',freqc)
    print('freqf',freqf)



    #行动概率计算
    #使用密度估计
    probactions1=np.zeros((npdfpts,3))
    probreal1=np.zeros((npdfpts,3))
    for i in range(npdfpts):
        sumfreq=freqf[i]+freqc[i]+freqr[i]
        if sumfreq>0.1:
            probactions1[i,0]=freqr[i]/sumfreq
            probactions1[i,1]=freqc[i]/sumfreq
            probactions1[i,2]=freqf[i]/sumfreq
        else:
            probactions1[i,:]=0.0

    if opname=='LA':
        alpha=0.3
        beta=0.45
        for i in range(npdfpts):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=0.0
                probreal1[i,1]=1.0
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1.0
                probreal1[i,1]=0.0
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(maxWridx-minWridx+1)
        print('Eap1={}, gamebreak={}'.format(Eap1,gamenumbreak))
    elif opname=='LARD':
        alpha=0.3
        beta=0.45
        zeta1=0.7
        zeta2=0.2
        for i in range(npdfpts):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(npdfpts)
        print('Eap1={}, gamebreak={}'.format(Eap1,gamenumbreak))
    elif opname=='TARD' or opname=='TARDAF':
        alpha=0.55
        beta=0.65
        zeta1=0.7
        zeta2=0.2
        for i in range(npdfpts):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/2/(npdfpts)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))

        #使用积分计算不同行动的比例
        #这是一个积分问题而不是一个密度转换问题，所以变量转换时不需要乘上导数
        #int_0^1 f(I)dI=int_0^1 g(Wr(I))dI
        #用数值方法求：
        nsects=400
        Isetstemp=np.linspace(0,1,nsects+1) #分为nsects段
        Isets=np.zeros(nsects)
        wrsets=np.zeros(nsects)
        dI=1/nsects
        racts=np.zeros((nsects,3))
        for i in range(nsects):
            Isets[i]=(Isetstemp[i+1]+Isetstemp[i])/2
            wrsets[i]=WRfromhandratio(Isets[i],g_WrwithIflop)
            if wrsets[i]<alpha:
                racts[i,0]=0.0*dI  #raise
                racts[i,1]=0.0*dI  #call
                racts[i,2]=1.0*dI  #fold
            elif wrsets[i]<beta:
                racts[i,0]=(1-zeta1)*dI
                racts[i,1]=zeta1*dI
                racts[i,2]=0.0*dI
            else:
                racts[i,0]=(1-zeta2)*dI
                racts[i,1]=zeta2*dI
                racts[i,2]=0.0*dI
        #print('racts=',racts)
        
        sacts=np.sum(racts,axis=0)
        print('sacts',sacts)
        print('ratio of acts raise={},call={},fold={}'.format(sacts[0]/np.sum(sacts),sacts[1]/np.sum(sacts),sacts[2]/np.sum(sacts)))
    elif opname=='MS':
        domains=[[0.0,0.4],[0.4,0.65],[0.65,1]]
        probreal1a=np.array(outDmodel(domains,npdfpts)).T
        probreal1[:,0]=probreal1a[:,2]
        probreal1[:,1]=probreal1a[:,1]
        probreal1[:,2]=probreal1a[:,0]
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/2/(npdfpts)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))


    plt.figure()
    plt.plot(actionwrrange,probactions1[:,0],label='raise eval')
    plt.plot(actionwrrange,probactions1[:,1],label='call eval')
    plt.plot(actionwrrange,probactions1[:,2],label='fold eval')
    #plt.plot(actionwrrange,probreal1[:,0],'--',label='raise real')
    #plt.plot(actionwrrange,probreal1[:,1],'--',label='call real')
    #plt.plot(actionwrrange,probreal1[:,2],'--',label='fold real ')

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    plt.show()
    return None




# 对手模型的重建
# 从服务器数据重建，所有的都能用
# log文件指定后，这里不用再指定文件
def statmodeldiffattriALLPF(opname1,decisionpt,gamenumbreak=0):

    datadealing4(gamenumbreak)

    # 对手fps决策点的可观测行动的数据
    opname=opname1 #'TARD'
    round='preflop'

    raisehdsetrec=[]
    raisetimesrec=0
    callhdsetrec=[]
    calltimesrec=0
    foldhdsetrec=[]
    foldtimesrec=0

    actshdsetrec=[]

    recAllgameHisTMP=recAllgameHistorySF
    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]

        if decisionpt=='fps':
            if playername==opname and streetnumb==1 and opponentac=='bigbet':
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                elif playeractn[0]=='f':
                    foldtimesrec+=1
                    if playercard:
                        foldhdsetrec.append(getwinrate(2,playercard,[]))
        elif decisionpt=='fpbr':
            if playername==opname and streetnumb==1 and opponentac[0]=='r':
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                elif playeractn[0]=='f':
                    foldtimesrec+=1
                    if playercard:
                        foldhdsetrec.append(getwinrate(2,playercard,[]))
        elif decisionpt=='prp':
            if playername==opname and streetnumb==1 and (opponentac=='bigbet' or opponentac[0]=='r'):
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                elif playeractn[0]=='f':
                    foldtimesrec+=1
                    if playercard:
                        foldhdsetrec.append(getwinrate(2,playercard,[]))

    print('raisetimesrec=',raisetimesrec)
    print('calltimesrec=',calltimesrec)
    print('foldtimesrec=',foldtimesrec)

    actshdsetrec=sorted(raisehdsetrec+callhdsetrec+foldhdsetrec)

    npdfpts=101
    actionwrrange=np.linspace(0,1,npdfpts)
    widthdens=0.02

    xstt=0.323032
    xend=0.852037
    minWridx=int(np.ceil(xstt*(npdfpts-1)))
    maxWridx=int(np.floor(xend*(npdfpts-1)))

    #估计所有的
    wrdensity=np.array(evalActionDensity(actionwrrange,actshdsetrec,widthdens)) #核密度估计分布
    plt.figure()
    plt.plot(actionwrrange,wrdensity)
    plt.ylim(0,5.2)
    plt.title('all actions wr dist')


    #使用核密度估计计算pdf-raise
    actiondensity=np.zeros(npdfpts)
    if raisetimesrec>0:
        actiondensity=np.array(evalActionDensity(actionwrrange,raisehdsetrec,widthdens)) #核密度估计分布
        print('minwr-raise=',np.min(raisehdsetrec))
        actiondensity[:minWridx]=0.0
        actiondensity[maxWridx:]=0.0


    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='density of 1326 hands')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity2,marker='o',label='probability on 169 types')
    #plt.plot(np.array(actiontpstatsorted)[:,1],actiondensity3,marker='>',label='density1 on 169 types')
    #plt.scatter(np.array(actiontpstatsorted)[:,1],actiondensity4,color='green',marker='<',label='density transfered from 169 types')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.legend(frameon=False,loc='upper right')
    plt.xlim(0,1)
    plt.grid() 
    plt.title('Pdf of raise at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-raise.pdf'.format(decisionpt,round,opname))

    #使用核密度估计计算pdf-call
    actiondensity5=np.zeros(npdfpts)
    if calltimesrec>0:
        actiondensity5=np.array(evalActionDensity(actionwrrange,callhdsetrec,widthdens)) #核密度估计分布
        print('minwr-call=',np.min(callhdsetrec))
        actiondensity5[:minWridx]=0.0
        actiondensity5[maxWridx:]=0.0
    
    plt.figure()
    plt.plot(actionwrrange,actiondensity5,label='density of 1326 hands')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.xlim(0,1)
    plt.grid() 
    plt.legend(frameon=False)
    plt.title('Pdf of call at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}-call.pdf'.format(decisionpt,round,opname))


    #使用核密度估计计算pdf-fold
    actiondensity7=np.zeros(npdfpts)
    if foldtimesrec:
        actiondensity7=np.array(evalActionDensity(actionwrrange,foldhdsetrec,widthdens)) #核密度估计分布
        print('minwr-fold=',np.min(foldhdsetrec))
        actiondensity7[:minWridx]=0.0
        actiondensity7[maxWridx:]=0.0


    #概率密度绘图
    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='raise eval')
    plt.plot(actionwrrange,actiondensity5,label='call eval')
    plt.plot(actionwrrange,actiondensity7,label='fold eval')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    #plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action pdf at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-pdf-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    sumacts=raisetimesrec+calltimesrec+foldtimesrec
    ratioacts=[raisetimesrec/sumacts,calltimesrec/sumacts,foldtimesrec/sumacts]
    print('ratioacts',ratioacts)
    freqr=actiondensity*raisetimesrec
    freqc=actiondensity5*calltimesrec
    freqf=actiondensity7*foldtimesrec

    plt.figure()
    plt.plot(freqr,label='freq raise')
    plt.plot(freqc,label='freq call')
    plt.plot(freqf,label='freq fold')


    #行动概率计算
    #使用密度估计
    probactions1=np.zeros((npdfpts,3))
    probreal1=np.zeros((npdfpts,3))
    for i in range(npdfpts):
        probactions1[i,0]=freqr[i]/(freqf[i]+freqc[i]+freqr[i])
        probactions1[i,1]=freqc[i]/(freqf[i]+freqc[i]+freqr[i])
        probactions1[i,2]=freqf[i]/(freqf[i]+freqc[i]+freqr[i])

    opplayers=['LA','LARD','TARD','MS']
    if opname=='LA':
        alpha=0.3
        beta=0.45
        for i in range(npdfpts):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=0.0
                probreal1[i,1]=1.0
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1.0
                probreal1[i,1]=0.0
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(maxWridx-minWridx+1)
        print('Eap1={}, gamebreak={}'.format(Eap1,gamenumbreak))
    elif opname=='LARD':
        alpha=0.3
        beta=0.45
        zeta1=0.7
        zeta2=0.2
        for i in range(npdfpts):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1])))/2/(npdfpts)
        print('Eap1={}, gamebreak={}'.format(Eap1,gamenumbreak))
    elif opname=='TARD':
        alpha=0.55
        beta=0.65
        zeta1=0.7
        zeta2=0.2
        for i in range(npdfpts):
            if actionwrrange[i]<alpha:
                probreal1[i,0]=0.0  #raise
                probreal1[i,1]=0.0  #call
                probreal1[i,2]=1.0  #fold
            elif actionwrrange[i]<beta:
                probreal1[i,0]=1-zeta1
                probreal1[i,1]=zeta1
                probreal1[i,2]=0.0
            else:
                probreal1[i,0]=1-zeta2
                probreal1[i,1]=zeta2
                probreal1[i,2]=0.0
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/2/(npdfpts)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))

        #使用积分计算不同行动的比例
        #这是一个积分问题而不是一个密度转换问题，所以变量转换时不需要乘上导数
        #int_0^1 f(I)dI=int_0^1 g(Wr(I))dI
        #用数值方法求：
        nsects=400
        Isetstemp=np.linspace(0,1,nsects+1) #分为nsects段
        Isets=np.zeros(nsects)
        wrsets=np.zeros(nsects)
        dI=1/nsects
        racts=np.zeros((nsects,3))
        for i in range(nsects):
            Isets[i]=(Isetstemp[i+1]+Isetstemp[i])/2
            wrsets[i]=WRfromhandratio(Isets[i],g_WrwithIflop)
            if wrsets[i]<alpha:
                racts[i,0]=0.0*dI  #raise
                racts[i,1]=0.0*dI  #call
                racts[i,2]=1.0*dI  #fold
            elif wrsets[i]<beta:
                racts[i,0]=(1-zeta1)*dI
                racts[i,1]=zeta1*dI
                racts[i,2]=0.0*dI
            else:
                racts[i,0]=(1-zeta2)*dI
                racts[i,1]=zeta2*dI
                racts[i,2]=0.0*dI
        #print('racts=',racts)
        
        sacts=np.sum(racts,axis=0)
        print('sacts',sacts)
        print('ratio of acts raise={},call={},fold={}'.format(sacts[0]/np.sum(sacts),sacts[1]/np.sum(sacts),sacts[2]/np.sum(sacts)))
    elif opname=='MS':
        domains=[[0.0,0.4],[0.4,0.65],[0.65,1]]
        probreal1a=np.array(outDmodel(domains,npdfpts)).T
        probreal1[:,0]=probreal1a[:,2]
        probreal1[:,1]=probreal1a[:,1]
        probreal1[:,2]=probreal1a[:,0]
        Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,0]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,2])))/2/(npdfpts)
        print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))


    plt.figure()
    plt.plot(actionwrrange,probactions1[:,0],label='raise eval')
    plt.plot(actionwrrange,probactions1[:,1],label='call eval')
    plt.plot(actionwrrange,probactions1[:,2],label='fold eval')
    if opname in opplayers:
        plt.plot(actionwrrange,probreal1[:,0],'--',label='raise real')
        plt.plot(actionwrrange,probreal1[:,1],'--',label='call real')
        plt.plot(actionwrrange,probreal1[:,2],'--',label='fold real ')

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    np.savetxt(opname+"-"+round+"-"+decisionpt+'full.csv',probactions1,delimiter =",",fmt ='%.f6')
    plt.show()

    return None




# 对手模型的重建
# 从服务器数据重建，所有的都能用
# log文件指定后，这里不用再指定文件
def statmodelserverALLPF(opname1,decisionpt,gamenumbreak=0,wdin=0.015):

    datadealing4(gamenumbreak)

    # 对手fps决策点的可观测行动的数据
    opname=opname1 #'TARD'
    round='preflop'

    '''
    numbactions=5
    nameaction1=[['fold'],['call'],['r1'],['r2'],['allin']]
    indexaction=[0,1,2,3,4]
    '''

    numbactions=3
    nameaction1=[['fold'],['call'],['r1','r2','allin']]
    indexaction=[0,1,2]




    acttimesrec=np.zeros(numbactions)
    acthdsetrec=[]
    for i in range(numbactions):
        acthdsetrec.append([])

    recAllgameHisTMP=recAllgameHistorySF
    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]
        otherscard=rec[6]

        if decisionpt=='fps':
            if playername==opname and streetnumb==1 and opponentac=='bigbet':
                for i in range(numbactions):
                    if playeractn in nameaction1[i]:
                        acttimesrec[indexaction[i]]+=1
                        if playercard:
                            wr=getwinrate(2,playercard,[])
                            acthdsetrec[indexaction[i]].append(wr)
        elif decisionpt=='fpbr':
            if playername==opname and streetnumb==1 and (opponentac[0]=='r' or opponentac=='allin'):
                for i in range(numbactions):
                    if playeractn in nameaction1[i]:
                        acttimesrec[indexaction[i]]+=1
                        if playercard:
                            wr=getwinrate(2,playercard,[])
                            acthdsetrec[indexaction[i]].append(wr)
        elif decisionpt=='prp':
            if playername==opname and streetnumb==1 and (opponentac[0]=='r' or opponentac=='allin' or opponentac=='bigbet'):
                for i in range(numbactions):
                    if playeractn in nameaction1[i]:
                        acttimesrec[indexaction[i]]+=1
                        if playercard:
                            wr=getwinrate(2,playercard,[])
                            acthdsetrec[indexaction[i]].append(wr)


    print('nameaction1=',nameaction1)
    print('acttimesrec=',acttimesrec,sum(acttimesrec))
    print('acttimesrec=',[len(x) for x in acthdsetrec])

    npdfpts=101
    actionwrrange=np.linspace(0,1,npdfpts)
    widthdens=wdin

    xstt=0.323032
    xend=0.852037
    minWridx=int(np.ceil(xstt*(npdfpts-1)))
    maxWridx=int(np.floor(xend*(npdfpts-1)))


    #使用核密度估计计算pdf
    actiondensall=[]

    plt.figure()
    for i in range(numbactions):
        actiondensity=np.zeros(npdfpts)
        if len(acthdsetrec[i])>0:
            actiondensity=np.array(evalActionDensity(actionwrrange,acthdsetrec[i],widthdens)) #核密度估计分布
            actiondensity[:minWridx]=0.0
            actiondensity[maxWridx:]=0.0
        actiondensall.append(actiondensity)
        plt.plot(actionwrrange,actiondensity,label=nameaction1[i])

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.legend(frameon=False,loc='upper right')
    plt.xlim(0,1)
    plt.grid() 
    plt.title('Pdf of raise at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}.pdf'.format(decisionpt,round,opname))


    sumacts=sum(acttimesrec)
    ratioacts=[x/sumacts for x in acttimesrec]
    print('ratioacts',ratioacts)
    freqall=np.array(actiondensall)
    for  i in range(numbactions):
        freqall[i,:]=freqall[i,:]*acttimesrec[i]


    #行动概率计算
    #使用密度估计
    probactions1=np.zeros((numbactions,npdfpts))
    for i in range(numbactions):
        for j in range(npdfpts):
            if sum(freqall[:,j])>0:
                probactions1[i,j]=freqall[i,j]/sum(freqall[:,j])
            else:
                probactions1[i,j]=0.0

    plt.figure()
    for i in range(numbactions):
        plt.plot(actionwrrange,probactions1[i,:],label=str(nameaction1[i])+' fulldata')

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    np.savetxt(opname+"-"+round+"-"+decisionpt+'full.csv',probactions1.T,delimiter =",",fmt ='%.6f')
    np.savetxt(opname+"-"+round+"-"+decisionpt+'fullpdf.csv',np.array(actiondensall).T,delimiter =",",fmt ='%.6f')
    
    
    plt.show()

    return None



# 对手模型的重建
# 从服务器数据重建，所有的都能用
# log文件指定后，这里不用再指定文件
def statmodelserverALLAF(opname1,decisionpt,gamenumbreak=0):

    datadealing4(gamenumbreak)

    # 对手fps决策点的可观测行动的数据
    opname=opname1 #'TARD'
    round='postflop'

    '''
    numbactions=5
    nameaction1=[['fold'],['call'],['r1'],['r2'],['allin']]
    indexaction=[0,1,2,3,4]
    '''

    numbactions=3
    nameaction1=[['fold'],['call'],['r1','r2','allin']]
    indexaction=[0,1,2]


    acttimesrec=np.zeros(numbactions)
    acthdsetrec=[]
    for i in range(numbactions):
        acthdsetrec.append([])

    recAllgameHisTMP=recAllgameHistorySF
    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]
        otherscard=rec[6]

        if decisionpt=='pra':
            if playername==opname and streetnumb==2 and opponentac[0]=='r':
                for i in range(numbactions):
                    if playeractn in nameaction1[i]:
                        acttimesrec[indexaction[i]]+=1
                        if playercard:
                            wr=getwinrate(2,playercard,boardcards[:3])
                            acthdsetrec[indexaction[i]].append(wr)

    print('nameaction1=',nameaction1)
    print('acttimesrec=',acttimesrec,sum(acttimesrec))
    print('acttimesrec=',[len(x) for x in acthdsetrec])

    npdfpts=101
    actionwrrange=np.linspace(0,1,npdfpts)
    widthdens=0.02

    xstt=1
    xend=0
    for i in range(numbactions):
        if xstt>min(acthdsetrec[i]):
            xstt=min(acthdsetrec[i])  #0.323032
        if xend<max(acthdsetrec[i]):
            xend=max(acthdsetrec[i])  #0.852037
    minWridx=int(np.ceil(xstt*(npdfpts-1)))
    maxWridx=int(np.floor(xend*(npdfpts-1)))


    #使用核密度估计计算pdf
    actiondensall=[]

    plt.figure()
    for i in range(numbactions):
        actiondensity=np.zeros(npdfpts)
        if len(acthdsetrec[i])>0:
            actiondensity=np.array(evalActionDensity(actionwrrange,acthdsetrec[i],widthdens)) #核密度估计分布
            actiondensity[:minWridx]=0.0
            actiondensity[maxWridx:]=0.0
        actiondensall.append(actiondensity)
        plt.plot(actionwrrange,actiondensity,label=nameaction1[i])

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.legend(frameon=False,loc='upper right')
    plt.xlim(0,1)
    plt.grid() 
    plt.title('Pdf of raise at {} {} of {}'.format(decisionpt,round,opname))
    plt.savefig('fig-pdf-ker-{}-{}-{}.pdf'.format(decisionpt,round,opname))


    sumacts=sum(acttimesrec)
    ratioacts=[x/sumacts for x in acttimesrec]
    print('ratioacts',ratioacts)
    freqall=np.array(actiondensall)
    for  i in range(numbactions):
        freqall[i,:]=freqall[i,:]*acttimesrec[i]


    #行动概率计算
    #使用密度估计
    probactions1=np.zeros((numbactions,npdfpts))
    for i in range(numbactions):
        for j in range(npdfpts):
            if sum(freqall[:,j])>0:
                probactions1[i,j]=freqall[i,j]/sum(freqall[:,j])
            else:
                probactions1[i,j]=0.0

    plt.figure()
    for i in range(numbactions):
        plt.plot(actionwrrange,probactions1[i,:],label=str(nameaction1[i])+' fulldata')

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname),fontsize=16) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    np.savetxt(opname+"-"+round+"-"+decisionpt+'full.csv',probactions1.T,delimiter =",",fmt ='%.6f')
    np.savetxt(opname+"-"+round+"-"+decisionpt+'fullpdf.csv',np.array(actiondensall).T,delimiter =",",fmt ='%.6f')
    
    
    plt.show()

    return None




# 对手模型的重建
# 不同属性表征的信息集的模型
# preflop轮用，因为preflop的特殊性
def statmodelserverinferPF(opname1,decisionpt,filenamefulldata,gamenumbreak=0,wdin=0.015,cpf=20):

    datadealing4(gamenumbreak)

    # 对手fps决策点的可观测行动的数据
    opname=opname1 #'TARD'
    round='preflop'

    raisehdsetrec=[]
    raisetpsetrec=np.zeros(169)
    raisetimesrec=0
    callhdsetrec=[]
    calltpsetrec=np.zeros(169)
    calltimesrec=0
    foldtimesrec=0

    wroftprec=np.zeros(169)
    for i in range(1326):
        handrk=get_handrank(IdxtoHand[i][0],IdxtoHand[i][1],0,0,0,0,0)
        wroftprec[handrk-1]=get_winrate(2,0,IdxtoHand[i][0],IdxtoHand[i][1],0,0,0,0,0)
    #print('wroftprec=',wroftprec)

    recAllgameHisTMP=recAllgameHistorySF
    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]

        if decisionpt=='fps':
            if playername==opname and streetnumb==1 and opponentac=='bigbet':
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                        raisetpsetrec[gethandrank(playercard,[])-1]+=1
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                        handrk=gethandrank(playercard,[])
                        #print('handrk=',handrk)
                        calltpsetrec[handrk-1]+=1
                elif playeractn[0]=='f':
                    foldtimesrec+=1
        elif decisionpt=='fpbr':
            if playername==opname and streetnumb==1 and opponentac[0]=='r':
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                        raisetpsetrec[gethandrank(playercard,[])-1]+=1
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                        handrk=gethandrank(playercard,[])
                        #print('handrk=',handrk)
                        calltpsetrec[handrk-1]+=1
                elif playeractn[0]=='f':
                    foldtimesrec+=1
        elif decisionpt=='prp':
            if playername==opname and streetnumb==1 and (opponentac=='bigbet' or opponentac[0]=='r'):
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,[]))
                        raisetpsetrec[gethandrank(playercard,[])-1]+=1
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,[]))
                        handrk=gethandrank(playercard,[])
                        #print('handrk=',handrk)
                        calltpsetrec[handrk-1]+=1
                elif playeractn[0]=='f':
                    foldtimesrec+=1

    print('raisetimesrec=',raisetimesrec)
    print('calltimesrec=',calltimesrec)
    print('foldtimesrec=',foldtimesrec)
    #print('raisehdsetrec',raisehdsetrec)
    #print('raisetpsetrec',raisetpsetrec.tolist())

    
    npdfpts=101
    xstt=0.323032
    xend=0.852037
    minWridx=int(np.ceil(xstt*(npdfpts-1)))
    maxWridx=int(np.floor(xend*(npdfpts-1)))
    actionwrrange=np.linspace(0,1,npdfpts)

    widthdens=wdin
    #使用核密度估计计算pdf
    actiondensity=np.array(evalActionDensity(actionwrrange,raisehdsetrec,widthdens)) #核密度估计分布
    actiondensity[:minWridx]=0.0
    actiondensity[maxWridx:]=0.0
    
    #使用核密度估计计算pdf
    actiondensity5=np.array(evalActionDensity(actionwrrange,callhdsetrec,widthdens)) #核密度估计分布
    actiondensity5[:minWridx]=0.0
    actiondensity5[maxWridx:]=0.0
    
    #fold情况下的信息进行估计
    roffold=foldtimesrec/(raisetimesrec+foldtimesrec+calltimesrec)
    if roffold>0.01:
        rofraise=raisetimesrec/(raisetimesrec+foldtimesrec+calltimesrec)
        print('ration of fold',roffold)
        cmprnew=cpf
        eshift=ActModeldistSe(1-roffold,cmprnew)
        Cv=[ActModeldistScv(handratiofromWR(wr),eshift,cmprnew) for wr in actionwrrange]

        #print('handratiofromWR=',[handratiofromWR(wr) for wr in actionwrrange])

        '''
        eshift1=ActModeldistSe(rofraise,cmprnew)
        Cv1=[ActModeldistScv(handratiofromWR(wr),eshift1,cmprnew) for wr in actionwrrange]
        plt.figure()
        plt.plot(actionwrrange,Cv)
        plt.plot(actionwrrange,Cv1)
        plt.figure()
        plt.plot(actionwrrange,1-np.array(Cv))
        plt.plot(actionwrrange,np.array(Cv)-np.array(Cv1))
        plt.plot(actionwrrange,np.array(Cv1))
        '''
        #print('Cv=',Cv)
        #anykey=input()

        freqc=np.array(actiondensity)*raisetimesrec
        freqr=np.array(actiondensity5)*calltimesrec
        #print('actionwrrange=',actionwrrange)

        freqfCv=(1-np.array(Cv))/roffold*foldtimesrec

        #flambdc=lambda x:1-x if x>0.0000001 else 0.0
        denfCvIwr=(1-np.array(Cv))/roffold*np.array([handderivwithWR(wr) for wr in actionwrrange])
        denfCvIwr[:minWridx]=0.0
        denfCvIwr[maxWridx:]=0.0
        #plt.figure()
        #plt.plot(actionwrrange,freqfCv)

        flambda=lambda i:(1-Cv[i])*(freqc[i]+freqr[i])/(Cv[i]) if i >= minWridx and i<=maxWridx else 0.0
        flambdb=lambda i: freqfCv[i] if flambda(i)> freqfCv[i] else flambda(i)
        freqf=[flambdb(i) for i in range(npdfpts)]
        #print('freqc=',freqc,np.sum(freqc))
        #print('freqr=',freqr,np.sum(freqr))
        #print('freqf=',freqf,np.sum(freqf))
        #folddensity=[freqf[i]/foldtimesrec for i in range(npdfpts)]
        folddensity=denfCvIwr
    else:
        freqf=np.zeros(npdfpts)

    #概率密度绘图
    pdfreal1=np.loadtxt(filenamefulldata[:-4]+'pdf.csv',delimiter=',')
    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='raise eval')
    plt.plot(actionwrrange,actiondensity5,label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,folddensity,label='fold eval')

    plt.plot(actionwrrange,pdfreal1[:,2],'--',label='raise fulldata')
    plt.plot(actionwrrange,pdfreal1[:,1],'--',label='call fulldata')
    if roffold>0.01:
        plt.plot(actionwrrange,pdfreal1[:,0],'--',label='fold fulldata')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    #plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action pdf at {} {} of {}".format(decisionpt,round,opname)) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-pdf-{}-{}-{}.pdf'.format(decisionpt,round,opname))


    #行动概率计算
    #使用密度估计
    probactions1=np.zeros((npdfpts,3))
    probreal1=np.loadtxt(filenamefulldata,delimiter=',')
    if roffold>0.01:
        for i in range(npdfpts):
            if freqf[i]>0 or actiondensity[i]*raisetimesrec>0 or actiondensity5[i]*calltimesrec>0:
                probactions1[i,0]=actiondensity[i]*raisetimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,1]=actiondensity5[i]*calltimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,2]=freqf[i]/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
    else:
        for i in range(npdfpts):
            if actiondensity[i]*raisetimesrec>0 or actiondensity5[i]*calltimesrec>0:
                probactions1[i,0]=actiondensity[i]*raisetimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,1]=actiondensity5[i]*calltimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)

    #注意全数据统计的是从fold到raise的，所以需要反一下
    Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,2]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,0])))/3/(maxWridx-minWridx+1)
    print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))

    plt.figure()
    plt.plot(actionwrrange,probactions1[:,0],label='raise eval')
    plt.plot(actionwrrange,probactions1[:,1],label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,probactions1[:,2],label='fold eval')
    plt.plot(actionwrrange,probreal1[:,2],'--',label='raise fulldata')
    plt.plot(actionwrrange,probreal1[:,1],'--',label='call fulldata')
    if roffold>0.01:
        plt.plot(actionwrrange,probreal1[:,0],'--',label='fold fulldata')

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname)) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    plt.show()
    return None








# 对手模型的重建
# 不同属性表征的信息集的模型
# postflop轮用，因为preflop的特殊性
def statmodelserverinferAF(opname1,decisionpt,filenamefulldata,gamenumbreak=0,wdin=0.015,cpf=20):

    datadealing4(gamenumbreak)

    # 对手fps决策点的可观测行动的数据
    opname=opname1 #'TARD'
    round='postflop'

    raisehdsetrec=[]
    raisetimesrec=0
    callhdsetrec=[]
    calltimesrec=0
    foldtimesrec=0

    recAllgameHisTMP=recAllgameHistorySF
    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]

        if decisionpt=='pra':
            if playername==opname and streetnumb==2 and opponentac[0]=='r':
                if playeractn[0]=='r' or playeractn=='allin':
                    raisetimesrec+=1
                    if playercard:
                        raisehdsetrec.append(getwinrate(2,playercard,boardcards[:3]))
                elif playeractn[0]=='c':
                    calltimesrec+=1
                    if playercard:
                        callhdsetrec.append(getwinrate(2,playercard,boardcards[:3]))
                elif playeractn[0]=='f':
                    foldtimesrec+=1


    print('raisetimesrec=',raisetimesrec)
    print('calltimesrec=',calltimesrec)
    print('foldtimesrec=',foldtimesrec)
    #print('raisehdsetrec',raisehdsetrec)
    #print('raisetpsetrec',raisetpsetrec.tolist())

    
    npdfpts=101
    xstt=0.110735 #0.323032
    xend=1.0 
    minWridx=int(np.ceil(xstt*(npdfpts-1)))
    maxWridx=int(np.floor(xend*(npdfpts-1)))
    actionwrrange=np.linspace(0,1,npdfpts)

    widthdens=wdin
    #使用核密度估计计算pdf
    actiondensity=np.array(evalActionDensity(actionwrrange,raisehdsetrec,widthdens)) #核密度估计分布
    actiondensity[:minWridx]=0.0
    actiondensity[maxWridx:]=0.0
    
    #使用核密度估计计算pdf
    actiondensity5=np.array(evalActionDensity(actionwrrange,callhdsetrec,widthdens)) #核密度估计分布
    actiondensity5[:minWridx]=0.0
    actiondensity5[maxWridx:]=0.0
    
    #fold情况下的信息进行估计
    roffold=foldtimesrec/(raisetimesrec+foldtimesrec+calltimesrec)
    if roffold>0.01:
        rofraise=raisetimesrec/(raisetimesrec+foldtimesrec+calltimesrec)
        print('ration of fold',roffold)
        cmprnew=cpf
        eshift=ActModeldistSe(1-roffold,cmprnew)
        Iofwr=[handratiofromWR(wr,g_WrwithIflop) for wr in actionwrrange]
        Cv=[ActModeldistScv(I,eshift,cmprnew) for I in Iofwr]

        freqc=np.array(actiondensity)*raisetimesrec
        freqr=np.array(actiondensity5)*calltimesrec

        freqfCv=(1-np.array(Cv))/roffold*foldtimesrec

        denfCvIwr=(1-np.array(Cv))/roffold*np.array([handderivwithWR(wr,g_derivIvsWrflop) for wr in actionwrrange])
        denfCvIwr[:minWridx]=0.0
        denfCvIwr[maxWridx:]=0.0

        flambda=lambda i:(1-Cv[i])*(freqc[i]+freqr[i])/(Cv[i]) if i >= minWridx and i<=maxWridx else 0.0
        flambdb=lambda i: freqfCv[i] if flambda(i)> freqfCv[i] else flambda(i)
        freqf=[flambdb(i) for i in range(npdfpts)]
        #print('freqc=',freqc,np.sum(freqc))
        #print('freqr=',freqr,np.sum(freqr))
        #print('freqf=',freqf,np.sum(freqf))
        #folddensity=[freqf[i]/foldtimesrec for i in range(npdfpts)]
        folddensity=denfCvIwr
    else:
        freqf=np.zeros(npdfpts)

    #概率密度绘图
    pdfreal1=np.loadtxt(filenamefulldata[:-4]+'pdf.csv',delimiter=',')
    plt.figure()
    plt.plot(actionwrrange,actiondensity,label='raise eval')
    plt.plot(actionwrrange,actiondensity5,label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,folddensity,label='fold eval')

    plt.plot(actionwrrange,pdfreal1[:,2],'--',label='raise fulldata')
    plt.plot(actionwrrange,pdfreal1[:,1],'--',label='call fulldata')
    if roffold>0.01:
        plt.plot(actionwrrange,pdfreal1[:,0],'--',label='fold fulldata')
    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    #plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action pdf at {} {} of {}".format(decisionpt,round,opname)) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-pdf-{}-{}-{}.pdf'.format(decisionpt,round,opname))


    #行动概率计算
    #使用密度估计
    probactions1=np.zeros((npdfpts,3))
    probreal1=np.loadtxt(filenamefulldata,delimiter=',')
    if roffold>0.01:
        for i in range(npdfpts):
            if freqf[i]>0 or actiondensity[i]*raisetimesrec>0 or actiondensity5[i]*calltimesrec>0:
                probactions1[i,0]=actiondensity[i]*raisetimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,1]=actiondensity5[i]*calltimesrec/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,2]=freqf[i]/(freqf[i]+actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
    else:
        for i in range(npdfpts):
            if actiondensity[i]*raisetimesrec>0 or actiondensity5[i]*calltimesrec>0:
                probactions1[i,0]=actiondensity[i]*raisetimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)
                probactions1[i,1]=actiondensity5[i]*calltimesrec/(actiondensity[i]*raisetimesrec+actiondensity5[i]*calltimesrec)


    Eap1=(np.sum(np.abs(probreal1[:,0]-probactions1[:,2]))+np.sum(np.abs(probreal1[:,1]-probactions1[:,1]))
        +np.sum(np.abs(probreal1[:,2]-probactions1[:,0])))/3/(maxWridx-minWridx+1)
    print('Eap1={},gamebreak={}'.format(Eap1,gamenumbreak))


    plt.figure()
    plt.plot(actionwrrange,probactions1[:,0],label='raise eval')
    plt.plot(actionwrrange,probactions1[:,1],label='call eval')
    if roffold>0.01:
        plt.plot(actionwrrange,probactions1[:,2],label='fold eval')

    plt.plot(actionwrrange,probreal1[:,2],'--',label='raise fulldata')
    plt.plot(actionwrrange,probreal1[:,1],'--',label='call fulldata')
    if roffold>0.01:
        plt.plot(actionwrrange,probreal1[:,0],'--',label='fold fulldata')

    plt.xlabel('Wr')
    plt.ylabel('probability')
    plt.grid() #绘制网格
    plt.xlim(0,1) #x轴绘制范围限制
    plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("Action probability at {} {} of {}".format(decisionpt,round,opname)) #调整x轴坐标刻度和标签
    plt.savefig('fig-action-prob-{}-{}-{}.pdf'.format(decisionpt,round,opname))

    plt.show()
    return None







# 翻牌前的动作概率模型
def statPlayerActProbPreflop():
    statPlayerActProb(1)
    return None

# 翻牌后的动作概率模型
def statPlayerActProbAfterflop():
    statPlayerActProb(2)
    return None






# 动作概率模型
def statPlayerActProb(streetstart):

    flg_WrpointFirst=True  #每一轮只考虑两个决策点个一次的标记

    if streetstart==1:
        strRound='Preflop'
        streetCalc=[1]
        xstt=0.323032
        xend=0.852037
    else:
        strRound='Afterflop'
        streetCalc=[2,3,4]
        xstt=0.0
        xend=1.0

    l_lstActName=['fold','check','call','r1','r2','r3','allin']
    nplayer=len(recAllagentsname) 
    allbots="-".join(recAllagentsname)
    handid=len(datausers)
    #统计对手的频率模型-反过来就是不同赢率对应的动作
    #记录对手做每个动作时的赢率，用一个字典表示，动作名表示字典的key，赢率用一个列表记录
    allplayeractdata={}
    allplayeractdatapck={}
    allplayeractdatapcl={}

    if flg_WrpointFirst:
        recAllgameHisTMP=recAllgameHistorySF
    else:
        recAllgameHisTMP=recAllgameHistoryS

    for rec in recAllgameHisTMP:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]
        #构成：{playername:{playeractn:[winrate,...]}}这样第一个字典
        if playername not in allplayeractdata:
            allplayeractdata[playername]={}
            allplayeractdatapck[playername]={}
            allplayeractdatapcl[playername]={}

        if playercard:
            if playeractn not in allplayeractdata[playername]:
                allplayeractdata[playername][playeractn]=[]
            playerwinrate=0.0

            if streetnumb in streetCalc:
                if streetnumb==1:
                    playerwinrate=getwinrate(nplayer,playercard,[])
                elif streetnumb==2:
                    playerwinrate=getwinrate(nplayer,playercard,boardcards[:3])
                elif streetnumb==3:
                    playerwinrate=getwinrate(nplayer,playercard,boardcards[:4])
                else:
                    playerwinrate=getwinrate(nplayer,playercard,boardcards)
                
                allplayeractdata[playername][playeractn].append(playerwinrate)
                if opponentac=='none' or  opponentac=='check' or opponentac=='call':
                    if playeractn not in allplayeractdatapck[playername]:
                        allplayeractdatapck[playername][playeractn]=[]
                    allplayeractdatapck[playername][playeractn].append(playerwinrate)
                else:
                    if playeractn not in allplayeractdatapcl[playername]:
                        allplayeractdatapcl[playername][playeractn]=[]
                    allplayeractdatapcl[playername][playeractn].append(playerwinrate)


    l_lstActRaiseName=['r1','r2','r3','allin']
    for playername  in recAllagentsname:
        lstraiseactwr=[]
        lstraiseactwrpck=[]
        lstraiseactwrpcl=[]
        for playeractn in l_lstActRaiseName:
            if playeractn in allplayeractdata[playername] and allplayeractdata[playername][playeractn]:
                lstraiseactwr+=allplayeractdata[playername][playeractn]
            if playeractn in allplayeractdatapck[playername] and allplayeractdatapck[playername][playeractn]:
                lstraiseactwrpck+=allplayeractdatapck[playername][playeractn]
            if playeractn in allplayeractdatapcl[playername] and allplayeractdatapcl[playername][playeractn]:
                lstraiseactwrpcl+=allplayeractdatapcl[playername][playeractn]
        allplayeractdata[playername]['raise']=lstraiseactwr
        allplayeractdatapck[playername]['raise']=lstraiseactwrpck
        allplayeractdatapcl[playername]['raise']=lstraiseactwrpcl

       
    #pdf -使用核密度估计方法
    decisionPTs=['pck','pcl'] #'rnd'
    for decisionpt in decisionPTs:
        allplayerApdfdatapck={}
        if decisionpt=='rnd':
            allplayeractdatatemp=allplayeractdata
        elif decisionpt=='pck':
            allplayeractdatatemp=allplayeractdatapck
        elif decisionpt=='pcl':
            allplayeractdatatemp=allplayeractdatapcl

        npdfpts=101
        for playername in allplayeractdatatemp.keys():
            allplayerApdfdatapck[playername]={}
            minWrAllacts=1.0
            maxWrAllacts=0.0
            sumNactions=0
            for playeractn in l_lstActName+['raise']:
                if playeractn in allplayeractdatatemp[playername] and allplayeractdatatemp[playername][playeractn]:
                    allplayerApdfdatapck[playername][playeractn]=[]
                    actionwrdata=np.array(allplayeractdatatemp[playername][playeractn])
                    #np.savetxt(playername+"-"+playeractn+"-"+decisionpt+"-"+strRound+'.txt',sorted(actionwrdata)) 
                    #minWinrate=np.min(actionwrdata)
                    #maxWinrate=np.max(actionwrdata)
                    minWridx=int(np.ceil(xstt*(npdfpts-1)))
                    maxWridx=int(np.floor(xend*(npdfpts-1)))
                    #print('minWridx',minWridx,maxWridx)
                    #if maxWinrate>maxWrAllacts:
                    #    maxWrAllacts=maxWinrate
                                        
                    actionwrrange=np.linspace(0,1,npdfpts)
                    allplayerApdfdatapck[playername][playeractn].append(actionwrrange.tolist())  #allplayerApdfdatapck[playername][playeractn][0]
                    #print('actionwrrange=',actionwrrange)

                    #使用核密度估计计算pdf
                    actiondensity=np.array(evalActionDensity(actionwrrange,actionwrdata)) #核密度估计分布
                    actiondensity[:minWridx]=0.0
                    actiondensity[maxWridx:]=0.0
                    allplayerApdfdatapck[playername][playeractn].append(actiondensity.tolist()) #allplayerApdfdatapck[playername][playeractn][1]
                    #print('actiondensity=',actiondensity)
                    #np.savetxt(playername+"-"+playeractn+"-"+decisionpt+"-"+strRound+'pdf.txt',actiondensity) 

                    #基于密度估计各赢率点对应的动作数
                    nactionswr=len(actionwrdata) #动作的总数
                    if playeractn !='raise':
                        sumNactions+=nactionswr
                    actionnumbers=np.array(actiondensity)*nactionswr
                    allplayerApdfdatapck[playername][playeractn].append(actionnumbers.tolist()) #allplayerApdfdatapck[playername][playeractn][2]
                    #print('actionnumbers=',nactionswr,actionnumbers)
                    print(playername+"-"+playeractn+"-"+decisionpt+"-"+strRound+'nactionswr=',nactionswr)
                    #np.savetxt(playername+"-"+playeractn+"-"+decisionpt+"-"+strRound+'actnum.txt',actionnumbers) 

                    #基于actionwrrange做cdf统计
                    ye=[]
                    for i in range(npdfpts):
                        yf=[thetastep(actionwrrange[i]-xi) for xi in actionwrdata]
                        ye.append(sum(yf)/nactionswr)
                    #使用核方法进行加权光滑：
                    n=len(ye)
                    actionCDFsmoothed=[]
                    w=0.05
                    for i in range(n):
                        yp=np.array([kernelepk(np.abs(actionwrrange[i]-xi)/w) for xi in actionwrrange])
                        ypy=yp*np.array(ye)
                        res=np.sum(ypy)/sum(yp)
                        actionCDFsmoothed.append(res)
                    allplayerApdfdatapck[playername][playeractn].append(actionCDFsmoothed) #allplayerApdfdatapck[playername][playeractn][3]
                    #计算pdf，二阶差分计算
                    ye=actionCDFsmoothed
                    xs=actionwrrange
                    fe=[]
                    h=xs[1]-xs[0]
                    for i in range(len(xs)):
                        if i==0:
                            fv=(-3*ye[i]+4*ye[i+1]-ye[i+2])/(2*h)
                        elif i==len(xs)-1:
                            fv= (3*ye[i]-4*ye[i-1]+ye[i-2])/(2*h)
                        else:
                            fv= (ye[i+1]-ye[i-1])/(2*h)
                        if fv<0: fv=0.0
                        fe.append(fv)
                    allplayerApdfdatapck[playername][playeractn].append(fe) #allplayerApdfdatapck[playername][playeractn][4]

                    #基于cdf得到的pdf密度估计各赢率点对应的动作数
                    actionnumbers=np.array(fe)*nactionswr
                    allplayerApdfdatapck[playername][playeractn].append(actionnumbers.tolist()) #allplayerApdfdatapck[playername][playeractn][5]

            #到此已经将核密度都算出来来
            #那么利用曲线假设和稳定指标来确定未知信息的密度
            if decisionpt=='pcl' and 'fold' not in allplayeractdata[playername]: #只有pcl决策点做
                if streetstart ==1 and recAllagentsstat[playername]['pffr']>0.01: #翻牌前的
                    print('pffr',streetstart,playername,recAllagentsstat[playername]['pffr'])
                    fratio=recAllagentsstat[playername]['pffr']
                    nactionfold=sumNactions/(1-fratio)*recAllagentsstat[playername]['pffr']

                    cmprnew=const_sc_compress

                    eshift=ActModeldistSe(1-fratio,cmprnew)
                    Cv=[ActModeldistScv(handratiofromWR(wr),eshift,cmprnew) for wr in actionwrrange]
                    freqc=allplayerApdfdatapck[playername]["call"][2]
                    freqr=allplayerApdfdatapck[playername]["raise"][2]

                    folddensity=[(1-Cv[i])/fratio for i in range(len(actionwrrange))]
                    freqf=[x*nactionfold for x in folddensity]

                    #freqf=[(1-Cv[i])*(freqc[i]+freqr[i])/(Cv[i]) for i in range(len(actionwrrange))]
                    #folddensity=[freqf[i]/nactionfold for i in range(len(actionwrrange))]
                    
                    #print('Cv=',Cv)
                    #print('actionwrrange=',actionwrrange)
                    #print('freqc=',np.sum(freqc))
                    #print('freqr=',np.sum(freqr))
                    #print('freqf=',np.sum(freqf))
                    #print('nactionfold=',nactionfold)
                    
                    allplayerApdfdatapck[playername]['fold']=[actionwrrange,folddensity,freqf,[],[],[]]

                    '''
                    folddensity=[(1-ActModeldistScv(wrScale(wr,xstt,xend),eshift,cmprnew))/recAllagentsstat[playername]['pffr'] for wr in actionwrrange]
                    foldactnums=[nactionfold*dens for dens in folddensity]
                    allplayerApdfdatapck[playername]['fold']=[actionwrrange,folddensity,foldactnums,[],[],[]]
                    '''
                    allplayeractdatatemp[playername]['fold']=[0.0,0.0] #随便添加一个便于后面判断fold已经补充进来了。但是raise不补充。
                    
                elif streetstart !=1 and recAllagentsstat[playername]['fr']>0.01: #翻牌后的
                    print('fr',streetstart,playername,recAllagentsstat[playername]['fr'])
                    nactionfold=sumNactions/(1-recAllagentsstat[playername]['fr'])*recAllagentsstat[playername]['fr']
                    cmprnew=const_sc_compress
                    eshift=ActModeldistSe(1-recAllagentsstat[playername]['fr'])
                    Cv=[ActModeldistScv(wr,eshift,cmprnew) for wr in actionwrrange]
                    freqc=allplayerApdfdatapck[playername]["call"][2]
                    freqr=allplayerApdfdatapck[playername]["raise"][2]
                    freqf=[(1-Cv[i])*(freqc[i]+freqr[i])/(Cv[i]) for i in range(len(actionwrrange))]
                    folddensity=[freqf[i]/nactionfold for i in range(len(actionwrrange))]
                    allplayerApdfdatapck[playername]['fold']=[actionwrrange,folddensity,freqf,[],[],[]]

                    #folddensity=[(1-ActModeldistScv(wr,eshift))/recAllagentsstat[playername]['fr'] for wr in actionwrrange]
                    #foldactnums=[nactionfold*dens for dens in folddensity]
                    #allplayerApdfdatapck[playername]['fold']=[actionwrrange,folddensity,foldactnums,[],[],[]]
                    allplayeractdatatemp[playername]['fold']=[0.0,0.0]
                    

            yid=2 #5,2 #利用核密度估计得到的概率密度乘以动作数进行求解
            #计算当前决策各动作的概率(比例)
            lstttdisp=np.zeros(npdfpts)
            for playeractn in l_lstActName:
                if playeractn in allplayeractdatatemp[playername] and allplayeractdatatemp[playername][playeractn]:
                    lstttdisp+=np.array(allplayerApdfdatapck[playername][playeractn][yid]) #actionnumbers
            #print('lstttdisp all acts=',lstttdisp) 

            for playeractn in l_lstActName:
                if playeractn in allplayeractdatatemp[playername] and allplayeractdatatemp[playername][playeractn]:
                    lstrtdisp=[] #一个赢率区间当前动作的占比
                    for i in range(npdfpts):
                        if lstttdisp[i]>1: #目前是手动设定
                            l_tmp_ratio=allplayerApdfdatapck[playername][playeractn][yid][i]/lstttdisp[i]
                        else:
                            l_tmp_ratio=0.0
                        lstrtdisp.append(l_tmp_ratio)
                    allplayerApdfdatapck[playername][playeractn].append(lstrtdisp) #allplayerApdfdatapck[playername][playeractn][6]

            #将r1，r2等统合到raise中
            lstraiseactrt=np.zeros(npdfpts)
            for playeractn in l_lstActRaiseName: #l_lstActRaiseName=['r1','r2','r3','allin']
                if playeractn in allplayeractdatatemp[playername] and allplayeractdatatemp[playername][playeractn]:
                    lstraiseactrt+=np.array(allplayerApdfdatapck[playername][playeractn][6])
            if 'raise' in allplayerApdfdatapck[playername]:
                allplayerApdfdatapck[playername]['raise'].append(lstraiseactrt.tolist())  #allplayerApdfdatapck[playername][playeractn][6]


        yid=1 #核密度估计的概率密度绘图，raise区分
        for playername in allplayeractdatatemp.keys():
            plt.figure() #绘图初始化
            for playeractn in l_lstActName:
                if playeractn in allplayerApdfdatapck[playername] and allplayerApdfdatapck[playername][playeractn]:
                    #print('allplayerApdfdatapck[playername][playeractn]=',allplayerApdfdatapck[playername][playeractn])
                    print('fig: decisionpt=',decisionpt,' name=',playername,' act=',playeractn," yid=",yid)
                    if allplayerApdfdatapck[playername][playeractn][yid]:
                        #绘图
                        plt.plot(allplayerApdfdatapck[playername][playeractn][0],
                        allplayerApdfdatapck[playername][playeractn][yid],label=playername+' '+playeractn)
            plt.grid() #绘制网格
            plt.xlim(0,1) #x轴绘制范围限制
            plt.legend(frameon=False) #显示图例
            plt.xlabel("winrate") #增加x轴说明
            plt.ylabel("action PDF") #增加y轴说明
            plt.title("action PDF at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
            plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+'-pdf-eval-bots-'+allbots+"-"+str(handid)+'.pdf')
            plt.close()

        yid=1 #核密度估计的概率密度绘图，raise综合
        for playername in allplayeractdatatemp.keys():
            plt.figure() #绘图初始化
            for playeractn in ['fold','check','call','raise']:
                if playeractn in allplayerApdfdatapck[playername] and allplayerApdfdatapck[playername][playeractn]:
                    print('fig: decisionpt=',decisionpt,' name=',playername,' act=',playeractn," yid=",yid)
                    if allplayerApdfdatapck[playername][playeractn][yid]:
                        #绘图
                        plt.plot(allplayerApdfdatapck[playername][playeractn][0],
                        allplayerApdfdatapck[playername][playeractn][yid],label=playername+' '+playeractn)
            plt.grid() #绘制网格
            plt.xlim(0,1) #x轴绘制范围限制
            plt.legend(frameon=False) #显示图例
            plt.xlabel("winrate") #增加x轴说明
            plt.ylabel("action PDF") #增加y轴说明
            plt.title("action PDF at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
            plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+'-pdf-eval2-bots-'+allbots+"-"+str(handid)+'.pdf')
            #plt.close()

        yid=6 #动作的占比绘图
        for playername in allplayeractdatatemp.keys():
            plt.figure() #绘图初始化
            for playeractn in l_lstActName:
                if playeractn in allplayerApdfdatapck[playername] and allplayerApdfdatapck[playername][playeractn]:
                    print('fig: decisionpt=',decisionpt,' name=',playername,' act=',playeractn," yid=",yid)
                    if allplayerApdfdatapck[playername][playeractn][yid]:
                        plt.plot(allplayerApdfdatapck[playername][playeractn][0],
                        allplayerApdfdatapck[playername][playeractn][yid],linewidth=2,label=playername+" "+playeractn) #绘制一条'or-'
            plt.grid() #绘制网格
            plt.xlim(0,1) #x轴绘制范围限制
            plt.ylim(0,1) #x轴绘制范围限制
            plt.legend(frameon=False) #显示图例
            plt.xlabel("winrate") #增加x轴说明
            plt.ylabel("action ratio") #增加y轴说明
            plt.title("action ratio at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
            plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+'-ratio-eval1-bots-'+allbots+"-"+str(handid)+'.pdf')
            plt.close()


        yid=6 #动作的占比绘图-使用raise替代r1和r2等
        for playername in allplayeractdatatemp.keys():
            plt.figure() #绘图初始化
            for playeractn in ['fold','check','call','raise']:
                if playeractn in allplayerApdfdatapck[playername] and allplayerApdfdatapck[playername][playeractn]:
                    print('fig: decisionpt=',decisionpt,' name=',playername,' act=',playeractn," yid=",yid)
                    if allplayerApdfdatapck[playername][playeractn][yid]:
                        plt.plot(allplayerApdfdatapck[playername][playeractn][0],
                        allplayerApdfdatapck[playername][playeractn][yid],linewidth=2,label=playername+" "+playeractn) #绘制一条'or-'
            #比较曲线
            if playername=='SC':
                if decisionpt=='pcl' and streetstart==1:
                    dist=[0.4,0.3,0.4]
                    xstt=0.323032
                    xend=0.852037
                    porig=outSmodel(dist,npdfpts,xstt,xend)
                    actnameorig=['fold','call','raise']
                elif decisionpt=='pcl' and streetstart!=1:
                    dist=[0.4,0.3,0.4]
                    porig=outSmodel(dist,npdfpts)
                    actnameorig=['fold','call','raise']
                elif decisionpt=='pck' and streetstart==1:
                    dist=[0.5,0.5]
                    porig=outSmodel(dist,npdfpts)
                    actnameorig=['check','raise']
                elif decisionpt=='pck' and streetstart!=1:
                    dist=[0.5,0.5]
                    porig=outSmodel(dist,npdfpts)
                    actnameorig=['check','raise']
                for i in range(len(porig)):
                    plt.plot(np.linspace(0,1,npdfpts),
                            porig[i],linewidth=2,label=playername+" "+actnameorig[i]+' orig')

            plt.grid() #绘制网格
            plt.xlim(0,1) #x轴绘制范围限制
            plt.ylim(0,1) #x轴绘制范围限制
            plt.legend(frameon=False) #显示图例
            plt.xlabel("winrate") #增加x轴说明
            plt.ylabel("action ratio") #增加y轴说明
            plt.title("action ratio at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
            plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+'-ratio-eval2-bots-'+allbots+"-"+str(handid)+'.pdf')
            #plt.close()

        if 0:
            yid=4 #利用cdf光滑后的差分得到的pdf绘图
            for playername in allplayeractdatatemp.keys():
                plt.figure() #绘图初始化
                for playeractn in l_lstActName:
                    if playeractn in allplayerApdfdatapck[playername] and allplayerApdfdatapck[playername][playeractn]:
                        if allplayerApdfdatapck[playername][playeractn][yid]:
                            plt.plot(allplayerApdfdatapck[playername][playeractn][0],
                            allplayerApdfdatapck[playername][playeractn][yid],linewidth=2,label=playername+" "+playeractn) #绘制一条'or-'
                plt.grid() #绘制网格
                plt.xlim(0,1) #x轴绘制范围限制
                plt.legend(frameon=False) #显示图例
                plt.xlabel("winrate") #增加x轴说明
                plt.ylabel("action PDF") #增加y轴说明
                plt.title("action PDF at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
                plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+'-pdf-eval1-bots-'+allbots+"-"+str(handid)+'.pdf')
                plt.close()

            yid=3 #光滑后的cdf绘图
            for playername in allplayeractdatatemp.keys():
                plt.figure() #绘图初始化
                for playeractn in l_lstActName:
                    if playeractn in allplayerApdfdatapck[playername] and allplayerApdfdatapck[playername][playeractn]:
                        if allplayerApdfdatapck[playername][playeractn][yid]:
                            plt.plot(allplayerApdfdatapck[playername][playeractn][0],
                            allplayerApdfdatapck[playername][playeractn][yid],linewidth=2,label=playername+" "+playeractn) #绘制一条'or-'
                plt.grid() #绘制网格
                plt.xlim(0,1) #x轴绘制范围限制
                plt.legend(frameon=False) #显示图例
                plt.xlabel("winrate") #增加x轴说明
                plt.ylabel("action CDF") #增加y轴说明
                plt.title("action CDF at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
                plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+'-cdf-eval1-bots-'+allbots+"-"+str(handid)+'.pdf')
                plt.close()

            yid=2 #利用pdf估计后的动作数量
            for playername in allplayeractdatatemp.keys():
                plt.figure() #绘图初始化
                for playeractn in l_lstActName+['raise']:
                    if playeractn in allplayerApdfdatapck[playername] and allplayerApdfdatapck[playername][playeractn]:
                        if allplayerApdfdatapck[playername][playeractn][yid]:
                            plt.plot(allplayerApdfdatapck[playername][playeractn][0],
                            allplayerApdfdatapck[playername][playeractn][yid],linewidth=2,label=playername+" "+playeractn) #绘制一条'or-'
                plt.grid() #绘制网格
                plt.xlim(0,1) #x轴绘制范围限制
                plt.legend(frameon=False) #显示图例
                plt.xlabel("winrate") #增加x轴说明
                plt.ylabel("action numbers") #增加y轴说明
                plt.title("action number at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
                plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+'-num-eval1-bots-'+allbots+"-"+str(handid)+'.pdf')
                plt.close()


    #pdf，cdf，action-ratio 统计方法
    if 0:
        decisionPTs=['rnd','pck','pcl']
        for decisionpt in decisionPTs:
            
            if decisionpt=='rnd':
                allplayeractdatatemp=allplayeractdata
            elif decisionpt=='pck':
                allplayeractdatatemp=allplayeractdatapck
            elif decisionpt=='pcl':
                allplayeractdatatemp=allplayeractdatapcl

            allplayerdispdata={}
            l_Interval=0.02 #赢率统计的间隔
            for playername in allplayeractdatatemp.keys():
                allplayerdispdata[playername]={}
                for playeractn in l_lstActName:
                    if playeractn in allplayeractdatatemp[playername] and allplayeractdatatemp[playername][playeractn]:
                        allplayerdispdata[playername][playeractn]=[]
                        lstwr=sorted(allplayeractdatatemp[playername][playeractn])
                        #print("lstwr=",lstwr)
                        if lstwr:
                            lstwrdisp=[] #x轴信息-赢率
                            lstfqdisp=[] #y轴-频率信息
                            ist=0 #从lstwr列表中统计数量

                            for x in range(int(1/l_Interval)):
                                lstwrdisp.append(l_Interval*x+l_Interval/2) #
                                count=0
                                while True and ist<len(lstwr):
                                    if lstwr[ist] < l_Interval*(x+1):
                                        count+=1
                                        ist+=1
                                        #print("len(lstwr)",len(lstwr)," ist=",ist)
                                    else:
                                        break
                                lstfqdisp.append(count)
                            lstpddisp=[y/len(lstwr)/l_Interval for y in lstfqdisp]#y轴-概率密度信息
                            lstcddisp=[]
                            cdsum=0.0
                            for y in lstfqdisp:
                                cdsum+=y
                                lstcddisp.append(cdsum/len(lstwr))
                            #print("lstwrdisp=",lstwrdisp)
                            #print("lstfqdisp=",lstfqdisp)
                            #print("lstpddisp=",lstpddisp)
                            allplayerdispdata[playername][playeractn].append(lstwrdisp)
                            allplayerdispdata[playername][playeractn].append(lstfqdisp)
                            allplayerdispdata[playername][playeractn].append(lstpddisp) #密度分布
                            allplayerdispdata[playername][playeractn].append(lstcddisp) #累积分布

                #统计当前动作在所有动作中的占比
                lstttdisp=[] #一个赢率区间所有动作的数量
                lstaction=[] #所有能统计的动作名列表
                for playeractn in l_lstActName:
                    if playeractn in allplayeractdatatemp[playername] and allplayeractdatatemp[playername][playeractn]:
                        lstaction.append(playeractn)
                for i in range(int(1/l_Interval)):
                    lstttdisp.append(sum([allplayerdispdata[playername][playeractn][1][i] for playeractn in lstaction]))
                print(decisionpt,playername)
                print("lstttdisp=",lstttdisp)
                np.savetxt(playername+'-'+allbots+"-"+strRound+"-"+decisionpt+'-lstttdisp.txt',lstttdisp)
                for playeractn in lstaction:
                    np.savetxt(playername+'-'+allbots+"-"+strRound+"-"+decisionpt+"-"+playeractn+'.txt',allplayerdispdata[playername][playeractn][1])
                    print("lstfqdisp=",allplayerdispdata[playername][playeractn][1])
                    lstrtdisp=[] #一个赢率区间当前动作的占比
                    for i in range(int(1/l_Interval)):
                        if lstttdisp[i]>0:
                            l_tmp_ratio=allplayerdispdata[playername][playeractn][1][i]/lstttdisp[i]
                        else:
                            l_tmp_ratio=0.0
                        lstrtdisp.append(l_tmp_ratio)
                    allplayerdispdata[playername][playeractn].append(lstrtdisp)
            #print('allplayerdispdata=',allplayerdispdata)

            yid=3 #设置绘图的内容为累积分布
            for playername in allplayeractdatatemp.keys():
                plt.figure() #绘图初始化
                for playeractn in l_lstActName:
                    if playeractn in allplayerdispdata[playername] and allplayerdispdata[playername][playeractn]:
                        plt.plot(allplayerdispdata[playername][playeractn][0],
                        allplayerdispdata[playername][playeractn][yid],
                        linewidth=2,label=playername+" "+playeractn) #绘制一条'or-'
                plt.grid() #绘制网格
                plt.xlim(0,1) #x轴绘制范围限制
                plt.legend(frameon=False) #显示图例
                plt.xlabel("winrate") #增加x轴说明
                plt.ylabel("action CDF") #增加y轴说明
                plt.title("action CDF at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
                plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+"-cdf-bots-"+allbots+"-"+str(handid)+'.pdf')
                plt.close()
            
            yid=2 #设置绘图的内容为概率密度
            for playername in allplayeractdatatemp.keys():
                plt.figure() #绘图初始化
                for playeractn in l_lstActName:
                    if playeractn in allplayerdispdata[playername] and allplayerdispdata[playername][playeractn]:
                        plt.plot(allplayerdispdata[playername][playeractn][0],
                        allplayerdispdata[playername][playeractn][yid],
                        linewidth=2,label=playername+" "+playeractn) #绘制一条'or-'
                plt.grid() #绘制网格
                plt.xlim(0,1) #x轴绘制范围限制
                plt.legend(frameon=False) #显示图例
                plt.xlabel("winrate") #增加x轴说明
                plt.ylabel("action PDF") #增加y轴说明
                plt.title("action PDF at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
                plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+"-pdf-bots-"+allbots+"-"+str(handid)+'.pdf')
                plt.close()

            yid=4  #动作比例统计
            for playername in allplayeractdatatemp.keys():
                plt.figure() #绘图初始化
                for playeractn in l_lstActName:
                    if playeractn in allplayerdispdata[playername] and allplayerdispdata[playername][playeractn]:
                        plt.plot(allplayerdispdata[playername][playeractn][0],
                        allplayerdispdata[playername][playeractn][yid],
                        linewidth=2,label=playername+" "+playeractn) #绘制一条'or-'
                plt.grid() #绘制网格
                plt.xlim(0,1) #x轴绘制范围限制
                plt.ylim(0,1) #x轴绘制范围限制
                plt.legend(frameon=False) #显示图例
                plt.xlabel("winrate") #增加x轴说明
                plt.ylabel("action ratio") #增加y轴说明
                plt.title("action ratio at "+decisionpt+' of '+strRound+' for '+playername,fontsize=16) #调整x轴坐标刻度和标签
                plt.savefig('fig-'+playername+'-'+decisionpt+'-'+strRound+"-ratio-bots-"+allbots+"-"+str(handid)+'.pdf')
                plt.close()

    #plt.show() #显示图片
    return None



# 所有动作的概率模型
def statPlayerActProbAll():

    l_lstActName=['fold','check','call','r1','r2','r3','allin']
    #统计对手的频率模型-反过来就是不同赢率对应的动作
    #记录对手做每个动作时的赢率，用一个字典表示，动作名表示字典的key，赢率用一个列表记录
    allplayeractdata={}
    for rec in recAllgameHistoryS:
        #print("rec=",rec)
        playername=rec[0]
        streetnumb=rec[1]
        boardcards=rec[2]
        opponentac=rec[3]
        playeractn=rec[4]
        playercard=rec[5]
        #构成：{playername:{playeractn:[winrate,...]}}这样第一个字典
        if playername not in allplayeractdata:
            allplayeractdata[playername]={}
        if playercard:
            if playeractn not in allplayeractdata[playername]:
                allplayeractdata[playername][playeractn]=[]
            playerwinrate=0.0
            if streetnumb==1:
                playerwinrate=getwinrate(nplayer,playercard,[])
            elif streetnumb==2:
                playerwinrate=getwinrate(nplayer,playercard,boardcards[:3])
            elif streetnumb==3:
                playerwinrate=getwinrate(nplayer,playercard,boardcards[:4])
            else:
                playerwinrate=getwinrate(nplayer,playercard,boardcards)
            allplayeractdata[playername][playeractn].append(playerwinrate)
        
    allplayerdispdata={}
    for playername in allplayeractdata.keys():
        allplayerdispdata[playername]={}
        for playeractn in allplayeractdata[playername].keys():
            allplayerdispdata[playername][playeractn]=[]
            lstwr=sorted(allplayeractdata[playername][playeractn])
            #print("lstwr=",lstwr)
            if lstwr:
                lstwrdisp=[] #x轴信息-赢率
                lstfqdisp=[] #y轴-频率信息
                ist=0 #从lstwr列表中统计数量
                l_Interval=0.02 #赢率统计的间隔

                for x in range(int(1/l_Interval)):
                    lstwrdisp.append(l_Interval*x+l_Interval/2) #
                    count=0
                    while True and ist<len(lstwr):
                        if lstwr[ist] < l_Interval*(x+1):
                            count+=1
                            ist+=1
                            #print("len(lstwr)",len(lstwr)," ist=",ist)
                        else:
                            break
                    lstfqdisp.append(count)
                lstpddisp=[y/len(lstwr)  for y in lstfqdisp]#y轴-概率密度信息
                #print("lstwrdisp=",lstwrdisp)
                #print("lstfqdisp=",lstfqdisp)
                #print("lstpddisp=",lstpddisp)
                allplayerdispdata[playername][playeractn].append(lstwrdisp)
                allplayerdispdata[playername][playeractn].append(lstfqdisp)
                allplayerdispdata[playername][playeractn].append(lstpddisp)

    for playeractn in l_lstActName:
        plt.figure() #绘图初始化
        for playername in allplayerdispdata.keys():
            if playeractn in allplayerdispdata[playername]:
                plt.plot(allplayerdispdata[playername][playeractn][0],allplayerdispdata[playername][playeractn][1],linewidth=2,label=playername+" "+playeractn+" frequencies") #绘制一条'or-'
        plt.grid() #绘制网格
        plt.legend(loc='upper left',frameon=False) #显示图例
        plt.xlim(0,1) #x轴绘制范围限制
        plt.xlabel("winrate") #增加x轴说明
        plt.ylabel("action times") #增加y轴说明
        plt.title("action times of winrate",fontsize=16) #调整x轴坐标刻度和标签
    
    for playeractn in l_lstActName:
        plt.figure() #绘图初始化
        #print(allplayerdispdata[playername][playeractn][0])
        #print(allplayerdispdata[playername][playeractn][2])
        for playername in allplayeractdata.keys():
            if playeractn in allplayerdispdata[playername]:
                plt.plot(allplayerdispdata[playername][playeractn][0],allplayerdispdata[playername][playeractn][2],linewidth=2,label=playername+" "+playeractn+" PDF") #绘制一条'or-'
        plt.grid() #绘制网格
        plt.xlim(0,1) #x轴绘制范围限制
        plt.legend(loc='upper left',frameon=False) #显示图例
        plt.xlabel("winrate") #增加x轴说明
        plt.ylabel("act PDF") #增加y轴说明
        plt.title("PDF of winrate",fontsize=16) #调整x轴坐标刻度和标签

    #print('allplayeractdata=',allplayeractdata)
    '''
    f = open("temp.md","w")
    file_json=json.dump(allplayeractdata,f)
    f.close()
    '''
    return None


# 玩家的赢钱统计
def statPlayerWinmoney():
    global recwinmoneypgame
    global recwinmoneytotal

    # 记录所有玩家的从开始到结束的所有局的赢钱数
    recwinmoneypgame={}
    # 记录所有玩家到从开始到结束的所有局的赢钱数
    recwinmoneytotal={}

    nplayer=len(recAllagentsname) 
    allbots="-".join(recAllagentsname)
    handid=len(datausers)

    for name in recAllagentsname:
        recwinmoneypgame[name]=[]
        recwinmoneytotal[name]=[]

    if 1:
        for data in datausers:
            #记录玩家每局赢钱数和总数
            for player in data['players']:
                recwinmoneypgame[player['name']].append(player['win_money'])
                if len(recwinmoneytotal[player['name']])>0:
                    tempwinmonyt=recwinmoneytotal[player['name']][-1]+player['win_money']
                else:
                    tempwinmonyt=player['win_money']
                recwinmoneytotal[player['name']].append(tempwinmonyt)

        '''
        filename='rec'+"-"+allbots+"-"+str(handid)+'-pgame.md'
        f = open(filename, 'wb')
        pickle.dump(recwinmoneypgame, f)
        f.close()

        filename='rec'+"-"+allbots+"-"+str(handid)+'-total.md'
        f = open(filename, 'wb')
        pickle.dump(recwinmoneytotal, f)
        f.close()
        '''

        d2=recwinmoneytotal
        plt.figure() #绘图初始化
        keys=[]
        for key in d2.keys():
            keys.append(key)
        datalenth=len(d2[keys[0]])
        x=range(datalenth)
        colorline=['k:','+r-','b--','g-.','y','m1']
        for i in range(len(keys)):
            plt.plot(x,d2[keys[i]],colorline[i],linewidth=2,label=keys[i]) #绘制一条'or-'
        plt.grid() #绘制网格
        plt.legend(loc='upper left',frameon=False) #显示图例
        plt.xlabel("game") #增加x轴说明
        plt.ylabel("winmoney") #增加y轴说明
        plt.title("Win money of each player",fontsize=16) #调整x轴坐标刻度和标签
    #plt.show()

    Result=copy.deepcopy(recwinmoneytotal)

    return Result,allbots,handid


# 比较两个结果文件的赢钱(收益)
def compareWinmoneyTwoResult(filename1,filename2,myname):

    datadealing(filename1)
    result1,allbots,handid=statPlayerWinmoney()

    datadealing(filename2)
    result2,allbots,handid=statPlayerWinmoney()

    l_ngames=0
    for result in [result1,result2]:
        for name in result.keys():
            l_ngames=len(result[name])
            for i in range(l_ngames):
                result[name][i]=result[name][i]/(i+1)*10 #转成mbb/g

    opname=''
    for name in result1.keys():
        if name != myname:
            opname=name

    plt.figure()
    x=np.linspace(1,l_ngames,l_ngames)
    caselist=[' with pred',' without pred']
    j=0
    for result in [result1,result2]:
        name=myname
        plt.plot(x,result[name],label=name+' vs '+opname+" "+caselist[j])
        print('name=',name,' wm=',result[name][-1])
        j+=1
    plt.xlabel("game") #增加x轴说明
    plt.ylabel("$W_m$(mbb/g)") #增加y轴说明
    plt.title("Win Money of Each Player",fontsize=16) #调整x轴坐标刻度和标签
    plt.grid() #绘制网格
    plt.legend(loc='upper left',frameon=False) #显示图例
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-winmoney.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-winmoney.pdf")
    plt.show()

    return None



# 玩家的特征统计并输出到文件中
def statPlayerFeature():

    for name in recAllagentsname:
        #统计相关的次数，并计算特征量
        #{"chand":总手数,"cpfvplay":翻牌前玩牌玩牌手数,"cpfvraise":翻牌前加注手数，
        # "cafraise":翻牌后加注和下注的次数,"cafcall":翻牌后跟注的次数,
        # "vpip":vpip值,'pfr':pfr,'af':af,'pfrdvpip':pfr/vpip}
        recAllagentsstat[name]={"chand":0,"cfhandpre":0,"cpfvplay":0,"cpfvraise":0,"cafraise":0,"cafcall":0,
        'cpfccl':0,'cpfcclfd':0,'cpfcclcl':0,'cpfcclrs':0,'cpfcck':0,'cpfcckck':0,'cpfcckrs':0,
        'cafccl':0,'cafcclfd':0,'cafcclcl':0,'cafcclrs':0,'cafcck':0,'cafcckck':0,'cafcckrs':0,
        'cpfall':0,'cpfallfd':0,'cpfallcl':0,'cafall':0,'cafallfd':0,'cafallcl':0,
        "vpip":0.0,'pfr':0.0,'af':0.0,'pfrdvpip':0.0,'rvpip':0.0,"rpfr":0.0,
        'br':0.0,'fr':0.0,'rr':0.0,'pfbr':0.0,'pffr':0.0,'pfrr':0.0}

    nplayer=len(recAllagentsname) 
    allbots="-".join(recAllagentsname)

    #用于记录整个过程中的特征参数
    vpipdata={}
    pfrdata={}
    afdata={}
    pfrdvpipdata={}
    pfbrdata={}
    pffrdata={}
    pfrrdata={}
    brdata={}
    frdata={}
    rrdata={}
    for name in recAllagentsname:
        vpipdata[name]=[]
        pfrdata[name]=[]
        afdata[name]=[]
        pfrdvpipdata[name]=[]
        pfbrdata[name]=[]
        pffrdata[name]=[]
        pfrrdata[name]=[]
        brdata[name]=[]
        frdata[name]=[]
        rrdata[name]=[]

    
    #处理每一局的结果数据
    
    handid=0
    for data in datausers:
        handid+=1

        #处理用于统计的动作和其它信息
        #记录到recAllgameHistory和recAllgameHistoryS中
        get_actioninfosetall(data,nplayer)
        
        if 1:
            for name in recAllagentsname:
                recAllagentsstat[name]["chand"]=handid
                recAllagentsstat[name]["vpip"]=int(recAllagentsstat[name]["cpfvplay"]/handid*100)
                recAllagentsstat[name]["pfr"]=int(recAllagentsstat[name]["cpfvraise"]/handid*100)
                recAllagentsstat[name]["rvpip"]=int(recAllagentsstat[name]["cpfvplay"]/(handid-recAllagentsstat[name]["cfhandpre"])*100)
                recAllagentsstat[name]["rpfr"]=int(recAllagentsstat[name]["cpfvraise"]/(handid-recAllagentsstat[name]["cfhandpre"])*100)

                if recAllagentsstat[name]["cafcck"]!=0:
                    recAllagentsstat[name]["br"]=int(recAllagentsstat[name]["cafcckrs"]/recAllagentsstat[name]["cafcck"]*100)
                else:
                    recAllagentsstat[name]["br"]=0
                if recAllagentsstat[name]["cafccl"]!=0:
                    recAllagentsstat[name]["fr"]=int(recAllagentsstat[name]["cafcclfd"]/recAllagentsstat[name]["cafccl"]*100)
                    recAllagentsstat[name]["rr"]=int(recAllagentsstat[name]["cafcclrs"]/recAllagentsstat[name]["cafccl"]*100)
                else:
                    recAllagentsstat[name]["fr"]=0
                    recAllagentsstat[name]["rr"]=0

                if recAllagentsstat[name]["cpfcck"]!=0:
                    recAllagentsstat[name]['pfbr']=int(recAllagentsstat[name]["cpfcckrs"]/recAllagentsstat[name]["cpfcck"]*100)
                else:
                    recAllagentsstat[name]['pfbr']=0
                if recAllagentsstat[name]["cpfccl"]!=0:
                    recAllagentsstat[name]['pffr']=int(recAllagentsstat[name]["cpfcclfd"]/recAllagentsstat[name]["cpfccl"]*100)
                else:
                    recAllagentsstat[name]['pffr']=0
                if recAllagentsstat[name]["cpfccl"]!=0:
                    recAllagentsstat[name]['pfrr']=int(recAllagentsstat[name]["cpfcclrs"]/recAllagentsstat[name]["cpfccl"]*100)
                else:
                    recAllagentsstat[name]['pfrr']=0


                if recAllagentsstat[name]["cafcall"]!=0:
                    recAllagentsstat[name]["af"]=(recAllagentsstat[name]["cafraise"]/recAllagentsstat[name]["cafcall"])
                else:
                    recAllagentsstat[name]["af"]=100
                if recAllagentsstat[name]["vpip"]!=0:
                    recAllagentsstat[name]["pfrdvpip"]=(recAllagentsstat[name]["pfr"]/recAllagentsstat[name]["vpip"])
                else:
                    recAllagentsstat[name]["pfrdvpip"]=0.0
                
                vpipdata[name].append(recAllagentsstat[name]["vpip"])
                pfrdata[name].append(recAllagentsstat[name]["pfr"])
                afdata[name].append(recAllagentsstat[name]["af"])
                pfrdvpipdata[name].append(recAllagentsstat[name]["pfrdvpip"])
                pfbrdata[name].append(recAllagentsstat[name]["pfbr"])
                pffrdata[name].append(recAllagentsstat[name]["pffr"])
                pfrrdata[name].append(recAllagentsstat[name]["pfrr"])
                brdata[name].append(recAllagentsstat[name]["br"])
                frdata[name].append(recAllagentsstat[name]["fr"])
                rrdata[name].append(recAllagentsstat[name]["rr"])
                '''
                print('name\tpfbr\tpffr\tpfrr\tbf\tfr\trr')
                print('{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(name
                ,pfbrdata[name][-1],pffrdata[name][-1],pfrrdata[name][-1],
                brdata[name][-1],frdata[name][-1],rrdata[name][-1]))
                print('recAllagentsstat[name]=',recAllagentsstat[name])
                '''
                #anykey=input()


    #计算特征量
    handid=len(datausers)
    for name in recAllagentsname:
        recAllagentsstat[name]["chand"]=handid
        recAllagentsstat[name]["vpip"]=int(recAllagentsstat[name]["cpfvplay"]/handid*100)
        recAllagentsstat[name]["pfr"]=int(recAllagentsstat[name]["cpfvraise"]/handid*100)
        recAllagentsstat[name]["rvpip"]=int(recAllagentsstat[name]["cpfvplay"]/(handid-recAllagentsstat[name]["cfhandpre"])*100)
        recAllagentsstat[name]["rpfr"]=int(recAllagentsstat[name]["cpfvraise"]/(handid-recAllagentsstat[name]["cfhandpre"])*100)

        recAllagentsstat[name]["br"]="{:.1f}".format(recAllagentsstat[name]["cafcckrs"]/recAllagentsstat[name]["cafcck"]*100)
        recAllagentsstat[name]["fr"]="{:.1f}".format(recAllagentsstat[name]["cafcclfd"]/recAllagentsstat[name]["cafccl"]*100)
        recAllagentsstat[name]["rr"]="{:.1f}".format(recAllagentsstat[name]["cafcclrs"]/recAllagentsstat[name]["cafccl"]*100)

        recAllagentsstat[name]['pfbr']="{:.1f}".format(recAllagentsstat[name]["cpfcckrs"]/recAllagentsstat[name]["cpfcck"]*100)
        recAllagentsstat[name]['pffr']="{:.1f}".format(recAllagentsstat[name]["cpfcclfd"]/recAllagentsstat[name]["cpfccl"]*100)
        recAllagentsstat[name]['pfrr']="{:.1f}".format(recAllagentsstat[name]["cpfcclrs"]/recAllagentsstat[name]["cpfccl"]*100)

        if recAllagentsstat[name]["cafcall"]!=0:
            recAllagentsstat[name]["af"]="{:.2f}".format(recAllagentsstat[name]["cafraise"]/recAllagentsstat[name]["cafcall"])
        else:
            recAllagentsstat[name]["af"]=100
        if recAllagentsstat[name]["vpip"]!=0:
            recAllagentsstat[name]["pfrdvpip"]="{:.2f}".format(recAllagentsstat[name]["pfr"]/recAllagentsstat[name]["vpip"])
        else:
            recAllagentsstat[name]["pfrdvpip"]=0.0
    print('recAllagentsstat=',recAllagentsstat)

    #将这些特征写入csv文件中
    featureslst=['chand','cfhandpre','cpfvplay','cpfvraise','cafraise','cafcall',
    'cpfccl','cpfcclfd','cpfcclcl','cpfcclrs','cpfcck','cpfcckck','cpfcckrs','cpfall','cpfallcl','cpfallfd',
    'cafccl','cafcclfd','cafcclcl','cafcclrs','cafcck','cafcckck','cafcckrs','cafall','cafallcl','cafallfd',
    'vpip','pfr','af','pfrdvpip','rvpip','rpfr','pfbr','pffr','pfrr','br','fr','rr']
    resfeaturelist=np.zeros((len(featureslst),len(recAllagentsname)))

    j=-1
    for name in recAllagentsname:
        j+=1
        for i in range(len(resfeaturelist)):
            resfeaturelist[i,j]=recAllagentsstat[name][featureslst[i]]
    featuresdata=[featureslst]+resfeaturelist.T.tolist()
    ftransposed = list()
    for i in range(len(featuresdata[0])):
        row = list()
        for sublist in featuresdata:
            row.append(sublist[i])
        ftransposed.append(row)
    featuresfinal=[["player"]+recAllagentsname]+ftransposed
    print('featuresfinal',featuresfinal)


    #保存特征数据
    filenameft='rec'+"-"+allbots+"-"+str(handid)+'-featrues.csv'
    np.savetxt(filenameft,featuresfinal,delimiter=',',fmt ='% s') 

    for name in recAllagentsname:
        plt.figure() #绘图初始化
        plt.plot(pfbrdata[name],marker='x',markevery=123,label='PFBR') #
        plt.plot(pffrdata[name],marker=',',markevery=111,label='PFFR') #
        plt.plot(pfrrdata[name],marker='o',markevery=103,label='PFRR') #
        plt.plot(brdata[name],marker='v',markevery=127,label='BR') #
        plt.plot(frdata[name],marker='<',markevery=119,label='FR') #
        plt.plot(rrdata[name],marker='>',markevery=107,label='RR') #
        plt.title("Features for {}".format(name),fontsize=16) #
        plt.xlabel("games") #增加x轴说明
        plt.ylabel("value(%)") #增加y轴说明
        plt.legend(frameon=False,loc='upper right') #显示图例
        #plt.xlim(-5,100) #x轴绘制范围限制
        plt.ylim(-5,110) #x轴绘制范围限制
        plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+name+"-features.svg")
        plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+name+"-features.pdf")

    '''
    plt.figure() #绘图初始化
    i=0
    for name in recAllagentsname:
        plt.plot(vpipdata[name],colorline[i],label=name) #
        i+=1
    plt.title("VPIP",fontsize=16) #
    plt.legend(frameon=False) #显示图例
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-vpip.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-vpip.pdf")

    plt.figure() #绘图初始化
    i=0
    for name in recAllagentsname:
        plt.plot(pfrdata[name],colorline[i],label=name) #
        i+=1
    plt.title("PFR",fontsize=16) #
    plt.legend(frameon=False) #显示图例
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-pfr.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-pfr.pdf")

    plt.figure() #绘图初始化
    i=0
    for name in recAllagentsname:
        plt.plot(afdata[name],colorline[i],label=name) #
        i+=1
    plt.title("AF",fontsize=16) #
    plt.legend(frameon=False) #显示图例
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-af.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-af.pdf")

    plt.figure() #绘图初始化
    i=0
    for name in recAllagentsname:
        plt.plot(pfrdvpipdata[name],colorline[i],label=name) #
        i+=1
    plt.title("PFR/VPIP",fontsize=16) #
    plt.legend(frameon=False) #显示图例
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-pfrdvpip.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-pfrdvpip.pdf")
    '''
    plt.show()
    return None


# 绘制不同风格玩家的特征数据
def displayplayerfeature():
    #输入数据用列表表示，便于从csv文件中读取
    playernamelst=['LooseAggressive','LoosePassive','TightAggressive','TightPassive']
    #V1
    '''
    playervpiplst=[78,63,50,20]
    playerpfrlst=[57,0,28,0]
    playeraflst=[5.1 ,0.48, 6.69, 1.14]
    playerpfrdvpiplst=[0.73 , 0 , 0.56, 0]
    '''
    #V2
    '''
    playervpiplst=[78,63,22,20]
    playerpfrlst=[57,0,4,0]
    playeraflst=[5.1 ,0.48, 6.63, 1.14]
    playerpfrdvpiplst=[0.73 , 0 , 0.18, 0]
    '''
    #V3
    playervpiplst=[79,  67, 30, 33]
    playerpfrlst=[57,   19, 12, 17]
    playeraflst=[5.114658926,   1.625632378,    6.443609023,    3.498905908]
    playerpfrdvpiplst=[0.721518987, 0.28358209, 0.4,    0.515151515]

    #dp
    playernamelst=['DP vs LA','DP vs LP','DP vs TA','DP vs TP'] #vs V2
    playervpiplst=[65,  48, 39, 39]
    playerpfrlst=[27,   25, 18, 18]
    playeraflst=[1.360995851,   6.993318486,    1.576815642,    4.214046823]
    playerpfrdvpiplst=[0.415384615, 0.520833333,    0.461538462,    0.461538462]

    vpipdata={}
    pfrdata={}
    afdata={}
    pfrdvpipdata={}
    i=0
    for name in playernamelst:
        vpipdata[name]=[playervpiplst[i]]
        pfrdata[name]=[playerpfrlst[i]]
        afdata[name]=[playeraflst[i]]
        pfrdvpipdata[name]=[playerpfrdvpiplst[i]]
        i+=1
    print(vpipdata)
    print(pfrdata)
    print(afdata)
    print(pfrdvpipdata)

    plt.figure() 
    i=0
    for name in playernamelst:
        plt.scatter(pfrdata[name][-1],vpipdata[name][-1],marker=markline[i],s=100,label=name)
        i+=1
    plt.xlim(-5,100) #x轴绘制范围限制
    plt.ylim(-5,100) #x轴绘制范围限制
    plt.xlabel("PFR(%)") #增加x轴说明
    plt.ylabel("VPIP(%)") #增加y轴说明
    plt.legend(frameon=False) #显示图例
    plt.grid()
    plt.title("VPIP vs PFR",fontsize=16) #
    plt.savefig("fig-player-styles-vpipvspfr.svg")
    plt.savefig("fig-player-styles-vpipvspfr.pdf")


    plt.figure() 
    i=0
    for name in playernamelst:
        plt.scatter(10*np.log10(afdata[name][-1]),vpipdata[name][-1],marker=markline[i],s=100,label=name)
        i+=1
    plt.xlim(-20,20) #x轴绘制范围限制
    plt.ylim(-5,100) #x轴绘制范围限制
    plt.xlabel("AF(dB)") #增加x轴说明
    plt.ylabel("VPIP(%)") #增加y轴说明
    plt.legend(frameon=False) #显示图例
    plt.grid()
    plt.title("VPIP vs AF",fontsize=16) #
    plt.savefig("fig-player-styles-vpipvsaf.svg")
    plt.savefig("fig-player-styles-vpipvsaf.pdf")


    plt.figure() 
    i=0
    for name in playernamelst:
        plt.scatter(pfrdvpipdata[name][-1]*100,vpipdata[name][-1],marker=markline[i],s=100,label=name)
        i+=1
    plt.xlim(-5,100) #x轴绘制范围限制
    plt.ylim(-5,100) #x轴绘制范围限制
    plt.xlabel("PFR/VPIP(%)") #增加x轴说明
    plt.ylabel("VPIP(%)") #增加y轴说明
    plt.legend(frameon=False) #显示图例
    plt.grid()
    plt.title("VPIP vs PFR/VPIP",fontsize=16) #
    plt.savefig("fig-player-styles-vpipvsprfdvpip.svg")
    plt.savefig("fig-player-styles-vpipvsprfdvpip.pdf")

    plt.show()
    return None




# 将acpc的log的信息转换为data数据
# 注意这是两人情况下用的，因为其中很多处理都是针对两人做的。
# 输入是log文件中的字符串和玩家的名字
# BigBlindfirst选项用于控制大盲注先行的情况，从acpc收集的数据看是有大盲注线性的。
# 而默认是小盲注先的。
params_sb = 50
params_bb = 100
params_stack = 20000
params_maxr = 3  #一轮最大raise次数
def ACPClogmsgTodata(msg,myname,BigBlindfirst=False,flagfoldseen=True):

    data={} #字典

    #print('in msg=',msg)
    m1=re.search("^STATE:(\d*):([^:]*):(.*):(.*):(.*)",msg.strip())
    #print('m1=',m1)
    
    hand_id=int(m1.group(1))
    actions=m1.group(2).strip()
    cards=m1.group(3)
    if BigBlindfirst:
        win_money=m1.group(4).split('|')
        playernames=m1.group(5).split('|')
        win_money.reverse()
        playernames.reverse()
    else:
        win_money=m1.group(4).split('|')
        playernames=m1.group(5).split('|')

    room_number=2
    position=playernames.index(myname)
    #print('position=',position)
    name=myname
    opposition=1-position
    opname=playernames[opposition]
    

    m2=re.search("([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)",actions)
    #print('m2=',m2,m2.group(0))
    preflop_actions = m2.group(1)
    flop_actions = m2.group(2)
    turn_actions = m2.group(3)
    river_actions = m2.group(4)
    lastaction=''
    if actions:
        lastaction=actions[-1]
    if preflop_actions=='':
        flagFirstAction=True
    else:
        flagFirstAction=False

    #print("cards",cards)
    m3 = re.search("([^\|]*)\|([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)",cards.strip())
    #print('m3=',m3,m3.group(0))
    if BigBlindfirst:
        hand_p1=m3.group(2)
        hand_p2=m3.group(1)
    else:
        hand_p1=m3.group(1)
        hand_p2=m3.group(2)
    flopcds=m3.group(3)
    turncds=m3.group(4)
    rivercds=m3.group(5)
    #print("flopcds=",flopcds)
    #print("turncds=",turncds)
    #print("rivercds=",rivercds)

    #位置和局序号
    data['hand_id']=hand_id

    #手牌
    if position==0 :
        data['private_card']=[hand_p1[:2],hand_p1[2:]]
    else:
        data['private_card']=[hand_p2[:2],hand_p2[2:]]

    #print('hand_p1=',hand_p1)
    #print('hand_p2=',hand_p2)
    if hand_p1 and hand_p2 :
        data['player_card']=[[hand_p1[:2],hand_p1[2:]],[hand_p2[:2],hand_p2[2:]]]
    else:
        if position==0 :
            data['player_card']=[[hand_p1[:2],hand_p1[2:]],[]]
        else:
            data['player_card']=[[],[hand_p2[:2],hand_p2[2:]]]
    
    ophand=data['player_card'][opposition]
    if ophand:
        ophandidx=HandtoIdx(cardint[ophand[0]],cardint[ophand[1]])
        data['ophandidx']=ophandidx

    if not flagfoldseen and lastaction=='f': #当对手手牌数据不可观测时，根据动作f来去掉
        data['player_card'][opposition]=[]


    #公共牌
    street=1
    actionstrings=[preflop_actions]
    data['public_card']=[]
    if flopcds :
        data['public_card']=[flopcds[:2],flopcds[2:4],flopcds[4:]]
        street=2
        actionstrings.append(flop_actions)
    if turncds :
        data['public_card'].append(turncds)
        street=3
        actionstrings.append(turn_actions)
    if rivercds :
        data['public_card'].append(rivercds)
        street=4
        actionstrings.append(river_actions)
    data['street']=street
    #print('actionstrings=',actionstrings)

    #动作和下注额
    #要特别注意：下面的代码是很多地方都是基于两人对局做了特殊处理的
    #A raise is valid if a) it raises by at least one chip, b) the player has sufficient money in their stack to
    #raise to the value, and c) the raise would put the player all-in (they have spent all their chips) or the
    #amount they are raising by is at least as large as the big blind and the raise-by amount for any other
    #raise in the round. 
    #注意acpc协议中r****实际是raiseto，就是全局加注到多少(而不是一个轮次的)
    #一次raise实际下注额是call上的+对手raise增加的额度+大盲注
    #那么raiseto的最小值为：原来已经下注的额度+前一次raise增加的额度+大盲注
    money_beted_now=[params_sb,params_bb] #两个人对局：位置0的下注额，位置1的下注额
    money_beted_rnd=[0,0,0,0] #四个轮次 
    actions=[]
    actions_all=[]
    tmp_lastAction=[] #最后一个动作，倒数第一个动作
    tmp_lastSecAction=[] #最后第二个动作，倒数第二个动作
    tmp_lastThiAction=[] #倒数第三个动作
    raisetimes=0
    raisemprev=0
    actpos=0    #各动作的所对应的玩家的位置
    for i  in range(len(actionstrings)):
        #tmp_lastRoundAction='' #当前轮的动作记录
        #tmp_lastMyRoundAction='' #当前轮的动作记录

        actions_remainder=actionstrings[i]
        actions_round=[]
        raisetimes=0 #一轮中raise的次数，各轮需要分别统计
        raisemprev=0 #一轮中一次raise增加的下注额度，各轮需要分别统计
        actsn=0      #一轮中动作的序号
        while actions_remainder != '':
            actsn+=1 #每一轮第一个动作序号为1，第二个动作序号为2，后续为3,4,5...
            if i==0: #preflop轮小盲先行
                actpos=1-actsn%2
            else: #flop/turn/river轮大盲先行
                actpos=actsn%2
            #print('actsn=',actsn)
            #print('actpos=',actpos)
            parsed_chunk = ''
            if actions_remainder.startswith("c"):
                #call 改成check
                if money_beted_now[actpos]==money_beted_now[1-actpos]:
                    actionstr="check"
                else:
                    actionstr="call"
                actions_round.append({"action":actionstr,"position": actpos})
                actions_all.append({"action":actionstr,"position": actpos})
                parsed_chunk = "c"
                money_beted_now[actpos]=money_beted_now[1-actpos]
            elif actions_remainder.startswith("r"):
                #print('actions_remainder=',actions_remainder)
                raise_amount = int(re.search("^r(\d*).*",actions_remainder).group(1))
                parsed_chunk = "r" + str(raise_amount)
                #print('raise_amount=',raise_amount)
                raisetimes+=1
                #raise的额度就是在call平基础上增加的额度
                #raisemprev=raise_amount-max(money_beted_now)
                #raise_amount就是raise后的下注额，即raiseto的额度
                money_beted_now[actpos]=raise_amount
                #注意acpc协议是全局的raiseto，而cisia则是当前轮的raiseto
                #acpc
                #actions_round.append({"action":"r" + str(raise_amount),"position": actpos}) #,"raise_amount":raise_amount
                #actions_all.append({"action":"r" + str(raise_amount),"position": actpos})
                #parsed_chunk = "r" + str(raise_amount)
                #cisia
                if i==0:
                    actions_round.append({"action":"r" + str(raise_amount),"position": actpos}) #,"raise_amount":raise_amount
                    actions_all.append({"action":"r" + str(raise_amount),"position": actpos})
                else:
                    raise_amount=raise_amount-money_beted_rnd[i-1]
                    actions_round.append({"action":"r" + str(raise_amount),"position": actpos}) #,"raise_amount":raise_amount
                    actions_all.append({"action":"r" + str(raise_amount),"position": actpos})
            elif actions_remainder.startswith("f"):
                actions_round.append({"action":"fold","position": actpos})
                actions_all.append({"action":"fold","position": actpos})
                parsed_chunk = "f"
            else:
                print("wrong action string")
            tmp_lastThiAction=tmp_lastSecAction
            tmp_lastSecAction=tmp_lastAction
            tmp_lastAction=[actions_round[-1]["action"],actpos]
            #tmp_lastMyRoundAction=tmp_lastRoundAction
            #tmp_lastRoundAction=actions_round[-1]["action"] #当前轮的动作记录
            
            actions_remainder = actions_remainder.replace(parsed_chunk,"",1)

        if money_beted_now[0]==money_beted_now[1]:
            money_beted_rnd[i]=money_beted_now[0]
        else:
            #print("not equal bet in round:",i)
            pass

        actions.append(actions_round)
    
    data['action_history']=actions


    #判断决策点信息
    #这里虽然可能轮次变化后最后一个动作可能是前一轮的，但很巧的是
    #下面的逻辑也没有问题，因为前一个动作若是r的画，那么轮次比如没有结束
    #所以面临的决策点是pcl是没有问题的。
    if position==0 :
        flagDecisionPt=1
    else:
        flagDecisionPt=0

    if tmp_lastAction:
        if tmp_lastAction[0][0]=="r":
            flagDecisionPt=1 #pcl point
        else:
            flagDecisionPt=0 #pck point
    if tmp_lastAction:
        if tmp_lastAction[1]==position:#当最后一个动作是自己的时就一定在轮次交界处
            #我方的动作必然是c或者f
            #因此对手的动作必然是call或者raise
            #若最后一个动作在第一轮，则对手必然是在pcl
            if street==2: # cc|flop，rc|flop
                flagOpDecisionPt=1 #pcl point,前一次对手面对的决策点类型
            else: # 比如：  rrc|turn
                if tmp_lastThiAction:
                    if tmp_lastThiAction[0][0]=='r':
                        flagOpDecisionPt=1
                    else: # 比如：crc|turn
                        flagOpDecisionPt=0 #pck point,前一次对手面对的决策点类型
                else:
                    flagOpDecisionPt=1
        else:
            if tmp_lastSecAction:
                if tmp_lastSecAction[0][0]=='r':
                    flagOpDecisionPt=1 
                else:
                    flagOpDecisionPt=0
            else:# 若倒数第二个动作不存在，且不是我做的倒数第一个动作，必然是对手做的，那么就是每局开始的时候
                flagOpDecisionPt=1
    else:#没有动作则无需考虑
        flagOpDecisionPt=-1

    data['LastAction']=tmp_lastAction
    #data['LastMyAction']=tmp_lastMyAction
    data['flagFirstAction']=flagFirstAction
    data['flagDecisionPt']=flagDecisionPt
    data['flagOpDecisionPt']=flagOpDecisionPt

    #玩家信息
    data['players']=[]
    for i in range(room_number):
        data['players'].append({"position":i,"money_bet":money_beted_now[i],"money_left":params_stack-money_beted_now[i],'total_money':params_stack})
    data['roundbet']=money_beted_rnd

    #legal_action
    legalact=[]
    #当前最小的加注额应等于大盲注+对手的加注额度(对手在平基础上增加的额度)
    # 当money_beted_now[actpos]，money_beted_now[1-actpos]不相等时，必然还在一个轮次内
    # 那么对手的加注额必然是：abs(money_beted_now[actpos]-money_beted_now[1-actpos])
    # 当money_beted_now[actpos]，money_beted_now[1-actpos]相等时，要么新一轮开始或者一轮前面对手c
    # 那么对手的加注额为0也等于abs(money_beted_now[actpos]-money_beted_now[1-actpos])
    # 所以最小加注额如下是对的：
    raisetorangemin=abs(money_beted_now[actpos]-money_beted_now[1-actpos])+params_bb
    
    if money_beted_now[actpos]==money_beted_now[1-actpos]:#这里是典型的基于2人的考虑的处理
        legalact.append("check")
        #通常相等的时候raise是没有多次的
        legalact.append("raise")
    else:
        legalact.append("call")
        legalact.append("fold")
        if raisetimes<params_maxr:
            legalact.append("raise")
    data["legal_actions"]=legalact
    if "raise" in legalact:
        raisetorangemin+=max(money_beted_now)
        if raisetorangemin>params_stack:
            raisetorangemin=params_stack

    #注意acpc的服务器的raise的额度可以自动调整的，默认是raiseto，若这个值过小，则会自动调整调整到call后加上该值 
    #所以提示的额度最好使用加注至(即raiseto)，分两种，一种是而且是全局的"raise_to_range"，
    #另一种是像自动化所那样使用一轮的raiseto，使用"raise_range"。
    if "raise" in legalact:#raise的范围是一轮中的值
        if street ==1:
            #acpc的全局raiseto
            data["raise_to_range"]=[raisetorangemin,params_stack]
            #cisia中的当前轮的raiseto
            data["raise_range"]=[raisetorangemin,params_stack]
        else:
            #acpc的全局raiseto
            data["raise_to_range"]=[raisetorangemin,params_stack]
            #cisia中的当前轮的raiseto。因为从第二轮开始street=2，要减去第一轮的下注，第一轮在money_beted_rnd列表中位置是0。
            data["raise_range"]=[raisetorangemin-money_beted_rnd[street-2],params_stack-money_beted_rnd[street-2]]
    
    #info,action_position
    data['info']='state'
    if lastaction=='f' or (hand_p1 and hand_p2) :
        actpos=2 #为了给出action_position=-1，所以设置为2
        data["info"]="result"
    data['position']=position

    #print("actions_all=",actions_all,len(actions_all))
    #print("actions=",actions,len(actions))
    #print("street=",street)
    #if street==2: print("str=",street,actions[street-1],len(actions[street-1]))
    #if street==3: print("str=",street,actions[street-1],len(actions[street-1]))
    #if street==4: print("str=",street,actions[street-1],len(actions[street-1]))
    if len(actions_all)==0:#当全局没有任何动作时是小盲
        data["action_position"]=0
    elif street==2 and len(actions[1])==0:#当flop没有任何动作时是大盲
        data["action_position"]=1
    elif street==3 and len(actions[2])==0:#当turn没有任何动作时是大盲
        data["action_position"]=1
    elif street==4 and len(actions[3])==0:#当river全局没有任何动作时是大盲
        data["action_position"]=1
    else:#其它情况下，当前需要动作的玩家是最后一个动作的玩家的对手
        data["action_position"]=1-actpos 


    #一局的结果可以直接算出来
    '''
    win_money=[0,0]
    if hand_p1 and hand_p2:
        rk1=gethandrank(data['player_card'][0],data['public_card'])
        rk2=gethandrank(data['player_card'][1],data['public_card'])
        if rk1>rk2:
            win_money=[money_beted_now[1],-money_beted_now[1]]
        elif rk1==rk2:
            win_money=[0,0]
        else:
            win_money=[-money_beted_now[0],money_beted_now[0]]
    if lastaction=='f':
        win_money=[0,0]
        foldpos=actions_all[-1]["position"]
        win_money[foldpos]=-money_beted_now[foldpos]
        win_money[1-foldpos]=money_beted_now[foldpos]
    '''

    if data["info"]=="result":
        for i in range(room_number):
            data['players'][i]["win_money"]=win_money[i]
    
    data['players'][position]["name"]=name
    data['players'][1-position]["name"]=opname

    return data



def testACPCtrans():

    str1='STATE:13:f:AdJc|6h3c:50|-50:PokerCNN_2pn_2017|PokerBot5_2pn_2017'
    print(ACPClogmsgTodata(str1,'PokerBot5_2pn_2017',True))
    str2='STATE:14:cr300c/r700f:JcAd|Js3c/8cAhKh:300|-300:PokerBot5_2pn_2017|PokerCNN_2pn_2017'
    print(ACPClogmsgTodata(str2,'PokerBot5_2pn_2017',True))


# ACPCserver平台给出的结果文件的处理
recAllresults=[]
#第三个参数flagsinglefile是一个标记处理单文件的标记，若为true，收集的数据仅是当前给定文件的
#若非flase那么是集合多个文件的。
def logfiledealing(filename,myname,flagsinglefile=True,bigblindfirst=False,flagfoldseen=True):
    global recAllresults
    
    if flagsinglefile:
        recAllresults=[]

    #打开文件，读取信息
    try:
        fIn = open(filename, 'r', encoding="utf8")
        resdata=fIn.readlines()
        fIn.close()
    except IOError:
        print("ERROR: Input file '" + filename +
                "' doesn't exist or is not readable")
        sys.exit(-1)

    lino=0
    for r in resdata:
        lino+=1
        if (r.count("STATE")>0):
            data=ACPClogmsgTodata(r.strip(),myname,bigblindfirst,flagfoldseen)
            #print('data=',data)
            recAllresults.append(data)
    return None



#显示比赛中双方5张牌是的分布的均匀性
def showfivecardsdist():

    mywrlst=[]
    opwrlst=[]
    mytplst=[]
    optplst=[]
    myhdlst=[]
    ophdlst=[]
    i=0
    for datares in recAllresults:
        hand1=datares['private_card']
        board1=datares['public_card']
        hand=[cardint[x] for x in hand1]
        board=[cardint[x] for x in board1[:3]]
        idx=datares['ophandidx']
        ophand1=[intcard[x] for x in IdxtoHand[idx]]
        ophand=[x for x in IdxtoHand[idx]]
        #wr1=getwinrate(2,hand1,board1[:3])
        #wr2=getwinrate(2,ophand1,board1[:3])
        wr1=get_winrate(2,3,hand[0],hand[1],board[0],board[1],board[2],0,0)
        wr2=get_winrate(2,3,ophand[0],ophand[1],board[0],board[1],board[2],0,0)
        id1=getCombinationId(hand+board,1,52)
        id2=getCombinationId(ophand+board,1,52)
        tp1=gethandrank(hand1,board1[:3])
        tp2=gethandrank(ophand1,board1[:3])
        mywrlst.append(wr1)
        opwrlst.append(wr2)
        mytplst.append(tp1)
        optplst.append(tp2)
        myhdlst.append(id1)
        ophdlst.append(id2)
        i+=1
        print('card:',hand1,board1,ophand1)
        print('i={},wr1={},wr2={},id1={},id2={},tp1={},tp2={}'.format(i,wr1,wr2,id1,id2,tp1,tp2))

    mywrlst1=sorted(mywrlst)
    opwrlst1=sorted(opwrlst)
    mytplst1=sorted(mytplst)
    optplst1=sorted(optplst)
    myhdlst1=sorted(myhdlst)
    ophdlst1=sorted(ophdlst)

    plt.figure()
    plt.hist(mywrlst1,bins=100,density=True,cumulative=False,rwidth=0.95,histtype='bar')
    plt.title('Wr')

    plt.figure()
    plt.hist(opwrlst1,bins=100,density=True,cumulative=False,rwidth=0.95,histtype='bar')
    plt.title('Wr op')

    plt.figure()
    plt.hist(mytplst1,bins=100,density=True,cumulative=False,rwidth=0.95,histtype='bar')
    plt.title('Type')
    
    plt.figure()
    plt.hist(optplst1,bins=100,density=True,cumulative=False,rwidth=0.95,histtype='bar')
    plt.title('Type op')
    
    plt.figure()
    plt.hist(myhdlst1,bins=100,density=True,cumulative=False,rwidth=0.95,histtype='bar')
    plt.title('Id')
    
    plt.figure()
    plt.hist(ophdlst1,bins=100,density=True,cumulative=False,rwidth=0.95,histtype='bar')
    plt.title('Id op')

    plt.show()

    return None
    

#用于处理手牌预测信息
#针对赢率分界的计算
def handpreddealing(filename):
    global rechandPredicted

    #记录从文件中读取的数据
    rechandPredicted={}

    #打开文件，读取信息
    try:
        fIn = open(filename, 'r', encoding="utf8")
        resdata=fIn.readlines()
        fIn.close()
    except IOError:
        print("ERROR: Input file '" + filename +
                "' doesn't exist or is not readable")
        sys.exit(-1)

    lino=0
    for r in resdata:
        lino+=1
        if (r.count('HANDPRED:')>0):
            #print('r=',r)
            m1=re.search("HANDPRED.*?:(.*?):.*?:.*?:.*\[(.*?)\]",r.strip())
            #print('m1=',m1)
            rechandid=int(m1.group(1))
            rechand=[int(x) for x in m1.group(2).split(',')]
            rechandPredicted[rechandid]=rechand

    #print('rechandPredicted=',rechandPredicted)
    
    predhandfigx=[]
    predhandfigy=[] #是否在范围内的信息，若再则为1，若不在则为0
    predhandfigy1=[] #期望赢率
    predhandfigy2=[] #考虑手牌的赢率
    predhandfigy3=[] #在已知牌后的赢率，要么是1要么是0
    for datares in recAllresults:
        rechandid=datares['hand_id']
        print('\nrechandid=',rechandid)
        idx=datares['ophandidx']
        print('ophand idx=',idx)
        hand1=datares['private_card']
        board1=datares['public_card']
        hand=[cardint[x] for x in hand1]
        board=[cardint[x] for x in board1]
        ophand=[intcard[x] for x in IdxtoHand[idx]]
        print('hand=',hand,'ophand=',IdxtoHand[idx])
        print('hand=',[intcard[x] for x in hand],'ophand=',ophand)
        print('board=',board1)
        print('best cards=',[intcard[x] for x in getbestfive(hand1,board1)],[intcard[x] for x in getbestfive(ophand,board1)])
        print('exp wr=',getwinrate(2,hand1,board1),getwinrate(2,ophand,board1))

        if rechandid in rechandPredicted:
            predhandfigx.append(rechandid)
            
            if idx in rechandPredicted[rechandid]:
                predhandfigy.append(1)
            else:
                predhandfigy.append(0)
                print('Not in range hand id=',rechandid)
                #anykey=input()

            nboard=len(board)
            if nboard<5:
                board+=([0]*(5-len(board)))
            wr1=getwinrate(2,hand1,board1)
            avgWr=0.0
            for id in rechandPredicted[rechandid]:
                avgWr+=get_winrate2p2s(hand[0],hand[1],IdxtoHand[id][0],IdxtoHand[id][1],board[0],board[1],board[2],board[3],board[4])
                if hand[0]==IdxtoHand[id][0] or hand[0]==IdxtoHand[id][1] or hand[1]==IdxtoHand[id][0] or hand[1]==IdxtoHand[id][1]:
                    print('error: same card in two sides')
                    #anykey=input()
            wr2=avgWr/len(rechandPredicted[rechandid])
            
            if nboard==5:
                hdrk1=get_handrank(hand[0],hand[1],board[0],board[1],board[2],board[3],board[4])
                #print('IdxtoHand[idx]=',IdxtoHand[idx],IdxtoHand[idx][0],IdxtoHand[idx][1])
                hdrk2=get_handrank(IdxtoHand[idx][0],IdxtoHand[idx][1],board[0],board[1],board[2],board[3],board[4])
                if hdrk1>hdrk2:
                    wr3=1.0
                elif hdrk1==hdrk2:
                    wr3=0.5
                else:
                    wr3=0.0
            else:
                wr3=get_winrate2p2s(hand[0],hand[1],IdxtoHand[idx][0],IdxtoHand[idx][1],board[0],board[1],board[2],board[3],board[4])
            predhandfigy1.append(wr1)
            predhandfigy2.append(wr2)
            predhandfigy3.append(wr3)
            print('{:.3f} {:.3f} {:.3f}'.format(wr1,wr2,wr3))
    
    for i in range(len(predhandfigy1)):
        print('{:.3f} {:.3f} {:.3f}'.format(predhandfigy1[i],predhandfigy2[i],predhandfigy3[i]))


    #平均误差-到当前局的
    dwr1=0.0
    dwr2=0.0
    e1=[]
    e2=[]
    rpds=[]
    for i in range(len(predhandfigy1)):
        dwr1+=abs(predhandfigy1[i]-predhandfigy3[i])
        dwr2+=abs(predhandfigy2[i]-predhandfigy3[i])
        e1.append(dwr1/(i+1))
        e2.append(dwr2/(i+1))
        rpds.append(sum(predhandfigy[:i+1])/(i+1))
    print('average error:',e1[-1],e2[-1])



    plt.figure()
    plt.plot(predhandfigx,predhandfigy,label='1 respresent In')
    plt.title('Hand in predited range')
    plt.xlabel('game')
    plt.ylabel('In range')
    plt.ylim(-0.1,1.2)
    plt.legend()

    plt.figure()
    plt.plot(predhandfigx,rpds,label='ratio till cur game')
    plt.title('ratio of true prediction')
    plt.xlabel('game')
    plt.ylabel('ratio')
    plt.legend()

    plt.figure()
    plt.plot(predhandfigx,e1,label='wr expected')
    plt.plot(predhandfigx,e2,label='wr predhand')
    plt.title("average error between wr_evaluated and wr_known op\'s hand")
    plt.xlabel('game')
    plt.ylabel('average error of wr')
    plt.legend()

    
    plt.figure()
    plt.plot(predhandfigx,predhandfigy1,label='wr expected')
    plt.plot(predhandfigx,predhandfigy3,label='wr knownhand')
    plt.title('wr comparison')
    plt.xlabel('game')
    plt.ylabel('wr')
    plt.ylim(-0.1,1.2)
    plt.legend()

    plt.figure()
    plt.plot(predhandfigx,predhandfigy2,label='wr predhand')
    plt.plot(predhandfigx,predhandfigy3,label='wr knownhand')
    plt.title('wr comparison')
    plt.xlabel('game')
    plt.ylabel('wr')
    plt.ylim(-0.1,1.2)
    plt.legend()

    plt.figure()
    plt.plot(predhandfigx,predhandfigy1,label='wr expected')
    plt.plot(predhandfigx,predhandfigy2,label='wr predhand')
    plt.title('wr comparison')
    plt.xlabel('game')
    plt.ylabel('wr')
    plt.ylim(-0.1,1.2)
    plt.legend()

    plt.show()

    return None



#用于处理手牌预测信息
#用于贝叶斯估计的计算
def handpreddealing2(filename,myname):
    global rechandPreddist

    #记录从文件中读取的数据
    rechandPreddist={}

    #打开文件，读取信息
    try:
        fIn = open(filename, 'r', encoding="utf8")
        resdata=fIn.readlines()
        fIn.close()
    except IOError:
        print("ERROR: Input file '" + filename +
                "' doesn't exist or is not readable")
        sys.exit(-1)

    lino=0
    flag_datauser=True
    for r in resdata:
        lino+=1
        if (r.count('HANDPROB:')>0):
            #print('r=',r)
            m1=re.search("HANDPROB.*?:(.*?):.*?:.*?:.*\[(.*?)\]",r.strip())
            #print('m1=',m1)
            rechandid=int(m1.group(1))
            rechand=[float(x) for x in m1.group(2).split(',')]
            rechandPreddist[rechandid]=rechand
        if (r.count("result")>1) and flag_datauser:
            #print("r=",r)
            r1=r.replace("result info:","").strip()
            r2=r1.replace("'","\"")
            #print(lino,":",r2)
            r3=re.search("(.*\})",r2).group(0)
            #print(lino,":",r3)
            r4=r3.replace("False","\"False\"")
            r5=r4.replace("True","\"True\"")
            datauser=json.loads(r5)
            flag_datauser=False
            #connect: {'info': 'connect', 'name': 'ph', 'room_number': 2, 'bots': ['LA'], 'game_number': 3000}
        if (r.count("connect")>1) and flag_datauser:
            r1=r.replace("connect:","").strip()
            r2=r1.replace("'","\"")
            r3=re.search("(.*\})",r2).group(0)
            gameinfo=json.loads(r3)

    #记录所有玩家的名称信息
    handid=gameinfo['game_number']
    opname=''
    allbots=""
    for player in datauser['players']:
        allbots+=player['name']
        if player['name']!=myname:
            opname=player['name']


    #print('rechandPredicted=',rechandPredicted)
    #input()
    
    predhandfigx=[]
    predhandfigy=[] #是否在范围内的信息，若再则为1，若不在则为0
    predhandfigy1=[] #期望赢率
    predhandfigy2=[] #考虑手牌的赢率
    predhandfigy3=[] #在已知牌后的赢率，要么是1要么是0
    predhandfigy4=[] #实际手牌的概率占所有手牌的总概率的比例
    predhandfigy5=[] #不是用手牌估计时，对手手牌的概率，通常是1/binom(50,2)，1/binom(47,2)，等与轮次相关
    saveidx=0
    savehandid=0
    for datares in recAllresults:
        rechandid=datares['hand_id']
        print('\nrechandid=',rechandid)
        idx=datares['ophandidx']
        print('ophand idx=',idx)
        hand1=datares['private_card']
        board1=datares['public_card']
        hand=[cardint[x] for x in hand1]
        board=[cardint[x] for x in board1]
        ophand=[intcard[x] for x in IdxtoHand[idx]]
        print('hand=',hand,'ophand=',IdxtoHand[idx])
        print('hand=',[intcard[x] for x in hand],'ophand=',ophand)
        print('board=',board1)
        print('best cards=',[intcard[x] for x in getbestfive(hand1,board1)],[intcard[x] for x in getbestfive(ophand,board1)])
        print('exp wr=',getwinrate(2,hand1,board1),getwinrate(2,ophand,board1))

        #print('datares=',datares)
        #anykey=input()

        

        #绘图数据准备
        if rechandid in rechandPreddist:
            predhandfigx.append(rechandid)
            saveidx=idx
            savehandid=rechandid 
            
            if rechandPreddist[rechandid][idx]>0:
                predhandfigy.append(1)
                predhandfigy4.append(rechandPreddist[rechandid][idx]/sum(rechandPreddist[rechandid]))
            else:
                predhandfigy.append(0)
                predhandfigy4.append(0)
                print('Not in range hand id=',rechandid)
                #anykey=input()

            if datares['street']==1:
                predhandfigy5.append(0.000816327) #N[1/Binomial[50, 2]]
            elif datares['street']==2:
                predhandfigy5.append(0.000925069) #N[1/Binomial[47, 2]]
            elif datares['street']==3:
                predhandfigy5.append(0.000966184) #NN[1/Binomial[46, 2]]
            else:
                predhandfigy5.append(0.0010101)   #N[1/Binomial[45, 2]]
                  
            
            wr1=getwinrate(2,hand1,board1) #期望赢率

            nboard=len(board)
            if nboard<5:
                board+=([0]*(5-len(board)))
            wr2=getwr2srangedist(hand,board,rechandPreddist[rechandid]) #考虑对手手牌分布的赢率
            
            if nboard==5:
                hdrk1=get_handrank(hand[0],hand[1],board[0],board[1],board[2],board[3],board[4])
                #print('IdxtoHand[idx]=',IdxtoHand[idx],IdxtoHand[idx][0],IdxtoHand[idx][1])
                hdrk2=get_handrank(IdxtoHand[idx][0],IdxtoHand[idx][1],board[0],board[1],board[2],board[3],board[4])
                if hdrk1>hdrk2:
                    wr3=1.0
                elif hdrk1==hdrk2:
                    wr3=0.5
                else:
                    wr3=0.0
            else:
                wr3=get_winrate2p2s(hand[0],hand[1],IdxtoHand[idx][0],IdxtoHand[idx][1],board[0],board[1],board[2],board[3],board[4])

            predhandfigy1.append(wr1)
            predhandfigy2.append(wr2)
            predhandfigy3.append(wr3)
            print('{:.3f} {:.3f} {:.3f}'.format(wr1,wr2,wr3))
    
    for i in range(len(predhandfigy1)):
        print('{:.3f} {:.3f} {:.3f}'.format(predhandfigy1[i],predhandfigy2[i],predhandfigy3[i]))


    #期望赢率和根据手牌的赢率到与实际赢率的比较。
    #平均误差-到当前局的
    dwr1=0.0
    dwr2=0.0
    e1=[]
    e2=[]
    rpds=[]  #到当前局的是否在预测范围内的平均值
    phcurg=[] #到当前局的估计手牌概率的平均值
    phcure=[] #到当前局的非手牌估计下的手牌概率的平均值
    for i in range(len(predhandfigy1)):
        dwr1+=abs(predhandfigy1[i]-predhandfigy3[i])
        dwr2+=abs(predhandfigy2[i]-predhandfigy3[i])
        e1.append(dwr1/(i+1))
        e2.append(dwr2/(i+1))
        rpds.append(sum(predhandfigy[:i+1])/(i+1))
        phcurg.append(sum(predhandfigy4[:i+1])/(i+1))
        phcure.append(sum(predhandfigy5[:i+1])/(i+1))
    print('average residual:',e1[-1],e2[-1])

    plt.figure()
    plt.plot(predhandfigx,rpds,label=opname)
    plt.title("Ratio of True Prediction for {}".format(opname))
    plt.xlabel('game')
    plt.ylabel('$P_{tp}$')
    plt.legend()
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-ptp.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-ptp.pdf")

    plt.figure()
    plt.plot(predhandfigx,phcurg,label=opname+' $P_{hp}$')
    plt.plot(predhandfigx,phcure,label=opname+' $P_{ep}$')
    plt.title("Average of Predicted Hand's probability for {}".format(opname))
    plt.xlabel('game')
    plt.ylabel('$M_{hp}$')
    plt.legend()
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-mhp.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-mhp.pdf")
    print('phcurg/phcure=',phcurg[-1],phcure[-1],phcurg[-1]/phcure[-1]*100)

    plt.figure()
    plt.plot(predhandfigx,e1,label='$Wr_{ep}$')
    plt.plot(predhandfigx,e2,label='$Wr_{hp}$')
    plt.title("Average Error of Wr for {}".format(opname))
    plt.xlabel('game')
    plt.ylabel('$E_{wr}$')
    plt.legend()
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-ewr.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-ewr.pdf")
    print('(e2-e1)/e1=',e2[-1],e1[-1],(e2[-1]-e1[-1])/e1[-1]*100)

    if 0:
        plt.figure()
        plt.plot(predhandfigx,predhandfigy,label='1 respresent In')
        plt.title('Hand in predited range')
        plt.xlabel('game')
        plt.ylabel('In range')
        plt.ylim(-0.1,1.2)
        plt.legend()

        plt.figure()
        plt.plot(predhandfigx,predhandfigy1,label='wr expected')
        plt.plot(predhandfigx,predhandfigy3,label='wr knownhand')
        plt.title('wr comparison')
        plt.xlabel('game')
        plt.ylabel('wr')
        plt.ylim(-0.1,1.2)
        plt.legend()

        plt.figure()
        plt.plot(predhandfigx,predhandfigy2,label='wr predhand')
        plt.plot(predhandfigx,predhandfigy3,label='wr knownhand')
        plt.title('wr comparison')
        plt.xlabel('game')
        plt.ylabel('wr')
        plt.ylim(-0.1,1.2)
        plt.legend()

        plt.figure()
        plt.plot(predhandfigx,predhandfigy1,label='wr expected')
        plt.plot(predhandfigx,predhandfigy2,label='wr predhand')
        plt.title('wr comparison')
        plt.xlabel('game')
        plt.ylabel('wr')
        plt.ylim(-0.1,1.2)
        plt.legend()

    #plt.show()
    print('idx=',saveidx,'rechandid=',savehandid)
    showhandpred(saveidx,rechandPreddist[savehandid],allbots,handid,opname)

    handres=copy.deepcopy(predhandfigx)
    ptpres=copy.deepcopy(rpds)
    ewrres=copy.deepcopy(e1)
    mphres=copy.deepcopy(phcurg)

    return handres,ptpres,ewrres,mphres,opname


#将手牌预测结果用一维和二维图显示
#输入：对手的手牌序号：idx; 预测的对手手牌的分布：handdist。
def showhandpred(idx,handdist,allbots,handid,opname):

    #一维绘图,用分布表示
    handids=np.linspace(0,1325,1326)
    plt.figure()
    plt.plot(handids,handdist,label='pdf')
    plt.scatter(idx,handdist[idx],label='predicted hand')
    plt.xlabel('handid')
    plt.ylabel('hand probability')
    plt.legend()
    plt.grid()
    plt.title('Predicted Hand Probability for {}'.format(opname))
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-pdist.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-pdist.pdf")


    #二维显示
    #169种牌，注意第二维大于第一维表示同色牌
    #probdist的内容是：
    # 22  23s ... 2As
    # 32o 33  ... 2As
    # ..  ..  ... ...
    # A2o 23s ... AA
    probdist=np.zeros((13,13))
    for i in range(1326):
        cd1=IdxtoHand[i][0]
        cd2=IdxtoHand[i][1]
        k1,k2=Cardbg2sm(cd1,cd2)
        probdist[k1,k2]+=handdist[i]

    #因为显示的形式与存储的数组的形式不一样所以需要做转换
    #probdisp的内容变为
    # AA  KAs ... 2As
    # AKo KK  ... 2Ks
    # ..  ..  ... ...
    # A2o k2o ... 22
    probdisp=np.zeros((13,13))
    for i in range(13):
        for j in range(13):
            probdisp[i,j]=probdist[12-j,12-i]


    #标记与上述矩阵无关，直接根据目标标记要求使用位置标记
    #要特别注意text的位置是横坐标的方向就是矩阵的列坐标方向
    #要特别注意text的位置是纵坐标的方向就是矩阵的行坐标方向
    #但这里是以矩阵的形式记录的
    strbase='AKQJT98765432'
    strdisp=[]
    for i in range(13):
        strdisp.append([])
        for j in range(13):
            #print('i=',i,' j=',j,strdisp[i])
            if (j>i):
                strdisp[i].append(strbase[j]+strbase[i]+'s')
            elif j==i:
                strdisp[i].append(strbase[i]+strbase[j])
            else:
                strdisp[i].append(strbase[j]+strbase[i]+'o')

    plt.figure(figsize=(8.8,6.6))
    plt.imshow(probdisp,interpolation = None,cmap='jet') #viridis
    #写第一行的时候，i=0，j在变，但在图上的位置是行坐标在变，所以j应该在横坐标位置
    #刚刚就是把j写在前面作为x的坐标
    for i in range(13):
        for j in range(13):
            plt.text(j-0.25,i+0.1,strdisp[i][j])
    plt.colorbar()
    #plt.xticks([])
    #plt.yticks([])
    for x in np.arange(0.5, 12.5, 1):
        plt.axvline(x,color='gray')
    for y in np.arange(0.5, 12.5, 1):
        plt.axhline(y,color='gray')
    cd1=IdxtoHand[idx][0]
    cd2=IdxtoHand[idx][1]
    k1,k2=Cardbg2sm(cd1,cd2)
    handstr=intcard[cd1]+intcard[cd2]
    print('handstr=',handstr)
    #注意text的位置不是正常直角坐标系，而是以图中ticks标记的坐标
    #注意这里的k1，k2是probdist中的索引:(k1,k2)
    #所以要转换成probdist的索引:(12-k2,12-k1)
    #然后又要转成text写的坐标:(12-k1,12-k2)
    plt.text(12-k1+0.25,12-k2-0.25,handstr,fontsize=8,color='white',backgroundcolor='green')
    plt.title('Contour of Predicted Hand Probability for {}'.format(opname))
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-pcountour.svg")
    plt.savefig("fig-"+allbots+"-"+str(handid)+"-"+opname+"-pcountour.pdf")
    #plt.show()


    return None


#比较取不同ns时的结果，即手牌估计开始的局数对于手牌估计的影响。
def compareDiffns(filelists,nslist,myname):

    handlist=[]
    ptplist=[]
    ewrlist=[]
    mphlist=[]
    opname=''
    for filelist in filelists:
        logfiledealing(filelist[0],myname)
        handres,ptp,ewr,mph,opname=handpreddealing2(filelist[1],myname)
        ptplist.append(ptp)
        ewrlist.append(ewr)
        mphlist.append(mph)
        handlist.append(handres)

    for i in range(len(ewrlist)):
        ewr=ewrlist[i]
        mph=mphlist[i]
        print('i=',i,'len(ewr)=',len(ewr),' len(mph)=',len(mph))

    plt.figure()
    i=0
    for ptp in ptplist:
        plt.plot(handlist[i],ptp,label=opname+' ns={}'.format(nslist[i]))
        i+=1
    plt.title("Comparison of Ratio of True Prediction for {}".format(opname))
    plt.xlabel('game')
    plt.ylabel('$P_{tp}$')
    plt.legend()
    plt.savefig("fig-compare-ptp-var-ns"+"-"+opname+".svg")
    plt.savefig("fig-compare-ptp-var-ns"+"-"+opname+".pdf")

    plt.figure()
    i=0
    for mph in mphlist:
        plt.plot(handlist[i],mph,label=opname+' ns={}'.format(nslist[i]))
        i+=1
    plt.title("Comparison of Predicted Hand's probability for {}".format(opname))
    plt.xlabel('game')
    plt.ylabel('$M_{hp}$')
    plt.legend()
    plt.savefig("fig-compare-mhp-var-ns"+"-"+opname+".svg")
    plt.savefig("fig-compare-mhp-var-ns"+"-"+opname+".pdf")

    plt.figure()
    i=0
    for ewr in ewrlist:
        plt.plot(handlist[i],ewr,label=opname+' ns={}'.format(nslist[i]))
        i+=1
    plt.title("Comparison of Average Error of Wr for {}".format(opname))
    plt.xlabel('game')
    plt.ylabel('$E_{Wr}$')
    plt.legend()
    plt.savefig("fig-compare-ewr-var-ns"+"-"+opname+".svg")
    plt.savefig("fig-compare-ewr-var-ns"+"-"+opname+".pdf")

    plt.show()
    return None






def testpltmat():
    '''
    a=np.array([[0.8,1.5],[0.5,1.2]])
    b=np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            b[i,j]=a[1-i,1-j]
    b=b.T
    plt.figure()
    plt.imshow(a)

    plt.figure()
    plt.imshow(b)
    plt.show()
    '''

    #打开文件，读取信息
    try:
        fIn = open('Hand2cards.dat', 'r', encoding="utf8")
        resdata=fIn.readlines()
        fIn.close()
    except IOError:
        print("ERROR: Input file '" + filename +
                "' doesn't exist or is not readable")
        sys.exit(-1)

    probdist=np.zeros((13,13))
    for res in resdata:
        reslist=res.split(',')
        k1=int(reslist[0])
        k2=int(reslist[1])
        probdist[k1,k2]=float(reslist[2])+float(reslist[3])*0.5


    probdisp=np.zeros((13,13))
    for i in range(13):
        for j in range(13):
            probdisp[i,j]=probdist[12-j,12-i]
    

    #标记与上述矩阵无关，直接根据目标标记要求使用位置标记
    #要特别注意text的位置是横坐标的方向就是矩阵的列坐标方向
    #要特别注意text的位置是纵坐标的方向就是矩阵的行坐标方向
    #但这里是以矩阵的形式记录的
    strbase='AKQJT98765432'
    strdisp=[]
    for i in range(13):
        strdisp.append([])
        for j in range(13):
            #print('i=',i,' j=',j,strdisp[i])
            if (j>i):
                strdisp[i].append(strbase[i]+strbase[j]+'s')
            elif j==i:
                strdisp[i].append(strbase[i]+strbase[j])
            else:
                strdisp[i].append(strbase[i]+strbase[j]+'o')

    plt.figure()
    plt.imshow(probdist,interpolation = 'none')
    

    plt.figure()
    plt.imshow(probdisp,interpolation = None,cmap='jet') #viridis
    #写第一行的时候，i=0，j在变，但在图上的位置是行坐标在变，所以j应该在横坐标位置
    #刚刚就是把j写在前面作为x的坐标
    for i in range(10):
        for j in range(5):
            plt.text(j-0.25,i+0.1,strdisp[i][j])
    plt.colorbar()
    #plt.xticks(np.arange(0.5, 12.5, 1))
    #plt.yticks(np.arange(0.5, 12.5, 1))
    #plt.xticks([])
    #plt.yticks([])
    for x in np.arange(0.5, 12.5, 1):
        plt.axvline(x,color='gray')
    for y in np.arange(0.5, 12.5, 1):
        plt.axhline(y,color='gray')
    
    cd1=52
    cd2=2
    k1,k2=Cardbg2sm(cd1,cd2)
    handstr=intcard[cd1]+intcard[cd2]
    print('handstr=',handstr,k1,k2)
    #注意text的位置不是正常直角坐标系，而是以图中ticks标记的坐标
    #注意这里的k1，k2是probdist中的索引:(k1,k2)
    #所以要转换成probdist的索引:(12-k2,12-k1)
    #然后又要转成text写的坐标:(12-k1,12-k2)
    plt.text(12-k1+0.25,12-k2-0.25,handstr,fontsize=6,color='red',backgroundcolor='yellow')

    plt.show()
    return None


def testACPClogmsgTodata():

    msg="STATE:5:cr1190c/cc/cc/cr4004f:3cTd|Ah4c/Kh6d2h/7s/3d:1190|-1190:LA|ph"
    myname='ph'
    print('msg=',msg)
    data=ACPClogmsgTodata(msg,myname)
    print('data=',data)


def testEapwithsigma():
    #MS
    opname1='MS'
    Eapdata=np.array([[0.08584260464031661,1326,0.5],
        [0.08147835153662934,2652,1.0],
        [0.07443046876271149,5304,2.0],
        [0.0643377781981896,10608,4.0],
        [0.06517372661358066,21216,8.0],
        [0.06237583445993274,42432,16.0],
        [0.062279483024799395,84864,32.0],
        [0.06411389223873737,127296,48.0],
        [0.06333769345168944,190944,72.0]])
    Eapdata1=np.zeros_like(Eapdata)
    Eapdata1[:,0]=Eapdata[:,2]
    Eapdata1[:,2]=Eapdata[:,0]
    Eapdata1[:,1]=Eapdata[:,1]

    #TARD
    opname2='TARD'
    Eapdata2=np.array([[0.09713460730242938,1326,0.5],
        [0.09911309329395974,2652,1.0],
        [0.07114819143331783,5304,2.0],
        [0.07042212543081507,10608,4.0],
        [0.0639938740768592,21216,8.0],
        [0.06382108561240912,42432,16.0],
        [0.06277656827377612,84864,32.0],
        [0.06317984570610809,127296,48.0],
        [0.06490937253797645,190944,72.0]])
    Eapdata3=np.zeros_like(Eapdata2)
    Eapdata3[:,0]=Eapdata2[:,2]
    Eapdata3[:,2]=Eapdata2[:,0]
    Eapdata3[:,1]=Eapdata2[:,1]

    for i in range(len(Eapdata1)):
        print("{} & {} & {}".format(Eapdata1[i,0],Eapdata1[i,1],Eapdata1[i,2]))
    
    plt.figure()
    plt.plot(Eapdata[:,2],Eapdata[:,0],marker="o",label='MS (vs LP)')
    plt.plot(Eapdata2[:,2],Eapdata2[:,0],marker="x",label='TARD (vs ph)')
    plt.xlabel('$\sigma$')
    plt.ylabel('$E_{ap}$')
    plt.grid() #绘制网格
    #plt.xlim(0,1) #x轴绘制范围限制
    #plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("$E_{ap}$ at fps preflop") #调整x轴坐标刻度和标签
    plt.savefig('fig-action-eap-fps-preflop.pdf')
    plt.show()
    return None



#------------------------------------------------------------
#获取动作信息集
#不要用call/raise这样的动作来描述，用bet来描述，就是投入的金额数来描述。
#记录到recAllgameHistory中的信息为：
#playername,streetflag,public_card,recOpLastbet,ourbet,player_card
#记录到recAllgameHistoryS中的信息为：
#playername,streetflag,public_card,recOpLastact,ouract,player_card
def get_actioninfosetall1(data,nplayer):
    global recAllagentsname
    global recAllgameHistory
    global recAllgameHistoryS
    global recAllgameHistoryF
    global recAllgameHistorySF #一个赢率只记录两个决策点各一次

    #--计算每一轮的commit出来，方便加上raise值的转换
    commitchipres=[] #--记录每一轮结束的commit
    chipsnres=[]     #记录每个人的commit情况
    foldflagres=[]   #记录每个人的fold情况

    for i in range(nplayer):
        chipsnres.append(0)
        foldflagres.append(0)

    chipsnres[0]=50
    chipsnres[1]=100

    streetflag=0
    recordsn=[0,0,0,0] #--用于记录各个轮次玩家的行动数量
    recOpLastbet="none"
    recOpLastbetS="none"

    #recAllagentsstat[name]={"chand":0,"cpfvplay":0,"cpfvraise":0,"cafraise":0,"cafcall":0,
    #   "vpip":0.0,'pfr':0.0,'af':0.0,'pfrdvpip':0.0} 
    cpfvplay={}
    cpfvraise={}
    cafraise={}
    cafcall={}
    cfhandpre={}

    cafcck={}
    cafcckck={}
    cafcckrs={}
    cafccl={}
    cafcclfd={}
    cafcclcl={}
    cafcclrs={}

    cpfccl={}
    cpfcclfd={}
    cpfcclrs={}
    cpfcclcl={}
    cpfcck={}
    cpfcckrs={}
    cpfcckck={}

    cpfall={}
    cpfallfd={}
    cpfallcl={}
    cafall={}
    cafallfd={}
    cafallcl={}

    for name in recAllagentsname:
        cpfvplay[name]=False
        cpfvraise[name]=False
        cafraise[name]=0
        cafcall[name]=0
        cfhandpre[name]=0

        cafcck[name]=0
        cafcckrs[name]=0
        cafcckck[name]=0
        cafccl[name]=0
        cafcclfd[name]=0
        cafcclcl[name]=0
        cafcclrs[name]=0

        cpfccl[name]=0
        cpfcclfd[name]=0
        cpfcclrs[name]=0
        cpfcclcl[name]=0
        cpfcck[name]=0
        cpfcckrs[name]=0
        cpfcckck[name]=0

        cpfall[name]=0
        cpfallfd[name]=0
        cpfallcl[name]=0
        cafall[name]=0
        cafallfd[name]=0
        cafallcl[name]=0

    for lstaction in data["action_history"]:#--#一局的所有动作列表

        #print('len(lstaction)',len(lstaction))
        tmp_preAct='none'
        flg_CKcaseCounted={}
        flg_CLcaseCounted={}
        flg_CKcaseidx={}  #标记每一轮玩家的面临决策点的第几次
        flg_CLcaseidx={}
        flg_CKcaseCur=False
        flg_CLcaseCur=False
        for name in recAllagentsname:
            flg_CKcaseCounted[name]=False
            flg_CLcaseCounted[name]=False
            flg_CKcaseidx[name]=0
            flg_CLcaseidx[name]=0

        if (len(lstaction)>0):
            #注意street=1表示preflop
            streetflag=streetflag+1  #根据data["action_history"]中的列表数量确定street即轮次
            tmp_i_cAct=0 #用于统计当前动作在这一轮中是第几个动作
            for dictactround in lstaction:#--#每一轮的动作字典
                playername=data["players"][dictactround['position']]['name']
                playerpos=data["players"][dictactround['position']]['position']
                tmp_i_cAct+=1
                if tmp_i_cAct==1:
                    if streetflag==1:
                        recOpLastbetS="bigbet"
                    else:
                        recOpLastbetS="none"

                #--start注意：这一段是针对两人做的处理
                #当处于大盲位且是第一轮时，若对手直接fold，那么我方就会失去决策的机会，这种情况要去除掉
                #处理时根据第一轮第一个动作进行判断，若第一个动作为fold，那么对手就失去了决策的机会要去掉
                #由于第一个动作必然是小盲注位(0号位)的行动，那么必然是1号位玩家失去机会
                if streetflag==1 and dictactround["action"]=="fold" and tmp_i_cAct==1:  
                        cfhandpre[data["players"][1]['name']]=1
                #--end注意----
                flg_outAction=False
                action=""
                recordsn[streetflag-1]=recordsn[streetflag-1]+1 #--记录这一轮的玩家的动作总数
                #--无论处于哪个位置，动作的意义的一样的
                #--check意味着不加钱，raise意味着本轮从0开始加到多少，call意味着把钱加到其他最大的值
                #--因为动作历史有顺序在里头，因此不会出现问题
                #--但是要记录fold的玩家，计算投注平衡时需要避开fold的玩家
                action1=dictactround["action"]
                if (dictactround["action"]== "call"):
                    if streetflag==1:
                        cpfvplay[playername]=True

                        #--start注意：这一段是针对两人做的处理
                        if (tmp_preAct=='none' or tmp_preAct[0]=='r'):
                            if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                                cpfall[playername]+=1
                                cpfallcl[playername]+=1
                            else:
                                if (not flg_CLcaseCounted[playername]):
                                    cpfccl[playername]+=1 #既可以call，fold，raise
                                    cpfcclcl[playername]+=1
                                    flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                        #if cpfccl[playername]>1:
                        #   print('now call cpfccl=',cpfccl[playername])
                        #   anykey=input()
                        #--end注意----

                    else:
                        cafcall[playername]+=1

                        #--start注意：这一段是针对两人做的处理
                        if tmp_preAct[0]=='r':
                            if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                                cafall[playername]+=1
                                cafallcl[playername]+=1
                            else:
                                if (not flg_CLcaseCounted[playername]):
                                    cafccl[playername]+=1
                                    cafcclcl[playername]+=1
                                    flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                            '''
                            if playername=='TARD':
                                print('street=',streetflag,' preact=',tmp_preAct)
                                print('flg_CLcaseidx',playername,flg_CLcaseidx[playername])
                                print('flg_CLcaseCur',flg_CLcaseCur,flg_CKcaseCur)
                                print('call at data',data)
                                flg_outAction=True
                                anykey=input()
                            '''
                        #--end注意----

                    #--获取记录的chipsnres数组中的最大值
                    chipmax=max(chipsnres)
                    #--设置当前position为该最大值
                    chipsnres[dictactround["position"]]=chipmax
                    action="betto"+str(chipmax)
                elif (dictactround["action"]== "check"):
                    action="betto"+str(chipsnres[dictactround["position"]])

                    #--start注意：这一段是针对两人做的处理
                    if streetflag==1 and (tmp_preAct=='call' or tmp_preAct=='check'):
                        if (not flg_CKcaseCounted[playername]):
                            cpfcck[playername]+=1
                            cpfcckck[playername]+=1
                            flg_CKcaseCounted[playername]=True
                        flg_CKcaseidx[playername]+=1
                        flg_CKcaseCur=True
                        flg_CLcaseCur=False

                    if streetflag>1 and (tmp_preAct=='none' or tmp_preAct=='call' or tmp_preAct=='check'):
                        if (not flg_CKcaseCounted[playername]):
                            cafcck[playername]+=1
                            cafcckck[playername]+=1
                            flg_CKcaseCounted[playername]=True
                        flg_CKcaseidx[playername]+=1
                        flg_CKcaseCur=True
                        flg_CLcaseCur=False
                    #--end注意----

                elif (dictactround["action"]== "fold"):
                    #--将fold信息记录到foldflagres数组中
                    foldflagres[dictactround["position"]]=1
                    action="fold"
                    
                    #--start注意：这一段是针对两人做的处理
                    if streetflag==1 and (tmp_preAct=='none' or tmp_preAct[0]=='r') :
                        if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                            cpfall[playername]+=1
                            cpfallfd[playername]+=1
                        else:
                            if (not flg_CLcaseCounted[playername]):
                                cpfccl[playername]+=1    #既可以call，fold，raise
                                cpfcclfd[playername]+=1
                                flg_CLcaseCounted[playername]=True
                        flg_CLcaseidx[playername]+=1
                        flg_CKcaseCur=False
                        flg_CLcaseCur=True
                        #if cpfccl[playername]>1:
                        #   print('fold cpfccl=',cpfccl[playername])
                        #   anykey=input()
                    if streetflag>1 and tmp_preAct[0]=='r':
                        if recOpLastbetS =='allin': #不用管是不是第一次，因为一局只有一次
                            cafall[playername]+=1
                            cafallfd[playername]+=1
                        else:
                            if (not flg_CLcaseCounted[playername]):
                                cafccl[playername]+=1
                                cafcclfd[playername]+=1
                                flg_CLcaseCounted[playername]=True
                        flg_CLcaseidx[playername]+=1
                        flg_CKcaseCur=False
                        flg_CLcaseCur=True
                    #--end注意----

                else: #-- raise,从第二个字符开始的字符串转换成数字
                    if streetflag==1:
                        cpfvplay[playername]=True
                        cpfvraise[playername]=True

                        #--start注意：这一段是针对两人做的处理
                        if (tmp_preAct=='none' or tmp_preAct[0]=='r'):
                            if (not flg_CLcaseCounted[playername]):
                                cpfccl[playername]+=1 #既可以call，fold，raise
                                cpfcclrs[playername]+=1
                                flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                        elif (tmp_preAct=='call' or tmp_preAct=='check'):
                            if (not flg_CKcaseCounted[playername]):
                                cpfcck[playername]+=1
                                cpfcckrs[playername]+=1
                                flg_CKcaseCounted[playername]=True
                            flg_CKcaseidx[playername]+=1
                            flg_CKcaseCur=True
                            flg_CLcaseCur=False
                        #--end注意----

                    else:
                        cafraise[playername]+=1

                        #--start注意：这一段是针对两人做的处理
                        if tmp_preAct=='none' or tmp_preAct=='call' or tmp_preAct=='check':
                            if (not flg_CKcaseCounted[playername]):
                                cafcck[playername]+=1
                                cafcckrs[playername]+=1
                                flg_CKcaseCounted[playername]=True
                            flg_CKcaseidx[playername]+=1
                            flg_CKcaseCur=True
                            flg_CLcaseCur=False
                        if tmp_preAct[0]=='r':
                            if (not flg_CLcaseCounted[playername]):
                                cafccl[playername]+=1
                                cafcclrs[playername]+=1
                                flg_CLcaseCounted[playername]=True
                            flg_CLcaseidx[playername]+=1
                            flg_CKcaseCur=False
                            flg_CLcaseCur=True
                        #--end注意----

                    
                    raiseamount=int(dictactround["action"][1:])
                    betval=0
                    if(streetflag==1):
                        betval= raiseamount
                        pot=150
                    elif(streetflag==2):
                        betval=commitchipres[0]+raiseamount
                        pot=commitchipres[0]*nplayer
                    elif(streetflag==3):
                        betval=commitchipres[1]+raiseamount
                        pot=commitchipres[1]*nplayer
                    else:
                        betval=commitchipres[2]+raiseamount
                        pot=commitchipres[2]*nplayer
                    chipsnres[dictactround["position"]]=betval
                    action="betto"+str(betval)
                    
                    if raiseamount<=2*pot:
                        action1="r1"
                    else:
                        action1="r2"
                    if betval>=20000:
                        action1="allin"
                    
                    '''
                    if raiseamount<=pot:
                        action1="r1"
                    elif raiseamount<=2*pot:
                        action1="r2"
                    elif raiseamount<=4*pot:
                        action1="r3"
                    else:
                        action1="r4"
                    if betval>=20000:
                        action1="allin"
                    if betval<1000:
                        action1='r1'
                    elif betval<10000:
                        action1='r2'
                    elif betval<20000:
                        action1='r3'
                    '''

                reconeact=[playername,streetflag,data['public_card'],recOpLastbet,action,data['player_card'][playerpos],data['player_card'][1-playerpos]]
                reconeactS=[playername,streetflag,data['public_card'],recOpLastbetS,action1,data['player_card'][playerpos],data['player_card'][1-playerpos]]
                recAllgameHistory.append(reconeact)
                recAllgameHistoryS.append(reconeactS)

                #if flg_outAction:
                #   print('reconeactS',reconeactS)
                #   flg_outAction=False

                #通过分析表明当对手raise变成allin时，我方只能是call，无论决策是raise还是call，所以会造成模糊
                #因此我们去掉对手是allin动作的情况再进行统计。
                if flg_CLcaseidx[playername]==1 and flg_CLcaseCur and recOpLastbetS != "allin":
                    recAllgameHistoryF.append(reconeact)
                    recAllgameHistorySF.append(reconeactS)
                
                if flg_CKcaseidx[playername]==1 and flg_CKcaseCur:
                    recAllgameHistoryF.append(reconeact)
                    recAllgameHistorySF.append(reconeactS)
                
                recOpLastbet=action     #bet to 的形式 +fold
                recOpLastbetS=action1   #check，call，fold，分级的r1，r2，等
                tmp_preAct=dictactround["action"] #原始形式的记录


            #--判断非fold玩家是否投注已经平衡，如果平衡了说明当前轮已经结束了，否则是未结束的
            #--投注平衡的标记，也是当前轮是否结束的标志
            equiflag=True
            equichip=0
            for i in range(nplayer):
                if (foldflagres[i]!=1): #--随便找一个非fold的玩家来记录一个当前的投注额
                    equichip=chipsnres[i]
                    break

            mnotfold=0 #--统计一下没有fold玩家的数量
            for v in foldflagres:
                if(v==0):
                    mnotfold=mnotfold+1

            #--只有当当前轮的动作记录的数量不小于非fold玩家数量时，才有可能达到平衡
            if(recordsn[streetflag-1]>=mnotfold):
                for i in range(nplayer):
                    if (foldflagres[i]!=1 and chipsnres[i]!=equichip):
                        equiflag=False
                        break
            else:
                equiflag=False

            if(equiflag):
                commitchipres.append(equichip)
                #print("street",streetflag, "is finished")
            else:
                #print("street",streetflag, "is not finished")
                pass
    
    return None





def testholecardsdist1():

    nplayer=2
    datausers=recAllresults
    dealgamenumber=0
    if dealgamenumber==0:
        numberbreak=len(datausers)
    else:
        numberbreak=dealgamenumber

    playnames=[ x['name'] for x in datausers[0]['players']]
    cardsdist={}
    for x in playnames:
        cardsdist[x]={'fps':{'wrs':[],'ids':[],'cds':[],'owrs':[],'oids':[],'ocds':[]},
        'fpbr':{'wrs':[],'ids':[],'cds':[],'owrs':[],'oids':[],'ocds':[]}}

    countact1=np.zeros((1326,3)) #fold,call,raise
    countact2=np.zeros((1326,3)) #fold,call,raise
    countactP=np.zeros(1326)
    countactO=np.zeros(1326)

    #p表示第一个玩家
    #o表示第二个玩家
    nameP=playnames[0]
    nameO=playnames[1]

    i=0
    for data in datausers:
        #处理用于统计的动作和其它信息
        #记录到recAllgameHistory和recAllgameHistoryS中
        #get_actioninfosetall1(data,nplayer)

        #posP,posO是第一个和第二个玩家在当前局中的位置
        if data['players'][0]['name']==nameP:
            posP=0
            posO=1
        else:
            posP=1
            posO=0
        
        act1=data['action_history'][0][0]['action']
        if act1 !='fold':
            act2=data['action_history'][0][1]['action']
        
        cardP=data['player_card'][posP]
        cardO=data['player_card'][posO]
        cardidP=int(getCombinationId([cardint[x] for x in cardP], 1,52))
        cardidO=int(getCombinationId([cardint[x] for x in cardO], 1,52))
        wrP=getwinrate(2,cardP,[])
        wrO=getwinrate(2,cardO,[])

        #第一个玩家位置是0，所以他面对fps，另一方则是fpbr/fpbc/或直接结束

        if posP==0:
            if act1[0]=='r':
                cardsdist[nameP]['fps']['wrs'].append(wrP)
                cardsdist[nameP]['fps']['ids'].append(cardidP)
                cardsdist[nameP]['fps']['cds'].append(cardP)
                cardsdist[nameP]['fps']['owrs'].append(wrO)
                cardsdist[nameP]['fps']['oids'].append(cardidO)
                cardsdist[nameP]['fps']['ocds'].append(cardO)
                countactP[cardidP-1]+=1
                countactO[cardidO-1]+=1
                
            
            if act1[0]=='r':
                countact1[cardidP-1,2]+=1
            elif act1[0]=='c':
                countact1[cardidP-1,1]+=1
            elif act1[0]=='f':
                countact1[cardidP-1,0]+=1
            '''
            if act1[0]=='f':
                countactO[cardidO-1]+=1
                cardsdist[nameO]['fpbr']['wrs'].append(wrO)
                cardsdist[nameO]['fpbr']['ids'].append(cardidO)
                cardsdist[nameO]['fpbr']['cds'].append(cardO)
                cardsdist[nameO]['fpbr']['owrs'].append(wrP)
                cardsdist[nameO]['fpbr']['oids'].append(cardidP)
                cardsdist[nameO]['fpbr']['ocds'].append(cardP)
            '''
        
        if posP==1 or False:
            if act1[0]=='f':
                cardsdist[nameO]['fps']['wrs'].append(wrO)
                cardsdist[nameO]['fps']['ids'].append(cardidO)
                cardsdist[nameO]['fps']['cds'].append(cardO)
            
            if act1[0]=='r':
                countact2[cardidO-1,2]+=1
            elif act1[0]=='c':
                countact2[cardidO-1,1]+=1
            elif act1[0]=='f':
                countact2[cardidO-1,0]+=1
            
            '''
            if act1[0]=='f':#  or True
                countactP[cardidP-1]+=1
                cardsdist[nameP]['fpbr']['wrs'].append(wrP)
                cardsdist[nameP]['fpbr']['ids'].append(cardidP)
                cardsdist[nameP]['fpbr']['cds'].append(cardP)
                cardsdist[nameP]['fpbr']['owrs'].append(wrO)
                cardsdist[nameP]['fpbr']['oids'].append(cardidO)
                cardsdist[nameP]['fpbr']['ocds'].append(cardO)
                #print('')
            '''

        i+=1
        if i>= numberbreak:
            break

    for i in range(1326):
        print(countact2[i,:])
    
    for i in range(1326):
        ctasum1=np.sum(countact1[i,:])
        countact1[i,:]=countact1[i,:]/ctasum1
        ctasum2=np.sum(countact2[i,:])
        countact2[i,:]=countact2[i,:]/ctasum2


    #第一个玩家的fps，fpbr(对手)的牌的分布
    playname=nameP
    decspt='fps'
    print('playname=',playname,decspt)

    '''
    nfps=len(cardsdist[playname]['fps']['ids'])
    print("nfps=",nfps,np.sum(countact1,axis=0))
    print('countact1',countact1)
    
    plt.figure()
    plt.plot(countact1[:,0],label='fold '+nameP)
    plt.plot(countact1[:,1],label='call '+nameP)
    plt.plot(countact1[:,2],label='raise '+nameP)
    plt.legend()

    plt.figure()
    plt.plot(countact2[:,0],label='fold '+nameO)
    plt.plot(countact2[:,1],label='call '+nameO)
    plt.plot(countact2[:,2],label='raise '+nameO)
    plt.legend()
    '''

    
    plt.figure()
    plt.plot(countactP,label=nameP)
    plt.xlabel('I')
    plt.legend()

    plt.figure()
    plt.plot(countactO,label=nameO)
    plt.xlabel('I')
    plt.legend()
    

    '''
    plt.figure()
    plt.hist(cardsdist[playname][decspt]['wrs'],bins=100,density=True,cumulative=False,rwidth=0.9,histtype='bar')
    plt.title('Wr Hist of two cards samples-{}'.format(playname))
    plt.xlabel('Wr')
    

    plt.figure()
    plt.hist(cardsdist[playname][decspt]['ids'],bins=100,density=True,cumulative=False,rwidth=0.9,histtype='bar')
    print("nfps=",len(cardsdist[playname][decspt]['ids']))
    plt.title('I Hist of two cards samples-{}'.format(playname))
    plt.xlabel('I')
    '''

    if decspt=='fpbr':
        plt.figure()
        plt.hist(cardsdist[playname][decspt]['oids'],bins=100,density=True,cumulative=False,rwidth=0.9,histtype='bar')
        print("nfps=",len(cardsdist[playname][decspt]['oids']))
        plt.title('I Hist of two cards samples-o'.format(playname))
        plt.xlabel('I')

        nfpbr=len(cardsdist[playname][decspt]['ids'])
        print("nfpbr",nfpbr)
        cdsrecord=[]
        for i in range(nfpbr):
            cdsrecord.append([cardsdist[playname][decspt]['cds'][i],cardsdist[playname][decspt]['ids'][i],
            cardsdist[playname][decspt]['ocds'][i],cardsdist[playname][decspt]['oids'][i]])
        np.savetxt('cdsrecord1.txt',cdsrecord,fmt='%s')

    plt.show()

    return None







def testholecardsdist():

    nplayer=2
    datausers=recAllresults
    dealgamenumber=0
    if dealgamenumber==0:
        numberbreak=len(datausers)
    else:
        numberbreak=dealgamenumber

    playnames=[ x['name'] for x in datausers[0]['players']]
    cardsdist={}
    for x in playnames:
        cardsdist[x]={'fps':{'wrs':[],'ids':[],'cds':[],'owrs':[],'oids':[],'ocds':[]},
        'fpbr':{'wrs':[],'ids':[],'cds':[],'owrs':[],'oids':[],'ocds':[]}}

    countact1=np.zeros((1326,3)) #fold,call,raise
    countact2=np.zeros((1326,3)) #fold,call,raise
    countactP=np.zeros(1326)
    countactO=np.zeros(1326)
    countactHP=np.zeros((52,2))
    countactHO=np.zeros((52,2))

    #p表示第一个玩家
    #o表示第二个玩家
    nameP=playnames[0]
    nameO=playnames[1]

    
    i=0
    for data in datausers:
        #处理用于统计的动作和其它信息
        #记录到recAllgameHistory和recAllgameHistoryS中
        #get_actioninfosetall1(data,nplayer)

        #posP,posO是第一个和第二个玩家在当前局中的位置
        if data['players'][0]['name']==nameP:
            posP=0
            posO=1
        else:
            posP=1
            posO=0
        
        act1=data['action_history'][0][0]['action']
        if act1 !='fold':
            act2=data['action_history'][0][1]['action']
        
        cardP=data['player_card'][posP]
        cardO=data['player_card'][posO]
        cardidP=int(getCombinationId([cardint[x] for x in cardP], 1,52))
        cardidO=int(getCombinationId([cardint[x] for x in cardO], 1,52))
        wrP=getwinrate(2,cardP,[])
        wrO=getwinrate(2,cardO,[])
        

        #第一个玩家位置是0，所以他面对fps，另一方则是fpbr/fpbc/或直接结束

        if posP==0:
            #print('%d,%d,%d,%d'%(cardint[cardP[0]]-1,cardint[cardP[1]]-1,cardint[cardO[0]]-1,cardint[cardO[1]]-1))
            if act1[0]=='f':
                cardsdist[nameP]['fps']['wrs'].append(wrP)
                cardsdist[nameP]['fps']['ids'].append(cardidP)
                cardsdist[nameP]['fps']['cds'].append(cardP)
                cardsdist[nameP]['fps']['owrs'].append(wrO)
                cardsdist[nameP]['fps']['oids'].append(cardidO)
                cardsdist[nameP]['fps']['ocds'].append(cardO)
                countactP[cardidP-1]+=1
                countactO[cardidO-1]+=1
                countactHP[cardint[cardP[0]]-1,0]+=1
                countactHP[cardint[cardP[1]]-1,1]+=1
                countactHO[cardint[cardO[0]]-1,0]+=1
                countactHO[cardint[cardO[1]]-1,1]+=1
                #print('%d,%d,%d,%d'%(cardint[cardP[0]]-1,cardint[cardP[1]]-1,cardint[cardO[0]]-1,cardint[cardO[1]]-1))
                print('%s%s|%s%s'%(cardP[0],cardP[1],cardO[0],cardO[1]))
                
        i+=1
        if i>= numberbreak:
            break




    #第一个玩家的fps，fpbr(对手)的牌的分布
    playname=nameP
    print('ndata=',len(datausers))
    decspt='fps'
    print('playname=',playname,decspt)
    print('hands=',np.sum(countactP),np.sum(countactO))

   
    plt.figure()
    plt.plot(countactP,label=nameP)
    plt.xlabel('I')
    plt.legend()

    plt.figure()
    plt.plot(countactO,label=nameO)
    plt.xlabel('I')
    plt.legend()

    plt.figure()
    plt.plot(countactHP[:,0],label=nameP+'CARD')
    plt.xlabel('I')
    plt.legend()

    plt.figure()
    plt.plot(countactHP[:,1],label=nameP+'CARD')
    plt.xlabel('I')
    plt.legend()

    plt.figure()
    plt.plot(countactHO[:,0],label=nameO+'CARD')
    plt.xlabel('I')
    plt.legend()

    plt.figure()
    plt.plot(countactHO[:,1],label=nameO+'CARD')
    plt.xlabel('I')
    plt.legend()
    


    plt.show()

    return None




def testEapwithsigmafig(Eapdata,Eapdata1):
    #MS
    opname1='MS'

    #TARD
    opname2='TARD'

    for i in range(len(Eapdata)):
        print("{} & {} & {}".format(Eapdata[i,0],Eapdata[i,1],Eapdata[i,2]))
    
    plt.figure()
    plt.plot(Eapdata[:,0],Eapdata[:,2],marker="o",label='MS (vs LP)')
    plt.plot(Eapdata1[:,0],Eapdata1[:,2],marker="x",label='TARD (vs ph)')
    plt.xlabel('$\sigma$')
    plt.ylabel('$E_{ap}$')
    plt.grid() #绘制网格
    #plt.xlim(0,1) #x轴绘制范围限制
    #plt.ylim(0,1.1) #x轴绘制范围限制
    plt.legend(frameon=False) #显示图例
    plt.title("$E_{ap}$ at fps preflop") #调整x轴坐标刻度和标签
    plt.savefig('fig-action-eap-fps-preflop.pdf')
    plt.show()
    return None







if __name__ == "__main__":
    
    #testACPClogmsgTodata()
    #print(handratiofromWR(0.5))
    #testpltmat()
    #anykey=input()
    #testEapwithsigma()
    #testACPCtrans()


    # AI获得的结果数据进行处理
    # ---------------------------
    # figure 1
    if 0:
        # sub figure (b) 
        #statmodeldiffattri('res-2p-ph-LA-20000.md','LA','fps',wdin=0.01)
        statmodeldiffattrinojson('dataph\\res-2p-ph-LA-20000-msg.md','LA','fps',wdin=0.01)

    if 0:
        # sub figure (b) 
        statmodeldiffattrinojson('dataph\\res-2p-ph-LARD-20000-msg.md','LARD','fps',wdin=0.02)


    # ---------------------------
    # figure 2
    if 0:
        # sub figure (a,b)
        #statmodeldiffattri('res-2p-ph-TARD-20000.md','TARD','fps',cpf=100)
        statmodeldiffattrinojson('dataph\\res-2p-ph-TARD-20000-msg.md','TARD','fps',cpf=100)

    if 0:
        # sub figure (c,d)
        #statmodeldiffattri('res-2p-ph-MS-20000.md','MS','fps',cpf=20)
        statmodeldiffattrinojson('dataph\\res-2p-ph-MS-20000-msg.md','MS','fps',cpf=20)



    # ---------------------------
    # figure 7/8
    filename='dataACPC\\PokerBot5.PokerCNN'
    if 1:
        opname='PokerCNN' #figure 7
        wdin1=0.02
        cpf1=10
        wdin2=0.02
        cpf2=10

    if 0:
        opname='PokerBot5' # figure 8
        wdin1=0.03
        cpf1=20
        wdin2=0.04
        cpf2=2
    
    if 1:
        decisionpt='fps'

        if 1:
            filelist=[]
            for i in range(1,11):
                filename1=filename+'.{}.0.log'.format(i)
                filename2=filename+'.{}.1.log'.format(i)
                
                filelist.append(filename1)
                filelist.append(filename2)
                
            for file in filelist:
                #注意从acpc下载的数据中是大盲注先行的
                logfiledealing(file,opname+'_2pn_2017',False,True)
            #np.savetxt('recAllresults.txt',recAllresults,fmt='%s')
            #testholecardsdist()
            statmodelserverALLPF(opname+'_2pn_2017',decisionpt)

        filelist=[]
        for i in range(1,2):
            filename1=filename+'.{}.0.log'.format(i)
            filename2=filename+'.{}.1.log'.format(i)
            
            filelist.append(filename1)
            filelist.append(filename2)

        recAllresults=[]
        for file in filelist:
            #注意从acpc下载的数据中是大盲注先行的
            logfiledealing(file,opname+'_2pn_2017',False,True,flagfoldseen=False)
        statmodelserverinferPF(opname+'_2pn_2017',decisionpt,opname+'_2pn_2017'+'-preflop-'+decisionpt+'full.csv',wdin=wdin1,cpf=cpf1)


    if 0:
        decisionpt='prp'

        if 0:
            filelist=[]
            for i in range(1,11):
                filename1=filename+'.{}.0.log'.format(i)
                filename2=filename+'.{}.1.log'.format(i)
                
                filelist.append(filename1)
                filelist.append(filename2)
                
            for file in filelist:
                #注意从acpc下载的数据中是大盲注先行的
                logfiledealing(file,opname+'_2pn_2017',False,True)
            #np.savetxt('recAllresults.txt',recAllresults,fmt='%s')
            #testholecardsdist()
            statmodelserverALLPF(opname+'_2pn_2017',decisionpt)

        filelist=[]
        for i in range(1,2):
            filename1=filename+'.{}.0.log'.format(i)
            filename2=filename+'.{}.1.log'.format(i)
            
            filelist.append(filename1)
            filelist.append(filename2)

        recAllresults=[]
        for file in filelist:
            #注意从acpc下载的数据中是大盲注先行的
            logfiledealing(file,opname+'_2pn_2017',False,True,flagfoldseen=False)
        statmodelserverinferPF(opname+'_2pn_2017',decisionpt,opname+'_2pn_2017'+'-preflop-'+decisionpt+'full.csv',wdin=wdin1,cpf=cpf1)


    if 0:
        decisionpt='pra'

        if 0:
            filelist=[]
            for i in range(1,11):
                filename1=filename+'.{}.0.log'.format(i)
                filename2=filename+'.{}.1.log'.format(i)
                
                filelist.append(filename1)
                filelist.append(filename2)
                
            for file in filelist:
                #注意从acpc下载的数据中是大盲注先行的
                logfiledealing(file,opname+'_2pn_2017',False,True)
            #np.savetxt('recAllresults.txt',recAllresults,fmt='%s')
            #testholecardsdist()
            statmodelserverALLAF(opname+'_2pn_2017',decisionpt)

        filelist=[]
        for i in range(1,2):
            filename1=filename+'.{}.0.log'.format(i)
            filename2=filename+'.{}.1.log'.format(i)
            
            filelist.append(filename1)
            filelist.append(filename2)

        recAllresults=[]
        for file in filelist:
            #注意从acpc下载的数据中是大盲注先行的
            logfiledealing(file,opname+'_2pn_2017',False,True,flagfoldseen=False)
        statmodelserverinferAF(opname+'_2pn_2017',decisionpt,opname+'_2pn_2017'+'-postflop-'+decisionpt+'full.csv',wdin=wdin2,cpf=cpf2)



    # ---------------------------
    # figure 3
    if 0:
        # sub figure (a)
        statmodeldiffattri('res-2p-ph-TARD-20000.md','TARD','fps',gamenumbreak=1326,cpf=100)
    if 0:
        # sub figure (b)
        statmodeldiffattri('res-2p-ph-MS-20000.md','MS','fps',gamenumbreak=1326,cpf=20)
    if 0:
        # sub figure (c,d)
        Eapdata=np.zeros((9,3))
        Eapdata[:,0]=[0.5,1,2,4,8,16,32,48,72]
        Eapdata[:,1]=[1326,2652,5304,10608,21216,42432,84864,127296,190944]
        i=0
        for gamebreak in [1326,2652,5304,10608,21216,42432,84864,127296,190944]:
            Eapdata[i,2]=statmodeldiffattri('res-2p-LP-MS-200000.md','MS','fps',gamebreak,cpf=20)
            i+=1
        
        datausers=[]
        Eapdata1=Eapdata.copy()
        i=0
        for gamebreak in [1326,2652,5304,10608,21216,42432,84864,127296,190944]:
            Eapdata1[i,2]=statmodeldiffattri('res-2p-ph-TARD-200000.md','TARD','fps',gamebreak,cpf=100)
            i+=1
        print('Eapdata=',Eapdata)
        print('Eapdata1=',Eapdata1)
        testEapwithsigmafig(Eapdata,Eapdata1)

    # ---------------------------
    # figure 4
    if 0:
        # sub figure (a)
        statmodeldiffattri('res-2p-ph-TARD-20000.md','TARD','prp',wdin=0.015,cpf=100)
    if 0:
        # sub figure (b)
        datausers=[]
        statmodeldiffattri('res-2p-RA-TARD-20000.md','TARD','prp',wdin=0.015,cpf=100)
    if 0:
        # sub figure (c)
        statmodeldiffattri('res-2p-ph-TARD-20000.md','TARD','prp',gamenumbreak=1326,wdin=0.015,cpf=100)
    if 0:
        # sub figure (d)
        statmodeldiffattri('res-2p-RA-TARD-20000.md','TARD','prp',gamenumbreak=1326,wdin=0.015,cpf=100)

    
    
    # ---------------------------
    # figure 5/6
    if 0:
        # sub figure (a)
        statmodeldiffattriAF('res-2p-ph-TARDAF-20000.md','TARDAF','pra',wdin=0.015,cpf=50)
    if 0:
        # sub figure (a)
        datausers=[]
        statmodeldiffattriAF('res-2p-RA-TARDAF-20000.md','TARDAF','pra',wdin=0.015,cpf=50)
    if 0:
        # sub figure (a)
        statmodeldiffattriAF('res-2p-ph-MSAF-20000.md','MSAF','pra',wdin=0.015,cpf=20)
    if 0:
        # sub figure (a)
        datausers=[]
        statmodeldiffattriAF('res-2p-RA-MSAF-20000.md','MSAF','pra',wdin=0.015,cpf=20)


    
    
    

    

    
    



