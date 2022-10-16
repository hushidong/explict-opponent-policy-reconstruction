#!/usr/bin/env python3
#_∗_coding: utf-8 _∗_


import sys
import io
import numpy as np
import matplotlib.pyplot as plt
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

# 常数
const_sc_compress=100  #s型曲线的压缩因子 10 for MS,50 for TARD
const_dm_prelmax=50   #匹配区间模型的相对概率最大值
const_dm_prelinter=25
const_dm_dx=0.01


#高斯核
def kernelrbf(x):
    y=np.exp(-x**2/2.0)/np.sqrt(2*np.pi)
    return y

#核密度估计
def evalActionDensity(xs,xf,w=0.03):  #w=0.03  #尺度
    fx1=[]
    n=len(xf)
    for i in range(len(xs)):
        yp=[kernelrbf(np.abs(xs[i]-xi)/w) for xi in xf]
        fx1.append(sum(yp)/n/w)
    return fx1


def thetastep(x):#仅考虑一维
    if x>0:
        y=1.0
    else:
        y=0.0
    return y

# Epanechikov 二次核
def kernelepk(x):
    if x>=-1 and x<=1:
        y=0.75*(1.0-x**2)
    else:
        y=0.0
    return y


#
#s型曲线函数
#输入为赢率，中心位移
#输出为s曲线的值
#act_model_dist_sigmoid_curve
def ActModeldistScv(wr,eshift,c=const_sc_compress):
    if wr<0 or wr >1:
        sres=1.0
    else:
        x=wr
        e=eshift
        sres=1/(1+np.exp(-c*(x-e)))
    return sres


#
#根据0-1范围曲线下积分面前求曲线的位移
#输入为曲线下面积，即：当前行动和比当前行动风险高的行动的概率。
#输出为s曲线的中心位移
#若赢率范围不是0-1，那么可以做一个赢率转换。
#xstt=0.323032 xend=0.852037
def ActModeldistSe(rdist,c=const_sc_compress):
    r=rdist
    eres=np.log((np.exp(c)-np.exp(c*r))/(-1+np.exp(c*r)))/c
    return eres


#赢率转换：输入是要考虑的赢率wrreal，但可能不在范围内，所以直接转成-1，若在范围内则扩展到0-1范围处理。
#输出的是：假设曲线的对应的赢率
#为什么要做转换：因为只有转换了才有可能
def wrScale(wrreal,xstt,xend):
    if wrreal<xstt:
        wrres=-1
    elif wrreal>xend:
        wrres=-1
    else:
        wrres=(wrreal-xstt)/(xend-xstt)
    return wrres


#
#计算赢率为wr情况时，给定动作总体概率下的各行动选择概率
#输入是赢率，动作总体概率列表，比如f:c:r1:r2，在大数据统计中是0.15:0.3:0.4:0.15，则输入[0.15,0.3,0.4,0.15]即可
#输出两个列表
#第一个列表是各动作的行动选择概率，第二个列表为分割各行动风险区域的s曲线上的点的值
def ActModelSProb(wr,dist,c=const_sc_compress):
    d=dist
    x=wr
    nact=len(dist)
    prob=[0]*nact
    ed=[ActModeldistSe(sum(d[i:]),c) for i in range(1,nact)]
    cv=[ActModeldistScv(x,e,c) for e in ed]
    pr=[1]+cv+[0]
    for i in range(nact):
        prob[i]=pr[i]-pr[i+1]
    return prob,cv


# 将行动概率曲线输出
# 输出：一个列表的列表，内部分别为各个动作对应从0到1的npts个点的赢率的行动概率
def outSmodel(dist,npts,xstt=0.0,xend=1.0):
    d=dist
    nact=len(d)
    wr=np.linspace(0,1,npts)
    pa=np.zeros((npts,nact))
    for i in range(npts):
        if wr[i]<xstt or wr[i]>xend:
            pa[i,:]=0.0
        else:
            prob,cv=ActModelSProb(wr[i],d)
            pa[i,:]=prob[:]
    pout=[]
    for j in range(nact):
        pout.append(pa[:,j].tolist())
    return pout


#
#计算匹配区间模型的相对行动概率
#输入是赢率，和各行动的匹配区间列表[[0,0.3],[0.3,0.45],[0.45,0.65],[0.65,1]]
#输出各动作的相对概率列表
def ActModelDmPrel(wr,domains):
    prelres=[]
    xp=wr
    for domain in domains:
        lowerbound=domain[0]
        upperbound=domain[1]
        pmax=const_dm_prelmax #匹配区间模型的相对概率最大值
        pinter=const_dm_prelinter
        dx=const_dm_dx
        if  (xp<lowerbound):
            if(lowerbound-xp<dx):
                #yp=(pinter-pmax)*(lowerbound-xp)**2/dx**2+pmax
                yp=(pinter-pmax)*(lowerbound-xp)/dx+pmax
            else:
                yp=dx/(lowerbound-xp)*pinter
        elif (xp>upperbound):
            if(xp-upperbound<dx):
                #yp=(pinter-pmax)*(xp-upperbound)**2/dx**2+pmax
                yp=(pinter-pmax)*(xp-upperbound)/dx+pmax
            else:
                yp=(dx)/(xp-upperbound)*pinter
        else:
            yp=pmax
        prelres.append(yp)
    return prelres

#
#计算赢率为wr情况时，给定动作总体概率下的各行动选择概率
#输入是赢率，匹配区域列表[[0,0.3],[0.3,0.45],[0.45,0.65],[0.65,1]]
#输出两个列表
#第一个列表是各动作的行动选择概率，第二个列表为相对概率值
def ActModelDProb(wr,domains):
    prel=ActModelDmPrel(wr,domains)
    prob=[]
    sumprel=sum(prel)
    for i in range(len(prel)):
        prob.append(prel[i]/sumprel)
    return prob,prel


# 将行动概率曲线输出
# 输出：一个列表的列表，内部分别为各个动作对应从0到1的npts个点的赢率的行动概率
def outDmodel(domains,npts,xstt=0.0,xend=1.0):
    wr=np.linspace(0,1,npts)
    nact=len(domains)
    pa=np.zeros((npts,nact))
    for i in range(npts):
        if wr[i]<xstt or wr[i]>xend:
            pa[i,:]=0.0
        else:
            prob,prel=ActModelDProb(wr[i],domains)
            pa[i,:]=prob[:]

    pout=[]
    for j in range(nact):
        pout.append(pa[:,j].tolist())
    return pout


# 将行动概率曲线输出
# 输出：一个列表的列表，内部分别为各个动作对应从0到1的npts个点的赢率的行动概率
# dcpt:是决策点类型，0是pck，1是pcl
def outBmodel(alpha,beta,gamma,zeta1,zeta2,npts,dcpt,xstt=0.0,xend=1.0):
    wr=np.linspace(0,1,npts)
    if dcpt==0:
        nact=2
        pa=np.zeros((npts,nact))
        for i in range(npts):
            if wr[i]<xstt or wr[i]>xend:
                pa[i,:]=0.0
            elif wr[i]< beta:
                pa[i,0]=zeta1
                pa[i,1]=1-zeta1
            else:
                pa[i,0]=zeta2
                pa[i,1]=1-zeta2
    else:
        nact=3
        pa=np.zeros((npts,nact))
        for i in range(npts):
            if wr[i]<xstt or wr[i]>xend:
                pa[i,:]=0.0
            elif wr[i] <alpha:
                pa[i,0]=1.0
                pa[i,1]=0.0
                pa[i,2]=0.0
            elif wr[i]< beta:
                pa[i,0]=0.0
                pa[i,1]=zeta1
                pa[i,2]=1-zeta1
            else:
                pa[i,0]=0.0
                pa[i,1]=zeta2
                pa[i,2]=1-zeta2
    pout=[]
    for j in range(nact):
        pout.append(pa[:,j].tolist())
    return pout

#测试基础风格的鞠策模型
def testopBmodel():
    alpha=0.55
    beta=0.65
    gamma=0.0
    zeta1=0.7
    zeta2=0.2
    dcpt=0
    npts=101
    pres=outBmodel(alpha,beta,gamma,zeta1,zeta2,npts,dcpt)
    if dcpt==1:
        actnames=['fold','call','raise']
    else:
        actnames=['check','raise']

    plt.figure()
    for i in range(len(pres)):
        plt.plot(np.linspace(0,1,npts),pres[i],linewidth=2,label=actnames[i])
    plt.xlabel('winrate')
    plt.ylabel('action probability')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.grid()
    plt.title('Action Probability about Winrate')
    plt.legend(loc='upper left',frameon=False)
    plt.savefig('fig-action-model-btype-prob.pdf')

    plt.show()
    return None




#测试D模型
def testopDmodel():

    d=[[0,0.3],[0.3,0.45],[0.45,0.65],[0.65,1]] #f,c,r1,r2
    npts=101
    wr=np.linspace(0,1,npts)
    pf=np.zeros(npts)
    pc=np.zeros(npts)
    pr1=np.zeros(npts)
    pr2=np.zeros(npts)
    prelf=np.zeros(npts)
    prelc=np.zeros(npts)
    prelr1=np.zeros(npts)
    prelr2=np.zeros(npts)
    for i in range(npts):
        prob,prel=ActModelDProb(wr[i],d)
        pf[i],pc[i],pr1[i],pr2[i]=prob
        prelf[i],prelc[i],prelr1[i],prelr2[i]=prel

    plt.figure()
    plt.plot(wr,prelf,linewidth=2,label='fold')
    plt.plot(wr,prelc,marker='s',markevery=7,linewidth=2,label='call')
    plt.plot(wr,prelr1,marker='d',markevery=7,linewidth=2,label='r1')
    plt.plot(wr,prelr2,marker='o',markevery=7,linewidth=2,label='r2')
    plt.xlabel('winrate')
    plt.ylabel('action relative probability')
    plt.xlim(0,1)
    plt.ylim(0,55)
    plt.grid()
    plt.title('Action Relative Probability about Winrate')
    plt.legend(loc='upper left',frameon=False)
    plt.savefig('fig-action-model-dtype-prel.pdf')

    plt.figure()
    plt.plot(wr,pf,linewidth=2,label='fold')
    plt.plot(wr,pc,marker='s',markevery=7,linewidth=2,label='call')
    plt.plot(wr,pr1,marker='d',markevery=7,linewidth=2,label='r1')
    plt.plot(wr,pr2,marker='o',markevery=7,linewidth=2,label='r2')
    plt.xlabel('winrate')
    plt.ylabel('action probability')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.grid()
    plt.title('Action Probability about Winrate')
    plt.legend(loc='upper left',frameon=False)
    plt.savefig('fig-action-model-dtype-prob.pdf')

    plt.show()


#测试s曲线的模型
def testopSmodel1():

    d=[0.15,0.3,0.4,0.15] #f,c,r1,r2
    npts=101
    wr=np.linspace(0,1,npts)
    pf=np.zeros(npts)
    pc=np.zeros(npts)
    pr1=np.zeros(npts)
    pr2=np.zeros(npts)
    cifc=np.zeros(npts)
    cicr1=np.zeros(npts)
    cir1r2=np.zeros(npts)
    for i in range(npts):
        prob,cv=ActModelSProb(wr[i],d)
        pf[i],pc[i],pr1[i],pr2[i]=prob
        cifc[i],cicr1[i],cir1r2[i]=cv

    plt.figure()
    plt.plot(wr,cifc,linewidth=2,label='between fold and call')
    plt.plot(wr,cicr1,marker='s',markevery=7,linewidth=2,label='between call and r1')
    plt.plot(wr,cir1r2,marker='d',markevery=7,linewidth=2,label='between r1 and r2')
    plt.xlabel('winrate')
    plt.ylabel('action dist')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.title('Action Distribution about Winrate')
    plt.legend(loc='upper left',frameon=False)
    plt.savefig('fig-action-model-stype-dist.pdf')

    plt.figure()
    plt.plot(wr,pf,linewidth=2,label='fold')
    plt.plot(wr,pc,marker='s',markevery=7,linewidth=2,label='call')
    plt.plot(wr,pr1,marker='d',markevery=7,linewidth=2,label='r1')
    plt.plot(wr,pr2,marker='o',markevery=7,linewidth=2,label='r2')
    plt.xlabel('winrate')
    plt.ylabel('action probability')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.grid()
    plt.title('Action Probability about Winrate')
    plt.legend(loc='upper left',frameon=False)
    plt.savefig('fig-action-model-stype-prob.pdf')

    plt.show()

    return None




#测试s曲线的模型
def testopSmodel(dist,xstt=0.0,xend=1.0):

    #d=[0.3,0.4,0.3] #f,c,r
    d=dist
    actnames=['fold','call','raise']
    npts=101
    nact=len(d)
    wr=np.linspace(0,1,npts)
    pa=np.zeros((npts,nact))
    ca=np.zeros((npts,nact-1))
    for i in range(npts):
        if wr[i]<xstt or wr[i]>xend:
            pa[i,:]=0.0
            ca[i,:]=0.0
        else:
            prob,cv=ActModelSProb(wr[i],d)
            pa[i,:]=prob[:]
            ca[i,:]=cv[:]

    plt.figure()
    for i in range(nact-1):
        plt.plot(wr,ca[:,i],linewidth=2,label='between {} and {}'.format(actnames[i],actnames[i+1]))
    plt.xlabel('winrate')
    plt.ylabel('action dist')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.title('Action Distribution about Winrate')
    plt.legend(loc='upper left',frameon=False)
    plt.savefig('fig-action-model-stype-dist.pdf')

    plt.figure()
    for i in range(nact):
        plt.plot(wr,pa[:,i],linewidth=2,label=actnames[i])
    plt.xlabel('winrate')
    plt.ylabel('action probability')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.grid()
    plt.title('Action Probability about Winrate')
    plt.legend(loc='upper left',frameon=False)
    plt.savefig('fig-action-model-stype-prob.pdf')

    plt.show()

    return None








if __name__ == "__main__":

    #testopDmodel()

    '''
    pclmodel=[0.4,0.3,0.4]
    testopSmodel(pclmodel,0.323032,0.852037)
    pckmodel=[0.5,0.5]
    testopSmodel(pckmodel)
    '''

    testopBmodel()



