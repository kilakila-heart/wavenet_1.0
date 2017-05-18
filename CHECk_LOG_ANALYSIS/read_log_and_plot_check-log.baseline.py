#--*coding:utf-8 --*
#先使用正则表达式将数据分别用列表和字典存储，之后看一下怎么去用描点还是什么方法画出这种图
import numpy as np
import re
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
#pattern_train_loss="train step \d+ - loss = \d\.\d+, t1_loss = \d\.\d+, t1_L1 = \d+\.\d+,(.*?) "# for match
pattern_train_loss_train_tl_L1="train step (.*?) - loss = (.*?), t1_loss = (.*?), t1_L1 = (.*?),(.*?)"
dict_train_loss={}
list_train_loss_step=[]
list_train_loss=[]
list_train_tl_L1=[]
dict_train_t1_L1={}

dict_val_loss={}
list_val_loss_step=[]
list_val_loss=[]
list_val_tl_L1=[]
dict_val_t1_L1={}

pattern_validation_loss_val_tl_L1="validation at step (.*?) - loss = (.*?), t1_loss = (.*?), t1_L1 = (.*?), (.*?)"
dict_val_loss={}
dict_val_t1_L1={}
file_line_number=0
filename='check-log.baseline.txt'
with open(filename) as file_log:
        for line in file_log:
                if re.match(pattern_train_loss_train_tl_L1,line) is not None:
                        searchObj=re.match(pattern_train_loss_train_tl_L1,line) 
                        if searchObj:
                                train_step=int(searchObj.group(1))
                                train_loss=float(searchObj.group(2))
                                train_tl_L1=float(searchObj.group(4))
                                dict_train_loss[train_step]=train_loss
                                list_train_loss_step.append(train_step)
                                list_train_loss.append(train_loss)
                                list_train_tl_L1.append(train_tl_L1)
                                dict_train_t1_L1[train_step]=train_tl_L1
                elif re.match(pattern_validation_loss_val_tl_L1,line) is not None:
                        searchObj=re.match(pattern_validation_loss_val_tl_L1,line)
                        if searchObj:                                
                                val_step = int(searchObj.group(1))
                                val_loss = float(searchObj.group(2))
                                val_tl_L1 = float(searchObj.group(4))
                                dict_val_loss[val_step]= val_loss
                                list_val_loss_step.append(val_step)
                                list_val_loss.append(val_loss)
                                list_val_tl_L1.append(val_tl_L1)
                                dict_val_t1_L1[val_step]=val_tl_L1
                file_line_number+=1
#排序是多余的。。
#dict_train_loss=OrderedDict(sorted(dict_train_loss.items(), key=lambda x: x[0]))
#dict_train_t1_L1=OrderedDict(sorted(dict_train_t1_L1.items(), key=lambda x: x[0]))
#dict_val_loss=OrderedDict(sorted(dict_val_loss.items(), key=lambda x: x[0]))
#dict_val_t1_L1=OrderedDict(sorted(dict_val_t1_L1.items(), key=lambda x: x[0]))
print file_line_number
# Train and test done, outputing convege graph
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

train_x = list_train_loss_step
train_y = list_train_loss
val_x=list_val_loss_step[1:] #直接截取一段画出就好了，这里二了
val_y=list_val_loss[1:]
# val_y=[100*i for i in val_y]
train_tl_L1=list_train_tl_L1
val_tl_L1=list_val_tl_L1[1:]

fig, ax = plt.subplots(figsize=(20, 18))
# plt.xlim(X.min() * 1.1, X.max() * 1.1)
# plt.ylim(C.min() * 1.1, C.max() * 1.1)

ax.plot(train_x, train_y, linewidth=1.0,label='train loss')
# print list_train_loss_step[1]
# print list_train_loss_step[-1]
ax.set_xlim(list_train_loss_step[1],list_train_loss_step[-1])
# supper = np.ma.masked_where(val_x < val_x[0], [0]*(val_x[0]-list_train_loss_step[1]))
# slower = np.ma.masked_where(val_x >=val_x[0], list_val_loss)
# smiddle = np.ma.masked_where(np.logical_or(s < lower, s > upper), s)
# ax.plot(np.arange(list_train_loss_step[1],val_x[0]-1),supper,val_x,slower , 'k--',linewidth=5.0,label='val loss')
ax.plot(val_x, val_y, '*',linewidth=1.0,label='val loss') #k--

ax.grid()
#plt.title('train loss and val loss', fontdict=font)
#plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
plt.xlabel('Step (10^2)', fontdict=font)
plt.ylabel('Loss ', fontdict=font)
ax.set_title("Train loss and Val loss")
# ax.set_xscale('log')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)

plt.show()

savefig('./my1'+filename+'.jpg')


fig1, ax1 = plt.subplots(figsize=(10, 8))
# plt.xlim(X.min() * 1.1, X.max() * 1.1)
# plt.ylim(C.min() * 1.1, C.max() * 1.1)

ax1.plot(list_train_loss_step,train_tl_L1 , linewidth=5.0,label='train t1_l1')
# print list_train_loss_step[1]
# print list_train_loss_step[-1]
ax1.set_xlim(list_train_loss_step[1],list_train_loss_step[-1])
# supper = np.ma.masked_where(val_x < val_x[0], [0]*(val_x[0]-list_train_loss_step[1]))
# slower = np.ma.masked_where(val_x >=val_x[0], list_val_loss)
# smiddle = np.ma.masked_where(np.logical_or(s < lower, s > upper), s)
# ax.plot(np.arange(list_train_loss_step[1],val_x[0]-1),supper,val_x,slower , 'k--',linewidth=5.0,label='val loss')
ax1.plot(val_x, val_tl_L1, linewidth=5.0,label='val t1_l1')

ax1.grid()
#plt.title('train loss and val loss', fontdict=font)
#plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
plt.xlabel('Step (10^2)', fontdict=font)
plt.ylabel('T1_L1_Loss ', fontdict=font)
ax1.set_title("Train T1_L1 and val T1_L1 ")
# ax1.set_xscale('log')
legend = ax1.legend(loc='upper center', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
# Tweak spacing to prevent clipping of ylabel
#plt.subplots_adjust(left=0.15)
plt.show()

savefig('./my2'+filename+'.jpg')
print "ok"

def list_mean(A,i):
    sum=0.0
    smooth_int=5
    if i<smooth_int:
        print "error"
    else:
        for j in range(smooth_int+1):
            sum+=A[i-j]
    return sum/(smooth_int+1)



#平滑曲线
for i in xrange(5,len(list_train_loss)-1):
    list_train_loss[i]=list_mean(list_train_loss,i)
for i in xrange(5,len(list_train_tl_L1)-1):
    list_train_loss[i]=list_mean(list_train_tl_L1,i)
for i in xrange(5,len(list_val_loss)-1):
    list_train_loss[i]=list_mean(list_val_loss,i)
for i in xrange(5,len(list_val_tl_L1)-1):
    list_train_loss[i]=list_mean(list_val_tl_L1,i)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

train_x = list_train_loss_step
train_y = list_train_loss
val_x=list_val_loss_step[1:] #直接截取一段画出就好了，这里二了
val_y=list_val_loss[1:]

train_tl_L1=list_train_tl_L1
val_tl_L1=list_val_tl_L1[1:]

fig, ax = plt.subplots(figsize=(10, 8))
# plt.xlim(X.min() * 1.1, X.max() * 1.1)
# plt.ylim(C.min() * 1.1, C.max() * 1.1)

ax.plot(train_x, train_y, linewidth=1.0,label='train loss')
# print list_train_loss_step[1]
# print list_train_loss_step[-1]
ax.set_xlim(list_train_loss_step[1],list_train_loss_step[-1])
# supper = np.ma.masked_where(val_x < val_x[0], [0]*(val_x[0]-list_train_loss_step[1]))
# slower = np.ma.masked_where(val_x >=val_x[0], list_val_loss)
# smiddle = np.ma.masked_where(np.logical_or(s < lower, s > upper), s)
# ax.plot(np.arange(list_train_loss_step[1],val_x[0]-1),supper,val_x,slower , 'k--',linewidth=5.0,label='val loss')
ax.plot(val_x, val_y, linewidth=1.0,label='val loss')

ax.grid()
#plt.title('train loss and val loss', fontdict=font)
#plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
plt.xlabel('Step (10^2)', fontdict=font)
plt.ylabel('Loss ', fontdict=font)
ax.set_title("Train loss and Val loss")
# ax.set_xscale('log')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()

savefig('./my3'+filename+'.jpg')


fig1, ax1 = plt.subplots(figsize=(10, 8))
# plt.xlim(X.min() * 1.1, X.max() * 1.1)
# plt.ylim(C.min() * 1.1, C.max() * 1.1)

ax1.plot(list_train_loss_step,train_tl_L1 , linewidth=1.0,label='train t1_l1')
# print list_train_loss_step[1]
# print list_train_loss_step[-1]
ax1.set_xlim(list_train_loss_step[1],list_train_loss_step[-1])
# supper = np.ma.masked_where(val_x < val_x[0], [0]*(val_x[0]-list_train_loss_step[1]))
# slower = np.ma.masked_where(val_x >=val_x[0], list_val_loss)
# smiddle = np.ma.masked_where(np.logical_or(s < lower, s > upper), s)
# ax.plot(np.arange(list_train_loss_step[1],val_x[0]-1),supper,val_x,slower , 'k--',linewidth=5.0,label='val loss')
ax1.plot(val_x, val_tl_L1, linewidth=1.0,label='val t1_l1')

ax1.grid()
#plt.title('train loss and val loss', fontdict=font)
#plt.text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', fontdict=font)
plt.xlabel('Step (10^2)', fontdict=font)
plt.ylabel('T1_L1_Loss ', fontdict=font)
ax1.set_title("Train T1_L1 and val T1_L1 ")
# ax1.set_xscale('log')
legend = ax1.legend(loc='upper center', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()

savefig('./my4'+filename+'.jpg')
print "ok"









