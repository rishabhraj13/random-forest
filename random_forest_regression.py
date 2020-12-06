from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt 

def float_conversion(f, y):          # convert the samples into float values        
    for i in range(len(f)):
        x = f[i]
        a = x[y].strip()
        x[y] = float(a)
        
def int_conversion(f, y):
    a = []
    for x in f:
        x[y] = float(x[y])
        a = a + [x[y]]
    m1 = float(min(a))
    m2 = float(max(a))
    
    ax = float(m2 - m1)
    ac = float(ax/3)      
    i = float(m1 + ac)
    j = float(m1 + (2*ac))
    
    for x in f:
        x[y] = float(x[y])
        if x[y] >= m1 and x[y] < i:
            x[y] = 0
        elif x[y] >= i and x[y] < j:
            x[y] = 1
        else:
            x[y] = 2
    return


def tree(f, algo, n, *arguments):    
    fs = []
    f_copy = list(f)
    l = len(f)
    s_f = int(l/n)
    for i in range(n):
        fd = []
        while len(fd) < s_f:
            l1 = len(f_copy)
            ind = randrange(l1)     # to generate random number in the range of length of dataset
            a = f_copy.pop(ind)
            fd = fd + [a]
        fs = fs + [fd]    
    
    K_fds = fs
    s = []
    for ij in range(len(K_fds)):
        k_fd = K_fds[ij]
        x_set = list(K_fds)             #train_set = x    test_set = y
        x_set.remove(k_fd)
        x_set = sum(x_set, [])
        y_set = []
                
        for j in range(len(k_fd)):
            x = k_fd[j]
            x1 = list(x)               #x_copy = x1      
            y_set = y_set + [x1]
            x1[-1] = 0
        pred = algo(x_set, y_set, *arguments)
        act = []
        
        for k in range(len(k_fd)):
            x = k_fd[k]
            act = act + [x[-1]]
        
        l = len(act)
        c = 0
        for i in range(l):
            if act[i] == pred[i]:
                c = c + 1
        d = float(l)
        r = (c/d)*100.0
        acc = r
        s = s + [acc]
    return s


def seperate(f, n):                    #n = n_feature
    c = []
    for i in range(len(f)):
        x = f[i]
        c= c + [x[-1]]
        
    s = set(c)
    class_values = list(s)
    bind = 100
    bval = 100
    bsc = 100
    b_gps = 0
    fs = []
    while len(fs) < n:        #fs = features
        l = len(f[0])
        a = l-1
        ind = randrange(a)
        if ind not in fs:
            fs = fs + [ind]
            
    for j in range(len(fs)):
        ind = fs[j]
        for k in range(len(f)):
            x = f[k]
            u = []
            v = []
            for ax in f:
                if ax[ind] < x[ind]:
                    u = u + [ax]
                else:
                    v = v + [ax]
            
            gps = (u,v)
            gini_ind = 0.0
            
            for m in range(len(class_values)):
                k = class_values[m]
                for mn in range(len(gps)):
                    gp = gps[mn]
                    a = []
                    s = len(gp)
                    if s == 0:
                        continue
                    for y1 in range(len(gp)):
                        y = gp[y1]
                        a = a + [y[-1]]
                    b = a.count(k)
                    prop = b / float(s)
                    prop_s = (1.0 - prop)
                    ab = (prop * prop_s)
                    gini_ind = gini_ind + ab
    
            g = gini_ind
            #g = gini_ind(gps, class_values)
            if g < bsc:
                bind = ind
                bval = x[ind]
                bsc = g
                b_gps = gps
    return {'index':bind, 'value':bval, 'groups':b_gps}

def terminal_nd1(gp):
    r = []
    for i in range(len(gp)):
        x = gp[i]
        r = r + [x[-1]]
    a = set(r)
    return max(a, key=r.count)

def divide(nd1, md, ms, n, d):
    g = 'groups'
    lt = 'left'
    rt = 'right'
    
    l, r = nd1[g]
    del(nd1[g])
    
    if not l or not r:
        nd1[lt] = terminal_nd1(l + r)
        nd1[rt] = terminal_nd1(l + r)
        return
   
    if d >= md:
        nd1[lt] = terminal_nd1(l)
        nd1[rt] = terminal_nd1(r)
        return
   
    if len(l) > ms:
        nd1[lt] = seperate(l,n)
        divide(nd1[lt], md, ms,n, d+1)
    else:
        nd1[lt] = terminal_nd1(l)
   
    if len(r) > ms:
        nd1[rt] = seperate(r,n)
        divide(nd1[rt], md, ms,n, d+1)    
    else:
        nd1[rt] = terminal_nd1(r)

def prediction(nd, x):
    v = 'value'
    i = 'index'
    l = 'left'
    r = 'right'
    d = dict
    if nd[v] > x[nd[i]]:
        if isinstance(nd[l], d):
            return prediction(nd[l], x)
        else:
            return nd[l]
    else:
        if isinstance(nd[r], d):
            return prediction(nd[r], x)
        else:
            return nd[r]

def random_forest(x_t, y_t, md, ms, ss, n_t, n):
    t = []
    ij = 0
    while ij < n_t:
        ij = ij+1
        s1 = []
        l = ((len(x_t)) * ss)
        n1 = round(l)          #n = n_sample
        while len(s1) < n1:
            a = len(x_t)
            j = randrange(a)
            s1 = s1 + [x_t[j]]
        s = s1
        
        r1 = seperate(s, n)
        divide(r1, md, ms, n, 1)
        t1 = r1
        t = t + [t1]
    pred = []
    for x in y_t:
        p = []
        for x1 in t:
            p = p + [prediction(x1,x)]
        c = set(p)
        bag = max(c,key = p.count)
        pred = pred + [bag]
    return(pred)


seed(1101)
data = []
file_path = 'C:/Project/hou_all.csv'
with open(file_path, 'r') as f1:
    file1 = reader(f1)
    for x in file1:        #row = x
        data  = data + [x]
f1 = data
a = len(f1)

f = f1[1:a]

length = len(f[0])

for i in range(0,length-1):
    float_conversion(f,i)
    
int_conversion(f, length-1)

x = []
y = []
z = []

no_of_fds = 5
size = 1
sample_size_tree = 1.0

m = sqrt(length-1)
no_of_features = int(m)

for no_of_trees in [1,2,5,10,20]:
    for depth in [6]: 
        x = x + [no_of_trees]
        accuracy = tree(f, random_forest, no_of_fds, depth, size, sample_size_tree, no_of_trees, no_of_features)
        print('Number of trees : %d' % no_of_trees)
        print('Depth: %d' % depth)
        print('Cross Validation Scores: %s' % accuracy)
        a = float(len(accuracy))
        b = sum(accuracy)
        s = (b/a)
        print('Mean Accuracy of Model: %.2f%%' % s)
        y = y + [s]

plt.plot(x,y)
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')
