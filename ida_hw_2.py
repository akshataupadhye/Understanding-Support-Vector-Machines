#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


file=pd.read_excel('D:/fall_2018/ida/assignments/HW2-Synth-Data.xls')


# In[3]:


file


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


plt.figure(figsize=(20,20))
plt.scatter(file.X, file.Y, c=file.CLASS)
plt.show()


# In[60]:


X = file.drop(['CLASS'], axis=1)


# In[61]:


Y = file['CLASS']


# In[8]:


from sklearn import tree


# In[9]:


from sklearn.model_selection import cross_val_score


# In[10]:


dc_tree1 = tree.DecisionTreeClassifier(random_state = 42)


# In[11]:


scores=cross_val_score(dc_tree1, X, Y, cv=5)


# In[12]:


scores


# In[13]:


avg= scores.mean()


# In[14]:


avg


# In[ ]:





# In[15]:


dc_tree1.fit(X,Y)


# In[16]:


predict = dc_tree1.predict(X)


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


# Parameters
n_classes = 2
plot_colors = "ryb"
plot_step = 0.02

plt.figure(figsize=(20,15))

clf = dc_tree1


for pairidx, pair in enumerate([[0, 1]]):
    # We only take the two corresponding features
    X = np.array(file.drop(['CLASS'], axis=1))[:,pair]
    y = file['CLASS']

    # Train
    clf.fit(X, Y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel("X")
    plt.ylabel("Y")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=i,
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()


# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


# Parameters
n_classes = 2
plot_colors = "ryb"
plot_step = 0.02

plt.figure(figsize=(20,15))

clf = dc_tree1


for pairidx, pair in enumerate([[0, 1]]):
    # We only take the two corresponding features
    X = np.array(file.drop(['CLASS'], axis=1))[:,pair]
    y = file['CLASS']

    # Train
    clf.fit(X, Y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel2)

    plt.xlabel("X")
    plt.ylabel("Y")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=i,
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()


# In[49]:


x= X.tolist()


# In[50]:


x


# In[51]:


y= Y.tolist()


# In[52]:


y


# In[63]:


def dec_tree_predict(file):
    if(file.X <= 9.8):
        if(file.Y >= -3.2):
            return 1
        else:
            if(file.X <=4.5):
                return 0
            else:
                if file.Y <= -4.3:
                    return 1
                else:
                    return 0
    else:
        return 0


# In[64]:


preds = []
for i in range(X.count().X):
    preds.append(dec_tree_predict(X.iloc[i]))


# In[65]:


preds


# In[66]:


from sklearn.metrics import accuracy_score


# In[67]:


accuracy_score(Y,preds)


# In[68]:


from sklearn.metrics import confusion_matrix


# In[69]:


confusion_matrix(Y, preds)


# In[70]:


from sklearn.metrics import classification_report


# In[71]:


print (classification_report(Y, preds))


# In[24]:


from sklearn.svm import LinearSVC


# In[25]:


svm_1 = LinearSVC(random_state = 42)


# In[26]:


scores=cross_val_score(svm_1, X, Y, cv=5)


# In[27]:


scores


# In[28]:


avg= scores.mean()


# In[29]:


avg


# In[57]:


from sklearn.model_selection import cross_validate


# In[58]:


cross_validate(svm_1, X, Y, cv=5)


# In[ ]:





# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# we create 40 separable points
np.random.seed(0)
X = np.array( file.drop(['CLASS'], axis=1))
Y = file.CLASS

# figure number
fignum = 1

# fit the model
for name, penalty in (('unreg', 1), ('reg', 0.05)):

    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)

    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
    # 2-d.
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy - np.sqrt(1 + a ** 2) * margin
    yy_up = yy + np.sqrt(1 + a ** 2) * margin

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = 0
    x_max = 100
    y_min = 0
    y_max = 100

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1

plt.show()


# In[ ]:





# In[ ]:





# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
#X, Y = make_blobs(n_samples=450, centers=2, random_state=6)
X = np.array( file.drop(['CLASS'], axis=1))
y = file.CLASS


# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, Y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()


# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
#X, Y = make_blobs(n_samples=450, centers=2, random_state=6)
X = np.array( file.drop(['CLASS'], axis=1))
y = file.CLASS


# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=500)
clf.fit(X, Y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()


# In[33]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
#X, Y = make_blobs(n_samples=450, centers=2, random_state=6)
X = np.array( file.drop(['CLASS'], axis=1))
y = file.CLASS


# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='rbf', C=1, gamma=0.1)
clf.fit(X, Y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()


# In[ ]:





# In[ ]:




