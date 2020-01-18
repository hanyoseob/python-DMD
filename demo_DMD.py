## import libraries

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy import linalg

sech = lambda x: 1/np.cosh(x)
tanh = lambda x: np.tanh(x)
exp = lambda x: np.exp(x)
transpose = lambda x: np.transpose(x)
hermitian = lambda x: np.conj(np.transpose(x))

## generate grid gemoetry
xi = np.linspace(-10, 10, 400)
t = np.linspace(0, 4*np.pi, 200)

dt = t[1] - t[0]

[xgrid, tgrid] = np.meshgrid(xi, t)

## create two spatio-temporal patterns
# f1 in complex [200, 400]
# f2 in complex [200, 400]
f1 = sech(xgrid + 3) * (1*exp(1j*2.3*tgrid))
f2 = (sech(xgrid) * tanh(xgrid)) * (2*exp(1j*2.8*tgrid))

f = f1 + f2

# singular value decomposition
# if f in [M, N],
# u in [M, M]
# s in [N, ]
# v in [N, N]
[m, n] = transpose(f).shape
[u, sdiag, vh] = linalg.svd(transpose(f), full_matrices=True)
s = np.zeros((m, n), dtype=sdiag.dtype)
s[:n, :n] = np.diag(sdiag)
v = hermitian(vh)

## plot
# f1, f2, f
fig = plt.figure(1)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(xgrid, tgrid, np.real(f1), cmap=plt.cm.Greys)
plt.title('f1(x, t)'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot_surface(xgrid, tgrid, np.real(f2), cmap=plt.cm.Greys)
plt.title('f2(x, t)'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot_surface(xgrid, tgrid, np.real(f), cmap=plt.cm.Greys)
plt.title('f(x, t) = f1(x, t) + f2(x, t)'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

plt.show()

# S of SVD
fig = plt.figure(2)
plt.plot(np.diag(s) / sum(np.diag(s)), 'ro')
plt.title('SVD: low rank property (rank = 2, two modes)')

plt.show()

# U and V of SVD
fig = plt.figure(3)
ax = fig.add_subplot(2, 1, 1)
plt.legend(ax.plot(np.real(u[:, 0:3])),
           ['1st mode of basis u',
            '2nd mode of basis u',
            '3rd mode; numerical round off'])
plt.title('Modes for basis of column space of ''f'' ')

ax = fig.add_subplot(2, 1, 2)
plt.legend(ax.plot(np.real(v[:, 0:3])),
           ['1st mode of basis v',
            '2nd mode of basis v',
            '3rd mode; numerical round off'])
plt.title('Modes for basis of row space of ''f'' ')

plt.show()

## dynamic mode decomposition (DMD)
# Linear system
X = transpose(f)

X1 = X[:, :-1]
X2 = X[:, 1:]

## STEP 1: singular value decomposition (SVD)
r = 2   # rank-r truncation
[m, n] = X1.shape
[U, Sdiag, Vh] = linalg.svd(X1, full_matrices=False)
S = np.zeros((m, n), dtype=Sdiag.dtype)
S[:n, :n] = np.diag(Sdiag)
V = hermitian(Vh)

Ur = U[:, :r]
Sr = S[:r, :r]
Vr = V[:, :r]

## STEP 2: low-rank subspace matrix
# (similarity transform, least-square fit matrix, low-rank subspace matrix)
# Atilde = np.dot(np.dot(np.dot(Ur.transpose(), X2), Vr), linalg.inv(Sr))
Atilde = np.dot(hermitian(Ur), np.dot(X2, np.dot(Vr, linalg.inv(Sr))))

## STEP 3: eigen decomposition
# W: eigen vectors
# D: eigen values
[m, n] = Atilde.shape
[Ddiag, W] = linalg.eig(Atilde)
D = np.zeros((m, n), dtype=Ddiag.dtype)
D[:n, :n] = np.diag(Ddiag)

## STEP 4: real space DMD mode
Phi = np.dot(X2, np.dot(Vr, np.dot(linalg.inv(Sr), W)))

sgm = np.diag(D)
omega = np.log(sgm)/dt

fig = plt.figure(4)
ax = fig.add_subplot(2, 1, 1)
plt.legend(ax.plot(np.real(u[:, 0:2])),
           ['1st mode of SVD',
            '2nd mode of SVD'])
plt.title('Modes for SVD ')

ax = fig.add_subplot(2, 1, 2)
plt.legend(ax.plot(np.real(Phi[:, 0:2])),
           ['1st mode of DMD',
            '2nd mode of DMD'])
plt.title('Modes for DMD')

plt.show()

## STEP 5: reconstruct the signal
x1 = X[:, 0]    # time = 0
b = np.dot(linalg.pinv(Phi), x1)

t_dyn = np.zeros((r, len(t)), dtype=X.dtype)

for i in range(len(t)):
    t_dyn[:, i] = b * np.exp(omega*t[i])

f_dmd = np.dot(Phi, t_dyn)

# f1, f2, f, f_rec
fig = plt.figure(1)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(xgrid, tgrid, np.real(f1), cmap=plt.cm.Greys)
plt.title('f1(x, t)'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot_surface(xgrid, tgrid, np.real(f2), cmap=plt.cm.Greys)
plt.title('f2(x, t)'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot_surface(xgrid, tgrid, np.real(f), cmap=plt.cm.Greys)
plt.title('f(x, t) = f1(x, t) + f2(x, t)'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot_surface(xgrid, tgrid, np.real(transpose(f_dmd)), cmap=plt.cm.Greys)
plt.title('f_dmd(x, t); DMD reconstruction'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

plt.show()

## additional informations
# predict furture state using time dynamics
t_ext = np.linspace(0, 8*np.pi, 400)

[xgrid_ext, tgrid_ext] = np.meshgrid(xi, t_ext)

t_ext_dyn = np.zeros((r, len(t_ext)), dtype=X.dtype)

##
for i in range(len(t_ext)):
    t_ext_dyn[:, i] = b * np.exp(omega * t_ext[i])

f_dmd_ext = np.dot(Phi, t_ext_dyn)

# f1, f2, f, f_rec
fig = plt.figure(1)
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot_surface(xgrid, tgrid, np.real(f), cmap=plt.cm.Greys)
plt.title('f(x, t) = f1(x, t) + f2(x, t)'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot_surface(xgrid, tgrid, np.real(transpose(f_dmd)), cmap=plt.cm.Greys)
plt.title('f_dmd(x, t); DMD reconstruction'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot_surface(xgrid_ext, tgrid_ext, np.real(transpose(f_dmd_ext)), cmap=plt.cm.Greys)
plt.title('f_dmd(x, t); DMD prediction'), plt.xlabel('x (spatial)'), plt.ylabel('t (temporal)')

plt.show()