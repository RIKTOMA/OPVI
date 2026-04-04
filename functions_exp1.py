import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as mat
import time
import math
from scipy.special import lambertw


#algortihm 
def alg_real(f_esterna,G_interna,y0,param_f=[],param_G=[],vincolo="cubo",param_C=[0,1],alpha=1.5,rho=10,epsilon=10**-1,n=5,L=10,tol=10**-5):
    def G(x):
        if len(param_G)>0:
            return G_interna(x,param_G)
        else:
            return G_interna(x)
    def f(x):
        if len(param_f)>0:
            return f_esterna(x,param_f)
        else:
            return f_esterna(x)
            
    if vincolo=="cubo":
        vu=param_C[1]
        vl=param_C[0]
        D=np.sqrt(n)*(vu-vl)
    if vincolo=="sfera":
        R=param_C
        D=2*np.sqrt(R)
    if vincolo=="simplesso":
        tot=param_C
        D=tot*np.sqrt(2)
        
    #Funzione Gap minty con k-tagli        
    def gapM(Y,z,F):
        Z=[F(y)@(z-y).T for y in Y]
        return np.max(Z)

    #Funzione Gap Stampacchia sul cubo [0,1]^n
    def gapS(z,F,n):
        ms=gp.Model("stampacchia")
        ms.setParam("OutPutFlag",0)
        ms.Params.LogToConsole = 0
        y=ms.addMVar(shape=n,lb=-100, name="y")
        if vincolo=="cubo":
            for i in range(n):
                ms.addConstr(y[i] <=vu, "a"+str(i))
                ms.addConstr(y[i]>=vl, "b"+str(i))
        if vincolo=="sfera":
            ms.addConstr(y@y<=R, "s")
        if vincolo=="simplesso":
            ms.addConstr(y>=0, "s1")
            ms.addConstr(np.ones(n).T@y<=tot, "s2")
        ms.setObjective(F(z)@z - F(z)@y, GRB.MAXIMIZE)
        ms.optimize()
        return (ms.ObjVal, y.X)

    #Funzione Gap Stampacchia sul cubo [0,1]^n
    def gapS1(z,F,Y,Eps,n):
        ms=gp.Model("stampacchia1")
        ms.Params.LogToConsole = 0
        y=ms.addMVar(shape=n,lb=-100, name="y")
        if vincolo=="cubo":
            for i in range(n):
                ms.addConstr(y[i] <=vu, "a"+str(i))
                ms.addConstr(y[i]>=vl, "b"+str(i))
        if vincolo=="sfera":
            ms.addConstr(y@y<=R, "s")
        if vincolo=="simplesso":
            ms.addConstr(y>=0, "s1")
            ms.addConstr(np.ones(n).T@y<=tot, "s2")
        for l in range(len(Y)):
            ms.addConstr(Eps[l]>=F(Y[l])@y-F(Y[l])@F(Y[l]))
        ms.setObjective(F(z)@z - F(z)@y, GRB.MAXIMIZE)
        ms.optimize()
        return (ms.ObjVal, y.X)

    #Funzione Gap Minty   sul cubo [0,1]^n, caso affine
    def gapMA(z,F,n):
        ms=gp.Model("minty_aff")
        ms.Params.LogToConsole = 0
        y=ms.addMVar(shape=n,lb=-100, name="y")
        if vincolo=="cubo":
            for i in range(n):
                ms.addConstr(y[i] <=vu, "a"+str(i))
                ms.addConstr(y[i]>=vl, "b"+str(i))
        if vincolo=="sfera":
            ms.addConstr(y@y<=R, "s")
        if vincolo=="simplesso":
            ms.addConstr(y>=0, "s1")
            ms.addConstr(np.ones(n).T@y<=tot, "s2")
        ms.setObjective(F(y)@z - F(y)@y, GRB.MAXIMIZE)
        ms.optimize()
        return (ms.ObjVal, y.X)

    #Funzione "distanza"
    def dist(F,x,y,lam,eps):
        ylam=(1-lam)*x + lam*y
        zlam=F(ylam)
        ris=(zlam@(x-ylam)-eps)/np.sqrt(1+zlam@zlam)
        #ris=(zlam@(x-ylam)-eps)
        return ris

    #Funzione "distanza" con norma 1
    def dist1(F,x,y,lam,eps):
        ylam=(1-lam)*x + lam*y
        zlam=F(ylam)
        u=np.append(zlam,1)
        v=np.append(x-ylam,-eps)
        ris=(u@v)/(u@u)*np.linalg.norm(u,1)
        return ris
    
    #trovare minimi lacoli con Golden search
    def gss(F,x,y,eps,tolerance=10**-2):
        b=1
        a=0
        invphi=(np.sqrt(5)-1)/2
        it=0
        while b - a > tolerance:
            c = b - (b - a) * invphi
            d = a + (b - a) * invphi
            if dist(F,x,y,c,eps) > dist(F,x,y,d,eps):
                b = d
            else:
                a = c
            it=it+1
        return [(b + a) / 2,it]

    
    l=np.sqrt(epsilon/L)/D
    E=2*D*np.sqrt(L*epsilon)
    
    #creo modello
    m=gp.Model("prob_penalizzato")
    m.setParam("OutPutFlag",0)
    m.Params.LogToConsole = 0

    #definisco le variabili
    x=m.addMVar(shape=n,lb=-100, name="x")
    eta=m.addVar(lb=0,name="eta")

    #Insieme C
    if vincolo=="cubo":
        for i in range(n):
            m.addConstr(x[i] <=vu, "a"+str(i))
            m.addConstr(x[i]>=vl, "b"+str(i))
    if vincolo=="sfera":
        m.addConstr(x@x<=R, "s")
    if vincolo=="simplesso":
        m.addConstr(x>=0, "s1")
        m.addConstr(np.ones(n).T@x<=tot, "s2")


    #punto di parenza
    Y=[]
    Y.append(y0)
    xmin=y0
    ynew=y0
    yold=y0
    cutmax=2000
    
    #Aggiungere un vincolo
    m.addConstr(eta >= G(y0)@x - G(y0)@y0 - epsilon, "c")
    Eps=[epsilon]

    
    #algoritmo
    cons={}
    j=0
    delta=0.95
    beta=0.05
    delta_k=delta
    lteor=np.sqrt(epsilon/L)/D
    opt=E
    new_eps=epsilon
    
    t=1
    s=1
    vero=0
    inizio=time.time()
    for i in range(10000):
        if rho>10**8:
            print(opt-E)
            break
        m.setObjective(f(x)+rho*eta)
        m.optimize()
        xmin=x.X
        valmin=m.ObjVal
        #print("valore della f "+str(valmin))
        psik=gapM(Y,xmin,G)
        if psik-new_eps>tol:
            rho=alpha*rho
        else:
            (opt,yopt)=gapS(x.X,G,n)
            #(opt,yopt)=gapS1(x.X,G,Y,Eps,n)
            [newl,it]=gss(G,xmin,yopt,psik)
            ynew=(1-newl)*xmin + newl*yopt
            yold=ynew
            #print(G(ynew)@(xmin-ynew)-new_eps)
            #condizione di Stop
            if G(ynew)@(xmin-ynew)-new_eps<tol:
                #print("Gap di stampacchia "+str(opt))
                #print("Nuovo epsilon "+str(new_eps))
                ynew=(1-lteor)*xmin + lteor*yopt
                if opt<E:
                    break
            if new_eps<10**(-5):
                break
            if opt<epsilon+tol:
                break
                
            jnew=j%cutmax
            if j>=cutmax:
                m.remove(cons["c"+str(jnew)])
                m.update()
            cons["c"+str(jnew)]=m.addConstr(eta >= G(ynew)@x - G(ynew)@ynew - new_eps)
            j=j+1
            Y.append(ynew)
            #Eps.append(new_eps)
    fine=time.time()        
    #print("Algoritmo realistico di dimensione "+str(n))
    #print("numero di incrementi di rho="+str(i-j),"rho="+str(rho))
    #print("numero di tagli="+str(j))
    #print("tempo di esecuzione "+str(fine-inizio))
    #print("Errore teorico "+str(E))
    #print("Gap di Stampacchia "+str(opt))
    #print(Eps)
    #return (xmin,valmin,j,gapMA(xmin,G,n)[0],E,fine-inizio,opt)
    return [xmin,valmin,j,0,E,fine-inizio,opt,i-j]

#Functions
def random_sA(n,aut):
    DIAG=np.diag(aut)
    Q=np.linalg.qr(np.random.random_sample((n,n)))[0]
    A=Q.T@DIAG@Q
    return A

def random_nsA(n,k):
    Q11=np.linalg.qr(np.random.normal(size=[n-k,n-k]))[0]
    Q=np.zeros([n,n])
    Q[0:n-k,0:n-k]=Q11

    iters=n-k
    v=np.zeros([iters,iters])
    d=np.random.random_sample(size=[iters])*n+1
    d=np.sort(d)
    d=np.flip(d)
    c=0
    for i in range(iters):
        if i>0:
            c=sum(abs(v[i,:i-1]))
        if d[i]-c<0:
            d[i]=(d[i]+c)*(1+np.random.normal()**2)
        tot=d[i]-c
        b=0
        w=np.zeros(iters-(i+1))
        for j in range(i+1,iters):
            a=np.random.random_sample()*(tot-b)/(j+2)
            w[j-(i+1)]=a*np.random.choice([-1,1])
            b=b+a
        np.random.shuffle(w)
        v[i,i]=d[i]
        v[i,i+1:]=w
        v[i+1:,i]=-w.T

    A11=Q11@v@Q11.T
    A=np.zeros([n,n])
    A[0:n-k,0:n-k]=A11
    return(A)

def random_nsA1(n,k):
    Q11=np.linalg.qr(np.random.normal(size=[n-k,n-k]))[0]
    Q=np.zeros([n,n])
    Q[0:n-k,0:n-k]=Q11

    iters=n-k
    v=np.zeros([iters,iters])
    d=np.random.random_sample(size=[iters])*n
    D=np.diag(d)
    H=np.random.normal(size=[iters,iters])*10
    T=(H-H.T)/2

    A11=Q11@(D+T)@Q11.T
    A=np.zeros([n,n])
    A[0:n-k,0:n-k]=A11
    return A
    
#Creo le funzioni
def f_linear(y,a):
    return a@y
    
def f_quadratic(y,param_f):
    A=param_f[0]
    b=param_f[1]
    c=param_f[2]
    return y.T@A@y + b.T@y +c

def gradf_quadratic(y,param_f):
    A=param_f[0]
    b=param_f[1]
    c=param_f[2]
    return (A+A.T)@y + b
    
def G_linear(y,A):
    return A@y

def G_affine(y,param_G):
    A=param_G[0]
    b=param_G[1]
    return A@y+b

def G_exp(y,param_G):
    A=param_G[0]
    b=param_G[1]
    c=param_G[2]
    d=param_G[3]
    z=c*np.exp(d*y)
    return A@y+b+z

def W(c,d):
    return c*d*np.exp(d)

def inner_prob(n,k,L,p,normb):
    c_coef=4
    c_offset=1
    d_coef=1
    d_offset=0.5
    A=random_nsA1(n,k)
    L_A=np.linalg.norm(A,2)
    d=np.random.random_sample((n,))*d_coef+d_offset
    c=np.random.random_sample((n,))*c_coef+c_offset
    
    c[-k:]=0
    c_max=np.max(c)
    d_max=np.max(d)
    si=np.argmax(W(c,d))
    gamma =abs(lambertw(L/((p+1)*c[si])))/d[si]
    while np.argmax(W(c,gamma*d))!=si:
        d=np.random.random_sample((n,))*d_coef+d_offset
        c=np.random.random_sample((n,))*c_coef+c_offset
        c[-k:]=0
        c_max=np.max(c)
        d_max=np.max(d)
        si=np.argmax(W(c,d))
        gamma =abs(lambertw(L/((p+1)*c[si])))/d[si]
        for i in range(len(c)):
            if c[i]!=0 and d[i]!=0 and d[i]-d[si]!=0:
                gamma_bar=np.log(c[si]*d[si]/(c[i]*d[i]))/(d[i]-d[si])
                if gamma<gamma_bar:
                    break 
    b=np.random.random_sample((n,))*2-1
    b[-k:]=0
    b=b/np.linalg.norm(b,2)*normb
    #b[-k:]=0
    d=d*gamma
    theta=p*W(c[si],d[si])/L_A
    A=A*theta
    L_A=np.linalg.norm(A,2)
    L_e=np.max(W(c,d))
    return [A,b,c,d,L_A,L_e]


#Solving numerical experiments
def esp(n,k,L,vincolo,param_C,iters,epsilon,rho,alpha,normb,seed=0):
    np.random.seed(seed)
    risultati=np.zeros((iters,7))
    for it in range(iters):
        #Genero f quadratica
        x_rand=np.random.random_sample(n,)*10-5
        Af=random_sA(n,np.random.random_sample((n,))*10)
        bf=-2*x_rand.T@Af
        cf=x_rand.T@Af@x_rand

        #genero di G
        q=np.random.random_sample()*4-2
        p=2**q
        A,b,c,d,L_A,L_e=inner_prob(n,k,L,p,normb)
        if vincolo=="cubo":
            [vl,vu]=param_C
            y0=(vu-vl)*np.array(np.random.random_sample((n,))) + vl
        if vincolo=="sfera":
            R=param_C
            z0=np.array(np.random.normal(size=(n,)))
            y0=R*z0/np.linalg.norm(z0,2)
        if vincolo=="simplesso":
            tot=param_C
            z0=np.array(np.random.normal(size=(n,)))
            y0=tot*abs(z0)/np.linalg.norm(z0,1)
        ris=alg_real(f_quadratic,G_exp,y0,[Af,bf,cf],[A,b,c,d],vincolo,param_C,alpha,rho,epsilon,n,L)
        #tempo esecuzione
        risultati[it][0]=ris[5]
        #numero di incrementi di rho
        risultati[it][1]=ris[7]
        #rho finale
        risultati[it][2]=rho*alpha**ris[7]
        #numero di tagli
        risultati[it][3]=ris[2]
        #stampacchia finale
        risultati[it][4]=ris[6]
        #errore teorico
        risultati[it][5]=ris[4]
        #rapporto gap stampacchia errore teorico 
        risultati[it][6]=ris[6]/ris[4]
    return risultati

#parametri is a dictionary with input:n,k,L,vincolo,param_c,iters,epsilon,rho,alpha
def tabelle(parametri,seed=0):
    tabs=np.zeros((len(parametri),7))
    j=0
    for ind in parametri:
        n,k,L,vincolo,iters,epsilon,rho,alpha,normb=parametri[ind]
        if vincolo=="cubo":
            param_c=[0,1]
        if vincolo=="sfera" or vincolo=="simplesso":
            param_c=1
        ris=esp(n,k,L,vincolo,param_c,iters,epsilon,rho,alpha,normb,seed)
        tabs[j]=np.mean(ris,0)
        #tabs[j][-1]=np.var(ris.T[4])
        j=j+1
    return tabs

def pytolatex(matrix):
    dims=np.shape(matrix)
    testo="&"
    for i in range(dims[0]):
        for j in range(dims[1]):
            if j<dims[1]-1:
                testo=testo+str(matrix[i][j])+"&"
            else:
                testo=testo+str(matrix[i][j])+"\\"+"\\"+"\n"+"&"
    return testo
