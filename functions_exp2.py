import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as mat
import time
import math
from scipy.special import lambertw
from scipy.optimize import direct
import gc
import random


def alg_real_grad(f_esterna,gradf_esterna,G_interna,y0,param_f=[],param_G=[],vincolo="cubo",param_C=[0,1],alpha=1.5,rho=10,epsilon=10**-1,N=4,V=3,L=10,tol=10**-5,MAXTIME=100):
    n=2*N*V
    
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
    def gradf(x):
        if len(param_f)>0:
            return gradf_esterna(x,param_f)
        else:
            return gradf_esterna(x)  
            
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


    
    
    #bound specifici del problema
    v=np.zeros([N,2*N*V])
    for i in range(N):
        v[i,i*V:(i+1)*V]=np.ones(V)
        v[i,(N+i)*V:(N+i+1)*V]=-np.ones(V)

    
    #Funzione Gap minty con k-tagli        
    def gapM(Y,z,F):
        Z=[F(y)@(z-y).T for y in Y]
        return np.max(Z)
        
    #SubGradiente Gap Minty con k-tagli
    def grad_gapM(Y,z,F,n,eps):
        Z=[F(y)@(z-y).T for y in Y]
        val=np.max(Z)
        if val<=0:
            return 0
        if val>0:
            return F(Y[np.argmax(Z)]) 
        
        return np.max(Z)

     #Trovare i minimi con direct
    def dist1(l,F,y,x):
        #F=args[0]
        #y=args[1]
        #x=args[2]
        ylam=(1-l)*x + l*y
        zlam=F(ylam)
        #ris=(zlam@(x-ylam)-eps)/np.sqrt(1+zlam@zlam)
        ris=zlam@(ylam-x)
        return ris
        
    #Funzione Gap Stampacchia sul cubo [0,1]^n
    def gapS(z,F,n):
        ms=gp.Model("stampacchia")
        ms.setParam('OutputFlag', 0)
        ms.Params.LogToConsole = 0
        y=ms.addMVar(shape=n,lb=0, name="y")
        if vincolo=="cubo":
            for i in range(N*V):
                ms.addConstr(y[i] <=vu, "a"+str(i))
                ms.addConstr(y[i]>=vl, "b"+str(i))
        if vincolo=="sfera":
            ms.addConstr(y@y<=R, "s")
        if vincolo=="simplesso":
            ms.addConstr(y>=0, "s1")
            ms.addConstr(np.ones(n).T@y<=tot, "s2")
        for i in range(N):
            ms.addConstr(y@v[i]==0, "c"+str(i))
        ms.setObjective(F(z)@z - F(z)@y, GRB.MAXIMIZE)
        ms.optimize()
        return (ms.ObjVal, y.X)



    #Funzione "distanza"
    def dist(F,x,y,lam,eps):
        ylam=(1-lam)*x + lam*y
        zlam=F(ylam)
        #ris=(zlam@(x-ylam)-eps)/np.sqrt(1+zlam@zlam)
        ris=zlam@(x-ylam)
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
            d1=dist(F,x,y,c,eps)
            d2=dist(F,x,y,d,eps)
            if d1 > d2:
                b = d
            else:
                a = c
            
            it=it+1
        return [(b + a) / 2,it]

    def proj(z,n):
        ms=gp.Model("projection")
        ms.Params.LogToConsole = 0
        y=ms.addMVar(shape=n,lb=0, name="y")
        if vincolo=="cubo":
            for i in range(N*V):
                ms.addConstr(y[i] <=vu, "a"+str(i))
                ms.addConstr(y[i]>=vl, "b"+str(i))
        if vincolo=="sfera":
            ms.addConstr(y@y<=R, "s")
        if vincolo=="simplesso":
            ms.addConstr(y>=0, "s1")
            ms.addConstr(np.ones(n).T@y<=tot, "s2")
        for i in range(N):
            ms.addConstr(y@v[i]==0, "c"+str(i))
        ms.setObjective(z@z+y@y-2*z@y, GRB.MINIMIZE)
        ms.optimize()
        return (ms.ObjVal, y.X)
    #Calcolo D
    m1=gp.Model("calcolo Diametro")
    m1.setParam('OutputFlag', 0)
    m1.Params.LogToConsole = 0


    #Funzione Gap Stampacchia sul cubo [0,1]^n
    ms=gp.Model("stampacchia")
    ms.setParam('OutputFlag', 0)
    ms.Params.LogToConsole = 0
    ys=ms.addMVar(shape=n,lb=0, name="ys")
    if vincolo=="cubo":
        for i in range(N*V):
            ms.addConstr(ys[i] <=vu, "a"+str(i))
            ms.addConstr(ys[i]>=vl, "b"+str(i))
    if vincolo=="sfera":
        ms.addConstr(ys@ys<=R, "s")
    if vincolo=="simplesso":
        ms.addConstr(ys>=0, "s1")
        ms.addConstr(np.ones(n).T@ys<=tot, "s2")
    for i in range(N):
        ms.addConstr(ys@v[i]==0, "c"+str(i))

    
    #definisco le variabili
    x1=m1.addMVar(shape=n,lb=0, name="x1")
    y1=m1.addMVar(shape=n,lb=0, name="y1")
    #Insieme C
    if vincolo=="cubo":
        for i in range(N*V):
            m1.addConstr(x1[i] <=vu, "ax"+str(i))
            m1.addConstr(x1[i]>=vl, "bx"+str(i))
            m1.addConstr(y1[i] <=vu, "ay"+str(i))
            m1.addConstr(y1[i]>=vl, "by"+str(i))
    if vincolo=="sfera":
        m1.addConstr(x1@x1<=R, "s")
    if vincolo=="simplesso":
        m.addConstr(x1>=0, "s1")
        m.addConstr(np.ones(n).T@x1<=tot, "s2")

    for i in range(N):
        m1.addConstr(x1@v[i]==0, "c"+str(i))
        m1.addConstr(y1@v[i]==0, "c"+str(i))
    m1.setObjective((x1-y1)@(x1-y1),GRB.MAXIMIZE)
    m1.optimize()
    D=np.sqrt(m1.ObjVal)
    l=np.sqrt(epsilon/L)/D
    E=2*D*np.sqrt(L*epsilon)
    
    #creo modello
    m=gp.Model("prob_penalizzato")
    m.setParam('OutputFlag', 0)
    m.Params.LogToConsole = 0

    #definisco le variabili
    x=m.addMVar(shape=n,lb=0, name="x")
    myAuxVar = m.addVars(V, vtype=GRB.CONTINUOUS, lb=0, name="myAuxVar" )
    t = m.addVars(V, vtype=GRB.CONTINUOUS, name="t" )
    eta=m.addVar(lb=0,name="eta")
    #Insieme C
    if vincolo=="cubo":
        for i in range(N*V):
            m.addConstr(x[i] <=vu, "a"+str(i))
            m.addConstr(x[i]>=vl, "b"+str(i))
    if vincolo=="sfera":
        m.addConstr(x@x<=R, "s")
    if vincolo=="simplesso":
        m.addConstr(x>=0, "s1")
        m.addConstr(np.ones(n).T@x<=tot, "s2")

    for i in range(N):
        m.addConstr(x@v[i]==0, "c"+str(i))
    #punto di parenza
    Y=[]
    Y.append(y0)
    xmin=y0
    ynew=y0
    yold=y0
    cutmax=10000
    
    #Aggiungere un vincolo
    m.addConstr(eta >= G(y0)@x - G(y0)@y0 - epsilon, "c")
    
    #algoritmo
    cons={}
    j=0
    lteor=np.sqrt(epsilon/L)/D
    (opt,yopt)=gapS(y0,G,n)
    new_eps=epsilon
    gamma=0.05
    xmin=y0
    ymin=y0
    valmin=f(xmin)
    val=[]
    val_gap=[]
    tempi=[]
    tempo_esterna=[]
    proxim=1
    inizio=time.time()
    mold=10000
    for i in range(10000):
        primo=time.time()
        
        for i1 in range(10000):
            gammak=gamma
            w=gradf(xmin)*gammak
            xold=xmin
            m.setObjective(rho*gammak*eta+1/2*(x-xmin+w)@(x-xmin+w))
            m.optimize()
            xmin=x.X
            #if abs(mold-m.ObjVal)<10**-3:
                #break
            #mold=m.ObjVal
            if np.sqrt((xold-xmin)@(xold-xmin))<10**-2:
                break
                #print(np.sqrt((xold-xmin)@(xold-xmin)))
                #print(m.ObjVal)
        #print("tempo di risoluzioner pblema sup", time.time()-primo,"numero iter",i1)
        tempo_esterna.append(time.time()-primo)
        valold=valmin
        valmin=f(xmin)
        val.append(valmin)
        val_gap.append(opt)
        tempi.append(time.time()-inizio)
        if time.time()-inizio>MAXTIME:
            break
        if eta.X>10**-10:
            rho=alpha*rho
        else:
            #(opt,yopt)=gapS(xmin,G,n)
            ms.setObjective((G(xmin)@xmin - G(xmin)@ys), GRB.MAXIMIZE)
            ms.optimize()
            yopt=ys.X
            opt=ms.ObjVal
            #print(opt)
            if  abs(valold-valmin)<10**-3 and opt<10**-2:
                val.append(valmin)
                val_gap.append(opt)
                tempi.append(time.time()-inizio)
                break
            #
            numb_inters=50
            ldir=direct(dist1,[(0,1)],args=(G,yopt,xmin),maxiter=numb_inters)
            newl=ldir.x
            #tempo_interna_lambda.append(time.time()-temp)
            ynew=(1-newl)*xmin + newl*yopt
            while (G(ynew)@xmin - G(ynew)@ynew-new_eps)<0:
                numb_inters=numb_inters+50
                ldir=direct(dist1,[(0,1)],args=(G,yopt,xmin),maxiter=numb_inters)
                newl=ldir.x
                #tempo_interna_lambda.append(time.time()-temp)
                ynew=(1-newl)*xmin + newl*yopt
                if numb_inters>200:
                    break
                    
            jnew=j%cutmax
            if j>=cutmax:
                m.remove(cons["c"+str(jnew)])
                m.update()
            cons["c"+str(jnew)]=m.addConstr(eta >= (G(ynew)@x - G(ynew)@ynew-new_eps))
            j=j+1
            Y.append(ynew)
        
    fine=time.time()        
    return [xmin,valmin,j,0,E,fine-inizio,opt,i-j,val,val_gap,tempi]


def alg_kayo(f_esterna,gradf_esterna,G_interna,y0,param_f=[],param_G=[],vincolo="cubo",param_C=[0,1],gamma0=1,eta0=1,r=0.5,epsilon=10**-1,N=4,V=3,L=10,tol=10**-5,MAXTIME=100):
    n=2*N*V
    
    def G(x):
        if len(param_G)>0:
            return G_interna(x,param_G)
        else:
            return G_interna(x)
    def gradf(x):
        if len(param_f)>0:
            return gradf_esterna(x,param_f)
        else:
            return gradf_esterna(x)

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
        
    v=np.zeros([N,2*N*V])
    for i in range(N):
        v[i,i*V:(i+1)*V]=np.ones(V)
        v[i,(N+i)*V:(N+i+1)*V]=-np.ones(V)

    #Proiezione
    mp=gp.Model("projection")
    mp.setParam('OutputFlag', 0)
    mp.Params.LogToConsole = 0
    yp=mp.addMVar(shape=2*N*V,lb=0, name="yp")
    if vincolo=="cubo":
        for i in range(N*V):
            mp.addConstr(yp[i] <=vu, "a"+str(i))
            mp.addConstr(yp[i]>=vl, "b"+str(i))
    if vincolo=="sfera":
        mp.addConstr(yp@yp<=R, "s")
    if vincolo=="simplesso":
        mp.addConstr(yp>=0, "s1")
        mp.addConstr(np.ones(n).T@yp<=tot, "s2")
    for i in range(N):
        mp.addConstr(yp@v[i]==0, "c"+str(i))
    


    #Funzione Gap Stampacchia
    ms=gp.Model("stampacchia")
    ms.setParam('OutputFlag', 0)
    ms.Params.LogToConsole = 0
    ys=ms.addMVar(shape=n,lb=0, name="ys")
    if vincolo=="cubo":
        for i in range(N*V):
            ms.addConstr(ys[i] <=vu, "a"+str(i))
            ms.addConstr(ys[i]>=vl, "b"+str(i))
    if vincolo=="sfera":
        ms.addConstr(ys@ys<=R, "s")
    if vincolo=="simplesso":
        ms.addConstr(ys>=0, "s1")
        ms.addConstr(np.ones(n).T@ys<=tot, "s2")
    for i in range(N):
        ms.addConstr(ys@v[i]==0, "c"+str(i))
    

    m=gp.Model("prob_penalizzato")
    m.setParam('OutputFlag', 0)
    m.Params.LogToConsole = 0

    #definisco le variabili
    x=m.addMVar(shape=n,lb=0, name="x")
    eta=m.addVar(lb=0,name="eta")

    #Insieme C
    if vincolo=="cubo":
        for i in range(N*V):
            m.addConstr(x[i] <=vu, "a"+str(i))
            m.addConstr(x[i]>=vl, "b"+str(i))
    if vincolo=="sfera":
        m.addConstr(x@x<=R, "s")
    if vincolo=="simplesso":
        m.addConstr(x>=0, "s1")
        m.addConstr(np.ones(n).T@x<=tot, "s2")

    for i in range(N):
        m.addConstr(x@v[i]==0, "c"+str(i))
    #punto di parenza
    xold=y0
    xnew=y0
    xbar=y0
    Sold=gamma0
    Snew=gamma0
    val=[]
    tempi=[]
    val_gap=[]
    
    inizio=time.time()
    for i in range(1000000):
        #print(time.time()-inizio)
        #print(MAXTIME)
        if time.time()-inizio>MAXTIME:
            break
        w=gradf(xnew)
        xold=xnew
        gammak=gamma0/np.sqrt(i+1)
        etak=eta0/(i+1)**(1/4)
        z=xold-gammak*(G(xold)+etak*w)
        mp.setObjective(z@z+yp@yp-2*z@yp, GRB.MINIMIZE)
        mp.optimize()
        xnew=yp.X
        Snew=Sold+gammak**r
        xbar=(Sold*xbar+xnew*gammak**r)/Snew
        Sold=Snew

        ms.setObjective(G(xbar)@xbar - G(xbar)@ys, GRB.MAXIMIZE)
        ms.optimize()
        errore_Stampacchia=ms.objVal
        
        if np.sqrt((xold-xnew)@(xold-xnew))<10**-3 and errore_Stampacchia<0.01:
                break
    tempi.append(time.time()-inizio)
    val.append(f(xbar))
    val_gap.append(errore_Stampacchia)
    fine=time.time()        
    ms.dispose()
    mp.dispose()
    return [xbar,fine-inizio,errore_Stampacchia,i,val,val_gap,tempi]


def alg_sayo(f_esterna,gradf_esterna,G_interna,y0,param_f=[],param_G=[],vincolo="cubo",param_C=[0,1],gamma0=1,eta0=1,r=0.5,epsilon=10**-1,N=4,V=3,L=10,tol=10**-5,MAXTIME=100,stop=10**-6):
    n=2*N*V
    
    def G(x):
        if len(param_G)>0:
            return G_interna(x,param_G)
        else:
            return G_interna(x)
    def gradf(x):
        if len(param_f)>0:
            return gradf_esterna(x,param_f)
        else:
            return gradf_esterna(x)

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
        
    v=np.zeros([N,2*N*V])
    for i in range(N):
        v[i,i*V:(i+1)*V]=np.ones(V)
        v[i,(N+i)*V:(N+i+1)*V]=-np.ones(V)

    #Calcolo D
    m1=gp.Model("calcolo Diametro")
    m1.setParam('OutputFlag', 0)
    m1.Params.LogToConsole = 0

    #definisco le variabili
    x1=m1.addMVar(shape=n,lb=0, name="x1")
    y1=m1.addMVar(shape=n,lb=0, name="y1")
    #Insieme C
    if vincolo=="cubo":
        for i in range(N*V):
            m1.addConstr(x1[i] <=vu, "ax"+str(i))
            m1.addConstr(x1[i]>=vl, "bx"+str(i))
            m1.addConstr(y1[i] <=vu, "ay"+str(i))
            m1.addConstr(y1[i]>=vl, "by"+str(i))
    if vincolo=="sfera":
        m1.addConstr(x1@x1<=R, "s")
    if vincolo=="simplesso":
        m1.addConstr(x1>=0, "s1")
        m1.addConstr(np.ones(n).T@x1<=tot, "s2")

    for i in range(N):
        m1.addConstr(x1@v[i]==0, "c"+str(i))
        m1.addConstr(y1@v[i]==0, "c"+str(i))
    m1.setObjective((x1-y1)@(x1-y1),GRB.MAXIMIZE)
    m1.optimize()
    D=np.sqrt(m1.ObjVal)
    #Proiezione
    mp=gp.Model("projection")
    mp.setParam('OutputFlag', 0)
    mp.Params.LogToConsole = 0
    yp=mp.addMVar(shape=2*N*V,lb=0, name="yp")
    if vincolo=="cubo":
        for i in range(N*V):
            mp.addConstr(yp[i] <=vu, "a"+str(i))
            mp.addConstr(yp[i]>=vl, "b"+str(i))
    if vincolo=="sfera":
        mp.addConstr(yp@yp<=R, "s")
    if vincolo=="simplesso":
        mp.addConstr(yp>=0, "s1")
        mp.addConstr(np.ones(n).T@yp<=tot, "s2")
    for i in range(N):
        mp.addConstr(yp@v[i]==0, "c"+str(i))

    #Funzione Gap Stampacchia
    ms=gp.Model("stampacchia")
    ms.setParam('OutputFlag', 0)
    ms.Params.LogToConsole = 0
    ys=ms.addMVar(shape=n,lb=0, name="ys")
    if vincolo=="cubo":
        for i in range(N*V):
            ms.addConstr(ys[i] <=vu, "a"+str(i))
            ms.addConstr(ys[i]>=vl, "b"+str(i))
    if vincolo=="sfera":
        ms.addConstr(ys@ys<=R, "s")
    if vincolo=="simplesso":
        ms.addConstr(ys>=0, "s1")
        ms.addConstr(np.ones(n).T@ys<=tot, "s2")
    for i in range(N):
         ms.addConstr(ys@v[i]==0, "c"+str(i))
    

    m=gp.Model("prob_penalizzato")
    m.setParam('OutputFlag', 0)
    m.Params.LogToConsole = 0

    #definisco le variabili
    x=m.addMVar(shape=n,lb=0, name="x")
    eta=m.addVar(lb=0,name="eta")

    #Insieme C
    if vincolo=="cubo":
        for i in range(N*V):
            m.addConstr(x[i] <=vu, "a"+str(i))
            m.addConstr(x[i]>=vl, "b"+str(i))
    if vincolo=="sfera":
        m.addConstr(x@x<=R, "s")
    if vincolo=="simplesso":
        m.addConstr(x>=0, "s1")
        m.addConstr(np.ones(n).T@x<=tot, "s2")

    for i in range(N):
        m.addConstr(x@v[i]==0, "c"+str(i))
    #punto di parenza
    xold=y0
    xnew=y0
    xbar=y0
    Sold=gamma0
    Snew=gamma0
    val=[]
    tempi=[]
    val_gap=[]
    #print(np.sqrt(gradf(np.ones(24)*0)@gradf(np.ones(24)*0)))
    inizio=time.time()
    for i in range(100000):
        #print(time.time()-inizio)
        #print(MAXTIME)
        if time.time()-inizio>MAXTIME:
            break
        w=gradf(xnew)
        xold=xnew
        gammak=gamma0
        etak=eta0/(i+1)**(1/2)
        z=xold-gammak*(G(xold)+etak*w)
        mp.setObjective(z@z+yp@yp-2*z@yp, GRB.MINIMIZE)
        mp.optimize()
        ynew=yp.X
        yw=gradf(ynew)
        z=xold-gammak*(G(ynew)+etak*yw)
        mp.setObjective(z@z+yp@yp-2*z@yp, GRB.MINIMIZE)
        mp.optimize()
        #print(err1(gamma0,D,0.5,eta0,i))
        xnew=yp.X
        xbarold=xbar
        fold=f(xbar)
        xbar=(i*xbar+xnew)/(i+1)
        fnew=f(xbar)
        #val.append(f(xbar))
        #tempi.append(time.time()-inizio)
        #condizione di Stop
        ms.setObjective(G(xbar)@xbar - G(xbar)@ys, GRB.MAXIMIZE)
        ms.optimize()
        errore_Stampacchia=ms.objVal
        if abs(fold-fnew)<10**-3 and errore_Stampacchia<0.01:
                break
    val.append(f(xbar))
    tempi.append(time.time()-inizio)
    val_gap.append(errore_Stampacchia)
    fine=time.time()        
    ms.dispose()
    mp.dispose()
    return [xbar,fine-inizio,errore_Stampacchia,i,val,val_gap,tempi]




def cost(c,y,player):
    return c[player]@y[player]

def price(alpha,beta,sigma,s,node):
    #print(sum(s[:,node]))
    return alpha-beta*(sum(s[:,node]))**sigma

def price_lin(alpha,beta,s,node):
    #print(sum(s[:,node]))
    return alpha-beta*(sum(s[:,node]))

def neg_earning(y,s,player,E,c,alpha,beta,sigma):
    tot=cost(c,y,player)
    for node in range(E):
        tot=tot-s[player,node]*price_lin(alpha,beta,s,node)
    return tot


def aggregate(y,s,c,alpha,beta,sigma,N,E):
    tot=0
    for player in range(N):
        tot=tot+neg_earning(y,s,player,E,c,alpha,beta,sigma)
    return tot

def gradient(y,s,c,alpha,beta,sigma,N,E):
    Grad=np.zeros((2*N*E))
    for player in range(N):
        for node in range(E):
            Grad[player*E+node]=c[player,node]
            Grad[N*E+player*E+node]=-alpha+beta*((sum(s[:,node]))**sigma+sigma*s[player,node]*(sum(s[:,node]))**(sigma-1))
    return Grad

def G_grad(z,param_G):
    c=param_G[0]
    alpha=param_G[1]
    beta=param_G[2]
    sigma=param_G[3]
    N=param_G[4]
    E=param_G[5]
    y=z[0:N*E].reshape((N,E))
    s=z[N*E:].reshape((N,E))
    return gradient(y,s,c,alpha,beta,sigma,N,E)

def f_aggr(z,param_f):
    c=param_f[0]
    alpha=param_f[1]
    beta=param_f[2]
    sigma=param_f[3]
    N=param_f[4]
    E=param_f[5]
    y=z[0:N*E].reshape((N,E))
    s=z[N*E:2*N*E].reshape((N,E))
    return aggregate(y,s,c,alpha,beta,sigma,N,E)

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

def f_aggr_quad(z,param_f):
    c=param_f[0]
    alpha=param_f[1]
    beta=param_f[2]
    sigma=param_f[3]
    N=param_f[4]
    E=param_f[5]
    y=z[0:N*E]
    s=z[N*E:2*N*E]
    q1=c.reshape(N*E)@y
    q2=-alpha*np.ones(N*E)@s
    Q=np.zeros((N*E,N*E))
    for i in range(E):
        Q[N*(i):N*(i+1),N*(i):N*(i+1)]=np.ones((N,N))

    q3=beta*(s@Q@s)
    return q1+q2+q3

def gradf_aggr_quad(z,param_f):
    c=param_f[0]
    alpha=param_f[1]
    beta=param_f[2]
    sigma=param_f[3]
    N=param_f[4]
    E=param_f[5]
    y=z[0:N*E]
    s=z[N*E:2*N*E]
    q1=c.reshape(N*E)
    q2=-alpha*np.ones(N*E)
    Q=np.zeros((N*E,N*E))
    for i in range(E):
        Q[N*(i):N*(i+1),N*(i):N*(i+1)]=np.ones((N,N))

    q3=2*beta*(Q@s)
    w=np.zeros(2*N*E)
    w[0:N*E]=q1
    w[N*E:2*N*E]=q2+q3
    return w

def f_aggr_gen(z,param_f):
    c=param_f[0]
    alpha=param_f[1]
    beta=param_f[2]
    sigma=param_f[3]
    N=param_f[4]
    E=param_f[5]
    y=z[0:N*E]
    s=z[N*E:2*N*E]
    q1=c.reshape(N*E)@y
    q2=-alpha*np.ones(N*E)@s
    Q=np.zeros((N*E,N*E))
    for i in range(E):
        Q[N*(i):N*(i+1),N*(i):N*(i+1)]=np.ones((N,N))
    
    q3=beta*(s@(Q@s)**sigma)
    return q1+q2+q3

def gradf_aggr_gen(z,param_f):
    c=param_f[0]
    alpha=param_f[1]
    beta=param_f[2]
    sigma=param_f[3]
    N=param_f[4]
    E=param_f[5]
    y=z[0:N*E]
    s=z[N*E:2*N*E]
    q1=c.reshape(N*E)
    q2=-alpha*np.ones(N*E)
    Q=np.zeros((N*E,N*E))
    for i in range(E):
        Q[N*(i):N*(i+1),N*(i):N*(i+1)]=np.ones((N,N))

    Qs=Q@s
    q3=beta*(Qs**sigma + Q@np.diag([sigma*Qs[i]**(sigma-1) for i in range(N*E)])@s)
    w=np.zeros(2*N*E)
    w[0:N*E]=q1
    w[N*E:2*N*E]=q2+q3
    return w


def cost(c,y,player):
    return c[player]@y[player]

def price1(alpha,beta,sigma,t,node):
    #print(t)
    return alpha-beta*t[node]

def neg_earning1(y,s,t,player,E,c,alpha,beta,sigma):
    tot=cost(c,y,player)
    for node in range(E):
        tot=tot-s[player,node]*price1(alpha,beta,sigma,t,node)
    return tot


def aggregate1(y,s,t,c,alpha,beta,sigma,N,E):
    tot=0
    for player in range(N):
        tot=tot+neg_earning1(y,s,t,player,E,c,alpha,beta,sigma)
    return tot

def f_aggr1(z,t,param_f):    
    c=param_f[0]
    alpha=param_f[1]
    beta=param_f[2]
    sigma=param_f[3]
    N=param_f[4]
    E=param_f[5]
    y=z[0:N*E]
    s=z[N*E:2*N*E]
    q1=c.reshape(N*E)@y
    q2=-alpha*np.ones(N*E)@s
    q3=0
    for i in range(E):
        for j in range(N):
            q3=q3+beta*(s[N*i+j]*t[i])
    return q1+q2+q3
    
def stampacchia_media(tempi,valori,MAXTIME):
    intervalli_tempo=np.arange(0,MAXTIME*2+1)/2
    v=np.zeros((len(tempi),MAXTIME*2+1))
    for it in tempi:
        MAT=np.zeros((len(intervalli_tempo),len(tempi[it])))
        for i in range(len(intervalli_tempo)):
            ind_t=0
            for t in tempi[it]:
                if t>=MAXTIME:
                    MAT[-1,ind_t]=1
                else:
                    if t-int(t)<0.25:
                        MAT[int(t)*2,ind_t]=1
                    if t-int(t)>=0.25 and  t-int(t)<0.75:
                        MAT[int(t)*2+1,ind_t]=1
                    if t-int(t)>0.75:
                        MAT[int(t)*2+2,ind_t]=1
                ind_t=ind_t+1
            inds=MAT[i]==1
            s=np.array(valori[it])
            if np.sum(MAT[i])<=10**-8:
                v[it][i]=v[it][i-1]
            else:
             v[it][i]=np.min(s[inds])
    print(MAT)         
    print(np.shape(MAT)) 
    print(s)
    print(v)
    return intervalli_tempo,np.mean(v,0)

def err1(gamma,D,b,eta0,k):
    return (D**2/(gamma*eta0))/(k+1)**(1-b)

def err2(gamma,D,b,eta0,C,k):
    return (D**2/gamma)/(k+1)+ (np.sqrt(2)*eta0*C*D)/((1-b)*(k+1)**b)


def test(iters,N,V,seed,epsilon=10**-6,MAXTIME=120,aIRG=False):
    ris=None
    ris_kayo=None
    ris_sayo=None
    
   

    tempi={}
    values={}
    stampacchia={}
    names=["Real","KaYo","SaYo"]
    np.random.seed(seed)
    for name in names:
        tempi[name]={}
        values[name]={}
        stampacchia[name]={}
    gamma=1
    beta=0.01
    sigma=1.05
    B=5


    L=N*beta*(2*sigma*(N*B)**(sigma-1)+(sigma)*(sigma-1)*B*(N*B)**(sigma-2))
    alpha=1.5
    
    gamma0=1
    eta0=0.1
    r=0.5
    dati={}
    it_dati=0
    print("oligopoly with ", N, "firms","and ",V, "locations")
    for it in range(iters):
        rho=1
        theta=10**-5
        print("instance", it+1)
        c=np.random.rand(N,V)*0.9+0.1
        y=np.random.rand(N,V)
        s=np.random.rand(N,V)

    


        vincolo="cubo"
        param_C=[0,B]
        [vl,vu]=param_C
        z0=(vu-vl)*np.array(np.random.random_sample((N*V,))) + vl
        y0=np.zeros(2*N*V)
        y0[0:N*V]=z0
        y0[N*V:]=z0
    
    
        minimo=1000
    
        
        
        
        ris=alg_real_grad(f_aggr_gen,gradf_aggr_gen,G_grad,y0,[c,gamma,beta,sigma,N,V],[c,gamma,beta,sigma,N,V],vincolo,param_C,alpha,rho,epsilon,N,V,L,tol=epsilon*10**(-1),MAXTIME=MAXTIME)
        if aIRG==True:
            ris_kayo=alg_kayo(f_aggr_gen,gradf_aggr_gen,G_grad,y0,[c,gamma,beta,sigma,N,V],[c,gamma,beta,sigma,N,V],vincolo,param_C,gamma0,eta0,r,minimo,N,V,L,MAXTIME=MAXTIME)
        ris_sayo=alg_sayo(f_aggr_gen,gradf_aggr_gen,G_grad,y0,[c,gamma,beta,sigma,N,V],[c,gamma,beta,sigma,N,V],vincolo,param_C,gamma0,eta0,r,minimo,N,V,L,MAXTIME=MAXTIME,stop=10**-6)
        
        if ris is not None:
            tempi["Real"][it]=np.array(ris[10])[-1]
            values["Real"][it]=np.array(ris[8])[-1]   
            stampacchia["Real"][it]=np.array(ris[9])[-1]
        if ris_kayo is not None:
            tempi["KaYo"][it]=np.array(ris_kayo[6])[-1]
            values["KaYo"][it]=np.array(ris_kayo[4])[-1]
            stampacchia["KaYo"][it]=np.array(ris_kayo[5])[-1]
        if ris_sayo is not None:
            tempi["SaYo"][it]=np.array(ris_sayo[6])[-1]
            values["SaYo"][it]=np.array(ris_sayo[4])[-1]
            stampacchia["SaYo"][it]=np.array(ris_sayo[5])[-1]

    return tempi,stampacchia,values,dati

def tab6(MAXTIME,iters,seed):
    Res=test(iters,4,3,seed,epsilon=10**-6,MAXTIME=MAXTIME,aIRG=True)
    table(Res,iters)

def tab7(MAXTIME,iters,seed):
    Res1=test(iters,4,3,seed,epsilon=10**-6,MAXTIME=MAXTIME,aIRG=False)
    time_plot(Res1,iters)
    Res2=test(iters,5,4,seed,epsilon=10**-7,MAXTIME=MAXTIME,aIRG=False)
    time_plot(Res2,iters)
    Res3=test(iters,6,5,seed,epsilon=10**-7.5,MAXTIME=MAXTIME,aIRG=False)
    time_plot(Res3,iters)
    Res4=test(iters,7,5,seed,epsilon=10**-7.5,MAXTIME=MAXTIME,aIRG=False)
    time_plot(Res4,iters)
    
def table(Res,iters):
    times=np.zeros([3,iters])
    stamp=np.zeros([3,iters])
    values=np.zeros([3,iters])
    for i in range(iters):
        times[0,i]=Res[0]["Real"][i]
        times[1,i]=Res[0]["KaYo"][i]
        times[2,i]=Res[0]["SaYo"][i]
        stamp[0,i]=Res[1]["Real"][i]
        stamp[1,i]=Res[1]["KaYo"][i]
        stamp[2,i]=Res[1]["SaYo"][i]
        values[0,i]=Res[2]["Real"][i]
        values[1,i]=Res[2]["KaYo"][i]
        values[2,i]=Res[2]["SaYo"][i]
    print("Mean time for Alg3/aIRG/IREG")
    print(np.mean(times,1))
    print("Max time for Alg3/aIRG/IREG")
    print(np.max(times,1))
    print("Mean inexactness for ALg3/aIRG/IREG")
    print(np.mean(stamp,1))
    print("Mean values for Alg3/aIRG/IREG")
    print(-np.mean(values,1))
    

def time_plot(Res,iters):
    times=np.zeros([2,iters])
    stamp=np.zeros([2,iters])
    values=np.zeros([2,iters])
    solved=np.zeros([2,iters])
    for i in range(iters):
        times[0,i]=Res[0]["Real"][i]
        times[1,i]=Res[0]["SaYo"][i]
        stamp[0,i]=Res[1]["Real"][i]
        stamp[1,i]=Res[1]["SaYo"][i]
        values[0,i]=Res[2]["Real"][i]
        values[1,i]=Res[2]["SaYo"][i]

    solved=np.zeros([2,iters])
    print("Solved instances Alg3/IREG")
    print(np.sum(stamp[0]<=0.01),np.sum(stamp[1]<=0.01))
    
    print("Median time for Alg3/IREG")
    print(np.median(times,1))
    
    print("Mean time for Alg3/IREG")
    print(np.mean(times,1))
    
    print("Max time for Alg3/IREG")
    print(np.max(times,1))
    
    
