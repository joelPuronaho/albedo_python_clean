import numpy as np

#region     *Set ranges for time period, age classes, and vintages for soil and product carbon*
# Define variables for time, age classes and carbon
t = np.arange(0,101)        # Time periods
scf_t = np.arange(0,201)    # scf periods, GAMS version has 200 indices for scf
a = np.arange(1,31)         # Age classes
k = np.arange(0,41)         # Product carbon vintages
j = np.arange(0,41)         # Soil carbon vintages
#endregion

#region     *Run parameters / sensitivity parameters*
pcScale = 0.15              # fraction at which scc is applied: scales prices"                      | PC =  Social Cost of Carbon
scfScale = 0                # fraction at which scf is applied: scales prices"  
constantCase = 0            # a switch for DICE and constant SCC/SCF cases: 0-DICE, 1-Constant"  
noLUC = 0                   # a switch for land-use change: 0: LUC on, 1: LUC off"
region = 1                  # region switch: 1-Jyväskylä, 2:Muhos, 3:Pudasjärvi"
albstren = 1                # albedo strength: scales the albedo difference (default value is 1)"
telast = 1                  # elasticity in timber demand function"                              
felast = 1                  # elasticity in food demand function"                                
calibRotation = None        # Calibration rotation in years -- region specific"
yinitShare = 0.5            # Initial farmland share" 
totalArea = 1               # Total land area (ha)"   
#endregion

#region     *General parameters, price and cost parameters*
timestep = 5            # period interval                                  
agyield = 1             # "Agricultural yield per hectare in a time period"
tcalibp = 55            # timber calibration price                         
fcalibp = 1000          # food calibration price                           
cht = 0                 # unit cost of timber harvest                      
chf = 0                 # unit cost of food harvest                        
creg = 1000             # regeneration cost                                
cf = None               # "per hectare cost of agriculture -- calibrated"

calibTimberVol = None   # "Volume of Calibration rotation age -- calculated"
calibInterest = None    # "Calibration interest rate"
yinit = None            # "Initial farmland area (ha)"
timberquality = None    # "timber quality index or a quality based price index"
 
tcalibq = None          # "timber calibration quantity -- calculated"
fcalibq = None          # "food calibration quantity -- calculated"

xinit = None            # "initial size of forest age-class (ha) -- calculated"

dfInit = None           # "- input data - Exogenous values for periodic **discount factor** (df)"
pcInit = None           # "- input data - Exogenous values for **carbon price**(pc)"
df = None               # "periodic discount factor"
pc = None               # "carbon price (� per CO2 ton)"
InitSCS = {}          # "Initial soil carbon stock -- calculated" - (j)
InitPCS = {}          # "Initial product carbon stock -- calculated" (k)
hxlowlim = 1e-8         # "minumum timber harvest in each period -- technical"           /1e-8/
hylowlim = 1e-8         # "minimum crop harvest in each period -- technical"             /1e-8/
#endregion

#region     *Read input data for DF, PC, SCF*
import pandas as pd
# Read input data from excel files for: discount factor(df), carbon price (pc), and social cost of forcing (scf)
dfInit1 = pd.read_excel("input_data/DataDF_H.xls")
pcInit1 = pd.read_excel("input_data/DataPC_H.xls")
scfInit1 = pd.read_excel("input_data/DataSCF_H.xls")

dfInit = dfInit1.values.flatten().tolist()[:101]
pcInit = pcInit1.values.flatten().tolist()[:101]
scfInit = scfInit1.values.flatten().tolist()
#endregion

#region     *Set DF, PC, SCF, depending on constantCase*
# Set calibInterest with input file discount factor values
calibInterest = dfInit[1]**(-1/5)-1
df = {}
pc = {}
scf = {}

# Set discount factor(df), carbon price (pc), and social cost of forcing (scf), based on if constantCase == 0 / 1
constantCase = 0
if constantCase == 0:
    for time in t:
        df[time] = dfInit[time]
        pc[time] = pcScale * pcInit[time]
    # GAMS version has 200 indices for scf
    for time2 in scf_t:
        scf[time2] = scfScale * scfInit[time2] 
elif constantCase == 1:
    for time in t:
        # df = (1 + r)^-n   <---> n = timestep * time
        df[time] = (1 + calibInterest)**(-timestep * time)
        pc[time] = pcScale * pcInit[0]

    # GAMS version has 200 indices for scf
    for time2 in scf_t:
        scf[time2] = scfScale * scfInit[0]
#endregion

#region     *Parameters for timber growth per site*
tqT0 = None                 # timber quality: age of first timber -- region specific
tqTscale = None             # "timber quality: development scale -- region specific"
tcalibtimberquality = None  # 
vb     = None               # "MSY yield -- region specific"
vc     = None               # "MSY rotation -- region specific"
vbeta  = None               # "shape parameter -- region specific"
valfa  = None               # "shape parameter - calculated"
vgamma = None               # "shape paremeter - calcualted"
v   = None                  # timber volume per hectare in age class
vmax   = None               # timber volume in oldest age class;

if (region == 2):
    # tuore kuusikko Muhos (source: KuusiMuhosMotti_long.xlsx)
    vb           = 5.53     # "MSY yield -- region specific"
    vc           = 87.10    # "MSY rotation -- region specific"
    vbeta        = 8.56     # "shape parameter -- region specific"
    tqT0         = 16.81    # timber quality: age of first timber -- region specific
    tqTscale     = 47.7     # "timber quality: development scale -- region specific"
    calibRotation = 55      # Calibration rotation in years -- region specific"
elif (region == 3):
    # tuore kuusikko Pudasjärvi (source: KuusiPudasjarviMotti_long.xlsx)
    vb           = 4.01     # "MSY yield -- region specific"
    vc           = 109.18   # "MSY rotation -- region specific"
    vbeta        = 5.78     # "shape parameter -- region specific"
    tqT0         = 16.56    # timber quality: age of first timber -- region specific
    tqTscale     = 68.41    # "timber quality: development scale -- region specific"
    calibRotation = 60      # Calibration rotation in years -- region specific"
else:
    # tuore kuusikko Jyväskylä (source: KuusiJyvaskylaMotti.xlsx)
    vb           = 7.98     # "MSY yield -- region specific"
    vc           = 70.96    # "MSY rotation -- region specific"
    vbeta        = 15.44    # "shape parameter -- region specific"
    tqT0         = 16.78    # timber quality: age of first timber -- region specific
    tqTscale     = 33.5     # "timber quality: development scale -- region specific"
    calibRotation = 50      # Calibration rotation in years -- region specific"
#endregion

#region     *Set parameters for biomass decay*
import math as math

vgamma = (vbeta - 1) / math.log(vbeta)
valfa = 1 / ((1 - (1 / vbeta))**vgamma)
v = {index: vb * vc * valfa * ((1 - vbeta**(timestep * (-index) / vc)))**vgamma for index in a}
vmax = vb * vc * valfa * ((1 - vbeta**(timestep * (-len(a) / vc))))**vgamma

#Parameters
gammav = 0.2026         # timber carbon density                                  
gammab = 0.2070         # carbon in other woody biomass proportional to v        
carbonshare = 0.5243    # carbon share of stemwood biomass    

# Soil carbon decay function parameters
beta1 = 0.52            #share residues (fast decay)  
beta2 = 0.34            #share residues (medium decay)
beta3 = 0.14            #share residues (slow decay)  
delta1= 0.2373          #decay rate (fast decay)      
delta2= 0.0295          #decay rate (medium decay)    
delta3= 0.0051          #decay rate (slow decay)      
#endregion

#region     *Soil Carbon Decay*

# *Carbon emissions at vintage j for carbon in share residues and soil carbon*
deltaS = {}             # soil carbon emissions from vintage j

# Biomass density by age class
b = None                #biomass per hectare by age class
bmax = None             #biomass per hectare in oldest age class

# Use carbon density parameters, share of stemwood, and vol. per ha. to calculate biomass density per ha. per time period
b = {a: ((gammav+gammab)/carbonshare)*v[a] for a in v}
bmax = ((gammav+gammab)/carbonshare)*vmax

# Calculate deltaS, based on the decay rates.
for i in j:
    if i < max(j):# - 1:
        e1 = beta1 * math.exp(-delta1 * timestep * (i))
        e2 = beta2 * math.exp(-delta2 * timestep * (i))
        e3 = beta3 * math.exp(-delta3 * timestep * (i))
        e_total1 = e1 + e2 + e3
        e4 = beta1 * math.exp(-delta1 * timestep * (i + 1))
        e5 = beta2 * math.exp(-delta2 * timestep * (i + 1))
        e6 = beta3 * math.exp(-delta3 * timestep * (i + 1))
        e_total2 = e4 + e5 + e6
        e7 = beta1 * math.exp(-delta1 * timestep * (i))
        e8 = beta2 * math.exp(-delta2 * timestep * (i))
        e9 = beta3 * math.exp(-delta3 * timestep * (i))
        e_total3 = e7 + e8 + e9
        result = (e_total1 - e_total2) / e_total3
        deltaS[i] = result
#endregion

#region     *Product carbon decay*
# Parameters
# Product carbon decay parameters
sharesolid = 0.6    # share of solidwood products             
sharepaper = 0.4    # share of paper products                 
lambdas    = 0.0231 # decay rate of solidwood product stock   
lambdap    = 0.3466 # decay rate of paper product stock       
# Product carbon decay
imCrel = 0.5        # immediate carbon release from wood processing /0.5/
deltaP = {}         # product carbon emissions from vintage k

# # Calculate deltaP, based on the decay rates.
for i in j:
    if i < max(j):
        result1 = (
        (sharesolid * math.exp(-lambdas * timestep * (i)) +
         sharepaper * math.exp(-lambdap * timestep * (i)) -
          (sharesolid * math.exp(-lambdas * timestep * (i + 1)) +
           sharepaper * math.exp(-lambdap * timestep * (i + 1)))) /
        (sharesolid * math.exp(-lambdas * timestep * (i )) +
         sharepaper * math.exp(-lambdap * timestep * (i)))
        )
        deltaP[i] = result1
#endregion

#region     *Albedo section*
# Parameters for albedo and warming
# *Albedo function parameters
albedoalfa  = 0.09132355       # intercept parameter                      
albedobeta  = 0.10644404       # slope parameter                          
albedogamma = 0.03216253       # power parameter                          
MWopenshrub = 1.254            # mean annual warming power of open shrub  
MWmature    = 1.412            # mean annual warming power of mature stand

# Stand albedo           
Alb = None          # stand albedo by age class
Albzero = None      # albedo of open shrub
Albmax = None       # albedo of oldest age class
w = {}              # mean annual warming power of stand albedo W per m2

# Calculate stand albedo for age class, open shrub and max albedo
# - use these to calculate the warming power per stand age
Alb = {i: albedoalfa + albedobeta * math.exp(-albedogamma * b[i]) for i in a}
Albzero = albedoalfa + albedobeta * math.exp(-albedogamma * 0)
Albmax  = albedoalfa + albedobeta * math.exp(-albedogamma * bmax)
w = {i: MWopenshrub + albstren * (MWmature - MWopenshrub)*(Alb[i] - Albzero) / (Albmax - Albzero) for i in a}
#endregion

#region     *Init agriculture and forest variables, and carbon stocks + Timber growth.*

##*** Calibration ***
# Initial values for agricultural and forest land area
yinit = yinitShare * totalArea
xinit = (1 - yinitShare) * totalArea / (calibRotation / timestep)

# "Total volume of timber at calibrated rotation age"
calibTimberVol = vb * vc * valfa * (1 - vbeta**(-calibRotation / vc))**vgamma

# Calculate timberquality per age-class
timberquality = {i: ( 1 - math.exp( -(i * timestep - tqT0 )/tqTscale ) ) if i > tqT0 / timestep else 0 for i in a}

# "Sum of timber quality at calibRotation / timestep"
tcalibtimberquality = sum(timberquality[i] for i in a if i == calibRotation / timestep)
# Total timber output at calibrated rotation
tcalibq  = xinit * calibTimberVol * tcalibtimberquality

# Agricultural yield at start
fcalibq  = agyield * yinit

cf = (fcalibp-cht) * agyield - (1 - (1 + calibInterest)**(-timestep)) * ( (tcalibp * tcalibtimberquality - chf) * calibTimberVol * (1 + calibInterest)**(-calibRotation) - creg)/(1 - (1 + calibInterest)**(-calibRotation))

# Calculate Initial product carbon stock for vintage k
InitPCS[0] = (1 - imCrel) * gammav * tcalibq/tcalibtimberquality
for i in range(max(k)):
    InitPCS[i + 1] = (1 - deltaP[i])*InitPCS[i]

# Calculate Initial soil carbon stock for vintage j
InitSCS[0] = gammab * tcalibq/tcalibtimberquality
for i in range(max(j)):
    InitSCS[i + 1] = (1 - deltaS[i])*InitSCS[i]
#endregion

#region *Hard coded from GAMS version*
# The below is added because the python version has different rounding with all the parameters / variables compared to GAMS

# Function to imitate GAMS behaviour with rounding (significant digits)
def round_significant(value, sig_digits):
    if value == 0:
        return 0
    else:
        return round(value, sig_digits - int(math.floor(math.log10(abs(value)))) - 1)

# Set hard coded on / off
hard_coded = 1
if hard_coded == 1:
    cf = 986.563039582848
    calibTimberVol = 351.987689353233
    calibInterest = 0.0501061417315183
    timberquality = {key: round_significant(value, 15) for key, value in timberquality.items()}
    tcalibq = 11.0705911414855
    dfInit = [round_significant(num, 15) for num in dfInit]
    pcInit = [round_significant(num, 15) for num in pcInit]
    scfInit = [round_significant(num, 15) for num in scfInit]
    df = {key: round_significant(value, 15) for key, value in df.items()}
    pc = {key: round_significant(value, 15) for key, value in pc.items()}
    InitSCS = {key: round_significant(value, 15) for key, value in InitSCS.items()}
    InitPCS = {key: round_significant(value, 15) for key, value in InitPCS.items()}
    tcalibtimberquality = 0.629032859747305
    valfa = 1.42371939186963
    vgamma = 5.27592359801612
    v = {index: vb * vc * valfa * ((1 - vbeta**(timestep * (-index) / vc)))**vgamma for index in a}
    v = {key: round_significant(value, 15) for key, value in v.items()}
    vmax = 793.217357339943
    b = {a: ((gammav+gammab)/carbonshare)*v[a] for a in v}
    bmax = 619.686876914821
    deltaS = {key: round_significant(value, 15) for key, value in deltaS.items()}
    deltaP = {key: round_significant(value, 15) for key, value in deltaP.items()}
    Alb = {key: round_significant(value, 15) for key, value in Alb.items()}
    w = {key: round_significant(value, 15) for key, value in w.items()}
#endregion

#region Hard coded variables with round() function (not in use anymore, above version more accurate imitation)
hard_coded_old = 0
if hard_coded_old == 1:
    cf = 986.563039582848
    calibTimberVol = 351.987689353233
    calibInterest = 0.0501061417315183
    timberquality = {key: round(value, 15) for key, value in timberquality.items()}
    tcalibq = 11.0705911414855
    dfInit = [round(num, 16) for num in dfInit]
    pcInit = [round(num, 13) for num in pcInit]
    scfInit = [round(num, 12) for num in scfInit]
    df = {key: round(value, 15) for key, value in df.items()}
    pc = {key: round(value, 14) for key, value in pc.items()}
    InitSCS = {key: round(value, 14) for key, value in InitSCS.items()}
    InitPCS = {key: round(value, 14) for key, value in InitPCS.items()}
    tcalibtimberquality = 0.629032859747305
    valfa = 1.42371939186963
    vgamma = 5.27592359801612
    v = {index: vb * vc * valfa * ((1 - vbeta**(timestep * (-index) / vc)))**vgamma for index in a}
    v = {key: round(value, 15) for key, value in v.items()}
    vmax = 793.217357339943
    b = {a: ((gammav+gammab)/carbonshare)*v[a] for a in v}
    bmax = 619.686876914821
    deltaS = {key: round(value, 15) for key, value in deltaS.items()}
    deltaP = {key: round(value, 15) for key, value in deltaP.items()}
    Alb = {key: round(value, 16) for key, value in Alb.items()}
    w = {key: round(value, 14) for key, value in w.items()}
#endregion

#       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       
###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
########################################################################################################################################
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   
###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     
#       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       

# ****************************************
# ************ PYOMO SECTION *************
# ****************************************

#region     *Setup model variables*
from pyomo.environ import *

# Setup model
model = ConcreteModel()

# Set indices for model
model.t = RangeSet(0,100)
model.scf_t = RangeSet(0,200)
model.a = RangeSet(1,30)
model.k = RangeSet(0,40)
model.j = RangeSet(0,40)

# Set positive variables for the model
model.y = Var(model.t, domain=NonNegativeReals)             # Agricultural area at time t
model.x = Var(model.t, model.a, domain=NonNegativeReals)    # Forest area at time t and age-class a

#* control variables
model.z = Var(model.t, model.a, domain=NonNegativeReals)    # Harvested forest area at time t and age-class a

#* resulting values
model.hx = Var(model.t, bounds=(hxlowlim, None))            # "quality weighted harvest(timber)"
model.hy = Var(model.t, bounds=(hylowlim, None))            # harvest (agriculture)
model.pt = Var(model.t, domain=NonNegativeReals)            # timber price
model.pf = Var(model.t, domain=NonNegativeReals)            # food price

# variables
model.wf = Var()                                            # welfare
model.ux = Var(model.t)                                     # periodic timber consumption utility
model.uy = Var(model.t)                                     # periodic food consumption utility
model.c = Var(model.t)                                      # total costs
model.cBCS = Var(model.t)                                   # change in biomass carbon stock
model.cPCS = Var(model.t)                                   # change product carbon stock
model.cSCS = Var(model.t)                                   # change in soil carbon stock
model.cTCS = Var(model.t)                                   # change in total carbon stock
model.PCS = Var(model.t, model.k)                           # product carbon stock by vintage
model.SCS = Var(model.t, model.j)                           # soil carbon stock by vintage
model.wp = Var(model.t)                                     # warming power of landscape albedo
#endregion

#region *Set initial values for variables*
# These are not explixitly set in GAMS, but GAMS sets variables to zero by default for all indices (verify this, Jussi?).
model.wf.value = 0
for t in model.t:
    model.y[t].value = 0
    for a in model.a:
        model.x[t, a].value = 0
        model.z[t, a].value = 0
    model.hx[t].value = hxlowlim
    model.hy[t].value = hxlowlim
    model.pt[t].value = 0
    model.pf[t].value = 0
    model.ux[t].value = 0
    model.uy[t].value = 0
    model.c[t].value = 0
    model.cBCS[t].value = 0
    model.cPCS[t].value = 0
    model.cSCS[t].value = 0
    model.cTCS[t].value = 0
    for k in model.k:
        model.PCS[t, k].value = 0
    for j in model.j:
        model.SCS[t, j].value = 0
    model.wp[t].value = 0
#endregion

#region     *Fix variable values and set bounds (x, y, z, carbon stocks change, harvests, prices)*
# *Set fixed values and boundaries for variables. GAMS version commented above the python expression*
#       x.fx('0',a) = xinit$(ord(a) LE calibRotation/timestep);
for t in model.t:
    if t == 0:
        for a in model.a:
            if a <= calibRotation / timestep:
                model.x[t, a].fix(xinit)
            else:
                model.x[t, a].fix(0)        # Fikastaan (Jussi). Antaa oikean luvun myös: "Number of nonzeros in inequality constraint Jacobian.:     6030"

#if(noLUC EQ 1,
#         y.fx(t) = yinit;
#else
#         y.up(t)  = totalArea;
#);
if noLUC == 1:
    for t in model.t:
        model.y[t].fix(yinit)
else:
    for t in model.t:
        model.y[t].setub(totalArea)

model.cBCS[100].setub((gammav + gammab) * vmax)
model.cPCS[100].setub(5)
model.cSCS[100].setub(5)

# SCS.fx('0',j)$(ord(j) GT 1)       =     InitSCS(j);
for j in model.j:
    if j > 0:
        model.SCS[0, j].fix(InitSCS[j])

# PCS.fx('0',k)$(ord(k) GT 1)       =     InitPCS(k);
for k in model.k:
    if k > 0:
        model.PCS[0, k].fix(InitPCS[k])
#endregion

#region *Set initial values and boundaries for variables*

# * initial values
# x.l(t,a)$(ord(t)>1) = ( (1 - yinitShare) * totalArea/(calibRotation/timestep) )$(ord(a) LE calibRotation/timestep);
for t in model.t:
    if t > 0:
        for a in model.a:
            if a <= calibRotation / timestep:
                model.x[t, a].value = (1 - yinitShare) * totalArea/(calibRotation/timestep)
            else:
                model.x[t, a].value = 0

# y.l(t)              = yinit;
for t in model.t:
    model.y[t].value = yinit

#z.l(t,a)            = ( x.l(t,a) )$(ord(a) EQ calibRotation/timestep);
for t in model.t:
    for a in model.a:
        if a == calibRotation / timestep:
            model.z[t, a].value = model.x[t, a].value
        else:
            model.z[t, a].value = 0

#hx.l(t)             = sum(a, z.l(t,a)*v(a)*timberquality(a));
for t in model.t:
    model.hx[t].value = sum(model.z[t,a].value * v[a] * timberquality[a] for a in model.a)

#hy.l(t)             = agyield*y.l(t);
for t in model.t:
    model.hy[t].value = agyield * model.y[t].value

#pt.l(t)             = tcalibp*((hx.l(t)/tcalibq)**(-telast));
for t in model.t:
    model.pt[t].value = tcalibp*((model.hx[t].value / tcalibq)**(-telast))

#pf.l(t)             = fcalibp*((hy.l(t)/fcalibq)**(-felast));
for t in model.t:
    model.pf[t].value = fcalibp * ((model.hy[t].value / fcalibq)**(-felast))
#endregion

#region     *Set model parameters*
# Model parameters. Calculated in the first part of the script - added as pyomo parameters for good practice
model.timestep = Param(initialize=timestep)
model.deltaS = Param(model.j, initialize=deltaS)
model.deltaP = Param(model.k, initialize=deltaP)
model.totalArea = Param(initialize=totalArea)
model.v = Param(model.a, initialize=v)
model.timberquality = Param(model.a, initialize=timberquality)
model.tcalibp = Param(initialize=tcalibp)
model.tcalibq = Param(initialize=tcalibq)
model.fcalibp = Param(initialize=fcalibp)
model.fcalibq = Param(initialize=fcalibq)
model.telast = Param(initialize=telast)
model.felast = Param(initialize=felast)
model.chf = Param(initialize=chf)
model.cht = Param(initialize=cht)
model.creg = Param(initialize=creg)
model.yinit = Param(initialize=yinit)
model.cf = Param(initialize=cf)
model.pc = Param(model.t, initialize=pc)
# GAMS version has 200 indices for scf, but it's used only in welfare constraint with t. If scf[0,200] used, has to be initialized with scf_t in the model.
model.scf = Param(model.scf_t, initialize=scf)
#model.scf = Param(model.t, initialize=scf)
model.df = Param(model.t, initialize=df)
model.imCrel = Param(initialize=imCrel)
model.gammav = Param(initialize=gammav)
model.gammab = Param(initialize=gammab)
model.w = Param(model.a, initialize=w)
model.MWopenshrub = Param(initialize=MWopenshrub)
model.agyield = Param(initialize=agyield)
#endregion

#       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       
###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
########################################################################################################################################
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   
###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     
#       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       

# ****************************************
# ********** MODEL CONSTRAINTS ***********
# ****************************************

#   *Defining model constraints*

#region     *xfirst, xnext, xlast, zcons, landcons
#xfirst(t)$( ord(t) LT card(t) )..
#         x(t+1,'1')  =e=   ( sum(a, z(t,a)) + y(t-1) - y(t) )$(ord(t) GT 1)
#                         + ( sum(a, z(t,a)) + yinit  - y(t) )$(ord(t) EQ 1);

# Constraint for x in first age class a == 1 
def xfirst_constraint(model, t):
    if t < max(model.t):
        if t > 0:
            return model.x[t + 1, 1] == sum(model.z[t, a] for a in model.a) + model.y[t - 1] - model.y[t]
        elif t == 0:
            return model.x[t + 1, 1] == sum(model.z[t, a] for a in model.a) + model.yinit - model.y[t]
    return Constraint.Skip
model.xfirst = Constraint(model.t, rule=xfirst_constraint)

# Constraint for x in "next" age classes a == 1 - 29 / 59
def xnext_constraint(model, t, a):
    if a < max(model.a) - 1 and t < max(model.t):
        return model.x[t + 1, a + 1] == model.x[t, a] - model.z[t, a]
    return Constraint.Skip
model.xnext = Constraint(model.t, model.a, rule=xnext_constraint)

# Constraint for x in the last age classe a == 30 / 60
def xlast_constraint(model, t, a):
    if a == max(model.a) - 1 and t < max(model.t):
        return model.x[t + 1, a + 1] == model.x[t, a] - model.z[t, a] + model.x[t, a + 1] - model.z[t, a + 1]
    return Constraint.Skip
model.xlast = Constraint(model.t, model.a, rule=xlast_constraint)

# Constraint for harvest - harvest area cannot exceed forest area
def zcons_constraint(model, t, a):
    # inequality constraints with only upper bounds:     3030 
    #return model.x[t, a] >= model.z[t, a]
    # inequality constraints with only lower bounds:     3030
    return model.x[t, a] - model.z[t, a] >= 0          # Jussi, matemaattisesti molemmat lienee sama asia, mutta ipopt antaa eri palautetta. ChatGPT:n mukaan tätä riviä tulisi käyttää, mutten ole pystynyt vahvistamaan
model.zcons = Constraint(model.t, model.a, rule=zcons_constraint)

# Constraint for land - agricultural area and forest area cannot exceed the total area
def landcons2_constraint(model, t):
    if t > 0:
        return model.totalArea == model.y[t - 1] + sum(model.x[t, a] for a in model.a)
    return Constraint.Skip
model.landcons2 = Constraint(model.t, rule=landcons2_constraint)
#endregion

#region     *harvx, harvy, utimber, ufood, cost, tprice, fprice*
# Constraint for timber harvest - Equals to harvested area *times* volume per age class *times* timberquality
def harvx_constraint(model, t):
    return model.hx[t] == sum(model.z[t, a] * model.v[a] * model.timberquality[a] for a in model.a)
model.harvx = Constraint(model.t, rule=harvx_constraint)

# Constraint for agricultural yield - agricultural area *times* agyield (parameter set for yield per ha.)
def harvy_constraint(model, t):
    return model.hy[t] == model.agyield * model.y[t]
model.harvy = Constraint(model.t, rule=harvy_constraint)

# Constraint for timber utility ** Add more info **
def utimber_constraint(model, t):
    if model.telast != 1:
        return model.ux[t] == model.tcalibp * model.tcalibq / (1 - model.telast) * ((model.hx[t] / model.tcalibq )**(1 - model.telast) - 1)
    if model.telast == 1:
        return model.ux[t] == model.tcalibp * model.tcalibq * log(model.hx[t] / model.tcalibq)
    return Constraint.Skip
model.utimber = Constraint(model.t, rule=utimber_constraint)

# Constraint for food utility ** Add more info **
def ufood_constraint(model, t):
    if model.felast != 1:
        return model.uy[t] == model.fcalibp * model.fcalibq / (1 - model.felast) * ((model.hy[t] / model.fcalibq)**(1 - model.felast) - 1)
    if model.felast == 1:
        return model.uy[t] == model.fcalibp * model.fcalibq * log(model.hy[t] / model.fcalibq)
    return Constraint.Skip
model.ufood = Constraint(model.t, rule=ufood_constraint)

# Constraint for overall costs - ** Add more info **
def cost_constraint(model, t):
    sum_condition = model.cht * sum(model.z[t, a] * model.v[a] for a in model.a) + model.chf * model.hy[t] + model.cf * model.y[t]
    if t > 0:
        return model.c[t] == sum_condition + model.creg * (sum(model.z[t,a] for a in model.a) + model.y[t - 1]  - model.y[t])
    elif t == 0:
        return model.c[t] == sum_condition + model.creg * (sum(model.z[t,a] for a in model.a) + model.yinit  - model.y[t])
    else:
        return Constraint.Skip
model.cost = Constraint(model.t, rule=cost_constraint)

# Constraint for timber price - ** Add more info **
def tprice_constraint(model, t):
    return model.pt[t] == model.tcalibp * ((model.hx[t] / model.tcalibq)**(-model.telast))
model.tprice = Constraint(model.t, rule=tprice_constraint)

# Constraint for food price - ** Add more info **
def fprice_constraint(model, t):
    return model.pf[t] == model.fcalibp * ((model.hy[t] / model.fcalibq)**(-model.felast))
model.fprice = Constraint(model.t, rule=fprice_constraint)
#endregion

#region     *welfare*
# Constraint for total welfare - ** Add more info **
def welfare_constraint(model):
    return model.wf == sum((model.ux[t] + model.uy[t] - model.c[t] + model.pc[t] * (11/3) * model.cTCS[t] - model.timestep * model.scf[t] * model.wp[t]) * model.df[t] for t in model.t)
model.welfare = Constraint(rule=welfare_constraint)
#endregion

#region     *Carbon stock changes*
# Constraint for youngest vintage of product carbon stock - ** Add more info **
def ProductCSzero_constraint(model, t, k):
    if k == 0:
        return model.PCS[t, 0] == (1 - model.imCrel) * model.gammav * sum(model.z[t, a] * model.v[a] for a in model.a)
    else:
        return Constraint.Skip
model.ProductCSzero = Constraint(model.t, model.k, rule=ProductCSzero_constraint)

# Constraint for product carbon stock - ** Add more info **
def ProductCS_constraint(model, t, k):
    if k < max(model.k) and t < max(model.t):
        return model.PCS[t + 1, k + 1] == model.PCS[t, k] * (1 - model.deltaP[k])
    else:
        return Constraint.Skip
model.ProductCS = Constraint(model.t, model.k, rule=ProductCS_constraint)

# Constraint for youngest vintage of soil carbon stock - ** Add more info **
def SoilCSzero_constraint(model, t, j):
    if j == 0:
        return model.SCS[t, 0] == model.gammab * sum(model.z[t, a] * v[a] for a in model.a)
    else:
        return Constraint.Skip
model.SoilCSzero = Constraint(model.t, model.j, rule=SoilCSzero_constraint)

# Constraint for soil carbon stock - ** Add more info **
def SoilCS_constraint(model, t, j):
    if j < max(model.j) and t < max(model.t):
        return model.SCS[t + 1, j + 1] == model.SCS[t, j] * (1 - model.deltaS[j])
    else:
        return Constraint.Skip
model.SoilCS = Constraint(model.t, model.j, rule=SoilCS_constraint)

# Constraint for change in biomass carbon stock - ** Add more info **
def ChangeBCS_constraint(model, t):
    if t < max(model.t):
        return model.cBCS[t] == (model.gammav + model.gammab) * (sum(model.v[a] * (model.x[t + 1, a] - model.x[t, a]) for a in model.a))
    else:
        return Constraint.Skip
model.ChangeBCS = Constraint(model.t, rule=ChangeBCS_constraint)

# Constraint for change in product carbon stock - ** Add more info **
def ChangePCS_constraint(model, t):
    if t < max(model.t):
        return model.cPCS[t] == sum(model.PCS[t + 1, k] for k in model.k) - sum(model.PCS[t, k] for k in model.k)
    else:
        return Constraint.Skip
model.ChangePCS = Constraint(model.t, rule=ChangePCS_constraint)

# Constraint for change in soil carbon stock - ** Add more info **
def ChangeSCS_constraint(model, t):
    if t < max(model.t):
        return model.cSCS[t] == sum(model.SCS[t + 1, j] for j in model.j) - sum(model.SCS[t, j] for j in model.j)
    else:
        return Constraint.Skip
model.ChangeSCS = Constraint(model.t, rule=ChangeSCS_constraint)

# Constraint for change in total carbon stock - ** Add more info **
def ChangeTCS_constraint(model, t):
    return model.cTCS[t] == model.cBCS[t] + model.cPCS[t] + model.cSCS[t]
model.ChangeTCS = Constraint(model.t, rule=ChangeTCS_constraint)
#endregion

#region     *Warm constraint*
# Constraint for warming power of landscape albedo - ** Add more info **
# **GAMS-versiossa "totalarea" eikä "totalArea"**
def Warm_constraint(model, t):
    return model.wp[t] == (100 / 51) * (sum(model.w[a] * (model.x[t, a] - model.z[t, a]) for a in model.a) + model.MWopenshrub * (model.totalArea - sum(model.x[t, a] - model.z[t, a] for a in model.a)))
model.Warm = Constraint(model.t, rule=Warm_constraint)
#endregion

#region     *Model Objective*
# Model objective - maximize welfare
def objective_rule(model):
    return model.wf
model.objective = Objective(rule=objective_rule, sense=maximize)
#endregion

#       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       
###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
########################################################################################################################################
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   
###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     
#       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       

#region *Solver options*
# ****************************************

# Set solver, solver options, and model info output
from pyomo.util.model_size import build_model_size_report

# Create and print report of model
report = build_model_size_report(model)
print("*** Model Report ***")
print(report)

# Set solver to Ipopt
solver = SolverFactory('ipopt')

# Use this to set maximum iterations
#solver.options['max_iter'] = 10

# Use this to set acceptable tolerance level of results (Ipopt) 
#solver.options['tol'] = 1e-06

# Set print detail level for Ipopt
solver.options['print_level'] = 5

solver_results = solver.solve(model, tee=True)
#endregion

#region *Print welfare result*
# Print welfare value
print("Objective result: model.wf.value")
print(model.wf.value)
#endregion

#region *Print variables used in the model (debugging)*
# Print variables used in the model
print("")
print("*Variables used in the model (debugging)")
print("*Variable bounds*")
variables = model.component_data_objects(Var)

var_count = {}
unique_vars = set()
for var in variables:
    # Basename of var
    base_name = var.name.split('[')[0]
    
    # Count the occurrences
    if base_name not in var_count:
        var_count[base_name] = 0
    var_count[base_name] += 1
    
    # Print
    if base_name not in unique_vars:
        print(f"Variable: {base_name}, Lower Bound: {var.lb}, Upper Bound: {var.ub}")
        unique_vars.add(base_name)

# Print the count of indices for each variable and the total number of variables
print("\nVariable Index Counts:")
total_variables = 0
for var_name, count in var_count.items():
    print(f"{var_name}: {count} indices")
    total_variables += count
print(f"\nTotal number of variables: {total_variables}")
#endregion

#       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       
###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
########################################################################################################################################
####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### 
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####   
###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     ###     
#       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       #       

#region ****** PRINT RESULTS *************
# ****************************************

#Parameter
keypars = {}            # "parameters put into excel"
scalarresults = {}      # "scalar results"
ma = {}                 # mean age of forests
TCS = {}                # total carbon stock (biomass & soils & products);

#ma(t)    = timestep*sum(a, x.l(t,a)*(ord(a)))/(0.000000001+sum(a, x.l(t,a)))-timestep/2;
for t in model.t:
    ma[t]  = timestep * sum(model.x[t, a].value * a for a in model.a) / (0.000000001 + sum(model.x[t, a].value for a in model.a)) - timestep / 2

#TCS(t)   = (gammav+gammab)*(sum(a, v(a)*x.l(t,a)))+sum(j, SCS.l(t,j))+sum(k, PCS.l(t,k));
for t in model.t:
    TCS[t] = (gammav + gammab) * (sum(model.v[a] * model.x[t, a].value for a in model.a)) + sum(model.SCS[t, j].value for j in model.j) + sum(model.PCS[t, k].value for k in model.k)

keypars['pcScale']       = pcScale + 1e-8
keypars['scfScale']      = scfScale + 1e-8
keypars['constantCase']  = constantCase + 1e-8
keypars['noLUC']         = noLUC + 1e-8
keypars['region']        = region
keypars['Alb_str']       = albstren + 1e-8
keypars['creg']          = creg
keypars['cf']            = cf
keypars['telast']        = telast
keypars['felast']        = felast

import pandas as pd
scalarresults = {
    'MODELSTAT': solver_results.solver.termination_condition,
    'SOLVESTAT': solver_results.solver.status,
    'welfare': model.wf.value,
    'wf_food': sum((model.uy[t].value - (model.chf * model.hy[t].value + model.cf * model.y[t].value)) * model.df[t] for t in model.t),
    'wf_timber': sum(
        (model.ux[t].value - (
            model.cht * model.hx[t].value
            + model.creg * (
                (sum(model.z[t, a].value for a in model.a) + model.y[t - 1].value - model.y[t].value) if t > 0
                else (sum(model.z[t, a].value for a in model.a) + model.yinit - model.y[t].value)
            )
        )) * model.df[t]
        for t in model.t
    )
}

if constantCase == 1:
    scalarresults['wf_carbon'] = sum(pcInit[0] * (11/3) * model.cTCS[t].value * model.df[t] for t in model.t)
    scalarresults['wf_albedo'] = -sum(model.timestep * scfInit[0] * model.wp[t].value * model.df[t] for t in model.t)
else:
    scalarresults['wf_carbon'] = sum(pcInit[t] * (11/3) * model.cTCS[t].value * model.df[t] for t in model.t)
    scalarresults['wf_albedo'] = -sum(model.timestep * scfInit[t] * model.wp[t].value * model.df[t] for t in model.t)

results = {
    'year': {}, 'SCC': {}, 'SCF': {}, 'y': {}, 'forest_area': {}, 'timber_harvest': {},
    'timber_price': {}, 'food_price': {}, 'Biomass_Carbon': {}, 'Soil_Carbon': {},
    'Product_Carbon': {}, 'Total_C-stock': {}, 'Change_in_TCS': {}, 'Albedo_warming': {},
    'TCS_per_forest_area': {}, 'mean_age': {}, 'utility': {}, 'u_food': {}, 'u_timber': {},
    'u_carbon': {}, 'u_albedo': {}, 'x': {}, 'z': {}, 'theta': {}
}

for t in model.t:
    results['year'][t] = model.timestep * (t) + 1e-8

    results['SCC'][t] = model.pc[t] + 1e-8
    results['SCF'][t] = model.scf[t] + 1e-8

    results['y'][t] = model.y[t].value
    results['forest_area'][t] = totalArea - model.y[t].value
    results['timber_harvest'][t] = model.hx[t].value
    results['timber_price'][t] = model.pt[t].value
    results['food_price'][t] = model.pf[t].value

    results['Biomass_Carbon'][t] = (gammav + gammab) * sum(model.v[a] * model.x[t, a].value for a in model.a)
    results['Soil_Carbon'][t] = sum(model.SCS[t, j].value for j in model.j)
    results['Product_Carbon'][t] = sum(model.PCS[t, k].value for k in model.k)

    results['Total_C-stock'][t] = TCS[t]
    results['Change_in_TCS'][t] = model.cTCS[t].value
    results['Albedo_warming'][t] = model.wp[t].value

    if t < len(model.t):
        results['TCS_per_forest_area'][t] = TCS[t] / (totalArea - model.y[t].value)
    results['mean_age'][t] = ma[t]
    results['utility'][t] = (
        model.ux[t].value + model.uy[t].value - model.c[t].value + model.pc[t] * (11 / 3) * model.cTCS[t].value
    )
    results['u_food'][t] = model.uy[t].value - (model.chf * model.hy[t].value + model.cf * model.y[t].value)
    results['u_timber'][t] = model.ux[t].value - (
        model.cht * model.hx[t].value +
        model.creg * (
            (sum(model.z[t, a].value for a in model.a) + model.y[t - 1].value - model.y[t].value) if t > 0
            else (sum(model.z[t, a].value for a in model.a) + model.yinit - model.y[t].value)
        )
    )

    if constantCase == 1:
        results['u_carbon'][t] = pcInit[0] * (11 / 3) * model.cTCS[t].value
        results['u_albedo'][t] = -model.timestep * scfInit[0] * model.wp[t].value
    else:
        results['u_carbon'][t] = pcInit[t] * (11 / 3) * model.cTCS[t].value
        results['u_albedo'][t] = -model.timestep * scfInit[t] * model.wp[t].value

for t in model.t:
    for a in model.a:
        results['x'].setdefault(t, {})[a] = model.x[t, a].value
        results['z'].setdefault(t, {})[a] = model.z[t, a].value
        results['theta'].setdefault(t, {})[a] = model.z[t, a].value / model.x[t, a].value if model.x[t, a].value > 0 else None

x_z_theta_data = []
for t in model.t:
    for a in model.a:
        x_val = results['x'][t][a]
        z_val = results['z'][t][a]
        theta_val = results['theta'][t][a]
        x_z_theta_data.append({
            'year': results['year'][t],
            'age_class': a,
            'x': x_val,
            'z': z_val,
            'theta': theta_val
        })

results_df = pd.DataFrame({
    'year': [results['year'][t] for t in model.t],
    'SCC': [results['SCC'][t] for t in model.t],
    'SCF': [results['SCF'][t] for t in model.t],
    'y': [results['y'][t] for t in model.t],
    'forest_area': [results['forest_area'][t] for t in model.t],
    'timber_harvest': [results['timber_harvest'][t] for t in model.t],
    'timber_price': [results['timber_price'][t] for t in model.t],
    'food_price': [results['food_price'][t] for t in model.t],
    'Biomass_Carbon': [results['Biomass_Carbon'][t] for t in model.t],
    'Soil_Carbon': [results['Soil_Carbon'][t] for t in model.t],
    'Product_Carbon': [results['Product_Carbon'][t] for t in model.t],
    'Total_C-stock': [results['Total_C-stock'][t] for t in model.t],
    'Change_in_TCS': [results['Change_in_TCS'][t] for t in model.t],
    'Albedo_warming': [results['Albedo_warming'][t] for t in model.t],
    'TCS_per_forest_area': [results['TCS_per_forest_area'][t] for t in model.t],
    'mean_age': [results['mean_age'][t] for t in model.t],
    'utility': [results['utility'][t] for t in model.t],
    'u_food': [results['u_food'][t] for t in model.t],
    'u_timber': [results['u_timber'][t] for t in model.t],
    'u_carbon': [results['u_carbon'][t] for t in model.t],
    'u_albedo': [results['u_albedo'][t] for t in model.t],
    'x': [results['x'][t] for t in model.t],
    'z': [results['z'][t] for t in model.t],
    'theta': [results['theta'][t] for t in model.t],
})

scalarresults_df = pd.DataFrame([scalarresults])
x_z_theta_df = pd.DataFrame(x_z_theta_data)
keypars_df = pd.DataFrame([keypars])

from openpyxl import * 
#with pd.ExcelWriter('output/albedo_model_results.xlsx', engine='openpyxl') as writer:
#    # Write keypars_df, scalarresults_df and results_df to xlsx
#    keypars_df.to_excel(writer, sheet_name='Results', index=False, startrow=0)
#    scalarresults_df.to_excel(writer, sheet_name='Results', index=False, startrow=4)
#    results_df.to_excel(writer, sheet_name='Results', index=False, startrow=8)

with pd.ExcelWriter('output/albedo_model_results.xlsx', engine='openpyxl') as writer:
    # Write keypars_df and scalarresults_df to the same sheet
    keypars_df.to_excel(writer, sheet_name='Keypars & Scalarresults', index=False, startrow=0)
    scalarresults_df.to_excel(writer, sheet_name='Keypars & Scalarresults', index=False, startrow=len(keypars_df) + 4)

    # Write results_df to a separate sheet
    results_df.to_excel(writer, sheet_name='Results', index=False, startrow=0)
#endregion