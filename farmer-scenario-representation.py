import numpy as np
import pyomo.environ as pyo

""""
This is a scenario representation of the farmer problem example.
  (section 1.1b, pg. 6-7, Birge & Louveax)

A farmer is trying to decide what to plant for the next season.
    - He has wheat (1), corn (2), and sugar beets (3)
    - He can buy (y), sell (w), and produce (x,x_yield) any of these three products.
    - He must buy / produce a min. amount of wheat / corn for his cattle.
    - Quota on sugar beets alters the price after a certain threshold.
Goal: max. the net proficts from purchasing, selling, and planting crops for next season.

However, there is no guarantees on the weather. The actual yield of the crops changes depending on weather -> aka uncertain.
Here, let's consider 3 scenarios: good, fair, or bad weather next year.

"""

# create pyomo model
model = pyo.ConcreteModel()

""" VARIABLES """

# land variables
model.x1=pyo.Var(within=pyo.NonNegativeReals) # = acres of land devoted to wheat
model.x2=pyo.Var(within=pyo.NonNegativeReals) # = areas of land devoted to corn
model.x3=pyo.Var(within=pyo.NonNegativeReals) # = acrease of land devoted to sugar beets

# selling decision variables
model.w1=pyo.Var(within=pyo.NonNegativeReals) # = tons of wheat sold
model.w2=pyo.Var(within=pyo.NonNegativeReals) # = tons of corn sold
model.w3=pyo.Var(within=pyo.NonNegativeReals) # = tons of sugar beats sold at favorable price
model.w4=pyo.Var(within=pyo.NonNegativeReals) # = tons of sugar beats sold at un-favorable price

# purchasing decision variables
model.y1=pyo.Var(within=pyo.NonNegativeReals) # = tons of wheat purchased
model.y2=pyo.Var(within=pyo.NonNegativeReals) # = tons of corn purchased

""" PARAMETERS """

# total area available on the farm.
model.total_acres=pyo.Param(initialize=500) # acre

# the farmer knows from experience that his mean yeild is roughly the following.
model.x1_yield=pyo.Param(initialize=2.5)         # T/acre
model.x2_yield=pyo.Param(initialize=3)            # T/acre
model.x3_yield=pyo.Param(initialize=20)    # T/acre

# the farmers know the cost of planting each product
model.x1_planting_cost=pyo.Param(initialize=150)         # $/acre
model.x2_planting_cost=pyo.Param(initialize=230)          # $/acre
model.x3_planting_cost=pyo.Param(initialize=260)   # $/acre

# mean selling prices observed by the farmer over the last decade for the products.
model.w1_selling_price=pyo.Param(initialize=170)         # $/T
model.w2_selling_price=pyo.Param(initialize=150)          # $/T

# the European Commision imposes a quota on sugar beats, which affects price
model.w3_selling_price=pyo.Param(initialize=36)     # $/T
model.w4_selling_price=pyo.Param(initialize=10)   # $/T

# the farmer needs a min. requirement of wheat and corn for his cattle feed.
model.x1_min_requirement=pyo.Param(initialize=200)         # T
model.x2_min_requirement=pyo.Param(initialize=240)          # T

# if the farmer doesn't produce min. requirement, he must buy wheat / corn at wholesale prices.
model.y1_purchase_price=pyo.Param(initialize=238)         # $/T
model.y2_purchase_price=pyo.Param(initialize=210)          # $/T

# cannot sell more than the max quota of sugar beets a certain threshold
model.x3_quota=pyo.Param(initialize=6000)

""" OPTIMIZATION """

# deteremine how much land to devote to each crop, maximizing E[profit]
model.obj=pyo.Objective( expr= model.x1*model.x1_planting_cost + model.x2*model.x2_planting_cost + model.x3*model.x3_planting_cost +\
                                model.y1*model.y1_purchase_price - model.w1*model.w1_selling_price +\
                                model.y2*model.y2_purchase_price - model.w2*model.w2_selling_price +\
                                -model.w3*model.w3_selling_price - model.w4*model.w4_selling_price    )

# total acres allocated cannot exceed total available acreas
@model.Constraint()
def total_acreage_allowed(model):
    return ( model.x1+model.x2+model.x3<=model.total_acres )

# must have at least 200 (T) of wheat
@model.Constraint()
def minimum_wheat_requirement(model):
    # total acres allotted * yield / acre + tons of wheat purchased - tons of wheat sold
    return ( model.x1*model.x1_yield + model.y1 - model.w1 >= model.x1_min_requirement)

# have at least 240 (T) of corn
@model.Constraint()
def minimum_corn_requirement(model):
    # total acres allotted * yield / acre + tons of corn purchased - tons of corn sold
    return ( model.x2*model.x2_yield + model.y2 - model.w2 >= model.x2_min_requirement)

# the total tons of sugar beats sold, at either price, must be equal to the amount produced.
@model.Constraint()
def sugar_beet_mass_balance(model):
    # sugar beets sold unfavorably + sold favorably <= total acreas allotted * yeild / acre
    return ( model.w3 + model.w4 <= model.x3_yield*model.x3 )

# the favorably priced beets cannot exceed 6000 (T)
@model.Constraint()
def sugar_beet_quota(model):
    # sugar beets sold at favorable price <= 6000 tons
    return ( model.w3 <= model.x3_quota )

# solve the deterministic LP
opt=pyo.SolverFactory('gurobi')
solver_result=opt.solve(model, 
                        tee=True)

""" RESULTS """

print("\nSurface (acres)")
print("\tWheat =", pyo.value(model.x1))
print("\tCorn =", pyo.value(model.x2))
print("\tBeets =", pyo.value(model.x3))

print("Yield (T)")
print("\tWheat =", pyo.value(model.x1)*pyo.value(model.x1_yield))
print("\tCorn =", pyo.value(model.x2)*pyo.value(model.x2_yield))
print("\tBeets =", pyo.value(model.x3)*pyo.value(model.x3_yield))

print("Sales (T)")
print("\tWheat =", pyo.value(model.w1))
print("\tCorn =", pyo.value(model.w2))
print("\tBeets =", pyo.value(model.w3)+pyo.value(model.w4))

print("Purchase (T)")
print("\tWheat =", pyo.value(model.y1))
print("\tCorn =", pyo.value(model.y2))
print("\tBeets = we never purchase.")

print("\nOverall Profit =", -pyo.value(model.obj))