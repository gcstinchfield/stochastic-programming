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

        

class Farmer:
    def __init__(self):
        self.total_acres=500
        self.crop_yield=self.Farmer_attr(units="T/acre",
                                         values={"wheat":2.5,
                                                 "corn":3,
                                                 "beets":20})
        self.planting_cost=self.Farmer_attr(units="$/acre",
                                            values={"wheat":150,
                                                    "corn":230,
                                                    "beets":260})
        self.selling_price=self.Farmer_attr(units="$/T",
                                            values={"wheat":170,
                                                    "corn":150,
                                                    "beets_favorable":36, 
                                                    "beets_unfavorable":10})
        self.min_requirement=self.Farmer_attr(units="T",
                                              values={"wheat":200,
                                                      "corn":240})
        self.purchase_price=self.Farmer_attr(units="$/T",
                                             values={"wheat":238,
                                                     "corn":210})
        self.beets_quota=6000
    
    class Farmer_attr:
        def __init__(self,units,values):
            self.units=units
            self.wheat=values["wheat"]
            self.corn=values["corn"]
            if len(values.keys())==3:
                self.beets=values["beets"]
            elif len(values.keys())==4:
                self.beets_favorable=values["beets_favorable"]
                self.beets_unfavorable=values["beets_unfavorable"]

    def build_deterministic_model(self):
                        
        # create pyomo model
        model = pyo.ConcreteModel()

        """ VARIABLES """

        # land variables [=] acres of land devoted to each crop
        model.x=pyo.Var(["wheat","corn","beets"], 
                        within=pyo.NonNegativeReals)

        # selling decision variables [=] tons of crop sold
        model.w=pyo.Var(["wheat","corn","beets_favorable", "beets_unfavorable"], 
                        within=pyo.NonNegativeReals)

        # purchasing decision variables [=] tons of crop purchased
        model.y=pyo.Var(["wheat","corn"], 
                        within=pyo.NonNegativeReals)

        """ OPTIMIZATION """

        # deteremine how much land to devote to each crop, maximizing E[profit]
        model.obj=pyo.Objective( expr= model.x["wheat"]*self.planting_cost.wheat + 
                                    model.x["corn"]*self.planting_cost.corn + 
                                    model.x["beets"]*self.planting_cost.beets +\
                                        model.y["wheat"]*self.purchase_price.wheat - 
                                        model.w["wheat"]*self.selling_price.wheat +\
                                            model.y["corn"]*self.purchase_price.corn 
                                            - model.w["corn"]*self.selling_price.corn +\
                                                -model.w["beets_favorable"]*self.selling_price.beets_favorable 
                                                -model.w["beets_unfavorable"]*self.selling_price.beets_unfavorable  )

        # total acres allocated cannot exceed total available acreas
        @model.Constraint()
        def total_acreage_allowed(model):
            return ( model.x["wheat"]+model.x["corn"]+model.x["beets"]<=self.total_acres )

        # must have at least 200 (T) of wheat
        @model.Constraint()
        def minimum_wheat_requirement(model):
            # total acres allotted * yield / acre + tons of wheat purchased - tons of wheat sold
            return ( model.x["wheat"]*self.crop_yield.wheat + model.y["wheat"] - model.w["wheat"] \
                    >= self.min_requirement.wheat)

        # have at least 240 (T) of corn
        @model.Constraint()
        def minimum_corn_requirement(model):
            # total acres allotted * yield / acre + tons of corn purchased - tons of corn sold
            return ( model.x["corn"]*self.crop_yield.corn + model.y["corn"] - model.w["corn"] \
                    >= self.min_requirement.corn)

        # the total tons of sugar beats sold, at either price, must be equal to the amount produced.
        @model.Constraint()
        def sugar_beet_mass_balance(model):
            # sugar beets sold unfavorably + sold favorably <= total acreas allotted * yeild / acre
            return ( model.w["beets_favorable"] + model.w["beets_unfavorable"] \
                    <= self.crop_yield.beets*model.x["beets"] )

        # the favorably priced beets cannot exceed 6000 (T)
        @model.Constraint()
        def sugar_beet_quota(model):
            # sugar beets sold at favorable price <= 6000 tons
            return ( model.w["beets_favorable"] <= self.beets_quota )
        
        self.deterministic_model=model

    def solve_pyomo_model(self,which):

        if which=="deterministic":
            # solve the deterministic LP
            opt=pyo.SolverFactory('gurobi')
            solver_result=opt.solve(self.deterministic_model, 
                                    tee=True)
            self.detereministic_result=solver_result
    
    def show_results(self):

        """ RESULTS """

        print("\nSurface (acres)")
        print("\tWheat =", pyo.value(self.deterministic_model.x["wheat"]))
        print("\tCorn =", pyo.value(self.deterministic_model.x["corn"]))
        print("\tBeets =", pyo.value(self.deterministic_model.x["beets"]))

        print("Yield (T)")
        print("\tWheat =", pyo.value(self.deterministic_model.x["wheat"])*self.crop_yield.wheat)
        print("\tCorn =", pyo.value(self.deterministic_model.x["corn"])*self.crop_yield.corn)
        print("\tBeets =", pyo.value(self.deterministic_model.x["beets"])*self.crop_yield.beets)

        print("Sales (T)")
        print("\tWheat =", pyo.value(self.deterministic_model.w["wheat"]))
        print("\tCorn =", pyo.value(self.deterministic_model.w["corn"]))
        print("\tBeets =", pyo.value(self.deterministic_model.w["beets_favorable"])
              +pyo.value(self.deterministic_model.w["beets_unfavorable"]))

        print("Purchase (T)")
        print("\tWheat =", pyo.value(self.deterministic_model.y["wheat"]))
        print("\tCorn =", pyo.value(self.deterministic_model.y["corn"]))
        print("\tBeets = we never purchase.")

        print("\nOverall Profit =", -pyo.value(self.deterministic_model.obj))

if __name__=="__main__":
    farmer=Farmer()
    farmer.build_deterministic_model()
    farmer.solve_pyomo_model("deterministic")
    farmer.show_results()