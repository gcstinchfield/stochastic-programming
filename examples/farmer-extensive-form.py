import pyomo.environ as pyo

""""
This is the extensive form of the farmer problem example.
  (section 1.1b, pg. 7-8, Birge & Louveax)

A farmer is trying to decide what to plant for the next season.
    - He has wheat (1), corn (2), and sugar beets (3)
    - He can buy (y), sell (w), and produce (x,x_yield) any of these three products.
    - He must buy / produce a min. amount of wheat / corn for his cattle.
    - Quota on sugar beets alters the price after a certain threshold.
Goal: max. the net proficts from purchasing, selling, and planting crops for next season.

However, there is no guarantees on the weather. The actual yield of the crops changes depending on weather -> aka uncertain.
Here, let's consider 3 scenarios: good, fair, or bad weather next year.
Link the different scenario variables together to get the best possible solution.

"""

class Farmer():
    def __init__(self,weather_types="fair"):

        # record the scenario (i.e. weather type)
        self.weathers=weather_types

        # if only 1 weather, deterministic.
        if type(weather_types)==str:

            # for scenario rep, the yield changes (+/-20%) based on weather.
            if weather_types == "good": predicted_yield=1.2
            if weather_types == "fair": predicted_yield=1
            if weather_types == "bad": predicted_yield=0.8

            self.crop_yield={"wheat":2.5*predicted_yield,
                            "corn":3*predicted_yield,
                            "beets":20*predicted_yield}

        # if > 1 weather, multiple scenarios.
        elif type(weather_types)==list:

            crop_yield_scenarios={}
            for weather in weather_types:

                # for scenario rep, the yield changes (+/-20%) based on weather.
                if weather == "good": predicted_yield=1.2
                if weather == "fair": predicted_yield=1
                if weather == "bad": predicted_yield=0.8

                crop_yield_scenarios[weather]={"wheat":2.5*predicted_yield,
                                                "corn":3*predicted_yield,
                                                "beets":20*predicted_yield}

            self.crop_yield=crop_yield_scenarios

        # these do not change, regardless of scenario.
        self.total_acres=500
        self.planting_cost={"wheat":150,
                            "corn":230,
                            "beets":260}
        self.planting_crops=["wheat","corn","beets"]
        self.selling_price={"wheat":170,
                            "corn":150,
                            "beets_favorable":36, 
                            "beets_unfavorable":10}
        self.selling_crops=["wheat", "corn", "beets_favorable", "beets_unfavorable"]
        self.min_requirement={"wheat":200,
                              "corn":240}
        self.purchase_price={"wheat":238,
                             "corn":210}
        self.purchasing_crops=["wheat","corn"]
        self.required_crops=self.purchasing_crops
        self.beets_quota=6000


    def build_deterministic_model(self):
                        
        # create pyomo model
        model = pyo.ConcreteModel()

        """ VARIABLES """

        # land variables [=] acres of land devoted to each crop
        model.x=pyo.Var(self.planting_crops, 
                        within=pyo.NonNegativeReals)

        # selling decision variables [=] tons of crop sold
        model.w=pyo.Var(self.selling_crops, 
                        within=pyo.NonNegativeReals)

        # purchasing decision variables [=] tons of crop purchased
        model.y=pyo.Var(self.purchasing_crops, 
                        within=pyo.NonNegativeReals)

        """ CONSTRAINTS """

        model.obj=pyo.Objective( expr= sum(model.x[planted_crop]*self.planting_cost[planted_crop] for planted_crop in self.planting_crops) \
                                        -sum(model.w[sold_crop]*self.selling_price[sold_crop] for sold_crop in self.selling_crops) \
                                         +sum(model.y[purchased_crop]*self.purchase_price[purchased_crop] for purchased_crop in self.purchasing_crops ))

        # total acres allocated cannot exceed total available acreas
        @model.Constraint()
        def total_acreage_allowed(model):
            return ( sum(model.x[planted_crop] for planted_crop in self.planting_crops) <=self.total_acres )

        # must have at least x of wheat,corn
        @model.Constraint(self.required_crops)
        def minimum_requirement(model, required_crop):
            return ( model.x[required_crop]*self.crop_yield[required_crop] + model.y[required_crop] - model.w[required_crop] \
                    >= self.min_requirement[required_crop])
        
        @model.Constraint()
        def sugar_beet_mass_balance(model):
            return ( model.w["beets_favorable"] + model.w["beets_unfavorable"] \
                    <= self.crop_yield["beets"]*model.x["beets"] )

        # the favorably priced beets cannot exceed 6000 (T)
        @model.Constraint()
        def sugar_beet_quota(model):
            return ( model.w["beets_favorable"] <= self.beets_quota )
        
        # add model to the class obj
        self.deterministic_model=model
    

    def build_extensive_form_model(self):
                        
        # create pyomo model
        model = pyo.ConcreteModel()

        """ VARIABLES """

        # land variables [=] acres of land devoted to each crop
        model.x=pyo.Var(self.planting_crops, 
                        within=pyo.NonNegativeReals)

        # selling decision variables [=] tons of crop sold
        model.w=pyo.Var(self.weathers,
                        self.selling_crops, 
                        within=pyo.NonNegativeReals)

        # purchasing decision variables [=] tons of crop purchased
        model.y=pyo.Var(self.weathers,
                        self.purchasing_crops, 
                        within=pyo.NonNegativeReals)

        """ CONSTRAINTS """

        model.obj=pyo.Objective( expr = sum(model.x[planted_crop]*self.planting_cost[planted_crop] for planted_crop in self.planting_crops) + \
                                        (-1/3)*sum(model.w[weather,sold_crop]*self.selling_price[sold_crop] for sold_crop in self.selling_crops for weather in self.weathers) \
                                             +(1/3)*sum(model.y[weather,purchased_crop]*self.purchase_price[purchased_crop] for purchased_crop in self.purchasing_crops for weather in self.weathers) )

        # total acres allocated cannot exceed total available acreas
        @model.Constraint()
        def total_acreage_allowed(model):
            return ( sum(model.x[planted_crop] for planted_crop in self.planting_crops) <= self.total_acres )

        # must have at least x of wheat,corn
        @model.Constraint(self.weathers, self.required_crops)
        def minimum_requirement(model, weather, required_crop):
            return ( model.x[required_crop]*self.crop_yield[weather][required_crop] + model.y[weather,required_crop] - model.w[weather,required_crop] \
                    >= self.min_requirement[required_crop])

        @model.Constraint(self.weathers)
        def sugar_beet_mass_balance(model, weather):
            return ( model.w[weather,"beets_favorable"] + model.w[weather,"beets_unfavorable"] \
                    <= self.crop_yield[weather]["beets"]*model.x["beets"] )

        # the favorably priced beets cannot exceed 6000 (T)
        @model.Constraint(self.weathers)
        def sugar_beet_quota(model, weather):
            return ( model.w[weather,"beets_favorable"] <= self.beets_quota )
        
        # add model to the class obj
        self.extensive_form_model=model


    def solve_pyomo_model(self,which):

        if which=="deterministic":
            # solve the deterministic form
            opt=pyo.SolverFactory('gurobi')
            solver_result=opt.solve(self.deterministic_model, 
                                    tee=False)
            self.detereministic_result=solver_result
        if which=="extensive":
            # solve the extensive form
            opt=pyo.SolverFactory('gurobi')
            solver_result=opt.solve(self.extensive_form_model, 
                                    tee=False)
            self.extensive_form_results=solver_result
    
    def show_results(self,which="deterministic"):

        if which=="deterministic":
            print("\nWeather =", self.weathers)
            print("\nSurface (acres)")
            print("\tWheat =", pyo.value(self.deterministic_model.x["wheat"]))
            print("\tCorn =", pyo.value(self.deterministic_model.x["corn"]))
            print("\tBeets =", pyo.value(self.deterministic_model.x["beets"]))

            print("Yield (T)")
            print("\tWheat =", pyo.value(self.deterministic_model.x["wheat"])*self.crop_yield["wheat"])
            print("\tCorn =", pyo.value(self.deterministic_model.x["corn"])*self.crop_yield["corn"])
            print("\tBeets =", pyo.value(self.deterministic_model.x["beets"])*self.crop_yield["beets"])

            print("Sales (T)")
            print("\tWheat =", pyo.value(self.deterministic_model.w["wheat"]))
            print("\tCorn =", pyo.value(self.deterministic_model.w["corn"]))
            print("\tBeets =", pyo.value(self.deterministic_model.w["beets_favorable"])
                +pyo.value(self.deterministic_model.w["beets_unfavorable"]))

            print("Purchase (T)")
            print("\tWheat =", pyo.value(self.deterministic_model.y["wheat"]))
            print("\tCorn =", pyo.value(self.deterministic_model.y["corn"]))
            print("\tBeets = we never purchase.")

            print("\nOverall Profit = $", round(-pyo.value(self.deterministic_model.obj),2))
        
        if which=="extensive":
            for weather in self.weathers:
                print("\nWeather =", weather)
                print("\nSurface (acres)")
                print("\tWheat =", pyo.value(self.extensive_form_model.x["wheat"]))
                print("\tCorn =", pyo.value(self.extensive_form_model.x["corn"]))
                print("\tBeets =", pyo.value(self.extensive_form_model.x["beets"]))

                print("Yield (T)")
                print("\tWheat =", pyo.value(self.extensive_form_model.x["wheat"])*self.crop_yield[weather]["wheat"])
                print("\tCorn =", pyo.value(self.extensive_form_model.x["corn"])*self.crop_yield[weather]["corn"])
                print("\tBeets =", pyo.value(self.extensive_form_model.x["beets"])*self.crop_yield[weather]["beets"])

                print("Sales (T)")
                print("\tWheat =", pyo.value(self.extensive_form_model.w[weather,"wheat"]))
                print("\tCorn =", pyo.value(self.extensive_form_model.w[weather,"corn"]))
                print("\tBeets =", pyo.value(self.extensive_form_model.w[weather,"beets_favorable"])
                                    +pyo.value(self.extensive_form_model.w[weather,"beets_unfavorable"]))

                print("Purchase (T)")
                print("\tWheat =", pyo.value(self.extensive_form_model.y[weather,"wheat"]))
                print("\tCorn =", pyo.value(self.extensive_form_model.y[weather,"corn"]))
                print("\tBeets = we never purchase.")

            print("\nE[Overall Profit] = $", round(-pyo.value(self.extensive_form_model.obj),2))

if __name__=="__main__":
    
    farmer=Farmer(["good","fair","bad"])
    farmer.build_extensive_form_model()
    farmer.solve_pyomo_model("extensive")
    farmer.show_results("extensive")