import pyomo.environ as pyo

from mpisppy.opt.ph import PH
import mpisppy.utils.sputils as sputils
from mpisppy.opt.ef import ExtensiveForm
from mpisppy.opt.lshaped import LShapedMethod

from mpi4py import MPI

""""
A farmer is trying to decide what to plant for the next season.
    - He has wheat (1), corn (2), and sugar beets (3)
    - He can buy (y), sell (w), and produce (x,x_yield) any of these three products.
    - He must buy / produce a min. amount of wheat / corn for his cattle.
    - Quota on sugar beets alters the price after a certain threshold.
Goal: max. the net proficts from purchasing, selling, and planting crops for next season.

However, there is no guarantees on the weather. The actual yield of the crops changes depending on weather -> aka uncertain.
Here, let's consider 3 scenarios: good, fair, or bad weather next year.
Link the different scenario variables together to get the best possible solution.

Here, we use MPI-SPPY to solve => https://mpi-sppy.readthedocs.io/en/latest/examples.html

"""

class Farmer():
    def __init__(self,weather_type):

        # for scenario rep, the yield changes (+/-20%) based on weather.
        global predicted_yield
        if weather_type == "good": predicted_yield=1.2
        if weather_type == "fair": predicted_yield=1
        if weather_type == "bad": predicted_yield=0.8

        self.crop_yield={"wheat":2.5*predicted_yield,
                        "corn":3*predicted_yield,
                        "beets":20*predicted_yield}

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

def build_deterministic_model(scenario):
                    
    # create pyomo model
    model = pyo.ConcreteModel()

    """ VARIABLES """

    # land variables [=] acres of land devoted to each crop
    model.x=pyo.Var(scenario.planting_crops, 
                    within=pyo.NonNegativeReals)

    # selling decision variables [=] tons of crop sold
    model.w=pyo.Var(scenario.selling_crops, 
                    within=pyo.NonNegativeReals)

    # purchasing decision variables [=] tons of crop purchased
    model.y=pyo.Var(scenario.purchasing_crops, 
                    within=pyo.NonNegativeReals)

    """ CONSTRAINTS """

    # now, we split objective into first / second stage varialbes for mpi-sppy
    model.planting_cost=sum(model.x[planted_crop]*scenario.planting_cost[planted_crop] for planted_crop in scenario.planting_crops)
    model.selling_cost=sum(model.w[sold_crop]*scenario.selling_price[sold_crop] for sold_crop in scenario.selling_crops)
    model.puchasing_cost=sum(model.y[purchased_crop]*scenario.purchase_price[purchased_crop] for purchased_crop in scenario.purchasing_crops)

    model.obj=pyo.Objective( expr= model.planting_cost - model.selling_cost + model.puchasing_cost )

    # total acres allocated cannot exceed total available acreas
    @model.Constraint()
    def total_acreage_allowed(model):
        return ( sum(model.x[planted_crop] for planted_crop in scenario.planting_crops) <= scenario.total_acres )

    # must have at least x of wheat,corn
    @model.Constraint(scenario.required_crops)
    def minimum_requirement(model, required_crop):
        return ( model.x[required_crop]*scenario.crop_yield[required_crop] + model.y[required_crop] - model.w[required_crop] \
                >= scenario.min_requirement[required_crop])
    
    @model.Constraint()
    def sugar_beet_mass_balance(model):
        return ( model.w["beets_favorable"] + model.w["beets_unfavorable"] \
                <= scenario.crop_yield["beets"]*model.x["beets"] )

    # the favorably priced beets cannot exceed 6000 (T)
    @model.Constraint()
    def sugar_beet_quota(model):
        return ( model.w["beets_favorable"] <= scenario.beets_quota )
    
    return model

def scenario_creator(scenario):
    # create parameters / model stored in obj for this scenario
    farmer_scenario=Farmer(scenario)
    model=build_deterministic_model(farmer_scenario)

    # add to the root node
    sputils.attach_root_node(model, model.planting_cost, [model.x])

    # add probability of scenario occuring
    model._mpisppy_probability = 1.0 / 3

    return model

if __name__=="__main__":
    
    all_scenarios = ["good", "fair", "bad"]

    # solve extensive form, via MPI-SPPY
    options = {"solver": "gurobi"}
    ef = ExtensiveForm(options, all_scenarios, scenario_creator)
    results = ef.solve_extensive_form()
    objval = ef.get_objective_value()
    print("Extensive form obj =", f"{objval:.1f}")  

    # solving using PH
    options = {
        "solvername": "gurobi_persistent",
        "PHIterLimit": 200,
        "defaultPHrho": 1,
        "convthresh": 1e-7,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
    }
    ph = PH(
            options,
            all_scenarios,
            scenario_creator,
    )
    conv, obj, _ =ph.ph_main()

    # accessing objective value?
    
    # solve using Bender's
    bounds = {name: -432000 for name in all_scenarios}
    options = {
        "root_solver": "gurobi",
        "sp_solver": "gurobi",
        "valid_eta_lb": bounds,
        "max_iter": 10,
    }
    ls = LShapedMethod(options, all_scenarios, scenario_creator)
    result = ls.lshaped_algorithm()