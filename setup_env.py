from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, Simulator, MeshcatVisualizer, StartMeshcat, Rgba
import numpy as np

# start the mesh 
meshcat = StartMeshcat()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

# add the kitchen model 
model_instances = Parser(plant).AddModels("./kitchen_model/kitchen.sdf")

plant.Finalize()
 

 
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# start the simulator 
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.Initialize()

# get context 
context = simulator.get_context()
diagram.ForcedPublish(context)


try:
    input()
except KeyboardInterrupt:
    print("\n程序已退出")
