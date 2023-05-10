import numpy as np
import waste_pickup_sim
import json
import random

sim_config = {	
	'sim_name': 'Hämeenlinna and nearby regions',
	'sim_runtime_days': 14, #14, # Simulation runtime in days <- Case Envor pidempi? (ruohomassa) # PITÄÄ MUUTTAA MYÖS OPTIMOIJAAN ! ! , routing_optimizer.cpp rivi 136 ! !, päivät työpäiviä, ei viikonloppuaj tms. vapaita mukana
	'pickup_sites_filename': 'geo_data/pickupSitesWithMassCoefficients.geojson', # JOS PAIKKOJA PALJON, KANNATTAA LISÄTÄ VIIVE JOTTEI REITITIN HAE LIIAN NOPEASTI
	'depots_filename': 'geo_data/sim_test_terminals.geojson', 
	'terminals_filename': 'geo_data/sim_test_terminals.geojson', # Nämä saadaaan oletetttavasti Lukelta (pickup_sites, depots ja terminals)
	'vehicle_template': {
		'load_capacity': 18, # Tonnes # Selvitettävä haastatteluissa, sekä käytetäänkö erilaisia ajoneuvoja biomassa vs. ruoho? # KÄÄNNETTÄVÄ MYÖS OPTIMOIJAN PUOLELLE EHKÄ
		'max_route_duration': 8*60 + 15, # Minutes (9h - 45min break = 8h 15min) # Selvitettävä haastatteluissa # KÄÄNNETTÄVÄ MYÖS OPTIMOIJAN PUOLELLE EHKÄ
#		'break_duration': 45, # Minutes # Break Happens after 1/2 of drivetime # Selvitettävä haastatteluissa # KÄÄNNETTÄVÄ MYÖS OPTIMOIJAN PUOLELLE  EHKÄ
#		'num_breaks_per_shift': 1,
		'pickup_duration': 15 # Minutes # Selvitettävä haastatteluissa # KÄÄNNETTÄVÄ MYÖS OPTIMOIJAN PUOLELLE 
	},
	'depots': [
		{
			'num_vehicles': 1 # Selvitettävä haastatteluissa
		},
		{
			'num_vehicles': 1 # Selvitettävä haastatteluissa
		}
	]
}

def hypothesis_test():
	"""
	"""
	# Tänne jotain toteutusta simulointien vertailuun eri oletuksilla?

	# Runs N simulation
	# logs them to list of jsons 
	pass 

def test_record():
	"""
	"""
	# List of jsons of sim runs 
	# Metadata on time avergae time of coptutaion and vehicle dricing time
	# config
	pass

random.seed(42)
np.random.seed(42)
waste_pickup_sim.preprocess_sim_config(sim_config, 'temp/sim_preprocessed_config.json')
sim = waste_pickup_sim.WastePickupSimulation(sim_config)
sim.sim_run()
sim.save_log()
sim.sim_record()