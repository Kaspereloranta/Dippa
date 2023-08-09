import numpy as np
import waste_pickup_sim
import json
import random
# coding: utf-8

sim_config = {	
	'sim_name': 'Biomass transportation to biogas facility',
	'sim_type': 2, # 1=Grass and straws, 2=manures
	'isTimeCriticalityConsidered' :  'False', # Boolean
	'sim_runtime_days': 14, # Simulation runtime in days, kalenterivuoden 2022 tyopaivat
	'pickup_sites_filename': 'geo_data/nearest_pickup_sitesKuivalannat.geojson',
	'depots_filename': 'geo_data/sim_test_terminals.geojson', # Nyt depots = terminals, vain 1, Biokaasulaitos. 
	'depot_capacity' : 14000, # yearly targeted input for facility
	'terminals_filename': 'geo_data/sim_test_terminals.geojson', 
	'vehicle_template': {
		'load_capacity': 45, # Keskim. arvaus
		'max_route_duration': 8*60 + 15, # Minutes (9h - 45min break = 8h 15min) 
		'break_duration': 45, # Minutes # Break Happens after 1/2 of drivetime 
		'num_breaks_per_shift': 1,
		'pickup_duration': 10, # Minutes # Tama 10 min = keruiden asetusaika, sama kaikille biomassoille. Simulaatiossa ja optimoinnissa huomioidaan keruun keston lineaarinen komponentti.
							  # Vakiokomponenttia yllapidetaan myos: routing_optimizer.cpp rivi 22
							  # Lin. kommponentti simulaatiossa: waste_pickup_sim.py rivit 254 ja 266  
							  #  Lin. kommponentti optimoijassa: LogisticsSimulation::pickup
		'load_TS_rate': 0.0
	},
	'depots': [
		{
			'num_vehicles': 10 # TESTAA MYOS ERI ARVOILLA
		}
	]
}

def hypothesis_test():
	"""
	"""
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