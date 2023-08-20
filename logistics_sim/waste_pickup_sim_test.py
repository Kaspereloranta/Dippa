import numpy as np
import waste_pickup_sim
import json
import random
# coding: utf-8

# BIOMASS_TYPES 1=GRASS AND STRAWS, 2=DRY MANURES, 3=SLURRY MANURES

sim_config = {	
	'sim_name': 'Biomass transportation to biogas facility',
	'isTimeCriticalityConsidered' :  'False', # Boolean
	'sim_runtime_days': 251, # Simulation runtime in days, Kalenterivuoden 2023 tyopaivat
	'pickup_sites_filename': 'geo_data/nearest_pickup_sites.geojson', #
	'depots_filename': 'geo_data/sim_test_terminals.geojson', # Nyt depots = terminals, vain 1, Biokaasulaitos. 
	'grass_capacity' : 28000,
	'drymanure_capacity': 14000,
	'slurrymanure_capacity': 14000,
	'terminals_filename': 'geo_data/sim_test_terminals.geojson', 
	'vehicle_template': {
		'load_capacity': 45, # Keskim. arvaus
		'max_route_duration': 8*60 + 15, # Minutes (9h - 45min break = 8h 15min) 
		'break_duration': 45, # Minutes # Break Happens after 1/2 of drivetime 
		'num_breaks_per_shift': 1,
		'pickup_duration': 10, # Minutes # Tama 10 min = keruiden asetusaika, sama kaikille biomassoille. 
							   # Simulaatiossa keruun kesto asetusaika + pickup_amount*pickup_rate
		'load_TS_rate': 0.0
	},
	'depots': [
		{
			'num_vehicles': 9 # Distributes equally between vehicle types. 
		}
	],
	'biomass_type_mapping':{
		'Hevoset ja ponit, kuivalanta yhteensa varastosta': 2,
		'Nautojen lietelanta elainsuojasta 2015': 3,
		'Lihakarjan lietelanta elainsuojasta' : 3,  
		'Lihakarjan kuivalanta yhteensa varastosta': 2,
		'Broilerit, kalkkunat ja muu siipikarja kuivalanta yhteensa varastosta' : 2,
		'Emakot ja porsaat lietelanta elainsuojasta' : 3,
		'Sikojenkuivalantavarastosta2016' : 2,
		'Emakot ja porsaat kuivalanta yhteensa varastosta' : 2, 
		'Sivuvirta: Kesantonurmi' : 1,
		'Sivuvirta: Olki' : 1,
		'Sivuvirta: Kuivaheinanurmet': 1
	}
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