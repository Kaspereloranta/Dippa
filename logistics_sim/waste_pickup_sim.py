import math
import time
import simpy
import random
import numpy as np
import queue as queue
import json
import functools
from geopy.distance import geodesic
from os.path import exists
from datetime import datetime
import os

from routing_api import get_distance_and_duration_matrix

def time_to_string(minutes):
	hours = math.floor(minutes/60)
	minutes -= hours*60
	days = math.floor(hours/24)
	hours -= days*24
	return f"{str(days):>3}d {str(hours).zfill(2)}:{str(math.floor(minutes)).zfill(2)}"

def tons_to_string(tons):
	return f"{tons:.3f}t"

def lonlat_to_string(lonlat):
	return f"({lonlat[0]:.5f}, {lonlat[1]:.5f})"

def to_percentage_string(number):
	return f"{number*100:.0f}%"

def heuristic_router(routing_input):
#    "pickup_sites": [
#        {
#            "capacity": 2,
#            "level": 0.9848136938048543,
#            "growth_rate": 8.10984801062072e-05	

	# Sort sites based on when they become full
	indexes_and_times_when_full = []
	for pickup_site_index, pickup_site in enumerate(routing_input['pickup_sites']):
		time_when_full = (pickup_site['capacity'] - pickup_site['level'])/pickup_site['growth_rate']
		indexes_and_times_when_full.append((pickup_site_index, time_when_full))
	indexes_and_times_when_full.sort(key=lambda index_and_time_when_full: index_and_time_when_full[1])

	# Assign destinations to vehicles, in interleaved order so that every vehicle gets the same number of sites
	vehicle_routes = []

	for vehicle in routing_input['vehicles']:
		home_depot = routing_input['depots'][vehicle['home_depot_index']]
		vehicle_routes.append([home_depot['location_index']])

	vehicle_route_durations = [0 for _ in range(len(routing_input['vehicles']))]
	durations_back_home = [0 for _ in range(len(routing_input['vehicles']))]
	for index, pickup_site_index_and_time_when_full in enumerate(indexes_and_times_when_full):
		vehicle_index = index % len(routing_input['vehicles'])
		vehicle = routing_input['vehicles'][vehicle_index]
		home_depot_location_index = routing_input['depots'][vehicle['home_depot_index']]['location_index']
		proposed_location_index = routing_input['pickup_sites'][pickup_site_index_and_time_when_full[0]]['location_index']
		# Calculate proposed route duration
		to_proposed_location_duration = routing_input['duration_matrix'][vehicle_routes[vehicle_index][-1]][proposed_location_index]
		from_proposed_location_to_home_depot_duration = routing_input['duration_matrix'][proposed_location_index][home_depot_location_index]
		proposed_route_duration = vehicle_route_durations[vehicle_index] + to_proposed_location_duration + from_proposed_location_to_home_depot_duration
		if proposed_route_duration <= vehicle['max_route_duration']:
			vehicle_routes[vehicle_index].append(proposed_location_index)
			vehicle_route_durations[vehicle_index] += to_proposed_location_duration
			durations_back_home[vehicle_index] = from_proposed_location_to_home_depot_duration

	for vehicle_index, vehicle in enumerate(routing_input['vehicles']):
		home_depot = routing_input['depots'][vehicle['home_depot_index']]
		vehicle_routes[vehicle_index].append(home_depot['location_index'])
		vehicle_route_durations[vehicle_index] += durations_back_home[vehicle_index]

	# Heuristic router only routes one day at a time.
	routing_output = {
		'days': [{
			'vehicles': [{
				'route': route
			} for route in vehicle_routes]
		}]
	}
	
	#print(routing_output)

	return routing_output


# Any entity in the simulation that has an index which should be mentioned in logging. We don't have other kinds of entities.
class IndexedSimEntity():

	def log(self, message):
		self.sim.log(f"{type(self).__name__} #{self.index}: {message}")

	def warn(self, message):
		self.sim.warn(f"{type(self).__name__}e #{self.index}: {message}")

	def __init__(self, sim, index):
		self.sim = sim
		self.index = index

		# Statistics
		self.total_run_time = 0


# Any entity in the simulation that is indexed and also has a location index
class IndexedLocation(IndexedSimEntity):

	def get_lonlat(self):
		return self.lonlat

	def __init__(self, sim, index, location_index):
		super().__init__(sim, index)
		self.location_index = location_index
		self.lonlat = sim.config['location_lonlats'][location_index]
		sim.locations[location_index] = self

# Pickup site
class PickupSite(IndexedLocation):

	def __init__(self, sim, index):

		super().__init__(sim, index, sim.config['pickup_sites'][index]['location_index'])

		self.capacity = sim.config['pickup_sites'][index]['capacity']
		self.level = sim.config['pickup_sites'][index]['level']
		# If isTimeCriticalyConsidered = False, TS_initial = TS_current always.
		self.TS_initial = sim.config['pickup_sites'][index]['TS-rate']
		self.TS_current = sim.config['pickup_sites'][index]['TS-rate']
		self.daily_growth_rate = sim.config['pickup_sites'][index]['daily_growth_rate']
		self.type = sim.config['pickup_sites'][index]['type'] # 1=GRASS, 2=DRYMANURE, 3=SLURRYMANURE
		self.Exact_type = sim.config['pickup_sites'][index]['Type']
		self.accumulation_days = sim.config['pickup_sites'][index]['accumulation_days']
		self.collection_rate =  sim.config['pickup_sites'][index]['collection_rate']
		self.levelListeners = []

		# Relevant only if timecriticality is considered.
		self.volume_loss = sim.config['pickup_sites'][index]['volume_loss_coefficient']
		self.moisture_loss = sim.config['pickup_sites'][index]['moisture_loss_coefficient']

		self.log(f"Initial level: {tons_to_string(self.level)} of {tons_to_string(self.capacity)} ({to_percentage_string(self.level / self.capacity)}), growth rate: {tons_to_string(self.daily_growth_rate)}/day, TS-rate: {tons_to_string(self.TS_initial)}")

		self.growth_process = sim.env.process(self.grow_daily_forever())
		if sim.config['isTimeCriticalityConsidered'] == 'True':
			self.drying_process = sim.env.process(self.dry_daily_forever())		

	# Put some amount into the containers at the site
	def put(self, amount):
		listeners_to_message_maybe = list(filter(lambda x: self.level < x[1], self.levelListeners))

		# To update TS-rate of site
		if self.level > 0:
			self.TS_current = (self.TS_current/100*self.level + self.TS_initial/100*amount)/(self.level+amount)*100
		else:
			self.TS_current = self.TS_initial

		self.level += amount
		listeners_to_message = list(filter(lambda x: self.level >= x[1], listeners_to_message_maybe))
		#if len(listeners_to_message):
			#self.log(f"Level increase past threshold for {len(listeners_to_message)} listeners.")
		for x in listeners_to_message:
			x[0](**x[2])

	# Get some amount from the containers at the site
	def get(self, amount):
		self.level -= amount
		if self.level <= 0:
			self.level = 0
			self.TS_current = 0

	def estimate_when_full(self):
		# Solve:
		# { level_at_time_x = level_now + (time_x - time_now)*growth_rate
		# { capacity = level_at_time_x
		# => capacity = level_now + (time_x - time_now)*growth_rate
		# => time_x = time_now + (capacity - level_now)/growth_rate
		if self.daily_growth_rate > 0:
			return self.sim.env.now + 24*60*(self.capacity - self.level)/self.daily_growth_rate
		else:
			return 0

	def addLevelListener(self, listener, threshold, data = None):
		#self.log("Added level listener")
		listener_info = (listener, threshold, data)
		self.levelListeners.append(listener_info)

	def removeLevelListener(self, listener):
		self.levelListeners = filter(lambda x: x[0] != listener, self.levelListeners)

	def grow_daily_forever(self):
		yield self.sim.env.timeout(1)
		day = 0
		while True:
			if(self.accumulation_days[day]==1):	
				self.put(self.daily_growth_rate)
			day += 1
			yield self.sim.env.timeout(24*60)

	def dry_daily_forever(self):
		if (self.sim.config['isTimeCriticalityConsidered'] == 'True'):
			while True:
				if self.level > 0:
					self.level -= self.level*pow(self.volume_loss,1/7)
					self.TS_current = (1-((1-pow(self.moisture_loss,1/7))*(1-self.TS_current/100)))*100
				else:
					self.level = 0
					self.TS_current = 0
				self.TS_current = max(0.0,self.TS_current)
				self.level = max(0.0,self.level)				
				yield self.sim.env.timeout(24*60)


	def TS_rate(self):
		if (self.sim.config['isTimeCriticalityConsidered'] == 'True'):
			return self.TS_current
		return self.TS_initial

	def biomass_type(self):
		return self.type

	def give_collection_rate(self):
		return self.collection_rate

# Vehicle
class Vehicle(IndexedSimEntity):	

	def __init__(self, sim, index, home_depot_index):
		super().__init__(sim, index)
		self.home_depot_index = home_depot_index

		# Load level, capacity, and TS-rate of the load. 
		self.load_capacity = sim.config['vehicle_template']['load_capacity']
		self.load_level = 0.0
		self.load_distribution = {} # Will include information of proportion of each Exact_type of biomass within the load.
		self.load_TS_rate = sim.config ['vehicle_template']['load_TS_rate']

		# Types of a vehicle. 1=Can pick grass and straws, 2=can pick dry manures, 
		# 3=can pick slurry manures
		if index % 3 == 0:
			self.type = 3
		elif index % 2 == 0:
			self.type = 2
		else:
			self.type = 1

		# Work shift
		self.max_route_duration = sim.config['vehicle_template']['max_route_duration']

		# Pickup duration
		self.pickup_duration = sim.config['vehicle_template']['pickup_duration']

		# Location and movement
		self.moving = False
		self.location_index = sim.depots[self.home_depot_index].location_index
		self.vehicle_odometer = 0
		self.wrong_sites_visited = 0

		# Drying of load
		if (self.sim.config['isTimeCriticalityConsidered'] == 'True'):
			self.drying_process = sim.env.process(self.load_drying_daily_forever())

		#self.log(f"At {type(sim.locations[self.location_index]).__name__} #{sim.locations[self.location_index].index}")

	# Get current location
	def get_lonlat(self):
		if self.moving == False:
			return self.sim.locations[self.location_index].lonlat
		else:
			# Interpolate between current source and destination locations
			route_step_fractional_progress = (self.sim.env.now - self.route_step_departure_time) / self.sim.duration_matrix[self.route[self.route_step]][self.route[self.route_step + 1]]
			if (route_step_fractional_progress > 1): 
				route_step_fractional_progress = 1 # After travel there may be time spent working at pickup site
			source_location_lonlats = self.sim.locations[self.route[self.route_step]].lonlat
			destination_location_lonlats = self.sim.locations[self.route[self.route_step + 1]].lonlat
			return (
				source_location_lonlats[0] + route_step_fractional_progress*(destination_location_lonlats[0] - source_location_lonlats[0]),
				source_location_lonlats[1] + route_step_fractional_progress*(destination_location_lonlats[1] - source_location_lonlats[1])
			)

	def put_load(self, value, ts, Exact_type):
		if(value > 0):
			self.update_TS(value,ts)
			# To maintain the information regarding to load distribution to actual biomass types.
			if Exact_type in self.load_distribution:
				for biomass_type in self.load_distribution:
					if biomass_type == Exact_type:
						self.load_distribution[biomass_type] = (self.load_distribution[biomass_type]*self.load_level+value)/(self.load_level+value)
					else:
						self.load_distribution[biomass_type] = (self.load_distribution[biomass_type]*self.load_level)/(self.load_level+value)
			else:
				for biomass_type in self.load_distribution:
					self.load_distribution[biomass_type] = (self.load_distribution[biomass_type]*self.load_level)/(self.load_level+value)
				self.load_distribution[Exact_type] = value/(self.load_level+value)
			
			self.load_level += value
			if (self.load_level > self.load_capacity):
				self.warn("Overload")

	def update_TS(self,amount,ts):
		if self.load_level > 0:
			self.load_TS_rate = (self.load_TS_rate/100*self.load_level + ts/100*amount)/(self.load_level+amount)*100
		else:
			self.load_TS_rate = ts
		self.load_TS_rate = max(0.0,self.load_TS_rate)

	# Assign route for vehicle
	def assign_route(self, route):
		self.route_activity = self.sim.env.process(self.run_assign_route(route))

	def run_assign_route(self, route):
		if len(route) > 0:
			self.moving = True
			moving_start_time = self.sim.env.now
			self.route = route
			for self.route_step in range(len(route) - 1):
				self.route_step_departure_time = self.sim.env.now
				depart_location = self.sim.locations[self.route[self.route_step]]
				arrive_location = self.sim.locations[self.route[self.route_step + 1]]
				self.log(f"Depart from {type(depart_location).__name__} #{depart_location.index}")
				yield self.sim.env.timeout(self.sim.duration_matrix[self.route[self.route_step]][self.route[self.route_step + 1]])
				self.record_distance_travelled(self.sim.distance_matrix[self.route[self.route_step]][self.route[self.route_step + 1]])
				self.log(f"Arrive at {type(arrive_location).__name__} #{arrive_location.index}")

				if isinstance(arrive_location, PickupSite):
					# Arrived at a pickup site
					pickup_site = arrive_location
					if pickup_site.biomass_type() == self.vehicle_type():
						if pickup_site.level > 0:
							if self.load_level + pickup_site.level > self.load_capacity: 
								# Can only take some
								get_amount = self.load_capacity - self.load_level
								pickup_site.get(get_amount)
								loadTS = pickup_site.TS_rate()
								self.put_load(get_amount,loadTS,pickup_site.Exact_type)
								if get_amount > 0:
									yield self.sim.env.timeout(self.pickup_duration + get_amount*pickup_site.give_collection_rate())
							else:
								# Can take all
								get_amount = pickup_site.level
								pickup_site.get(get_amount)
								loadTS = pickup_site.TS_rate()
								self.put_load(get_amount,loadTS,pickup_site.Exact_type)
								if get_amount > 0:
									yield self.sim.env.timeout(self.pickup_duration + get_amount*pickup_site.give_collection_rate())
							self.log(f"Pick up {tons_to_string(get_amount)} from pickup site #{pickup_site.index} with {tons_to_string(pickup_site.level)} remaining. Vehicle load {tons_to_string(self.load_level)} / {tons_to_string(self.load_capacity)}")
						else:
							self.log(f"Nothing to pick up at pickup site #{pickup_site.index}")			
					else:
						self.warn(f"Vehicle #{self.index} of type #{self.vehicle_type()} arrived at site #{pickup_site.index} of type #{pickup_site.biomass_type()}!")
						self.wrong_sites_visited += 1
					
				elif isinstance(arrive_location, Depot):
					# Arrived at a terminal
					depot = arrive_location
					if self.load_level > 0:
						depot.receive_biomass(self.load_level, self.load_TS_rate, self.type, self.load_distribution)
						self.log(f"Vehicle #{self.index} of type #{self.type} dumped load of #{self.load_level} to the biogas plant.")
						self.log(f"Vehicle #{self.index} load distribution: {self.load_distribution}")			
					self.load_level = 0
					self.load_distribution.clear()
					self.load_TS_rate = 0
					

			# Mark as not moving at final destination
			self.moving = False
			moving_end_time = self.sim.env.now
			self.total_run_time += moving_end_time - moving_start_time
			self.location_index = route[-1]

	def load_drying_daily_forever(self):
		if (self.sim.config['isTimeCriticalityConsidered'] == 'True'):
			while True:
				if(self.load_level > 0):
					self.load_level -= self.load_level*pow(0.01,1/7)
					self.load_TS_rate = (1-((1-pow(0.05,1/7))*(1-self.load_TS_rate/100)))*100
				else:
					self.load_level = 0
					self.load_TS_rate = 0
				self.load_TS_rate = max(0.0,self.load_TS_rate)
				self.load_level = max(0.0,self.load_level)
				yield self.sim.env.timeout(24*60)

	def record_distance_travelled(self, distance_driven):
		"""
		Remove later?
		"""
		self.vehicle_odometer += distance_driven

	def vehicle_type(self):
		return self.type

# Depot where the vehicles start from in the beginning of the day and go to at the end of the day
class Depot(IndexedLocation):

	def __init__(self, sim, index):
		super().__init__(sim, index, sim.config['depots'][index]['location_index'])

		# Biomass storage level and TS-rate of storage, total biomass received, consumption rate (tons/day), production_stoppage_days 
		# (storage_level < 0) and days of overfilling,  boolean for if the yearly demand is satisfied, counter for unnecessary imports 
		# (imports that occured after satisfying yearly demand) amount of dilution water consumed druing the biogas production. 
		# (Dilution water is required if TS > 15 %)

		# Number witin each Depot's variable represents the type of biomass of interest
		# 1=Grass and straw, 2=Dry manure, 3=Slurry manure

		self.storage_level_1 = sim.config['depots'][index]['storage_level_1']
		self.storage_level_2 = sim.config['depots'][index]['storage_level_2']
		self.storage_level_3 = sim.config['depots'][index]['storage_level_3']

		self.cumulative_biomass_received_1 = 0
		self.cumulative_biomass_received_2 = 0
		self.cumulative_biomass_received_3 = 0

		self.is_yearly_demand_satisfied_1 = False
		self.is_yearly_demand_satisfied_2 = False
		self.is_yearly_demand_satisfied_3 = False

		self.consumption_rate_1 = sim.config['depots'][index]['consumption_rate_1']
		self.consumption_rate_2 = sim.config['depots'][index]['consumption_rate_2']
		self.consumption_rate_3 = sim.config['depots'][index]['consumption_rate_3']

		self.capacity_1 = sim.config['depots'][index]['capacity_1']
		self.capacity_2 = sim.config['depots'][index]['capacity_2']
		self.capacity_3 = sim.config['depots'][index]['capacity_3']

		self.production_stoppage_counter = 0
		self.overfilling_counter = 0
		self.unnecessary_imports_counter = 0
		self.dilution_water = 0
		self.storage_TS = 14.7 # Based on the example two input combos under which circumstances simulation is initialized

		self.storage_distribution = sim.config['depots'][index]['storage_distribution']
											    # List of dicts.
											    # At index = 0, dict of distribution of grass and straws within storage, 
											    # at index = 1, dict of distribution of dry manures within storage
												# and at index = 2, dict of distribution of slurry manures within storage

		self.log(f"initialized storage distribution {self.storage_distribution}")

		self.production_process = sim.env.process(self.produce_biogas_forever())

		if (self.sim.config['isTimeCriticalityConsidered'] == 'True'):
			self.drying_process = sim.env.process(self.storage_drying_daily_forever())

	def storage_sum(self):
		return self.storage_level_1 + self.storage_level_2 + self.storage_level_3
	
	def avoid_negative_storage_levels(self):
		self.storage_level_1 = max(0.0,self.storage_level_1)
		self.storage_level_2 = max(0.0,self.storage_level_2)
		self.storage_level_3 = max(0.0,self.storage_level_3)
		if self.storage_sum() <= 0:
			self.storage_TS = 0

	def update_storage_distribution_if_negative(self,storagelevel,type):
		if storagelevel <= 0:
			for proportion in self.storage_distribution[type-1]:
				self.storage_distribution[type-1][proportion] = 0

	def biomass_consumption(self):
		self.storage_level_1 -= self.consumption_rate_1
		self.storage_level_2 -= self.consumption_rate_2
		self.storage_level_3 -= self.consumption_rate_3
		self.update_storage_distribution_if_negative(self.storage_level_1,1)
		self.update_storage_distribution_if_negative(self.storage_level_2,2)
		self.update_storage_distribution_if_negative(self.storage_level_3,3)
		self.avoid_negative_storage_levels()

	def produce_biogas_forever(self):
		yield self.sim.env.timeout(1)
		while True:
			if self.storage_sum() > 0:
				if self.storage_TS > 15:
					# Amount of water to dilute the storage's content to TS=15% (analytical solution)
					self.dilution_water += 14/3*self.storage_TS*self.storage_sum()/100 + pow(self.storage_TS,2)*self.storage_sum()
					self.warn(f"Dilution water consumed. Total consumption: {self.dilution_water}")
					self.storage_TS = 15
					self.log(f"Storage TS after dilution: {self.storage_TS}")
				self.biomass_consumption()

			elif self.storage_sum() <= 0:
				self.production_stoppage_counter += 1
				self.storage_TS = 0
				self.warn(f"Production stoppage! Stoppages total: {self.production_stoppage_counter}")

			self.log(f"Grass and straw storage: {self.storage_level_1}.")
			self.log(f"Dry manure storage: {self.storage_level_2}.")
			self.log(f"Slurry manure storage: {self.storage_level_3}.")
			self.log(f"Storage distribution: {self.storage_distribution}")			
			self.log(f"TS rate of biogas facility: {self.storage_TS}")			
			yield self.sim.env.timeout(24*60)


	def update_TS(self, amount, ts):
		if self.storage_sum() > 0:
			self.storage_TS = (self.storage_TS/100*self.storage_sum() + ts/100*amount)/(self.storage_sum()+amount)*100
		else:
			self.storage_TS = ts

		self.storage_TS = max(0.0,self.storage_TS)
		self.log(f"Storage TS updated. TS now: {self.storage_TS}")


	def update_storage_distribution(self, storagelevel, amount, load_distribution, type):
		# Called within receive_biomass. To maintain the correctness of storage distribution to biomass types.
		if(amount > 0):
			for exact_biomass_type in self.storage_distribution[type]:
				if exact_biomass_type in load_distribution:
					self.storage_distribution[type][exact_biomass_type] = (self.storage_distribution[type][exact_biomass_type]*storagelevel + amount*load_distribution[exact_biomass_type])/(storagelevel + amount)
				else:
					self.storage_distribution[type][exact_biomass_type] = (self.storage_distribution[type][exact_biomass_type]*storagelevel)/(storagelevel + amount)

	def receive_biomass(self, received_amount, received_TS, type, load_distribution):
		biomass_mapping = {
            1: (self.storage_level_1, self.cumulative_biomass_received_1, self.is_yearly_demand_satisfied_1, self.capacity_1),
            2: (self.storage_level_2, self.cumulative_biomass_received_2, self.is_yearly_demand_satisfied_2, self.capacity_2),
            3: (self.storage_level_3, self.cumulative_biomass_received_3, self.is_yearly_demand_satisfied_3, self.capacity_3),
        }

		storage_level, cumulative_biomass_received, is_yearly_demand_satisfied, capacity = biomass_mapping[type]
		storage_level = max(0.0,storage_level)

		if is_yearly_demand_satisfied:
			self.unnecessary_imports_counter += 1
			self.warn(f"Unnecessary import. Yearly demand already satisfied. Unnecessary imports {self.unnecessary_imports_counter}.")
			return

		self.update_TS(received_amount, received_TS)
		# To update the storage distribution within the exact biomass types
		self.update_storage_distribution(storage_level, received_amount, load_distribution, type-1)

		storage_level += received_amount
		cumulative_biomass_received += received_amount

		if storage_level > capacity:
			self.overfilling_counter += 1
			self.warn(f"Overfilling at biogas facility! Overfills: {self.overfilling_counter}.")

		if cumulative_biomass_received >= capacity:
			is_yearly_demand_satisfied = True
			self.warn(f"Yearly demand for biomass satisfied! Cumulative amount: {cumulative_biomass_received}, capacity: {capacity}")

	  # Assign the modified values back to the original attributes
		if type == 1:
			self.storage_level_1 = storage_level
			self.cumulative_biomass_received_1 = cumulative_biomass_received
			self.is_yearly_demand_satisfied_1 = is_yearly_demand_satisfied

		elif type == 2:
			self.storage_level_2 = storage_level
			self.cumulative_biomass_received_2 = cumulative_biomass_received
			self.is_yearly_demand_satisfied_2 = is_yearly_demand_satisfied
		elif type == 3:
			self.storage_level_3 = storage_level
			self.cumulative_biomass_received_3 = cumulative_biomass_received
			self.is_yearly_demand_satisfied_3 = is_yearly_demand_satisfied

	def storage_drying_daily_forever(self):
		if (self.sim.config['isTimeCriticalityConsidered'] == 'True'):
			while True:

				if(self.storage_sum() > 0):
					if(self.storage_level_1 > 0):
						self.storage_level_1 -= self.storage_level_1*pow(0.01,1/7)
					else:
						self.storage_level_1 = 0
					if(self.storage_level_2 > 0):
						self.storage_level_2 -= self.storage_level_2*pow(0.01,1/7)
					else:
						self.storage_level_2 = 0
					if(self.storage_level_3 > 0):
						self.storage_level_3 -= self.storage_level_3*pow(0.01,1/7)
					else:
						self.storage_level_3 = 0					
					self.storage_TS = (1-((1-pow(0.05,1/7))*(1-self.storage_TS/100)))*100
				else:
					self.storage_level_1 = 0								
					self.storage_level_2 = 0
					self.storage_level_3 = 0
					self.storage_TS = 0		

				self.avoid_negative_storage_levels()
				self.storage_TS = max(0.0,self.storage_TS)
				yield self.sim.env.timeout(24*60)


# Terminal where waste is brought to at the end of the day, before returning to depot
class Terminal(IndexedLocation):

	def __init__(self, sim, index):
		super().__init__(sim, index, sim.config['terminals'][index]['location_index'])

		# Biomass storage level and TS-rate of storage, total biomass received, consumption rate (tons/day), production_stoppage_days 
		# (storage_level < 0) and days of overfilling,  boolean for if the yearly demand is satisfied, counter for unnecessary imports 
		# (imports that occured after satisfying yearly demand) amount of dilution water consumed druing the biogas production. 
		# (Dilution water is required if TS > 15 %)

		# Number witin each Depot's variable represents the type of biomass of interest
		# 1=Grass and straw, 2=Dry manure, 3=Slurry manure

		self.storage_level_1 = sim.config['terminals'][index]['storage_level_1']
		self.storage_level_2 = sim.config['terminals'][index]['storage_level_2']
		self.storage_level_3 = sim.config['terminals'][index]['storage_level_3']

		self.cumulative_biomass_received_1 = 0
		self.cumulative_biomass_received_2 = 0
		self.cumulative_biomass_received_3 = 0

		self.is_yearly_demand_satisfied_1 = False
		self.is_yearly_demand_satisfied_2 = False
		self.is_yearly_demand_satisfied_3 = False

		self.consumption_rate_1 = sim.config['terminals'][index]['consumption_rate_1']
		self.consumption_rate_2 = sim.config['terminals'][index]['consumption_rate_2']
		self.consumption_rate_3 = sim.config['terminals'][index]['consumption_rate_3']

		self.capacity_1 = sim.config['terminals'][index]['capacity_1']
		self.capacity_2 = sim.config['terminals'][index]['capacity_2']
		self.capacity_3 = sim.config['terminals'][index]['capacity_3']

		self.production_stoppage_counter = 0
		self.overfilling_counter = 0
		self.unnecessary_imports_counter = 0
		self.dilution_water = 0
		self.storage_TS = 15 # Assuming that the storage's TS is within the acceptable range at the beginning of simulation.

# Simulation
class WastePickupSimulation():

	def log(self, message):
		log_message = f'{time_to_string(self.env.now)} - {message}'
		print(log_message)
		# Add to csv log for later vis
		self.action_log.append(log_message)		

	def warn(self, message):
		warn_message = f"{time_to_string(self.env.now)} WARNING - {message}"
		print(warn_message)	
		# Add to csv log for later vis
		self.action_log.append(warn_message)
		self.sim_records['warnings'].append(warn_message) 

	def __init__(self, config):		
		self.config = config
		self.run_start = f"{datetime.now()}".replace(':', '-')

		# Create SimPy environment
		self.env = simpy.Environment()

		# For gathering statistics
		self.action_log = []
		self.sim_records = {'warnings' : []}

		# Distance and duration matrixes
		self.distance_matrix = config['distance_matrix']
		self.duration_matrix = config['duration_matrix']

		# Create a list of locations so that we can easily check the type of a location. These will be populated by any IndexedLocation
		self.locations = [None for _ in config['location_lonlats']]

		# Create pickup sites as objects
		self.pickup_sites = [PickupSite(self, i) for i in range(len(config['pickup_sites']))]

		# Create depots as objects
		self.depots = [Depot(self, i) for i in range(len(config['depots']))]

		# Create terminals as objects
		self.terminals = [Terminal(self, i) for i in range(len(config['terminals']))]

		# Create vehicles as objects
		self.vehicles = []
		for depot_index, depot in enumerate(config['depots']):
			for i in range(depot['num_vehicles']):
				self.vehicles.append(Vehicle(self, len(self.vehicles), depot_index))

		# Monitor pickup site levels
		for site in self.pickup_sites:
			site.addLevelListener(self.site_full, site.capacity, {"site": site})
		self.daily_monitoring_activity = self.env.process(self.daily_monitoring(config))
		
		# Daily vehicle routing
		self.routing_output = None # No routes planned yet. The value None will cause them to be planned
		self.daily_routing_activity = self.env.process(self.daily_routing())	

		# Vehicle and pickup site tracking for animation on map
		self.vehicle_tracking_activity = self.env.process(self.vehicle_animation_tracking())
		self.pickup_site_tracking_activity = self.env.process(self.pickup_site_animation_tracking())

		# Route logs
		self.route_logs = [[] for _ in self.vehicles]

		# Pickup site logs
		self.pickup_site_logs = [[] for _ in self.pickup_sites]

	def site_full(self, site):
		self.warn(f"Site #{site.index} is full.")
	
	def vehicle_animation_tracking(self):
		while True:
			#self.log(f"Vehicle locations: {', '.join(map(lambda x: lonlat_to_string(x.get_lonlat()), self.vehicles))}")
			for vehicle in self.vehicles:
				if vehicle.moving:
					self.route_logs[vehicle.index].append({
						"time": self.env.now,
						"lonlat": vehicle.get_lonlat(),
						"load_level": vehicle.load_level,
						"load_capacity": vehicle.load_capacity
					})
			yield self.env.timeout(1)

	def pickup_site_animation_tracking(self):
		while True:
			#self.log(f"Vehicle locations: {', '.join(map(lambda x: lonlat_to_string(x.get_lonlat()), self.vehicles))}")
			for pickup_site in self.pickup_sites:
				self.pickup_site_logs[pickup_site.index].append({
					"time": self.env.now,
					"lonlat": pickup_site.get_lonlat(),
					"level": pickup_site.level,
					"capacity": pickup_site.capacity
				})
			yield self.env.timeout(15)

	def daily_monitoring(self,sim_config):
		while True:
			#self.log(f"Monitored levels: {', '.join(map(lambda x: to_percentage_string(x.level/x.capacity), self.pickup_sites))}")
			yield self.env.timeout(24*60)

	def daily_routing(self):
		while True:
			# Request routing when not currently available
			if self.routing_output == None or len(self.routing_output['days']) == 0:
				# Input to routing optimizer
				routing_input = {
					'pickup_sites': list(map(lambda pickup_site: {
						'capacity': pickup_site.capacity,
						'level': pickup_site.level,
						'growth_rate': pickup_site.daily_growth_rate,
						'location_index': pickup_site.location_index,
						'TS_initial': pickup_site.TS_initial, 
						'TS_current': pickup_site.TS_current, 
						'type' : pickup_site.type,
						'Exact_type' : pickup_site.Exact_type,
						'accumulation_days' : pickup_site.accumulation_days,
						'collection_rate' : pickup_site.collection_rate,
	  					'volume_loss_coefficient': pickup_site.volume_loss, 
						'moisture_loss_coefficient' : pickup_site.moisture_loss
					}, self.pickup_sites)),
					'depots': list(map(lambda depot: {
						'location_index': depot.location_index,
						'storage_level_1': depot.storage_level_1,
						'storage_level_2': depot.storage_level_2,
						'storage_level_3': depot.storage_level_3,
						'storage_distribution' : depot.storage_distribution,
						'cumulative_biomass_received_1' : depot.cumulative_biomass_received_1,
						'cumulative_biomass_received_2' : depot.cumulative_biomass_received_2,
						'cumulative_biomass_received_3' : depot.cumulative_biomass_received_3,
						'is_yearly_demand_satisfied_1' : depot.is_yearly_demand_satisfied_1,
						'is_yearly_demand_satisfied_2' : depot.is_yearly_demand_satisfied_2,
						'is_yearly_demand_satisfied_3' : depot.is_yearly_demand_satisfied_3,
						'consumption_rate_1' : depot.consumption_rate_1,
						'consumption_rate_2' : depot.consumption_rate_2,
						'consumption_rate_3' : depot.consumption_rate_3,
						'capacity_1' : depot.capacity_1,
						'capacity_2' : depot.capacity_2,
						'capacity_3' : depot.capacity_3,
						'production_stoppage_counter' : depot.production_stoppage_counter,
						'overfilling_counter' : depot.overfilling_counter,
						'unnecessary_imports_counter' : depot.unnecessary_imports_counter,
						'storage_TS' : depot.storage_TS,
						'dilution_water' : depot.dilution_water
					}, self.depots)),
					'terminals': list(map(lambda terminal: {
						'location_index': terminal.location_index,
						'storage_level_1': terminal.storage_level_1,
						'storage_level_2': terminal.storage_level_2,
						'storage_level_3': terminal.storage_level_3,
						'cumulative_biomass_received_1' : terminal.cumulative_biomass_received_1,
						'cumulative_biomass_received_2' : terminal.cumulative_biomass_received_2,
						'cumulative_biomass_received_3' : terminal.cumulative_biomass_received_3,
						'is_yearly_demand_satisfied_1' : terminal.is_yearly_demand_satisfied_1,
						'is_yearly_demand_satisfied_2' : terminal.is_yearly_demand_satisfied_2,
						'is_yearly_demand_satisfied_3' : terminal.is_yearly_demand_satisfied_3,
						'consumption_rate_1' : terminal.consumption_rate_1,
						'consumption_rate_2' : terminal.consumption_rate_2,
						'consumption_rate_3' : terminal.consumption_rate_3,
						'capacity_1' : terminal.capacity_1,
						'capacity_2' : terminal.capacity_2,
						'capacity_3' : terminal.capacity_3,
						'production_stoppage_counter' : terminal.production_stoppage_counter,
						'overfilling_counter' : terminal.overfilling_counter,
						'unnecessary_imports_counter' : terminal.unnecessary_imports_counter,
						'storage_TS' : terminal.storage_TS,
						'dilution_water' : terminal.dilution_water
					}, self.terminals)),
					'vehicles': list(map(lambda vehicle: {
						'load_capacity': vehicle.load_capacity,
						'home_depot_index': vehicle.home_depot_index,
						'max_route_duration': vehicle.max_route_duration,
						'load_TS_rate' : vehicle.load_TS_rate,
						'type' : vehicle.type
					}, self.vehicles)),
					'distance_matrix': self.config['distance_matrix'],
					'duration_matrix': self.config['duration_matrix']
				}

				filename = 'temp/routing_input.json'
				os.makedirs(os.path.dirname(filename), exist_ok=True)
				with open(filename, 'w') as outfile:
					json.dump(routing_input, outfile, indent=4)

				# Comment/uncomment: heuristic router
				# self.routing_output = heuristic_router(routing_input)

				# Comment/uncomment: genetic algorithm router

				filename = 'log/routing_optimizer_log.txt'
				os.makedirs(os.path.dirname(filename), exist_ok=True)
				#os.system(f"routing_optimizer > {filename}") # *** # Windows
				os.system(f"./routing_optimizer > {filename}") # *** # Linux
				with open('temp/routing_output.json') as infile:
					self.routing_output = json.load(infile)

			# Assign routes
			for vehicle_index, vehicle_routing_output in enumerate(self.routing_output['days'][0]['vehicles']):
				self.vehicles[vehicle_index].assign_route(vehicle_routing_output['route'])

			# We have used the first day of multiple days of routing output. Remove that day from the routing output
			self.routing_output['days'] = self.routing_output['days'][1:]

			# Wait 24h until next route optimization
			yield self.env.timeout(24*60)


	def sim_run(self):
		start_time = time.time()
		self.env.run(until=self.config["sim_runtime_days"]*24*60)
		end_time = time.time()
		self.total_time = end_time-start_time # Excuding config preprocessing
		self.log(f"Simulation finished with {self.total_time}s of computing")

		self.warn(f"SIMULATION STATS")
		for depot_index, depot in enumerate(self.depots):
			self.warn(f"Production stoppages at depot #{depot_index}: {depot.production_stoppage_counter} times.")
			self.warn(f"Dilution water consumed depot #{depot_index}: {depot.dilution_water} tons.")
			self.warn(f"Unnecessary imports at depot #{depot_index}: {depot.unnecessary_imports_counter} times.")
			self.warn(f"Overfillings at depot #{depot_index}: {depot.overfilling_counter} times.")

		totalWrongvisits = 0
		totalOdometer = 0
		for vehicle_index, vehicle in enumerate(self.vehicles):
			self.warn(f"Wrong sites visited by vehicle #{vehicle_index}: {vehicle.wrong_sites_visited} times.")
			self.warn(f"Odometer of the vehicle #{vehicle_index}: {vehicle.vehicle_odometer/1000} km.")
			totalWrongvisits += vehicle.wrong_sites_visited
			totalOdometer += vehicle.vehicle_odometer/1000

		self.warn(f"Wrong sites total: {totalWrongvisits} times.")
		self.warn(f"Kilometers driven total: {totalOdometer} km.")

		filename = f"log/routes_log_{self.run_start}.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'w') as f:
			print("x,y,t,v,l,c", file=f)
			for vehicle_index, vehicle_log in enumerate(self.route_logs):
				for sample in vehicle_log:
					print(f"{sample['lonlat'][0]},{sample['lonlat'][1]},{sample['time']},{vehicle_index},{sample['load_level']},{sample['load_capacity']}", file=f)

		filename = f"log/pickup_sites_log_{self.run_start}.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'w') as f:
			print("x,y,t,l,c", file=f)
			for pickup_site_log in self.pickup_site_logs:
				for sample in pickup_site_log:
					print(f"{sample['lonlat'][0]},{sample['lonlat'][1]},{sample['time']},{sample['level']},{sample['capacity']}", file=f)


	def save_log(self):
		"""
		TODO: Log processor somwhere else
		"""
		# Get log from saved Save log as csv file
		json_log = json.dumps(self.action_log)
		filename = f"log/sim_log_{self.run_start}.json"
		# TODO! select folder structure for saving data
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(f'{filename}', 'w') as f:
			json.dump(json_log, f, indent=4)


	def sim_record(self):
		"""
		"""
		# Create a dict (record_[time.now.to_string]that records
		self.sim_records['computational_time'] = self.total_time # comptutational time
		self.sim_records['vehicles_driving_stats'] = [{'index':v.index, 'runtime':v.total_run_time, 'distance':v.vehicle_odometer} for v in self.vehicles] # time of vehicles driving
		#self.sim_records['vehicles_distance'] = [[v.index, v.vehicle_odometer] for v in self.vehicles] # time of vehicles driving
		# vechicle driving distance
		# level listeners alerts # are added in warnings level
		filename = f"log/sim_record_{self.run_start}.json"

		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(f'{filename}', 'w') as f:
			json.dump(self.sim_records, f, indent=4)

		print(f"Simulation record saved to {filename}")


def preprocess_sim_config(sim_config, sim_config_filename):

	# Create configurations for pickup sites using known data and random values	
	sim_config['pickup_sites'] = []
	with open(sim_config['pickup_sites_filename']) as pickup_sites_file:
		pickup_sites_geojson = json.load(pickup_sites_file)
	for pickup_site in pickup_sites_geojson['features']:
		pickup_site_config = {
			**pickup_site['properties'],
			'lonlats': tuple(pickup_site['geometry']['coordinates']),
			'capacity': pickup_site['properties']['Clustermasses'],
			'daily_growth_rate' : pickup_site['properties']['Clustermasses']/sim_config['sim_runtime_days'], 
			'level' : pickup_site['properties']['Clustermasses']*np.random.uniform(0, 0.8),
			'TS_initial' : pickup_site['properties']['TS-rate'],
			'type' : sim_config['biomass_type_mapping'][pickup_site['properties']['Type']],
			'Exact_type' : pickup_site['properties']['Type'],
			'volume_loss_coefficient' : 0.01, # Weekly-basis (Relevant if time-criticality is considered)
			'moisture_loss_coefficient' : 0.05 # Weekly-basis (Relevant if time-criticality is considered)
		}
		if pickup_site_config['type'] == 1:
			# Accumulation_days to include 3 ones representing the cuttings occuring three times a year. 
			# Location of 1 within the list is randomized. When cuttings occur, level of site jumps 
			# from 0 -> annual_amount/3
			pickup_site_config['daily_growth_rate'] = pickup_site['properties']['Clustermasses']/3
			pickup_site_config['accumulation_days'] = [0]*(sim_config['sim_runtime_days'])

			# To randomize wich sites have grass and straw to collect at the beginning of simulation.
			if random.randint(1,10) <= 4:
				pickup_site_config['level'] = pickup_site['properties']['Clustermasses']/3
			else:
				pickup_site_config['level'] = 0

			# Randomize when the cuttings occur within a site		
			pickup_site_config['accumulation_days'][random.randint(int(166/365*sim_config['sim_runtime_days']-5),int(166/365*sim_config['sim_runtime_days']+5))] = 1
			pickup_site_config['accumulation_days'][random.randint(int(212/365*sim_config['sim_runtime_days']-5),int(212/365*sim_config['sim_runtime_days']+5))] = 1
			pickup_site_config['accumulation_days'][random.randint(int(258/365*sim_config['sim_runtime_days']-5),int(258/365*sim_config['sim_runtime_days']+5))] = 1

			pickup_site_config['collection_rate'] = 1/1.2 # Grass and straws

		elif pickup_site_config['type'] == 2:
			pickup_site_config['accumulation_days'] = [1]*(sim_config['sim_runtime_days'])
			pickup_site_config['collection_rate'] = 1/1 # Dry manure

		elif pickup_site_config['type'] == 3:
			pickup_site_config['accumulation_days'] = [1]*(sim_config['sim_runtime_days'])
			pickup_site_config['collection_rate'] = 1/1.6 # Slurry manure

		sim_config['pickup_sites'].append(pickup_site_config)

	# Create configurations for terminals using known data
	sim_config['terminals'] = []
	with open(sim_config['terminals_filename']) as terminals_file:
		terminals_geojson = json.load(terminals_file)
	for terminal in terminals_geojson['features']:
		terminal_config = {
			**terminal['properties'],
			'lonlats': tuple(terminal['geometry']['coordinates']),
			'storage_level_1' : sim_config['grass_capacity']*0.175/46,
			'storage_level_2' : sim_config['drymanure_capacity']*0.6/46,
			'storage_level_3' : sim_config['slurrymanure_capacity']*4.5/46,
			'capacity_1' : sim_config['grass_capacity'],
			'capacity_2' : sim_config['drymanure_capacity'],
			'capacity_3' : sim_config['slurrymanure_capacity'],
			'consumption_rate_1' : sim_config['grass_capacity']*0.175 / sim_config['sim_runtime_days'],
			'consumption_rate_2' : sim_config['drymanure_capacity']*0.6 / sim_config['sim_runtime_days'],
			'consumption_rate_3' : sim_config['slurrymanure_capacity']*4.5 / sim_config['sim_runtime_days']
		}
		sim_config['terminals'].append(terminal_config)
		
	# Create configurations for depots
	with open(sim_config['depots_filename']) as depots_file:
		depots_geojson = json.load(depots_file)
	for index, depot in enumerate(depots_geojson['features']):
		
		# NOTE: Storage levels, consumption rates, and storage_distribution are now initialized based on the example 2 of LuKe's input combos.

		depot_config = {
			**depot['properties'],
			**sim_config['depots'][index],
			'lonlats': tuple(depot['geometry']['coordinates']),
			'storage_level_1' : sim_config['grass_capacity']*0.175/46,
			'storage_level_2' : sim_config['drymanure_capacity']*0.6/46,
			'storage_level_3' : sim_config['slurrymanure_capacity']*4.5/46,
			'storage_distribution' : [{'Sivuvirta: Kuivaheinanurmet' : 1/3, 'Sivuvirta: Kesantonurmi' : 2/3, 'Sivuvirta: Olki' : 0.0,},
			{'Lihakarjan kuivalanta yhteensa varastosta' : 1.0, 'Hevoset ja ponit, kuivalanta yhteensa varastosta': 0.0, 'Broilerit, kalkkunat ja muu siipikarja kuivalanta yhteensa varastosta' : 0.0, 'Emakot ja porsaat kuivalanta yhteensa varastosta' : 0.0,},
			{'Emakot ja porsaat lietelanta elainsuojasta' : 1.0, 'Lihakarjan lietelanta elainsuojasta' : 0.0, }],
			'capacity_1' : sim_config['grass_capacity'],
			'capacity_2' : sim_config['drymanure_capacity'],
			'capacity_3' : sim_config['slurrymanure_capacity'],
			'consumption_rate_1' : sim_config['grass_capacity']*0.175 / sim_config['sim_runtime_days'],
			'consumption_rate_2' : sim_config['drymanure_capacity']*0.6 / sim_config['sim_runtime_days'],
			'consumption_rate_3' : sim_config['slurrymanure_capacity']*4.5 / sim_config['sim_runtime_days']
		}

		sim_config['depots'][index] = depot_config

	# Collect lonlats of everything into a list of location lonlats. Store the location indexes
	def set_location_index_and_get_lonlats(x, location_index):
		x['location_index'] = location_index
		return x['lonlats']

	sim_config['location_lonlats'] = list(map(lambda x: set_location_index_and_get_lonlats(x[1], x[0]), enumerate([*sim_config['pickup_sites'], *sim_config['terminals'], *sim_config['depots']])))

	# Load previous sim config
	try:
		with open(sim_config_filename) as cached_sim_config_file:
			cached_sim_config = json.load(cached_sim_config_file)
	except:
		cached_sim_config = None

	# Use previous distance and duration matrixes if the locations match within 0.001 deg of absolute error
	if cached_sim_config != None and np.sum(np.absolute(np.array(cached_sim_config['location_lonlats']) - np.array(sim_config['location_lonlats']))) < 0.001:
		sim_config['distance_matrix'] = cached_sim_config['distance_matrix']
		sim_config['duration_matrix'] = cached_sim_config['duration_matrix']
	else:
		# Calculate geodesic distance and duration matrixes
		#geodesic_distance_matrix = np.ndarray((len(sim_config['location_lonlats']), len(sim_config['location_lonlats'])), dtype=np.float32)
		#geodesic_duration_matrix = np.ndarray((len(sim_config['location_lonlats']), len(sim_config['location_lonlats'])), dtype=np.float32)
		#for b_index, b in enumerate(sim_config['location_lonlats']):
			#for a_index, a in enumerate(sim_config['location_lonlats']):
				#geodesic_distance_matrix[b_index, a_index] = geodesic((a[1], a[0]), (b[1], b[0])).m  #geodesic() uses latlon
				#geodesic_duration_matrix[b_index, a_index] = geodesic_distance_matrix[b_index, a_index] / 1000 / 60 * 60 # / 1000m/km / 60km/h / 60min/h
		#sim_config['geodesic_distance_matrix'] = geodesic_distance_matrix.tolist()
		#sim_config['geodesic_duration_matrix'] = geodesic_duration_matrix.tolist()

		# Get routing API based distance and duration matrixes
		distance_and_duration_matrixes = get_distance_and_duration_matrix(sim_config['location_lonlats'])
		sim_config['distance_matrix'] = distance_and_duration_matrixes["distance_matrix"].tolist()
		sim_config['duration_matrix'] = distance_and_duration_matrixes["duration_matrix"].tolist()

	os.makedirs(os.path.dirname(sim_config_filename), exist_ok=True)
	with open(sim_config_filename, 'w') as outfile:
		json.dump(sim_config, outfile, indent=4)
