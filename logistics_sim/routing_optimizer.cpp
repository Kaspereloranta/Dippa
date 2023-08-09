// Copyright 2022 Häme University of Applied Sciences
// Authors: Olli Niemitalo, Genrikh Ekkerman
//
// This work is dual-licensed under the MIT and Apache 2.0 licenses and is distributed without any warranty.

// Compile with one of:
// g++ routing_optimizer.cpp -std=c++20 -march=native -I. -O3 -fcoroutines -ffast-math -fopenmp -o routing_optimizer
// g++-10 routing_optimizer.cpp -std=c++20 -march=native -I. -O3 -fcoroutines -ffast-math -fopenmp -o routing_optimizer

#include <stdio.h>
#include "ga.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <random>
#include <omp.h>
#include <coroutine>
#include <sstream>
#include "fschuetz04/simcpp20.hpp"
#include "nlohmann/json.hpp"
using json = nlohmann::json;

int debug = 1; // 0: no printf, 1: printf for genetic algo, 2: all printf
const float pickup_duration = 10;

enum LocationType {LOCATION_TYPE_DEPOT, LOCATION_TYPE_PICKUP_SITE, LOCATION_TYPE_TERMINAL};

struct IndexedLocation
{
  int location_index;
};

struct RoutingInputPickupSite: public IndexedLocation
{
  static const LocationType locationType = LOCATION_TYPE_PICKUP_SITE;
  float capacity;
  float level;
  float growth_rate;
  int max_num_visits;
  float total_mass;
  int times_collected;
  float TS_initial;
  float TS_current;
  float volume_loss_coefficient;
  float moisture_loss_coefficient;
};

void from_json(const json &j, RoutingInputPickupSite &x)
{
  j.at("capacity").get_to(x.capacity);
  j.at("level").get_to(x.level);
  j.at("growth_rate").get_to(x.growth_rate);
  j.at("location_index").get_to(x.location_index);
   j.at("total_mass").get_to(x.total_mass);
  j.at("times_collected").get_to(x.times_collected);
  j.at("TS_initial").get_to(x.TS_initial);
  j.at("TS_current").get_to(x.TS_current);
  j.at("volume_loss_coefficient").get_to(x.volume_loss_coefficient);
  j.at("moisture_loss_coefficient").get_to(x.moisture_loss_coefficient);
}

struct RoutingInputDepot: public IndexedLocation
{
  static const LocationType locationType = LOCATION_TYPE_DEPOT;
	double storage_level ;
  double cumulative_biomass_received;
  bool is_yearly_demand_satisfied;
  double consumption_rate;
	double capacity;
	int production_stoppage_counter;
	int overfilling_counter;
	int unnecessary_imports_counter;
  double storage_TS;
  float dilution_water;
};

void from_json(const json &j, RoutingInputDepot &x)
{
  j.at("location_index").get_to(x.location_index);
  j.at("storage_level").get_to(x.storage_level);
  j.at("cumulative_biomass_received").get_to(x.cumulative_biomass_received);
  j.at("is_yearly_demand_satisfied").get_to(x.is_yearly_demand_satisfied);
  j.at("consumption_rate").get_to(x.consumption_rate);
  j.at("capacity").get_to(x.capacity);
  j.at("production_stoppage_counter").get_to(x.production_stoppage_counter);
  j.at("overfilling_counter").get_to(x.overfilling_counter);
  j.at("unnecessary_imports_counter").get_to(x.unnecessary_imports_counter);
  j.at("storage_TS").get_to(x.storage_TS);
  j.at("dilution_water").get_to(x.dilution_water);
}

struct RoutingInputTerminal: public IndexedLocation
{
  static const LocationType locationType = LOCATION_TYPE_TERMINAL;
  double storage_level ;
	double cumulative_biomass_received;
	bool is_yearly_demand_satisfied;
	double consumption_rate;
	double capacity;
	int production_stoppage_counter;
	int overfilling_counter;
	int unnecessary_imports_counter;
};

void from_json(const json &j, RoutingInputTerminal &x)
{
  j.at("location_index").get_to(x.location_index);
  j.at("storage_level").get_to(x.storage_level);
  j.at("cumulative_biomass_received").get_to(x.cumulative_biomass_received);
  j.at("is_yearly_demand_satisfied").get_to(x.is_yearly_demand_satisfied);
  j.at("consumption_rate").get_to(x.consumption_rate);
  j.at("capacity").get_to(x.capacity);
  j.at("production_stoppage_counter").get_to(x.production_stoppage_counter);
  j.at("overfilling_counter").get_to(x.overfilling_counter);
  j.at("unnecessary_imports_counter").get_to(x.unnecessary_imports_counter);
}

struct RoutingInputVehicle: public IndexedLocation
{
  float load_capacity;
  int home_depot_index;
  int max_route_duration;
  float load_TS_rate;
};

void from_json(const json &j, RoutingInputVehicle &x)
{
  j.at("load_capacity").get_to(x.load_capacity);
  j.at("home_depot_index").get_to(x.home_depot_index);
  j.at("max_route_duration").get_to(x.max_route_duration);
  j.at("load_TS_rate").get_to(x.load_TS_rate);
}

struct LocationTypeAndSpecificIndex
{
  LocationType locationType;
  int specific_index;
};

struct RoutingInput
{
  std::vector<RoutingInputPickupSite> pickup_sites;
  std::vector<RoutingInputDepot> depots;
  std::vector<RoutingInputTerminal> terminals;
  std::vector<RoutingInputVehicle> vehicles;
  std::vector<std::vector<float>> distance_matrix;
  std::vector<std::vector<float>> duration_matrix;

  // Will be calculated from the above:
  int output_num_days;
  int sim_duration_days;
  int sim_duration;
  std::vector<int> gene_to_pickup_site_index;
  int num_pickup_site_visits_in_genome;
  int num_breaks_in_genome;
  int num_genes;
  std::vector<LocationTypeAndSpecificIndex> location_index_info;
};

void from_json(const json &j, RoutingInput &x)
{
  j.at("pickup_sites").get_to(x.pickup_sites);
  j.at("depots").get_to(x.depots);
  j.at("terminals").get_to(x.terminals);
  j.at("vehicles").get_to(x.vehicles);
  j.at("distance_matrix").get_to(x.distance_matrix);
  j.at("duration_matrix").get_to(x.duration_matrix);
}

// Store location indexes and location types for easy access
template<class T> void preprocess_indexed_locations(RoutingInput &x, std::vector<T> locations) {
  for (int i = 0; i < locations.size(); i++) {
    int location_index = locations[i].location_index;
    if (x.location_index_info.size() <= location_index) x.location_index_info.resize(location_index + 1);
    x.location_index_info[location_index] = { T::locationType, i };
  }
}

void preprocess_routing_input(RoutingInput &x) {
  // Do some preprocessing calculations
  // Location index interpretation
  preprocess_indexed_locations<RoutingInputPickupSite>(x, x.pickup_sites);
  preprocess_indexed_locations<RoutingInputDepot>(x, x.depots);
  preprocess_indexed_locations<RoutingInputTerminal>(x, x.terminals);
  // Simulation length
  x.output_num_days = 228; // Get routes for 228 days # <- SIMULOINNIN PITUUS ! ! 
  x.sim_duration_days = x.output_num_days + 0; // 0 days marginal
  x.sim_duration = x.sim_duration_days*24*60; // * 24h/day * 60min/h
  // The relationship between genes and pickup sites
  x.num_pickup_site_visits_in_genome = 0;
  for (int i = 0; i < x.pickup_sites.size(); i++) {
    RoutingInputPickupSite site = x.pickup_sites[i];
    site.max_num_visits = (int)ceil((site.growth_rate*x.sim_duration + site.level)/(site.capacity*0.8));
    for (int j = x.num_pickup_site_visits_in_genome; j < x.num_pickup_site_visits_in_genome + site.max_num_visits; j++) {
      x.gene_to_pickup_site_index.push_back(i);
    }
    x.num_pickup_site_visits_in_genome += site.max_num_visits;
  }
  if (debug >= 2) printf("num_pickup_site_visits_in_genome = %d\n", x.num_pickup_site_visits_in_genome);
  x.num_breaks_in_genome = ((int)(x.sim_duration / (24*60) + 0.5)) * x.vehicles.size();
  x.num_genes = x.gene_to_pickup_site_index.size() + x.num_breaks_in_genome;
}

// Routing output data structures. The vectors are made of correct size, except for the route vector which is of unknown length.

struct RoutingOutputVehicle {
  std::vector<int> route;  
};

struct RoutingOutputDay {
  std::vector<RoutingOutputVehicle> vehicles;
  RoutingOutputDay(RoutingInput &routingInput): vehicles(routingInput.vehicles.size()) {}
};

struct RoutingOutput {
  std::vector<RoutingOutputDay> days;
  RoutingOutput(RoutingInput &routingInput): days(routingInput.sim_duration_days, routingInput) {}
};

void to_json(json& j, const RoutingOutputVehicle& x) {
  j = json{{"route", x.route}};
}

void to_json(json& j, const RoutingOutputDay& x) {
  j = json{{"vehicles", x.vehicles}};
}

void to_json(json& j, const RoutingOutput& x) {
  j = json{{"days", x.days}};
}

// Forward declaration of vehicle and pickup site classes
class VehicleState;
class PickupSiteState;
class DepotState;

// Logistics simulation class definition and member function declarations
class LogisticsSimulation: public HasCostFunction<int16_t> {
public:
  // Config
  simcpp20::simulation<> *sim;
  RoutingInput &routingInput;

  // Routing. This is here so that we don't need to allocate the memory for it multiple times.
  RoutingOutput routingOutput;

  // State
  std::vector<VehicleState> vehicles;
  std::vector<PickupSiteState> pickupSites;
  std::vector<DepotState> depots;
  int totalNumPickupSiteOverloadDays;
  double totalOdometer;
  double totalOvertime;
  double dilutionWater;
  int productionStoppages;
  int overFillings;
  int unnecessaryImports;

  // Member functions
  double costFunction(const std::vector<int16_t> &genome, double earlyOutThreshold = std::numeric_limits<double>::max());
  double pickup(int vehicleIndex, int pickupSiteIndex);
  void receive(int vehicleIndex, int depotIndex);
  std::string locationString(int locationIndex);

  // Space for printing
  std::vector<char> tempStr;

  // Constructor
  LogisticsSimulation(RoutingInput &routingInput);
private:

  simcpp20::event<> runVehicleRouteProcess(simcpp20::simulation<> &sim, int vehicleIndex, int day);
  simcpp20::event<> runDailyProcess(simcpp20::simulation<> &sim);
};

// Pickup site state class definition
struct PickupSiteState {
  float capacity;
  float level; // Material level
  float growth_rate;
  int locationIndex;
  float total_mass;
  int times_collected = 0;
  float TS_initial;  
  float TS_current;
  float  volume_loss_coefficient;
  float moisture_loss_coefficient;
};

// Vehicle state class definition and member function definitions
struct VehicleState {
  float loadLevel; // Load level
  float odometer; // Total distance traveled
  float overtime; // Total overtime accumulated
  float load_TS_rate;

  // Vehicle en route or not
  bool enRoute; // true: vehicle is en route, so no new route can be started, false: vehicle can start a new route

  // Vehicle location and movement state to allow vehicle monitoring
  bool moving;
  int locationIndex; // If moving: source location index, if moving: current location index
  int destinationLocationIndex; // Destination location index
  double departureTime; // If moving: time when departed from source location
};

// Depot state class definition
struct DepotState {
	double storage_level;
  double storage_TS;
	double cumulative_biomass_received;
  bool is_yearly_demand_satisfied;
  double consumption_rate;
  double capacity;
	int production_stoppage_counter;
	int overfilling_counter;
	int unnecessary_imports_counter;
  double dilution_water;
};

std::string LogisticsSimulation::locationString(int locationIndex) {
  std::stringstream ss;
  switch(routingInput.location_index_info[locationIndex].locationType) {
    case LOCATION_TYPE_PICKUP_SITE:
      ss << "pickup site #" << routingInput.location_index_info[locationIndex].specific_index;
      break;
    case LOCATION_TYPE_TERMINAL:
      ss << "terminal #" << routingInput.location_index_info[locationIndex].specific_index;
      break;
    case LOCATION_TYPE_DEPOT:
      ss << "depot #" << routingInput.location_index_info[locationIndex].specific_index;
      break;
    default:
      ss << "unknown location #" << locationIndex;
  }
  return ss.str();
}

simcpp20::event<> LogisticsSimulation::runVehicleRouteProcess(simcpp20::simulation<> &sim, int vehicleIndex, int day) {
  // Necessary variables
  VehicleState &vehicle = vehicles[vehicleIndex];

  int homeDepotIndex = routingInput.vehicles[vehicleIndex].home_depot_index;
  std::vector<int> &route = routingOutput.days[day].vehicles[vehicleIndex].route;
  
  if (route.size() == 0) {
    // Empty route
    if (debug >= 2) printf("%gh Vehicle #%d: no route for day %d\n", sim.now()/60, vehicleIndex, day);
  } else {
    // There are some locations to visit
    if (vehicle.enRoute == true) {
      // The previous route has not yet been finished
      if (debug >= 2) printf("%gh WARNING Vehicle #%d: can't start a new shift while still working on the previous one\n", sim.now()/60, vehicleIndex);
    } else {
      // Start route
      vehicle.enRoute = true;
      double shiftStartTime = sim.now();      
      for (int routeStep = 0; routeStep < route.size(); routeStep++) {

        // # TO CONSIDER THE LINEAR COMPONENT OF THE PICKUP DURATION BASED:
        // double collection_rate = 1/1.6; // Slurry manure
        double collection_rate = 1/1 ; // Dry manure
        // double collection_rate = 1/1.2; // Grass and straws

        vehicle.destinationLocationIndex = route[routeStep];
        if (vehicle.locationIndex == vehicle.destinationLocationIndex) {
          // No movement necessary
          vehicle.moving = false;
        } else {
          // Travel
          vehicle.moving = true;
          vehicle.departureTime = sim.now();
          if (debug >= 2) printf("%gh Vehicle #%d: depart from %s\n", sim.now()/60, vehicleIndex, locationString(vehicle.locationIndex).c_str());
          co_await sim.timeout(routingInput.duration_matrix[vehicle.locationIndex][vehicle.destinationLocationIndex]);
          vehicle.odometer += routingInput.distance_matrix[vehicle.locationIndex][vehicle.destinationLocationIndex];
          // Arrive at destination
          vehicle.moving = false;
          vehicle.locationIndex = vehicle.destinationLocationIndex;
          if (debug >= 2) printf("%gh Vehicle #%d: arrive at %s\n", sim.now()/60, vehicleIndex, locationString(vehicle.locationIndex).c_str());
          // Do work depending on the arrived at location type          
          switch(routingInput.location_index_info[vehicle.locationIndex].locationType) {
            case LOCATION_TYPE_PICKUP_SITE:
              {
                // Pickup work
                int pickup_site_index = routingInput.location_index_info[vehicle.destinationLocationIndex].specific_index;
                double collectedAmount = pickup( vehicleIndex, pickup_site_index);
                // pickup_duration being the constant term, collection_rate*collectedAmount linear term.
                co_await sim.timeout(pickup_duration + collection_rate*collectedAmount);              
              }
              break;
            case LOCATION_TYPE_TERMINAL:
              // No work at terminal
              {
                int terminal_index = routingInput.location_index_info[vehicle.destinationLocationIndex].specific_index;
              }
              break;
            case LOCATION_TYPE_DEPOT:
              // No work at depot
              {
                if (debug >= 2) printf("%gh Vehicle #%d: dump whole load of %f t at %s\n", sim.now()/60, vehicleIndex, vehicle.loadLevel, locationString(vehicle.locationIndex).c_str());
                int depot_index = routingInput.location_index_info[vehicle.destinationLocationIndex].specific_index;
                receive(vehicleIndex,depot_index);
                vehicle.loadLevel = 0;
                vehicle.load_TS_rate = 0;
              }
              break;
          }
        }
      }
      // Finish route
      vehicle.enRoute = false;      
      // Calculate shift duration and overtime
      double shiftDuration = sim.now() - shiftStartTime;
      if (shiftDuration > routingInput.vehicles[vehicleIndex].max_route_duration) {
        vehicle.overtime += shiftDuration - routingInput.vehicles[vehicleIndex].max_route_duration;
      }
    }
  }

  co_return;
}

simcpp20::event<> LogisticsSimulation::runDailyProcess(simcpp20::simulation<> &sim) {

  for (int day = 0; day < routingInput.sim_duration_days; day++) {

    for (int vehicleIndex = 0; vehicleIndex < vehicles.size(); vehicleIndex++) {
      // Start vehicle shift for current day
      runVehicleRouteProcess(sim, vehicleIndex, day);
      /*
      // Drying process within the vehicles
        if(vehicles[vehicleIndex].loadLevel > 0){
          vehicles[vehicleIndex].loadLevel -= vehicles[vehicleIndex].loadLevel*pow(0.01,1/7);
          vehicles[vehicleIndex].load_TS_rate = (1-((1-pow(0.05,1/7))*(1-vehicles[vehicleIndex].load_TS_rate/100)))*100;
        }
				else {
          vehicles[vehicleIndex].loadLevel = 0
          vehicles[vehicleIndex].load_TS_rate = 0
        }
      */
      }

    for (int depotIndex = 0; depotIndex < depots.size(); depotIndex++) {

      /*
      // Drying process within the biogas plant
      if (depots[depotIndex].storage_level > 0){
				depots[depotIndex].storage_level -= depots[depotIndex].storage_level*pow(0.01,1/7)
				depots[depotIndex].storage_TS = (1-((1-pow(0.05,1/7))*(1-depots[depotIndex].storage_TS/100)))*100
      }
      else {
        depots[depotIndex].storage_level = 0
        depots[depotIndex].storage_TS = 0
      }
      */
      // Biogas production process (resource consumption):     
      if (depots[depotIndex].storage_level > 0){
        // If dilution is needed
        if (depots[depotIndex].storage_TS > 15){
          // Amount of water to dilute the storage's content to TS=15% (analytical solution)
				  depots[depotIndex].dilution_water += 14/3*depots[depotIndex].storage_TS*depots[depotIndex].storage_level/100 + pow(depots[depotIndex].storage_TS,2)*depots[depotIndex].storage_level;
				  depots[depotIndex].storage_TS = 15;
        }
        depots[depotIndex].storage_level -= depots[depotIndex].consumption_rate;
      }
      if (depots[depotIndex].storage_level <= 0){
        depots[depotIndex].production_stoppage_counter += 1;
        depots[depotIndex].storage_TS = 0;
      }
      depots[depotIndex].storage_level = std::max(0.0,depots[depotIndex].storage_level);
    }

    // Increase pickup site levels
    for (int pickupSiteIndex = 0; pickupSiteIndex < pickupSites.size(); pickupSiteIndex++) {
      
      float put_amount = routingInput.pickup_sites[pickupSiteIndex].growth_rate;
      // To update TS-rate of site
			pickupSites[pickupSiteIndex].TS_current = (pickupSites[pickupSiteIndex].TS_current/100*pickupSites[pickupSiteIndex].level 
                                                + pickupSites[pickupSiteIndex].TS_initial/100*put_amount)/(pickupSites[pickupSiteIndex].level+put_amount)*100;
      pickupSites[pickupSiteIndex].level += put_amount;

      // TODO: 
      // Calculation of accumulation with noises to variable put_amount (above) to enable calculating the new TS-rate.

      /* 
      // For manures
      
      // Declare and initialize the random number generator and distribution
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<double> dist(0.0, 1.0);
      pickupSites[pickupSiteIndex].level += routingInput.pickup_sites[pickupSiteIndex].growth_rate*24*60
                                            + std::max(-10.0, std::min(dist(gen), 10.0))/20*
                                            routingInput.pickup_sites[pickupSiteIndex].growth_rate*24*60;
      /* 
      // For grass and straws
      std::random_device rd;
      std::mt19937 gen(rd());
      std::discrete_distribution<> dist({13, 1});  // Probability distribution for choices
    
      int isFulfilled = 0;
      if (138240 <= sim.now() && sim.now()<= 158400 && routingInput.pickup_sites[pickupSiteIndex].times_collected == 0) {
          isFulfilled = dist(gen);
          routingInput.pickup_sites[pickupSiteIndex].level = routingInput.pickup_sites[pickupSiteIndex].total_mass*isFulfilled;
          if (isFulfilled==1) {
              ++routingInput.pickup_sites[pickupSiteIndex].times_collected;
          }
      } else if (180000 <= sim.now() && sim.now()<= 200160 && routingInput.pickup_sites[pickupSiteIndex].times_collected == 1) {
          isFulfilled = dist(gen);
          routingInput.pickup_sites[pickupSiteIndex].level = routingInput.pickup_sites[pickupSiteIndex].total_mass*isFulfilled;
          if (isFulfilled==1) {
              ++routingInput.pickup_sites[pickupSiteIndex].times_collected;
          }
      } else if (221760 <= sim.now() && sim.now()<= 241920 && routingInput.pickup_sites[pickupSiteIndex].times_collected == 2) {
          isFulfilled = dist(gen);
          routingInput.pickup_sites[pickupSiteIndex].level = routingInput.pickup_sites[pickupSiteIndex].total_mass*isFulfilled;
          if (isFulfilled==1) {
              ++routingInput.pickup_sites[pickupSiteIndex].times_collected;
          }
      }
      */
      /*
      // Drying process within the pickup sites
      for (int pickupSiteIndex = 0; pickupSiteIndex < pickupSites.size(); pickupSiteIndex++) {
        if (pickupSites[pickupSiteIndex].level > 0){
          pickupSites[pickupSiteIndex].level -= pickupSites[pickupSiteIndex].level*pow(pickupSites[pickupSiteIndex].volume_loss_coefficient,1/7);
          pickupSites[pickupSiteIndex].TS_current = (1-((1-pow(pickupSites[pickupSiteIndex].moisture_loss_coefficient,1/7))*(1-pickupSites[pickupSiteIndex].TS_current/100)))*100;
        }
        else {
          pickupSites[pickupSiteIndex].level = 0
          pickupSites[pickupSiteIndex].TS_current = 0
        }
      }
      */
      if (pickupSites[pickupSiteIndex].level > routingInput.pickup_sites[pickupSiteIndex].capacity) {
        totalNumPickupSiteOverloadDays++;
        if (debug >= 2) printf("%gh WARNING Site %d overload\n", sim.now()/60, pickupSiteIndex);
      }
    }
    for (int pickupSiteIndex = 0; pickupSiteIndex < pickupSites.size(); pickupSiteIndex++) {
      if (debug >= 2) printf("%d%%, ", (int)floor(pickupSites[pickupSiteIndex].level / routingInput.pickup_sites[pickupSiteIndex].capacity * 100 + 0.5));
    }
    if (debug >= 2) printf("\n");
  co_await sim.timeout(24*60);
  }
  co_return;
}

double LogisticsSimulation::pickup(int vehicleIndex, int pickupSiteIndex) {

  // co_await sim.timeout(pickup_duration);              

  // TÄNNE TARVITTAESSA MUUTOKSIA, SAAKO TYHJENTÄÄ VAIN OSAN VAI TULEEKO OLLA TÄYSI TMS.
  // VOIDAAN ESTÄÄ VAJAISSA KÄYNTI TÄÄLLÄ

  double collectedAmount;

  if (pickupSites[pickupSiteIndex].level == 0) // <- TÄHÄN VOI MUUTTAA JOS ASETETAAN RAJA MISSÄ VOI KÄYDÄ JA MISSÄ EI
  { 
    // Unnecessary visit, nothing to pick up
    if (debug >= 2) printf("%gh Vehicle #%d: nothing to pick up at site #%d\n", sim->now()/60, vehicleIndex, pickupSiteIndex);
    collectedAmount = 0;
  }
  else if (vehicles[vehicleIndex].loadLevel == routingInput.vehicles[vehicleIndex].load_capacity) 
  {
    // Unnecessary visit, no unused load capacity left
    if (debug >= 2) printf("%gh Vehicle #%d: has no capacity left to pick anything at pickup site #%d with %f t remaining\n", sim->now()/60, vehicleIndex, pickupSiteIndex, pickupSites[pickupSiteIndex].level);
    collectedAmount = 0;
  } 
  else if (vehicles[vehicleIndex].loadLevel + pickupSites[pickupSiteIndex].level > routingInput.vehicles[vehicleIndex].load_capacity) 
  {
    // The vehicle cannot take everything
    collectedAmount = (routingInput.vehicles[vehicleIndex].load_capacity - vehicles[vehicleIndex].loadLevel);
    pickupSites[pickupSiteIndex].level -= collectedAmount;
    if (debug >= 2) printf("%gh Vehicle #%d: reaches its capacity taking %f t from pickup site #%d with %f t remaining\n", sim->now()/60, vehicleIndex, collectedAmount, pickupSiteIndex, pickupSites[pickupSiteIndex].level);
    vehicles[vehicleIndex].load_TS_rate = (vehicles[vehicleIndex].load_TS_rate/100*vehicles[vehicleIndex].loadLevel + pickupSites[pickupSiteIndex].TS_current/100*collectedAmount)/(vehicles[vehicleIndex].loadLevel+collectedAmount)*100;
    vehicles[vehicleIndex].loadLevel = routingInput.vehicles[vehicleIndex].load_capacity;
  }
  else 
  {
  // The vehicle empties the site
    collectedAmount = pickupSites[pickupSiteIndex].level;
    vehicles[vehicleIndex].load_TS_rate = (vehicles[vehicleIndex].load_TS_rate/100*vehicles[vehicleIndex].loadLevel + pickupSites[pickupSiteIndex].TS_current/100*collectedAmount)/(vehicles[vehicleIndex].loadLevel+collectedAmount)*100;
    vehicles[vehicleIndex].loadLevel += pickupSites[pickupSiteIndex].level;
    if (debug >= 2) printf("%gh Vehicle #%d: picks up all of %f t of pickup site #%d\n", sim->now()/60, vehicleIndex, pickupSites[pickupSiteIndex].level, pickupSiteIndex);
    pickupSites[pickupSiteIndex].level = 0;
    pickupSites[pickupSiteIndex].TS_current = 0;
  }
  return collectedAmount;
}

void LogisticsSimulation::receive(int vehicleIndex, int depotIndex){

  if (depots[depotIndex].is_yearly_demand_satisfied) {
    depots[depotIndex].unnecessary_imports_counter++;
    return;
  }

	depots[depotIndex].storage_TS = (depots[depotIndex].storage_TS/100*depots[depotIndex].storage_level +
                              vehicles[vehicleIndex].load_TS_rate/100*vehicles[vehicleIndex].loadLevel)
                              /(depots[depotIndex].storage_level+vehicles[vehicleIndex].loadLevel)*100;


  depots[depotIndex].storage_level += vehicles[vehicleIndex].loadLevel;
  depots[depotIndex].cumulative_biomass_received += vehicles[vehicleIndex].loadLevel;

  if (depots[depotIndex].storage_level > depots[depotIndex].capacity ) {
    depots[depotIndex].overfilling_counter++;
  }

  if ( depots[depotIndex].cumulative_biomass_received >= depots[depotIndex].capacity ) {
    depots[depotIndex].is_yearly_demand_satisfied = true;
  }
  return;
}


// Calculate cost function from components
double costFunctionFromComponents(double totalOdometer, double totalNumPickupSiteOverloadDays, double totalOvertime, double dilutionWater, int productionStoppages, int overFillings, int unnecessaryImports) {
  return totalOdometer*(50.0/100000.0*2) // Fuel price: 2 eur / L, fuel consumption: 50 L / (100 km)
  + totalNumPickupSiteOverloadDays*50.0 // Penalty of 50 eur / overload day / pickup site
  + totalOvertime*(50.0/60) // Cost of 50 eur / h for overtime work  
  + dilutionWater*1
  + productionStoppages*500
  + overFillings*1000
  + unnecessaryImports*100;
}

// Logistics simulation class member function: cost function
double LogisticsSimulation::costFunction(const std::vector<int16_t> &genome, double earlyOutThreshold) {
  // Interpret genome into routes  
  if (debug >= 2) printf("Genome size: %d\n", routingInput.num_genes);
  if (debug >= 2) printf("First non-pickup-site gene: %d\n", routingInput.num_pickup_site_visits_in_genome);
  int locus = 0;
  double totalOdometerLowerBound = 0;
  for (int day = 0; day < routingInput.output_num_days; day++) {
    for (int vehicleIndex = 0; vehicleIndex < routingInput.vehicles.size(); vehicleIndex++) {
      std::vector<int> &route = routingOutput.days[day].vehicles[vehicleIndex].route;
      route.clear(); // Empty any previously stored routes
      route.push_back(routingInput.depots[routingInput.vehicles[vehicleIndex].home_depot_index].location_index); // Start at depot
      while (genome[locus++] < routingInput.num_pickup_site_visits_in_genome) {
        int pickupSiteIndex = routingInput.gene_to_pickup_site_index[genome[locus - 1]];
        int locationIndex = routingInput.pickup_sites[pickupSiteIndex].location_index;
        if (route[route.size() - 1] != locationIndex) { // Is the vehicle departing from another location?
          route.push_back(locationIndex); // If yes, then add the location to the route
          totalOdometerLowerBound += routingInput.distance_matrix[route[route.size() - 2]][route[route.size() - 1]];
        }
      }
      if (route.size() < 2) {
        route.clear(); // If there is only the depot in the route, empty the route
      } else {        
        route.push_back(route[0]); // Else add the depot also at the end
        totalOdometerLowerBound += routingInput.distance_matrix[route[route.size() - 2]][route[route.size() - 1]];
      }
    }
    double costLowerBound = costFunctionFromComponents(totalOdometerLowerBound, 0, 0, 0, 0, 0, 0);
    if (costLowerBound >= earlyOutThreshold) return std::numeric_limits<double>::max();
  }

  // Initialization for simulation
  // Initialize vehicles
  for (int vehicleIndex = 0; vehicleIndex < vehicles.size(); vehicleIndex++) {
    VehicleState &vehicleState = vehicles[vehicleIndex];
    vehicleState.loadLevel = 0;
    vehicleState.odometer = 0;
    vehicleState.overtime = 0;
    vehicleState.load_TS_rate = 0;
    vehicleState.enRoute = false;
    vehicleState.moving = false;
    vehicleState.locationIndex = routingInput.depots[routingInput.vehicles[vehicleIndex].home_depot_index].location_index;
  }
  // Initialize pickup sites
  for (int pickupSiteIndex = 0; pickupSiteIndex < pickupSites.size(); pickupSiteIndex++) {
    PickupSiteState &pickupSiteState = pickupSites[pickupSiteIndex];
    pickupSiteState.capacity = routingInput.pickup_sites[pickupSiteIndex].capacity;
    pickupSiteState.level = routingInput.pickup_sites[pickupSiteIndex].level;
    pickupSiteState.growth_rate = routingInput.pickup_sites[pickupSiteIndex].growth_rate;
    pickupSiteState.locationIndex = routingInput.pickup_sites[pickupSiteIndex].location_index;
    pickupSiteState.total_mass = routingInput.pickup_sites[pickupSiteIndex].total_mass;
    pickupSiteState.times_collected = routingInput.pickup_sites[pickupSiteIndex].times_collected; // = 0
    pickupSiteState.TS_initial = routingInput.pickup_sites[pickupSiteIndex].TS_initial;
    pickupSiteState.TS_current = routingInput.pickup_sites[pickupSiteIndex].TS_current;
    pickupSiteState.volume_loss_coefficient = routingInput.pickup_sites[pickupSiteIndex].volume_loss_coefficient; // = 0.01
    pickupSiteState.moisture_loss_coefficient = routingInput.pickup_sites[pickupSiteIndex].moisture_loss_coefficient; // = 0.05

  }

  // Initialize depots
  for (int depotIndex = 0; depotIndex < depots.size(); depotIndex++) {
    DepotState &depotState = depots[depotIndex];
    depotState.storage_level = routingInput.depots[depotIndex].storage_level;
    depotState.cumulative_biomass_received = 0;
    depotState.is_yearly_demand_satisfied = false;
    depotState.consumption_rate = routingInput.depots[depotIndex].consumption_rate;
    depotState.capacity = routingInput.depots[depotIndex].capacity;
    depotState.production_stoppage_counter = 0;
    depotState.overfilling_counter = 0;
    depotState.unnecessary_imports_counter = 0;
    depotState.storage_TS = routingInput.depots[depotIndex].storage_TS; // = 15
    depotState.dilution_water = routingInput.depots[depotIndex].dilution_water; // = 0
  }  
  
  // Initialize cost components
  int totalNumPickupSiteOverloadDays = 0;
  int productionStoppages = 0;
  int overFillings = 0;
  int unnecessaryImports = 0;
  double dilutionWater = 0;

  // Simulate
  simcpp20::simulation<> sim;
  this->sim = &sim;
  runDailyProcess(sim);
  sim.run();
  double totalOvertime = 0;
  double totalOdometer = 0;
  for (int vehicleIndex = 0; vehicleIndex < vehicles.size(); vehicleIndex++) {
    totalOvertime += vehicles[vehicleIndex].overtime;
    if (debug >= 2) printf("Vehicle #%d overtime: %g h\n", vehicleIndex, vehicles[vehicleIndex].overtime/60);
    if (debug >= 2) printf("Vehicle #%d odometer reading: %g km\n", vehicleIndex, vehicles[vehicleIndex].odometer/1000);
    totalOdometer += vehicles[vehicleIndex].odometer;
  }
  for (int depotIndex = 0; depotIndex < depots.size(); depotIndex++) {
    productionStoppages += depots[depotIndex].production_stoppage_counter;
    overFillings += depots[depotIndex].overfilling_counter;
    unnecessaryImports += depots[depotIndex].unnecessary_imports_counter;
    dilutionWater += depots[depotIndex].dilution_water;
  }
  if (debug >= 2) printf("Total overtime: %g h\n", totalOvertime/60);
  if (debug >= 2) printf("Total odometer: %g km\n", totalOdometer/1000);
  if (debug >= 2) printf("Total pickup site overload days: %d\n", totalNumPickupSiteOverloadDays);
  if (debug >= 2) printf("Total production stoppages: %g h\n", float(productionStoppages));
  if (debug >= 2) printf("Total unnecessary imports to the biogas plant: %g h\n", float(unnecessaryImports));
  if (debug >= 2) printf("Total overfillings within the biogas plant: %g h\n", float(overFillings));
  if (debug >= 2) printf("Total consumption of dilution water: %g h\n", dilutionWater);
  return costFunctionFromComponents(totalOdometer, totalNumPickupSiteOverloadDays, totalOvertime, dilutionWater, productionStoppages, overFillings, unnecessaryImports);
}

// Simulation class constructor
LogisticsSimulation::LogisticsSimulation(RoutingInput &routingInput):
routingInput(routingInput), routingOutput(routingInput), vehicles(routingInput.vehicles.size()), pickupSites(routingInput.pickup_sites.size()), depots(routingInput.depots.size()) {}

int main() {
  // Read routing optimization input
  std::ifstream f("temp/routing_input.json");
  auto routingInputJson = json::parse(f);
  auto routingInput = routingInputJson.get<RoutingInput>();
  // Preprocess routing optimization input
  preprocess_routing_input(routingInput);
  std::vector<HasCostFunction<int16_t>*> logisticsSims;
  for (int i = 0; i < omp_get_max_threads(); i++) {
    logisticsSims.push_back(new LogisticsSimulation(routingInput));
  }
/*
  int *testGenome = new int[routingInput.num_genes];
  for (int i = 0; i < routingInput.num_genes; i++) {
    testGenome[i] = i;
  }
  logisticsSims[0]->costFunction(testGenome); */

/*
  int testGenome[] = {130,86,59,68,99,101,82,75,79,76,72,84,80,94,28,25,111,30,120,12,6,18,0,2,126,114,104,36,138,117,48,63,67,34,89,93,53,23,90,96,46,41,40,108,43,103,134,132,136,122,137,139,142,143,128,133,129,118,125,131,123,135,116,140,119,121,141,127,124,92,5,100,71,74,85,39,22,81,61,51,113,95,73,106,70,88,11,91,112,109,31,54,16,19,97,98,29,21,57,8,47,58,10,27,7,45,52,32,9,24,55,3,13,87,1,17,115,83,65,37,26,56,60,4,102,105,77,44,49,110,50,20,33,107,69,62,14,35,15,64,38,42,78,66};
  debug++;
  printf("Cost: %f\n", logisticsSims[0]->costFunction(testGenome));
  debug--;

*/

  Optimizer<int16_t> optimizer(routingInput.num_genes, logisticsSims);
  
  // TÄÄLLÄ MÄÄRÄTÄÄN KUINKA MONTA KIERROSTA GEENIAJOJA TEHDÄÄN, VAIKUTTA OPTIMOINNIN NOPEUTEEN, VOIDAAN MYÖS LISÄTÄ GEENEJÄ JOS HALUTAAN TARKENTAA LASKENTAA
  
  int numGenerations = 2000000; // 40000
  int numFinetuneGenerations = 1000000; // 20000
  int numGenerationsPerStep = 100;
  //optimizer.initPopulation();

  int generationIndex = 0;
  for (; generationIndex < numGenerations; generationIndex += numGenerationsPerStep) {
    if (debug >= 1) printf("%d,%f\n", generationIndex, optimizer.best.cost);
    optimizer.optimize(numGenerationsPerStep, false);
  }
  for (; generationIndex < numGenerations + numFinetuneGenerations; generationIndex += numGenerationsPerStep) {
    if (debug >= 1) printf("%d,%f\n", generationIndex, optimizer.best.cost);
    optimizer.optimize(numGenerationsPerStep, true);
  }
  if (debug >= 1) printf("%d,%f\n", generationIndex, optimizer.best.cost);

  debug++;
  logisticsSims[0]->costFunction(optimizer.best.genome);
  debug--;

  for (int i = 0; i < omp_get_max_threads(); i++) {
    delete logisticsSims[i];
  }

  // Get routes
  std::vector<int16_t> &genome = optimizer.best.genome;

  printf("\nBest genome:\n");
  for (int i = 0; i < routingInput.num_genes; i++) {
    printf("%d,", genome[i]);
  }
  printf("\n\n");
  LogisticsSimulation logisticsSim(routingInput);
  logisticsSim.costFunction(genome); // Get routeStartLoci
  json j = logisticsSim.routingOutput;
  std::ofstream o("temp/routing_output.json");
  o << std::setw(4) << j << std::endl;

  return 0;

}
