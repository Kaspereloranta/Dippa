import json
import random

num_days = 228
num_vehicles = 9
num_locations = 395  # 0 to 394 inclusive
start_location = 395  # Start and end location index
end_location = 395

def generate_random_route(num_locations, start_location, end_location):
    route_length = random.randint(2, 10)
    route = []
    current_location = start_location

    for _ in range(route_length - 1):
        next_location = random.randint(0, num_locations - 1)
        while next_location == current_location:
            next_location = random.randint(0, num_locations - 1)
        if next_location == 394:
            break
        route.append(next_location)
        current_location = next_location

    route.append(end_location)
    return route


data = {
    "days": [
        {
            "vehicles": [
                {
                    "route": generate_random_route(num_locations, start_location, end_location)
                }
                for _ in range(num_vehicles)
            ]
        }
        for _ in range(num_days)
    ]
}

output_json = json.dumps(data, indent=4)

with open("temp/routing_output.json", "w") as json_file:
    json_file.write(output_json)
