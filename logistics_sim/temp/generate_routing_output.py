import json
import random

num_days = 228
num_vehicles = 9
num_locations = 395  # 0 to 394 inclusive

data = {
    "days": [
        {
            "vehicles": [
                {"route": [random.randint(0, num_locations - 1) for _ in range(random.randint(2, 10))]}
                for _ in range(num_vehicles)
            ]
        }
        for _ in range(num_days)
    ]
}

output_json = json.dumps(data, indent=4)

with open("temp/routing_output.json", "w") as json_file:
    json_file.write(output_json)
