import subprocess
import os
import re


def parse_instance_info(raw_info):
    instance_ids = []
    pattern = r"\n(\d+)"
    matches = re.findall(pattern, raw_info)

    for i, instance_id in enumerate(matches):
        print(f"{i+1}. Instance ID: {instance_id}")
        instance_ids.append(instance_id)

    return instance_ids

def main():
    # Get instance information using subprocess
    raw_info = subprocess.getoutput('vastai show instances')

    # Parse the instance IDs
    instance_ids = parse_instance_info(raw_info)

    # Let user select an instance
    choice = int(input("Select an instance by number: "))
    selected_instance_id = instance_ids[choice - 1]

    # Run vastai copy command to get the onstart.log
    # subprocess.run(f"vastai copy {selected_instance_id}:/root/onstart.log .", shell=True)
    subprocess.run(f"vastai copy {selected_instance_id}:/root/.choline/choline_setup_log.txt .", shell=True)
    

    # Read and display the onstart.log
    with open("choline_setup_log.txt", 'r') as f:
        print(f.read())
    
    # Delete the onstart.log file
    os.remove("choline_setup_log.txt")

if __name__ == "__main__":
    main()