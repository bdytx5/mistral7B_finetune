# this will read machine params like GPU and Storage params from the choline json, and then ask u to select an instance 
# then it will use the software env info in the choline.json file to create an onstart script to set up the environment 
## then it will monitor the status of that machine and alert u if it fails 
import os
import json
import subprocess
import time
import argparse
import argparse
import os
import subprocess
import json
import torch
import re 
import yaml 
import os
import paramiko
import scp 
from scp import SCPClient
from fnmatch import fnmatch
import traceback
import yaml 

import re


def get_choline_yaml_data():
    with open('./choline.yaml', 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    
    return data



def get_choline_json_data():
    with open('./choline.json', 'r') as f:
        return json.load(f)




def custom_sort_key(offer, expected_storage_gb, expected_runtime_hr, expected_upload_gb=1.0):
    dph = offer['dph_base']
    storage_cost_per_hr = (offer['storage_cost'] / (30 * 24)) * expected_storage_gb
    total_storage_cost = storage_cost_per_hr * expected_runtime_hr
    download_cost = expected_storage_gb * offer['inet_down_cost']
    upload_cost = expected_upload_gb * offer['inet_up_cost']
    total_cost = (dph + storage_cost_per_hr) * expected_runtime_hr + download_cost + upload_cost
    
    return total_cost




def sort_offers_by_custom_criteria(offers, expected_storage_gb, expected_runtime_hr):
    return sorted(offers, key=lambda x: custom_sort_key(x, expected_storage_gb, expected_runtime_hr))




def pretty_print_offer(offer, index):
    print(f"########### VASTAI OFFERS {index + 1} ###########")
    print(f"Cost per hour: {offer['dph_base']}")
    print(f"Internet download speed: {offer['inet_down']}")
    print(f"Internet upload speed: {offer['inet_up']}")
    print(f"DL Performance: {offer['dlperf']}")
    print(f"Reliability: {offer['reliability2']}")
    print(f"Storage cost: {offer['storage_cost']}")
    print(f"Internet download cost: {offer['inet_down_cost']}")
    print(f"Internet upload cost: {offer['inet_up_cost']}")
    print("#######################################\n\n")







def read_setup_from_choline_yaml():
    choline_data = get_choline_yaml_data()
    
    setup_script = choline_data.get('setup_script', '')
    cwd = os.getcwd()

    setup_script_path = os.path.join(cwd, ".choline", "choline_setup.sh")

    # Save the script to a file
    with open(setup_script_path, "w") as f:
        f.write(setup_script)

    return setup_script_path



def ssh_copy_directory(scp, ssh, local_path, remote_base_path):
    # ignore_patterns = read_cholineignore()
    ignore_patterns = []
    cwd = os.getcwd()
    for root, dirs, files in os.walk(local_path):
        for file_name in files:
            local_file = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_file, cwd)
            # if should_ignore(relative_path, ignore_patterns):
            #     continue
            remote_file = os.path.join(remote_base_path, relative_path).replace('\\', '/')
            remote_dir = os.path.dirname(remote_file)
            
            stdin, stdout, stderr = ssh.exec_command(f"mkdir -p {remote_dir}")
            stdout.read()
            scp.put(local_file, remote_file)



def ssh_copy(username, host, port, src, dest):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy)
    client.connect(host, port=port, username=username)
    with SCPClient(client.get_transport()) as scp:
        if os.path.isdir(src):
            ssh_copy_directory(scp, client, src, dest)
        else:
            relative_path = os.path.relpath(src, os.getcwd())
            remote_file = os.path.join(dest, relative_path)
            scp.put(src, remote_file)
    client.close()




def generate_vast_onstart_script():
    # Create .choline directory if it doesn't exist
    choline_dir = os.path.join(os.getcwd(), ".choline")
    if not os.path.exists(choline_dir):
        os.makedirs(choline_dir)

    script_lines = [
        "#!/bin/bash",
        "mkdir -p ~/.choline",  # Create the directory if it doesn't exist
        "echo '0' > ~/choline.txt",
        "while [ ! -f ~/.choline/choline_setup.sh ]; do",
        "  sleep 1",
        "done",
        "sleep 5",  # Allow time for the full script to arrive
        "echo 'running setup script' > ~/choline.txt",
        ". ~/.choline/choline_setup.sh >> ~/.choline/choline_setup_log.txt 2>&1",
        "sh -x ~/.choline/choline_setup.sh >> ~/.choline/choline_setup_log.txt 2>&1"  # Run the script from its expected directory
    ]

    script_content = "\n".join(script_lines)
    setup_script_path = os.path.join(choline_dir, "choline_onstart.sh")

    with open(setup_script_path, "w") as f:
        f.write(script_content)

    return setup_script_path



def search_offers(additional_query=""):
    # query = f"cuda_vers == {cuda_version}"
    query = f""
    if additional_query:
        query = f"{query} {additional_query}"

    command = ["vastai", "search", "offers", "--raw", query]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Error in search:", result.stderr.decode())
        return []
    
    offers = json.loads(result.stdout.decode())
    return offers




def create_instance(instance_id, options):
    command = ["vastai", "create", "instance", str(instance_id)] + options
    print(command)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("res {}".format(result))

    if result.returncode != 0:
        print("Error in instance creation:", result.stderr.decode())
        return None

    stdout_str = result.stdout.decode()
    try:
        match = re.search(r"'new_contract': (\d+)", stdout_str)
        if match:
            new_contract = match.group(1)
            print(f"New contract ID: {new_contract}")
        else:
            print("Failed to find new contract ID.")
            return None
    except Exception as e:
        print(f"Error while parsing: {e}")
        return None

    print("Instance created successfully.")
    return new_contract



# def run_monitor_instance_script(instance_id, max_checks=30):
#     subprocess.Popen(["sudo", "python", "./monitor.py", "--id", str(instance_id), "--max_checks", str(max_checks)])

def run_monitor_instance_script(instance_id, max_checks=30):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    monitor_script_path = os.path.join(script_dir, "monitor.py")

    subprocess.Popen(["sudo", "python", monitor_script_path, "--id", str(instance_id), "--max_checks", str(max_checks)])

def main():
    # choline_data = get_choline_json_data()
    choline_data = get_choline_yaml_data()

    hardware_filters = choline_data.get('hardware_filters', {})
    cpu_ram_str = hardware_filters.get('cpu_ram', '')
    gpu_name = hardware_filters.get('gpu_name', '')
    disk_space_str = hardware_filters.get('disk_space', '')
    image = choline_data.get('image', 'python:3.8')
    startup_script_path = generate_vast_onstart_script()

    read_setup_from_choline_yaml()

    # Extract operator and value from the disk_space string
    match = re.match(r"([<>!=]+)(\d+)", cpu_ram_str)
    if match:
        cpu_ram_operator, cpu_ram_value = match.groups()
        cpu_ram_value = int(cpu_ram_value)
    else:
        print("Invalid disk_space string in JSON. Using default values.")
        return 
    # Extract operator and value from the disk_space string
    match = re.match(r"([<>!=]+)(\d+)", disk_space_str)
    if match:
        disk_space_operator, disk_space_value = match.groups()
        disk_space_value = int(disk_space_value)
    else:
        print("Invalid disk_space string in JSON. Using default values.")
        return 

    query = f"gpu_name={gpu_name} disk_space {disk_space_operator} {disk_space_value} cpu_ram > {cpu_ram_value}"
    print(query)
    offers = search_offers(query)
    exp_storage = disk_space_value

    offers = sort_offers_by_custom_criteria(offers, expected_storage_gb=exp_storage, expected_runtime_hr=2)

    if offers:
        print("Five cheapest offers:")
        for i, offer in enumerate(offers[:5]):
            pretty_print_offer(offer, i)

        choice = int(input("Select an offer by entering its number (1-5): ")) - 1
        selected_offer = offers[choice]
        pretty_print_offer(selected_offer, choice)

        confirmation = input("Would you like to proceed? (y/n): ")
        if confirmation.lower() == 'y':
            instance_id = selected_offer["id"]
            options = ["--image", image, "--disk", str(disk_space_value), "--onstart", startup_script_path, "--env", "-e TZ=PDT -e XNAME=XX4 -p 22:22 -p 8080:8080"]
            contract_id = create_instance(instance_id, options)
            print(int(contract_id))
            run_monitor_instance_script(instance_id=int(contract_id))
            print(f"Instance creation request complete. Now setting up your instance with id {instance_id}. Run 'choline status 'instance id'' to check the logs for your setup.")
        else:
            print("Operation canceled.")
    else:
        print("No suitable offers found.")

if __name__ == "__main__":
    main()
