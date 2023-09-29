# ### gets run as python monitor --id vastai_id 

# ##### will loop in the background and check for the choline.txt file signifying the completion of the startup script 

# ##### will also check for failures  

import subprocess
import os
import json
import time
import argparse
import platform
import paramiko
from scp import SCPClient
from fnmatch import fnmatch
import traceback
import yaml 

import unittest
import os


print("Exception traceback:")
traceback.print_exc()

import yaml

def get_choline_yaml_data():
    with open('choline.yaml', 'r') as f:
        data = yaml.safe_load(f)
    return data


def get_choline_json_data():
    with open('choline.json', 'r') as f:
        data = json.load(f)
    return data


def send_alert(message):
    if platform.system() == "Linux" or platform.system() == "Darwin":
        os.system(f'echo "{message}" | wall')



def read_cholineignore():
    ignore_patterns = []
    try:
        with open('./.cholineignore', 'r') as f:
            ignore_patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        pass
    return ignore_patterns


def should_ignore(path, ignore_patterns):
    for pattern in ignore_patterns:
        if fnmatch(path, pattern):
            return True
    return False

def get_ssh_details(vastai_id):
    result = subprocess.run(f"vastai ssh-url {vastai_id}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ssh_url = result.stdout.strip()
    if ssh_url.startswith('ssh://'):
        ssh_url = ssh_url[6:]
    username, rest = ssh_url.split('@')
    host, port = rest.split(':')
    return username, host, port


def ssh_copy_directory(scp, ssh, local_path, remote_base_path):
    ignore_patterns = read_cholineignore()
    cwd = os.getcwd()
    for root, dirs, files in os.walk(local_path):
        for file_name in files:
            local_file = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_file, cwd)
            if should_ignore(relative_path, ignore_patterns):
                continue
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


def check_for_choline_txt(vastai_id):
    result = subprocess.run(f"vastai copy {vastai_id}:/root/choline.txt ~/.choline", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(str(result))
    
    if result.returncode != 0 or "Invalid src_id" in result.stdout or "Invalid src_full_path" in result.stdout:
        print("Machine not yet operational. Waiting...")
        return False

    print(f"Detected Operational Machine {vastai_id}.")
    return True


# def send_choline_setup(vastai_id, max_retries=100):
#     # needs to read the setup script from choline.yaml and write it to .choline/choline_setup.sh then send it 
#     username, host, port = get_ssh_details(vastai_id)
#     current_path = os.getcwd()
#     local_path = os.path.join(current_path, '.choline')
#     remote_path = '/root'
#     retries = 0
#     while retries < max_retries:
#         try:
#             ssh_copy(username, host, port, local_path, remote_path)
#             print("Sent setup script")
#             break
#         except Exception as e:
#             print(f"Failed to send choline_setup: {e}. Waiting 5 seconds before retrying.")
#             time.sleep(5)
#             retries += 1


def send_choline_setup(vastai_id, max_retries=100):
    choline_data = get_choline_yaml_data()
    setup_script = choline_data.get('setup_script', '')

    local_path = os.path.join(os.getcwd(), '.choline')
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    setup_script_path = os.path.join(local_path, 'choline_setup.sh')

    with open(setup_script_path, 'w') as f:
        f.write(setup_script)

    username, host, port = get_ssh_details(vastai_id)
    remote_path = '/root'
    retries = 0
    while retries < max_retries:
        try:
            ssh_copy(username, host, port, local_path, remote_path)
            print("Sent setup script to instance.")
            break
        except Exception as e:
            print(f"Error to send choline setup: {e}. Waiting 20 seconds before retrying.")
            time.sleep(20)
            retries += 1

            
def send_upload_locations(vastai_id, upload_locations, max_retries=100):
    username, host, port = get_ssh_details(vastai_id)
    current_path = os.getcwd()
    remote_path = '/root'
    for location in upload_locations:
        retries = 0
        local_path = os.path.join(current_path, location)
        while retries < max_retries:
            try:
                ssh_copy(username, host, port, local_path, remote_path)
                print(f"Sent location {local_path}")
                break
            except Exception as e:
                print(f"Failed to send {location}: {e}. Waiting 20 seconds before retrying.")
                time.sleep(20)
                retries += 1

    complete_file = os.path.join(current_path, '.choline/choline_complete.txt')
    with open(complete_file, 'w') as f:
        f.write('0')

    retries = 0
    while retries < max_retries:
        try:
            ssh_copy(username, host, port, complete_file, remote_path)
            print("Data sync complete")
            break
        except Exception as e:
            print(f"Failed to send data sync completion indicator to machine: {e}. Waiting 20 seconds before retrying.")
            time.sleep(20)
            retries += 1

def main(vastai_id, max_checks):
    checks = 0
    ignore_patterns = read_cholineignore()
    print("waiting 25 seconds for machine startup...")
    time.sleep(25)
    
    while checks < max_checks:
        try:
            if check_for_choline_txt(vastai_id): # signifys machine is operational 
                choline_data = get_choline_yaml_data()
                upload_locations = choline_data.get('upload_locations', [])
                print("Sending Setup Script")
                send_choline_setup(vastai_id) # for setting up the instance 
                print("Sending Upload Locations")
                send_upload_locations(vastai_id, upload_locations) # transer required data 
                return
            print("waiting to try again")
            time.sleep(6)
            checks += 1
        except Exception as e:
            import traceback
            print("Error setting up machine. This may not be severe, and we will retry momentarily")
            traceback.print_exc()
            print("trying again in 5 seconds")
            time.sleep(5)
            continue
    
    if checks >= max_checks:
        print("failed to setup machine. We reccomend you try to launch a different machine, as this is likely an issue with the machine")
        send_alert(f"Instance {vastai_id} has failed to start up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor VastAI instance for startup completion.")
    parser.add_argument("--id", type=str, required=True, help="VastAI instance ID to monitor")
    parser.add_argument("--max_checks", type=int, default=1000, help="Maximum number of checks before declaring failure")
    args = parser.parse_args()
    main(args.id, args.max_checks)








# def send_choline_setup(vastai_id, max_retries=100):
#     print("in send setup script")
#     current_path = os.getcwd()
#     retries = 0
#     while retries < max_retries:
#         result = subprocess.run(f"vastai copy {current_path}/cholineSetupPayload/ {vastai_id}:/root/", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         print("res {}".format(str(result)))
#         if result.returncode == 0:
#             print("Sent setup script")
#             break
#         print("Failed to send choline_setup.sh. Waiting 5 seconds before retrying.")
#         time.sleep(5)
#         retries += 1

# def send_upload_locations(vastai_id, upload_locations, max_retries=100):
#     current_path = os.getcwd()
#     for location in upload_locations:
#         retries = 0
#         while retries < max_retries:
#             result = subprocess.run(f"vastai copy {location} {vastai_id}:/root/", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#             if result.returncode == 0:
#                 print("Sent location {}".format({current_path}/{location}))
#                 break
#             print(f"Failed to send {location}. Waiting 5 seconds before retrying.")
#             time.sleep(5)
#             retries += 1

#     with open(f'{current_path}/choline_complete.txt', 'w') as f:
#         f.write('0')

#     retries = 0
#     while retries < max_retries:
#         result = subprocess.run(f"vastai copy {current_path}/choline_complete.txt {vastai_id}:/root/choline_complete.txt", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if result.returncode == 0:
#             break
#         print("Failed to send choline_complete.txt. Waiting 5 seconds before retrying.")
#         time.sleep(5)
#         retries += 1






    # class TestVastaiSSHFunctions(unittest.TestCase):
#     def test_send_choline_setup(self):
#         instance_id = "7093888"  # Replace with your instance ID
#         send_choline_setup(instance_id)
#         # Check logs or destination to confirm file has been transferred
        
#     def test_send_upload_locations(self):
#         instance_id = "7093888"  # Replace with your instance ID
#         upload_locations = ["/Users/brettyoung/Desktop/dev/choline/dev/simple_startup/test_payload"]  # Replace with your test file name
#         send_upload_locations(instance_id, upload_locations)
#         # Check logs or destination to confirm file has been transferred

#     def test_all(self):
#         self.test_send_choline_setup()
#         self.test_send_upload_locations()

# if __name__ == '__main__':
#     unittest.main()

