goal is to .... 

-> standardized way for creating instances from the choline.yaml format 
-> monitoring success / failure of vanilla startup 
-> machine setup of correct python and req.txt and file syncing from yaml 


Probably best to seperate file syncing code from machine creation for better organization 


Notes 


-> there is a size limit for the vastai onstart script -- so we will use our own 

-> the choline.yaml file format is still under consideration, and is created using the choline init command, which is also still under development

-> choline init gets the system info for things like python versions and req.txt, and also asks for the files that need to be transfered over, and what GPU's you want to use 





how it works 


-> after creation command, a monitor script it started that constantly checks for the choline.txt file on the instance 
-> onstart is a script that writes a choline.txt file signifying completion and also waits for choline_setup.sh to arrive 
-> one monitor detects choline.txt, it sends the choline_setup.sh and the rest of the files to be sent to the instance 
-> after that, it will wait until it revcieves a choline_sync_complete.txt file, which signifies its ready to run the onstart command 




UPDATES 

- i switched from json to yaml because yaml is better yo 

