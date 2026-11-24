# import python-openrecon-server as starting point
FROM python-openrecon-server AS base

COPY . .

# Label with version of the app will be set automatically by the build.py script

# Command line will be added automatically by the build.py script

# Docker image metadata label `com.siemens-healthineers.magneticresonance.OpenRecon.metadata:1.1.0`
# will also be set automatically by the build.py script with the base64-encoded JSON text load from
# the json file provided in the app directory
