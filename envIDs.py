""" This script prints the environment IDs for
    OpenAI Atari """

from gym import envs
allEnvs = envs.registry.all()
envIDs = [envSpec.id for envSpec in allEnvs]
envIDs.sort()
for ID in envIDs:
    print(ID)
