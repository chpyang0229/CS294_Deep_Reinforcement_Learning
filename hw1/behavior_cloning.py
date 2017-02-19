import pickle
import numpy as np

# read the file
envname = 'HalfCheetah-v1'
f=open(envname+'.pkl','rb')
status_actor_file=pickle.load(f)
observations=status_actor_file['observations']
actions=status_actor_file['actions']