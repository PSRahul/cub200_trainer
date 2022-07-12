podman run --rm --gpus device=0 -it \
-v "/mnt/data/home/rsr/MasterThesis/:/home/psrahul/MasterThesis/" \
--name rsr_trainer \
--shm-size=2gb \
localhost/rsr/trainer :latest
