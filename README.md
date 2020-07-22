# Fetch Block Construction
- Produce up to 25 blocks, which is the max number supported by Mujoco 1.5
    - Max 17 blocks with uniform sampling of collision-free init positions on the table
- Set novel goal configurations with a goal position vector for each block
- Preset goal configurations
    - Single tower
    - Multiple towers
    - Pyramid
- Randomize block lengths to produce cuboids
- Optional rotational control for action space

## Installation
1. From the project root: 
`pip install -e .`

## Docker Build
Building the Docker image is optional, but makes it much easier to continuously deploy to EC2, GCP, etc. The Docker image is designed to support `rlkit-relational`. The example scripts there will ask you to input the name of your Docker image.

1. You need a mujoco license file. Move the text license into the root `fetch-block-construction` directory, and rename it as `mjkey.txt`
2. From the project root, run: 
`docker build -t <docker_username>/<image_name>:<image_tag> .`
3. `docker push <docker_username>/<image_name>:<image_tag>`

### Credits
Initial stack environment code based on
[gym-fetch-stack](https://github.com/CDMCH/gym-fetch-stack


### Citation
If you find this code useful, please cite:

    @inproceedings{li19relationalrl,
      Author = {Li, Richard and
      Jabri, Allan and Darrell, Trevor and Agrawal, Pulkit},
      Title = {Towards Practical Multi-object Manipulation using Relational Reinforcement Learning},
      Booktitle = {ICRA},
      Year = {2020}
    }
