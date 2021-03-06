# Z-GAN

![Generated birds](./media/sample-C-32.png "Birds gnerated from the GAN")

A Generative Adversarial Network (GAN) used to generate images. Requires a CUDA
GPU. Developed on Linux, but running it on other platforms should not be too
much of a problem. To install all dependencies, create a virtual environment and
run `make py-deps` or simply `pip install -r requirements.txt`.

## Quick overview

- `./dirk/apps.py`: short "applications" that are called by `./cli.py`
- `./dirk/training.py`: Implements the `Trainer` that is responsible for
  maintaining the generator and discriminator as well as their respective
  optimisers during training. It implements the training loop and mechanisms for
  loading and saving the models.
- `./dirk/sidecar.py`: Responsible for logging and other periodic tasks during
  training. This functionality could have been implemented directly in the
  `Trainer`, but to keep the code of the latter cleaner, it was moved into its
  own module.
- `./dirk/dataset.py`: Contains the dataset that loads images from the given
  directory. It assumes that the dataset fits into the main memory of the host.
  Preprocessing of the images is done here.
- `./dirk/models.py`: The implementation of the generator and discriminator.
  Note that the architecture of both is defined in the configuration files (e.g.
  `./config/default.yaml`), and the models implemented in `./dirk/models.py`
  merely construct a torch module from the configuration files. This allows for
  some basic versioning of model architecture across runs and allows us to
  easily load older models, even after changing `./dirk/models.py`. Note that,
  as of the current commit, only feed-forward architectures are possible. This
  could be changed in the future. To make the architectures specified in the
  configuration files more parametrisable, variables can be used. If a string
  value is encountered in either one of the `args` or `kwargs` fields of an
  entry in the `layers` list in the configuration file, they are evaluated using
  python's `eval` with the context given in the `context` field of the
  configuration.

## Create a *project*

To create a new generator-discriminator pair (a *project*), run:

    ./cli.py create

This will create a new directory inside `./zoo/`. More information is logged to
stderr. The default configuration from `./configs/default.yaml` is used. You
probably want to create your own configuration file. To use e.g.
`./configs/my-config.yaml`, run:

    ./cli.py create --config ./configs/my-config.yaml

The configuration file is then copied into the project's folder where it can be
further modified.

## Training

To start training, run:

    ./cli.py train

The training can be interrupted at any time and resumed by simply calling
`./cli.py train` again. To change the current project to e.g. `<project-name>`,
run:

    ./cli.py focus <project-name>

See `./cli.py --help` for more information.

## Inference

To generate images from the current project, run:

    ./cli.py show

To generate images from the current project and only show the best ones
(according to the discriminator), run:

    ./cli.py inference

To generate images and only show the ones that get a score (according to the
discriminator) above a certain threshold, run:

    ./cli.py inference --threshold 0.5

Note that the value of the threshold is highly dependant on the most recent
project, and should be chosen on a per-project basis.

## Web

The `Dockerfile` creates an image that can be used to start a server serving
images from the project pointed to with `./web-project`. To build and run the
docker image, run:

    docker build -t zgan .
    docker run -p 8080:8080 --gpus all zgan

Note that the container requires GPU access and thus the Nvidia Container
Toolkit. The server exposes http://127.0.0.1:8080/best and
http://127.0.0.1:8080/threshold.

## Project directory layout

Each command in the `./cli.py` can be passed a `--zoo` parameter. This is a
directory in which the folders of projects live, each containing the
configuration and checkpoints of one generator-discriminator pair. For a zoo in
e.g. `./zoo/` the `./zoo/latest/` symlink points to the project that is
currently in focus. This symlink will be updated when either `./cli.py create`
or `./cli.py focus` are called. Within each project, the `latest.cp` symlink
points to the project's most recent checkpoint.

The model architecture of both the generator and discriminator is determined by
the `config.yaml` in the project. The way this is done should be
self-explanatory.

## Weights and Biases

Samples generated during training are sent to Weights and Biases
([link](https://wandb.ai/)). This can be disabled in the configuration file. An
example can be seen [here](https://wandb.ai/nlsfnr/zgan/runs/326tnjos).

## Viking

The University of York's compute cluster "Viking" was used to train some of the
GANs in this project. To this end, some scripts can be found in `./viking/`.

![Generated birds](./media/sample-B-32.png "Birds gnerated from the GAN")
