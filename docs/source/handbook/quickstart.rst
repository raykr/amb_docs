快速开始
============================
.. 这里可以套骏哥儿README.md使用流程，同时借鉴MARLlib的Quick Start

安装
---------------------
.. conda、环境、依赖安装

创建conda环境
^^^^^^^^^^^^^^^

使用anaconda管理实验环境，对实验环境进行隔离管理，确保依赖包的版本稳定。

.. code-block:: bash

    # This will create an Anaconda environment named amb.    
    conda env create -f amb.yml

安装StarCraftII游戏引擎
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

切换到你想要安装StarCraftII游戏引擎的目录，然后执行以下指令：

.. code-block:: bash

    wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    unzip -P iagreetotheeula SC2.4.10.zip 
    rm -rf SC2.4.10.zip

    cd StarCraftII/
    wget https://raw.githubusercontent.com/Blizzard/s2client-proto/master/stableid.json

然后将安装StarCraftII的路径添加到bash或者zsh的系统变量。

.. code-block:: bash

    # for bash
    vim ~/.bashrc
    export SC2PATH="/path/to/your/StarCraftII"

最后需要将 ``amb/envs/smac/SMAC_Maps`` 和 ``amb/envs/smacv2/SMAC_Maps`` 文件夹复制到 ``StarCraftII/Maps`` 下。

安装Google Research Football
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

首先安装一些系统级别的依赖（只有linux需要）：

.. code-block:: bash

    sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

然后通过 ``pip`` 安装 ``GRF``

.. code-block:: bash

    pip install gfootball

使用示例
---------------------
.. Usage Example

训练单个算法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -u single_train.py --env <env_name> --algo <algo_name> --exp_name <exp_name> --run single

训练基于扰动的攻击算法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
    
    python -u single_train.py --env <env_name> --algo <perturbation_algo_name> --exp_name <exp_name> --run perturbation --victim <victim_algo_name> --victim.model_dir <dir/to/your/model>

训练对抗性内鬼算法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -u single_train.py --env <env_name> --algo <traitor_algo_name> --exp_name <exp_name> --run traitor --victim <victim_algo_name> --victim.model_dir <dir/to/your/model>

同时训练对抗双方算法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # In dual training, "angel" and "demon" are two competitive teams, where we only train "angel" but fix "demon".
    python -u dual_train.py --env <env_name> --angel <angel_algo_name> --demon <demon_algo_name> --exp_name <exp_name> --run dual

从文件中加载受害者参数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # It will load environment and victim configurations from JSON, together with the victim's checkpoints in "models/" directory
    python -u single_train.py --algo <adv_algo_name> --exp_name <exp_name> --run [traitor|perturbation] --load_victim <dir/to/victim/results>
    # In dual training, you can load angel and demon separately, even from single training checkpoint.
    python -u dual_train.py --env <env_name> --load_angel <dir/to/angel/results> --load_victim <dir/to/demon/results> --exp_name <exp_name> --run dual


配置项
---------------------

环境配置
^^^^^^^^^^^^^^^
.. 这里首先放一个code_block，罗列一下环境的配置项，然后再详细介绍每个配置项的含义

.. code-block:: bash
    
    # senario name
    map_name: 3s_vs_4z
    # choose from FP (Feature Pruned) and EP (Environment Provided)
    state_type: FP 
    # where to save the replay video
    replay_dir: ""
    # replay video's prefix
    replay_prefix: ""


算法配置
^^^^^^^^^^^^^^^
.. 同上
.. code-block:: bash
    
    # seed:
    # whether to use the specified seed
    seed_specify: True
    # seed
    seed: 1
    # device:
    # whether to use CUDA
    cuda: True
    # whether to set CUDA deterministic
    cuda_deterministic: True
    # arg to torch.set_num_threads
    torch_threads: 4
    # train:
    # number of parallel environments for training data collection
    n_rollout_threads: 20
    # number of total steps
    num_env_steps: 10000000
    # max length of an episode
    episode_length: 150
    # number of warmup steps
    warmup_steps: 50000
    # number of steps per train
    train_interval: 1000
    # ratio of training iterations to train_interval
    update_per_train: 0.05
    # logging interval
    log_interval: 20000
    # evaluation interval
    eval_interval: 100000
    # whether to use linear learning rate decay
    use_linear_lr_decay: False
    # if set, load models from this directory; otherwise, randomly initialise the models
    model_dir: ~
    # eval:
    # whether to use evaluation
    use_eval: True
    # number of parallel environments for evaluation
    n_eval_rollout_threads: 10
    # number of episodes per evaluation
    eval_episodes: 20
    # render:
    # whether to use render
    use_render: False
    # number of episodes to render
    render_episodes: 10
    # model:
    # network parameters
    # hidden sizes for mlp module in the network
    hidden_sizes: [256, 256]
    # activation function, choose from sigmoid, tanh, relu, leaky_relu, selu
    activation_func: relu
    # final activation function, choose from sigmoid, tanh, relu, leaky_relu, selu
    final_activation_func: tanh
    # whether to use feature normalization
    use_feature_normalization: True
    # initialization method for network parameters, choose from xavier_uniform_, orthogonal_, ...
    initialization_method: orthogonal_
    # optimizer parameters
    # actor learning rate
    lr: 0.0005
    # critic learning rate
    critic_lr: 0.0005
    # recurrent parameters
    # whether to use rnn policy (data is chunked for training)
    use_recurrent_policy: False
    # number of recurrent layers
    recurrent_n: 1
    # algo:
    # discount factor
    gamma: 0.99
    # off-policy buffer size
    buffer_size: 5000
    # training batch size
    batch_size: 1000
    # coefficient for target model soft update
    polyak: 0.005
    # exploration noise
    expl_noise: 0.1
    # the number of steps to look ahead
    n_step: 1
    # whether to clip gradient norm
    use_max_grad_norm: False
    # max gradient norm
    max_grad_norm: 10.0
    # whether to share parameter among actors
    share_param: True
    # whether to use policy active masks
    use_policy_active_masks: True
    # logger:
    # logging directory
    log_dir: "./results"

