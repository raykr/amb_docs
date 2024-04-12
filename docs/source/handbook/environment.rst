环境介绍
============================
.. 此处按骏哥儿README列出的环境来，内容借鉴MARLlib的介绍，其中需要包含一张环境的图片、一段话简介、相关链接、安装方法、使用方法。最后可以一个表格总结所有环境（任务模式、可观测性、动作空间、观测空间维度、全局状态、全局状态维度、回报、交互模式等）。例如下面的示例：

.. _SMAC:

SMAC
---------------------

.. image:: ../_static/images/env/smac.png

SMAC 是 WhiRL 基于暴雪星际争霸 II RTS 游戏的协作多智能体强化学习 (MARL) 领域的研究环境。 
SMAC 利用暴雪的星际争霸 II 机器学习 API 和 DeepMind 的 PySC2 为自主代理提供便捷的接口，以便与星际争霸 II 交互、获取观察结果并执行操作。
与 PySC2 不同，SMAC 专注于去中心化的微观管理场景，其中游戏的每个单元都由单独的 RL 代理控制。

相关链接: `oxwhirl/smac <https://github.com/oxwhirl/smac>`_

环境特征
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "环境", "任务模式", "可观测性", "动作空间", "回报", "交互模式"
    :widths: 20, 20, 20, 20, 20, 20

    "SMAC", "合作", "部分可观测", "离散", "稠密/稀疏", "同时"

安装方法
~~~~~~~~~~~~~~~~~~~~

1. 安装 StarCraft II。 切换到要安装StarCraftII的目录，然后运行以下命令：

.. code-block:: bash

    wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    unzip -P iagreetotheeula SC2.4.10.zip
    rm -rf SC2.4.10.zip
    cd StarCraftII/
    wget https://raw.githubusercontent.com/Blizzard/s2client-proto/master/stableid.json


在 ``~/. bashrc`` 中添加以下行：

.. code-block:: bash

    export SC2PATH="/path/to/your/StarCraftII"


将 ``amb/envs/smac/SMAC_Maps`` 和 ``amb/envs/smacv2/SMAC_Maps`` 目录复制到 ``StarCraftII/Maps`` 。

1. 安装SMAC。 

.. code-block:: bash

    pip install git+https://github.com/oxwhirl/smac.git


使用方法
~~~~~~~~~~~~~~~~~~~~

更改 ``amb/configs/env_cfgs/smac.yaml`` 配置项：

.. code-block:: yaml

    map_name: 3s_vs_4z
    state_type: FP # choose from FP (Feature Pruned) and EP (Environment Provided)
    replay_dir: ""
    replay_prefix: ""

训练时指定环境：

.. code-block:: bash

    python -u single_train.py --env smac --algo mappo --run single   

.. hint:: 项目可自动查找环境名称对应的配置项，保持名称相同即可。

.. _SMACv2:

SMACv2
---------------------


.. _MAMuJoCo:

Multi-Agent MuJoCo
---------------------


.. _MPE:

PettingZoo MPE
---------------------

.. _GRF:

Google Research Football
-------------------------------


.. _Gym:

Gym
---------------------

.. _Toy:

Toy Example
---------------------

.. _MetaDrive:

MetaDrive
---------------------

.. _Bi-DexHands:

Bi-DexHands
---------------------

.. image:: ../_static/images/env/bi-dexhands.gif

Bi-DexHands（`bi-dexhands.ai <https://bi-dexhands.ai/>`_）是基于Issac Gym构建的模拟人类灵巧双手操作的任务集合（例如移交、举起、投掷、放置等）。

相关链接: `PKU-MARL/DexterousHands <https://github.com/PKU-MARL/DexterousHands>`_, `bi-dexhands.ai <https://bi-dexhands.ai/>`_ 

环境特征
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "环境", "任务模式", "可观测性", "动作空间", "回报", "交互模式"
    :widths: 20, 20, 20, 20, 20, 20

    "dexhands", "合作", "部分可观测", "连续", "\-", "同时"

安装方法
~~~~~~~~~~~~~~~~~~~~

1. 首先正确安装IsaacGym，有关IsaacGym安装的详细信息可以在 `Isaac Gym <https://developer.nvidia.com/isaac-gym>`_ 找到。

.. code-block:: bash

    cd isaacgym/python
    pip install -e .

2. 正确安装Vulkan SDK，有关Vulkan驱动的详细信息可以在 `Vulkan <https://vulkan.lunarg.com/sdk/home>`_ 找到。

.. code-block:: bash

    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
    sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.275-jammy.list https://packages.lunarg.com/vulkan/1.3.275/lunarg-vulkan-1.3.275-jammy.list
    sudo apt update
    sudo apt install vulkan-sdk 

使用方法
~~~~~~~~~~~~~~~~~~~~

更改 ``amb/configs/env_cfgs/dexhands.yaml`` 配置项：

.. code-block:: yaml

    task: ShadowHandCatchOver2Underarm
    hands_episode_length: 75 

训练时指定环境：

.. code-block:: bash

    python -u single_train.py --env dexhands --algo mappo --run single


.. _Quads:

Quadrotor Swarms
---------------------

.. raw:: html

   <style>
       .row {
           display: flex;
           flex-wrap: wrap;
       }

       .column {
           flex: 50%;
           max-width: 50%;
           padding: 0 4px;
       }

       .column img {
           margin-top: 8px;
           vertical-align: middle;
           width: 100%;
       }
   </style>

.. raw:: html

   <div class="row">
       <div class="column">
           <img src="../_static/images/env/quads1.gif" alt="Image 1">
       </div>
       <div class="column">
           <img src="../_static/images/env/quads2.gif" alt="Image 2">
       </div>
   </div>
   <div class="row">
        <div class="column">
           <img src="../_static/images/env/quads3.gif" alt="Image 3">
        </div>
        <div class="column">
           <img src="../_static/images/env/quads4.gif" alt="Image 4">
        </div>
    </div>
    <br>

Quadrotor Swarms 是从 `gym_art <https://github.com/amolchanov86/gym_art>`_ 延伸出来的飞行动力学模拟器，扩展到支持四旋翼无人机群的模拟任务。

相关链接: `Zhehui-Huang/quad-swarm-rl <https://github.com/Zhehui-Huang/quad-swarm-rl/>`_

环境特征
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "环境", "任务模式", "可观测性", "动作空间", "回报", "交互模式"
    :widths: 20, 20, 20, 20, 20, 20

    "quads", "合作", "部分可观测", "连续", "\-", "同时"

安装方法
~~~~~~~~~~~~~~~~~~~~

已在源码中集成，无需额外安装。

使用方法
~~~~~~~~~~~~~~~~~~~~

更改 ``amb/configs/env_cfgs/quads.yaml`` 配置项：

.. code-block:: yaml

    scenario: quadrotor_multi
    conf:
        quads_use_numba: True
        quads_num_agents: 2
        quads_mode: static_same_goal
        quads_episode_duration: 15.0
        quads_neighbor_encoder_type: no_encoder
        quads_neighbor_hidden_size: 0
        quads_neighbor_obs_type: none
        quads_neighbor_visible_num: 0
        replay_buffer_sample_prob: 0.75
        quads_obs_repr: xyz_vxyz_R_omega
        quads_collision_hitbox_radius: 2.0
        quads_collision_falloff_radius: 1.0
        quads_use_obstacles: False
        quads_obst_density: 0.2
        quads_obst_size: 1.0
        quads_obst_spawn_area: [6.0, 6.0]
        quads_use_downwash: False
        quads_room_dims: [10., 10., 10.]
        quads_view_mode: ['topdown', 'chase', 'global']
        quads_render: False
        quads_domain_random: False
        quads_obst_density_random: False
        quads_obst_size_random: False
        quads_obst_density_min: 0.05
        quads_obst_density_max: 0.2
        quads_obst_size_min: 0.3
        quads_obst_size_max: 0.6
        quads_collision_reward: 0.0
        quads_collision_smooth_max_penalty: 10.0
        quads_obst_collision_reward: 0.0
        anneal_collision_steps: 0.0
        with_pbt: False
        visualize_v_value: False

训练时指定环境：

.. code-block:: bash

    python -u single_train.py --env quads --algo mappo --run single


.. _Network:

Network System Control
-----------------------------

.. image:: ../_static/images/env/network.png

Network System Control网络系统控制环境是一个模拟交通信号控制的多智能体强化学习环境。在这个环境中，智能体需要通过控制交通信号来优化交通流量。可用的四个场景分别是：

- ATSC Grid：合成交通网格中的自适应交通信号控制。
- ATSC Monaco：摩纳哥市真实交通网络中的自适应交通信号控制。
- CACC Catch-up：协作自适应巡航控制，用于追赶领先车辆。
- CACC Slow-down：协作自适应巡航控制，用于跟随领先车辆减速。

相关链接: `cts198859/deeprl_network <https://github.com/cts198859/deeprl_network>`_

环境特征
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "环境", "任务模式", "可观测性", "动作空间", "回报", "交互模式"
    :widths: 20, 20, 20, 20, 20, 20

    "network", "合作", "部分可观测", "离散", "\-", "同时"

安装依赖
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install eclipse-sumo sumolib traci seaborn ipdb


使用方法
~~~~~~~~~~~~~~~~~~~~

更改 ``amb/configs/env_cfgs/network.yaml`` 配置项：

.. code-block:: yaml

    scenario: large_grid
    network_cfg: config_ma2c_nc_grid.ini
    output_dir: results/network_data # from root of repo

训练时指定环境：

.. code-block:: bash

    python -u single_train.py --env network --algo mappo --run single


.. _Voltage:

Voltage Control
---------------------

.. image:: ../_static/images/env/voltage.png

Active Voltage Control on Power Distribution Networks (MAPDN)是电力分布网络上分布式/分散式有源电压控制的环境，也是可用于批量的状态最先进的多智能体Actor-Critic算法的训练。环境的实现遵循PyMARL中提供的多智能体环境框架。

相关链接: `Future-Power-Networks/MAPDN <https://github.com/Future-Power-Networks/MAPDN>`_

环境特征
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "环境", "任务模式", "可观测性", "动作空间", "回报", "交互模式"
    :widths: 20, 20, 20, 20, 20, 20

    "voltage", "合作", "部分可观测", "离散", "\-", "同时"

安装依赖
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install pandapower


使用方法
~~~~~~~~~~~~~~~~~~~~

更改 ``amb/configs/env_cfgs/voltage.yaml`` 配置项：

.. code-block:: yaml

    env: "voltage_control"

    env_args:
        "voltage_barrier_type": "l1" # "l1", "l2", "bowl", "courant_beltrami", "bump"
        "voltage_weight": 1.0
        "q_weight": 0.1
        "line_weight": null
        "dq_dv_weight": null
        "history": 1
        "pv_scale": 1.0
        "demand_scale": 1.0
        "state_space": ["pv", "demand", "reactive", "vm_pu", "va_degree"]
        "v_upper": 1.05
        "v_lower": 0.95
        "data_path": "data/case33_3min_final" # from the root of valtage_control environment folder
        "episode_limit": 240 # in time intervals
        "action_scale": 0.8
        "action_bias": 0
        "mode": distributed # distributed / decentralised
        "reset_action": True
        "seed": 0

训练时指定环境：

.. code-block:: bash

    python -u single_train.py --env voltage --algo mappo --run single
