环境介绍
============================
.. 此处按骏哥儿README列出的环境来，内容借鉴MARLlib的介绍，其中需要包含一张环境的图片、一段话简介、官方链接、安装方法、使用方法。最后可以一个表格总结所有环境（任务模式、可观测性、动作空间、观测空间维度、全局状态、全局状态维度、回报、交互模式等）。例如下面的示例：

.. _SMAC:

SMAC
---------------------

.. image:: ../_static/images/env/smac.png

SMAC 是 WhiRL 基于暴雪星际争霸 II RTS 游戏的协作多智能体强化学习 (MARL) 领域的研究环境。 
SMAC 利用暴雪的星际争霸 II 机器学习 API 和 DeepMind 的 PySC2 为自主代理提供便捷的接口，以便与星际争霸 II 交互、获取观察结果并执行操作。
与 PySC2 不同，SMAC 专注于去中心化的微观管理场景，其中游戏的每个单元都由单独的 RL 代理控制。

官方链接: `oxwhirl/smac <https://github.com/oxwhirl/smac>`_

环境特征
~~~~~~~~~~~~~~~~~~~~

.. csv-table::
    :header: "环境", "任务模式", "可观测性", "动作空间", "回报", "交互模式"
    :widths: 20, 20, 20, 20, 20, 20

    "SMAC", "合作", "部分可观测", "离散", "稠密/稀疏", "同时"

安装方法
~~~~~~~~~~~~~~~~~~~~

安装 StarCraft II
++++++++++++++++++++++

切换到要安装StarCraftII的目录，然后运行以下命令：

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

安装SMAC
++++++++++++++++++++++

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