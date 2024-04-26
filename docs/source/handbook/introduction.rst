平台简介
============

“逐鹿”多智能体强化学习鲁棒性评测平台是由北京航空航天大学复杂关键软件环境全国重点实验室刘祥龙教授实验室开发的，对于多智能体强化学习的鲁棒性进行评测的平台。


平台研究背景
---------------------

以深度学习为代表的人工智能技术的迅速发展，为以多智能体强化学习为代表的智能决策领域带来了新的变革，极大地促进了自动驾驶、无人集群控制、金融决策等智能应用的发展，创造了惊人的社会价值。例如，在无人驾驶场景中，智能决策算法替代了人类决策，降低驾驶员的工作强度，并减少了交通事故；在无人集群控制中，智能决策算法通过自主决策，在国防安全、智能制造、抢险救灾等应用中发挥了不可替代的作用，极大减少了人员投入与人员安全风险；在金融决策中，智能决策算法通过实时判断当前状态，从而决定当前操作，具有远超人类的效率。以智能决策为首的人工智能技术被认为是下一个时代的基础技术，正在引领人类社会新的变革。在深度学习为多智能体强化学习和多智能体强化学习带来成功的同时，其脆弱性与鲁棒性也为强化学习和多智能体强化学习带来了较大的安全性风险，使得多智能体强化学习不鲁棒、不可靠、不可控。

在多智能体强化学习部署过程中，智能体可能受到多样化不确定性影响。以合作类多智能体强化学习为例，其成功的重要原因是假设每一个智能体均最大化总体的奖励函数，这既是多智能体强化学习算法成功的前提，也为算法的理论分析与优化带来了重要的便利。但是，在实际算法部署过程中，智能体可能由于软硬件错误而做出随机不可控的动作，又或是被攻击者控制，做出最坏的动作。例如，在美国DARPA资助的OFFSET无人机群控制任务中，由于环境不确定性高、危险性高，无人机可能因耗尽电力而停飞，或在执行任务过程中因外力损毁。在此情况下，基于现有的多智能体强化学习训练出的策略需要能够在同伴的状态不可控的情况下继续完成鲁棒决策，从而完成给定任务；在自动驾驶场景中，如果假设所有无人车均共同合作遵循道路安全规范，则不遵守交通规则和安全规范的驾驶者可以被看作一个“攻击者”。无人车如果始终假设其他车辆均会遵循道路安全规范，在遇到“攻击者”时，将因为无法应对其他无人车的非合作行为，从而产生交通事故。实现鲁棒的多智能体强化学习是保障多智能体强化学习部署过程中的可信、可控的必由之路。


平台实现功能
---------------------

平台实现评测标准多样化、鲁棒性能可调优、评测结果可落地、模型环境可自定的多智能体强化学习鲁棒性一体化评测流程。

对于评测标准多样化而言，本平台根据马尔可夫决策过程中的状态、动作、环境、奖励函数四个维度，模拟真实环境中的多样化不确定性，对于多智能体强化学习训练、部署过程中的多样化鲁棒性缺失问题进行全面、一体化的评测，集成超过10种鲁棒性评测方法，保障算法落地过程中全面、可靠的部署。

对于鲁棒性能可调优而言，本平台集成超过20种训练技巧，并支持对于不同算法、不同环境子任务、不同鲁棒性种类的训练技巧评测。用户可以根据自身部署场景的需要，自行根据需要的鲁棒性种类，选取本平台中最优的算法及训练技巧，或对于自身设计的算法进行参数调优，从而保障算法部署过程中的最优性能。

对于评测结果可落地而言，本平台集成无人机控制、无人车控制、电网控制、灵巧手控制等高保真多智能体强化学习环境，并对于这些环境进行全方位的鲁棒性评价与测试，其结果将有助于这些领域的工作者在应用多智能体强化学习算法的过程中，评估不同多智能体强化学习算法在其领域内的性能和鲁棒性，并决定使用哪种多智能体强化学习算法、哪种多智能体强化学习技巧进行训练。

对于模型环境可自定而言，本平台将多智能体强化学习智能体抽象为Agent类，对于本平台未集成的自定义多智能体强化学习算法实现，只需令待评测智能体满足Agent类规定的接口，即可将其上传至本平台进行评测。对于自定义环境而言，本平台满足任意基于Gym环境接口的算法评测，用户可以根据自身需要，灵活方便地进行自定义。


项目成员
============

项目成员来自北京航空航天大学复杂关键软件环境全国重点实验室，包括：

Faculty
-----------------

.. raw:: html

   <center>
        <figure>
           <div align="left">
                <div style="display:inline-block; width:35%; vertical-align:top">
                    <img src="../_static/images/person/liuxianglong.jpg" style="width:200px; height:300px; margin-top: 10px;"/>
                </div>
                <div style="display:inline-block; width:60%">
                    <h1 align="left">Xianglong Liu</h1>
                    <p align="left" class="note">Professor</p>
                    <p align="left"> <a href="http://www.nlsde.buaa.edu.cn/">State Key Laboratory of Software Development Environment</a><br>
                    School of Computer Science and Engineering, Beihang University, China</p>
                    <p align="left"> Office: Room G606, New Main Building<br>
                        Address: 37 Xueyuan Road, Haidian, Beijing, China 100191<br>
                        Tel.: +86-10-8233-8092 <br>
                        Email: xlliu@nlsde.buaa.edu.cn / xlliu@buaa.edu.cn
                    </p>
                    <p align="left">
                        <a href="http://sites.nlsde.buaa.edu.cn/~xlliu/">Homepage</a>
                    </p>
                </div>
            </div>
        </figure> 
    </center>


PhD Students
-----------------

.. raw:: html

   <center>
       <figure>
           <div align="left">
                <div style="display:inline-block; width:35%; vertical-align:top">
                    <img src="../_static/images/person/lisimin.jpg" style="width:200px; height:300px; margin-top: 10px;"/>
                </div>
                <div style="display:inline-block; width:60%">
                    <h1 align="left">李思民</h1>
                    <p align="left" class="note">博士三年级学生</p>
                    <p align="left"> 
                        <a href="http://www.nlsde.buaa.edu.cn/">北航复杂关键软件环境全国重点实验室</a><br>
                        北航计算机学院
                    </p>
                    <p align="left"> 
                        邮箱：lisiminsimon@buaa.edu.cn<br>
                        地址：北京航空航天大学新主楼G602
                    </p>
                    <p align="left">
                        主页：<a href="https://siminli.github.io">https://siminli.github.io</a>
                    </p>
                </div>
            </div>
        </figure> 
    </center>


.. raw:: html

   <center>
       <figure>
           <div align="left">
                <div style="display:inline-block; width:35%; vertical-align:top">
                    <img src="../_static/images/person/yuxin.jpg" style="width:200px; height:300px; margin-top: 10px;"/>
                </div>
                <div style="display:inline-block; width:60%">
                    <h1 align="left">于鑫</h1>
                    <p align="left" class="note">博士五年级学生</p>
                    <p align="left"> 
                        <a href="http://www.nlsde.buaa.edu.cn/">北航复杂关键软件环境全国重点实验室</a><br>
                        北航计算机学院
                    </p>
                    <p align="left"> 
                        邮箱：nlsdeyuxin@buaa.edu.cn<br>
                        地址：北京航空航天大学新主楼G612
                    </p>
                    <p align="left">
                        主页：<a href="https://xinyu-site.github.io/">https://xinyu-site.github.io/</a>
                    </p>
                </div>
            </div>
        </figure> 
    </center>


MPhil Students
-----------------

.. raw:: html

    <center>
        <figure>
            <div align="left">
                <div style="display:inline-block; width:30%" align="center">
                    <div  style="margin:10px">
                        <img src="../_static/images/person/guojun.jpg" style="height:250px;"/>
                    </div>
                    <h5>郭骏</h5>
                </div>
                <div style="display:inline-block; width:30%" align="center">
                    <div  style="margin:10px">
                        <img src="../_static/images/person/jingzonglei.jpg" style="height:250px;"/>
                    </div>
                    <h5>景宗雷</h5>
                </div>
            </div>
        </figure> 
    </center>

Undergraduate Students
----------------------------------

.. raw:: html

    <center>
        <figure>
            <div align="left">
                <div style="display:inline-block; width:30%" align="center">
                    <div  style="margin:10px">
                        <img src="../_static/images/person/zhouboyang.jpg"/>
                    </div>
                    <h5>周伯阳</h5>
                </div>
            </div>
        </figure> 
    </center>
