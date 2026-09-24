#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 校准语料：64 段，校准 / 验证 / 测试 分离。

协议要求（02_补实验执行协议.md §3）：
  * 校准、验证、测试分离，语料与分割哈希进 manifest，不能用测试集反复调参；
  * 覆盖中文、英文、问答、叙述与少量代码，比例按目标任务预先规定；
  * 每段先喂真实前缀把状态推离零轨迹，再在分层位置采样（由 capture_calib_p1.py 做）。

分段比例（面向“中文课程助手”这个目标任务，中文为主、英文与代码各占少量）：
    校准 32 = 中文叙述 10 + 中文问答 10 + 英文叙述 4 + 英文问答 4 + 代码 4
    验证 16 = 中文叙述 5 + 中文问答 5 + 英文叙述 2 + 英文问答 2 + 代码 2
    测试 16 = 中文叙述 5 + 中文问答 5 + 英文叙述 2 + 英文问答 2 + 代码 2

用法：
    python p1_corpus.py --manifest p1_corpus_manifest.json   # 写清单（含哈希）
    python p1_corpus.py --list                              # 打印每段长度
    from p1_corpus import SEGMENTS, text_for

注意：所有正文都是为本项目写的原创内容（不是仓库里的演示提示词），测试段从不出现在
校准/验证里，三份分割的 sha1 都写进 manifest，换语料后哈希必变。
"""

import argparse
import hashlib
import json
import os

SPLIT_PLAN = {
    "calib": [("cn_narr", 12), ("cn_qa", 10), ("en_narr", 4), ("en_qa", 4), ("code", 2)],
    "val":   [("cn_narr", 5), ("cn_qa", 5), ("en_narr", 2), ("en_qa", 2), ("code", 2)],
    "test":  [("cn_narr", 5), ("cn_qa", 5), ("en_narr", 2), ("en_qa", 2), ("code", 2)],
}

# 素材按分割分区，保证"校准 / 验证 / 测试"三份的原始素材互不重叠
# （同一分割内部允许复用素材，但组合与顺序不同）。区间是 [start, end)。
SPLIT_RANGE = {
    "cn_narr": {"calib": (0, 6), "val": (6, 9), "test": (9, 12)},
    "cn_qa":   {"calib": (0, 8), "val": (8, 14), "test": (14, 20)},
    "en_narr": {"calib": (0, 4), "val": (4, 6), "test": (6, 8)},
    "en_qa":   {"calib": (0, 4), "val": (4, 6), "test": (6, 8)},
    "code":    {"calib": (0, 4), "val": (4, 5), "test": (5, 6)},
}

SYSTEMS = [
    "你是一名大学课程助手，回答准确、简洁，必要时给出例子。",
    "你是一名理工科助教，先给结论再解释原因，术语用中文。",
    "你是一名课程答疑助手，遇到公式时写出关键步骤，不要编造数据。",
    "你是一名学习伙伴，用通俗语言解释概念，并指出常见的理解误区。",
]

# ---------------------------------------------------------------- 中文叙述
# 每段 400~500 字，用于把循环状态推到“真实长轨迹”，同时提供叙述型分布的激活。
CN_NARR = [
    "微积分的出现并不是某一天被某个人凭空想出来的。十七世纪的欧洲同时面对三类问题："
    "求曲线下的面积、求曲线的切线、以及求物体运动的瞬时速度。开普勒整理行星观测数据时"
    "反复做求和，费马与笛卡尔用代数方法处理切线，巴罗已经发现求面积与求切线之间有互逆的"
    "味道。牛顿把这些线索统一到流数法里，用速度的瞬时变化描述运动；莱布尼茨则从差分与"
    "求和出发，设计了今天仍在用的微分与积分记号。两人的工作独立完成，但因为符号系统更"
    "便于运算与传播，莱布尼茨的写法最终成为通用语言。真正让微积分站得住脚的，是后来对"
    "极限的严格化：从柯西到魏尔斯特拉斯，人们用数列与邻域替换了含糊的无穷小，把直观的"
    "几何想象变成可以验证的推理。这段历史说明，数学工具常常先被使用、再被理解，最后才被"
    "严格化，而每一次严格化都会带来新的问题与新的分支。",
    "计算机的存储层次本质上是在速度、容量与价格之间做折中。寄存器位于处理器内部，"
    "一两个时钟周期就能访问，但总数极少；一级缓存把最常用的数据留在核心旁边，容量只有"
    "几十千字节；二级与三级缓存逐级放大容量、降低速度；主存的容量以吉字节计，但要经历"
    "几十到上百个周期的等待；固态硬盘与磁盘则把容量扩展到太字节，代价是毫秒级延迟。"
    "为了让这套层次真正发挥作用，程序需要表现出时间局部性与空间局部性：反复使用同一批"
    "数据，或者按顺序访问连续内存。矩阵乘法按行还是按列遍历、结构体数组还是数组结构体，"
    "性能差异往往来自缓存命中率而不是算法本身。理解层次结构的人，写出的代码常常看起来"
    "朴素，却在实测中快出数倍，原因就在这里。",
    "中国的季风气候与海陆热力差异密切相关。冬季陆地降温快，西伯利亚与蒙古高原形成强冷高压，"
    "气流从内陆吹向海洋，带来干冷空气；夏季陆地升温快，亚洲大陆形成热低压，海洋上的暖湿"
    "气流源源不断地输送到内陆，降水集中在几个月内。副热带高压的位置决定了雨带在哪里停留，"
    "它的北跳与南退带来华南前汛期、江淮梅雨与华北雨季的相继出现。同一条雨带的异常停留，"
    "既可能造成长江流域的洪涝，也可能让华北出现干旱。青藏高原的存在抬升了高空气流，"
    "使季风环流更稳定；海温异常则通过大气环流在数月尺度上调制降水。因此预报中国的降水，"
    "既要看海温与大气环流，也要看地形与季节内振荡，单一指标往往无法解释全部现象。",
    "蛋白质折叠问题问的是：一条由氨基酸组成的链，如何在没有外部指导的情况下迅速找到"
    "唯一的天然构象。安芬森的实验表明，序列本身包含了构象的全部信息，因为变性的蛋白能在"
    "合适条件下自发恢复活性。但按随机搜索计算，可能的构象数量大到不可能在生物学时间尺度上"
    "穷举，于是人们提出能量漏斗的图景：序列设计出的是一个偏向天然态的能量景观，链沿着"
    "漏斗快速下滑，先在局部形成二级结构，再逐步组装成三级结构。分子伴侣帮助链避开错误的"
    "聚集路径，二硫键与金属离子则进一步锁定构象。计算上，这个问题的难度仍然很高，"
    "预测方法的进步主要来自对共进化信息的利用与大量结构数据的训练。",
    "博弈中的搜索算法可以用棋类来说明。极小极大搜索把对局看成两方轮流选择的树，每一层"
    "交换考虑立场：自己做选择时取最大收益，对手做选择时取最小收益。完全展开的树大得不"
    "可想象，于是引入深度限制与评估函数，把不完整的信息压缩成一个分数。α-β 剪枝利用"
    "已经找到的边界提前放弃无望的分支，在不改变结果的前提下大幅减少访问节点。蒙特卡洛"
    "树搜索则换了一条路：随机模拟大量对局，用胜率估计节点的价值，并把更多时间分配给"
    "看起来有希望的走法。这些方法背后是同一个权衡——把有限的计算预算花在最可能改变"
    "决策的地方，而不是均匀地探索整棵树。",
    "热力学第二定律与信息论里的熵共享同一套数学结构。热力学熵衡量的是宏观状态对应多少"
    "微观实现方式；信息熵衡量的是一个随机变量的平均不确定度。把两者放在一起看，"
    "就会发现信息与物理过程之间存在换算关系：擦除一比特信息至少要向环境释放一份最小的"
    "热量，这就是兰道尔原理。现代计算设备的能耗主要不是花在逻辑门上，而是花在把数据"
    "在存储层次之间搬运，这个现象与上面的原理并不矛盾，因为搬运意味着状态的改变。"
    "反过来说，如果计算是可逆的，理论上可以做到极低耗散，这也是可逆计算与量子计算"
    "研究的一个动机。熵的概念把抽象的“不确定”与具体的“代价”连接了起来。",
    "供需曲线是经济学最基础的图示，但它常被误读。需求曲线描述的是在其他条件不变时，"
    "价格与需求量之间的关系，而不是把价格当成横轴、数量当成纵轴的物理定律。曲线的位置"
    "由收入、偏好、替代品价格等因素决定，价格变化引起的是沿着曲线移动，其他因素变化"
    "才会让曲线整体平移。把这两类变化混为一谈，就会得出“涨价必然减少总收入”这类错误"
    "结论。均衡价格由供给与需求共同决定，但现实中价格还受信息、习惯、政策与市场结构"
    "影响，短期可能长期偏离均衡。因此模型的用处不是预测某个具体价格，而是帮助我们"
    "分清哪些变量在起作用、方向是什么。",
    "从伽利略把镜片装进管子开始，望远镜就一直是把物理原理变成观测能力的典型例子。"
    "折射望远镜用透镜聚光，口径决定分辨能力，色差则限制了成像质量；牛顿用反射镜绕开"
    "色差，于是反射式结构成为后来大型设备的主流。口径越大，收集的光越多、理论分辨率"
    "越高，但镜面形变、大气湍流与跟踪精度都会成为新的瓶颈，这就是自适应光学存在的"
    "原因：用可变形镜面实时补偿大气的扰动。把望远镜送上太空，解决的是同一类问题的"
    "另一个极端——彻底避开大气，用更稳定的环境换更长的曝光。光谱分析则让望远镜不仅"
    "是“看得更远”的工具，还能判断天体的成分与运动状态。",
    "元素周期表的建立过程说明了分类的力量。早期化学家积累了元素的性质与原子量数据，"
    "门捷列夫把元素按原子量排列，发现性质呈现周期性，并大胆为未发现的元素留下空位、"
    "预测其性质。后来原子序数取代原子量成为排序依据，周期律的解释才落到电子排布上："
    "外层电子的填充顺序决定了化学性质的重复出现。周期表的价值不只是把信息摊在一张图上，"
    "而是提供了一个可以推理的结构——看到元素在表中的位置，就能大致判断它的活性、"
    "常见价态与成键倾向。这与计算机中的索引结构有相似之处：好的组织方式能让检索与"
    "推理都变得廉价。",
    "语言与思维的关系是心理学与语言学的经典争论。强版本的说法认为语言决定思维，"
    "弱版本则承认语言会影响注意与记忆的偏好。颜色词的研究提供了较清楚的证据："
    "当两种颜色在语言中共享同一个名称时，人们对它们的区分速度会变慢，但这种影响在"
    "需要快速判断的任务中更明显，在慢速的、有意识比较的任务中就减弱。空间方位表达"
    "方式不同的语言使用者，在记忆方位时会采用不同的参照系。这些结果更支持“语言调制"
    "而非“语言决定”。对课程助手来说，这个结论有直接含义：改写问题的表述方式，"
    "会改变模型关注的侧面，所以提示词设计与术语统一并不是装饰。",
    "光在介质中的传播速度与折射率的关系解释了海市蜃楼与光纤通信。光进入折射率较大的"
    "介质时方向偏折，而折射率随温度或密度变化时，光线就会弯折。沙漠上方的热空气与"
    "上层冷空气形成折射率梯度，远处景物的光线被弯曲到观察者眼中，于是出现倒影。"
    "在光纤里，“全反射”让光被限制在纤芯中传播，损耗主要由材料吸收与散射决定。"
    "同一套几何光学原理还被用在棱镜分光、透镜设计与大气折射修正上，说明基础模型的"
    "适用范围往往比最初发现它的场景更广。",
    "统计推断的核心问题是：从有限的样本出发，能对总体说多少话。抽样分布把“样本均值”"
    "当成随机变量，于是可以计算它的标准差，也就是标准误。置信区间的含义常被误读为"
    "“参数落在区间内的概率”，正确的频率解释是：如果重复抽样并每次构造区间，"
    "长期来看有固定比例的区间会覆盖真值。假设检验则把问题改成“如果真值在某个位置，"
    "观测到这样极端数据的概率有多大”。这些工具都依赖抽样方式与模型的正确性，"
    "所以报告结果时说明样本如何取得，比给出一个精确的小数更重要。",
]

# ---------------------------------------------------------------- 中文问答
# (问, 答)，答 60~120 字；一段语料由 4 组拼成。
CN_QA = [
    ("梯度下降为什么沿负梯度方向走？",
     "因为梯度指向函数上升最快的方向，取负号就是让函数值下降最快的方向。"
     "严格说这是一阶近似下最优的局部下降方向，学习率决定每一步走多远，"
     "步长过大会来回震荡甚至发散。"),
    ("牛顿第一定律为什么叫惯性定律？",
     "因为它说明物体本身具有保持原有运动状态的性质：不受外力时，静止的继续静止，"
     "匀速直线运动的继续保持。这里的“不受外力”是理想化条件，现实中用受力平衡来近似。"),
    ("熵在信息论里怎样定义？",
     "对一个离散随机变量，熵等于每个取值概率乘以它的负对数再求和，单位通常用比特。"
     "它衡量的是平均不确定度，也可以理解为用最优编码表示该变量所需的平均码长下界。"),
    ("光合作用分为哪两个阶段？",
     "光反应和暗反应。光反应在类囊体膜上进行，把光能转化为 ATP 与 NADPH 并释放氧气；"
     "暗反应在基质中进行，用 ATP 与 NADPH 固定二氧化碳，生成糖类等有机物。"),
    ("傅里叶变换的直觉是什么？",
     "把一个信号拆成不同频率的正弦与余弦分量的叠加，用每个频率的振幅与相位描述它。"
     "时域上复杂的卷积在频域变成相乘，这也是它在滤波与信号处理里特别好用的原因。"),
    ("递归和迭代有什么本质区别？",
     "表达方式不同：递归把问题拆成同类的子问题，依赖调用栈保存中间状态；"
     "迭代用显式循环变量推进。两者可以互相改写，但递归更接近数学定义，"
     "而迭代通常更省栈空间。"),
    ("通货膨胀对储蓄有什么影响？",
     "如果名义利率低于通胀率，实际利率为负，存款的购买力会缩水。"
     "所以衡量储蓄收益要看实际利率，也就是名义利率减去通胀率；"
     "这也解释了为什么高通胀时期人们会转向实物资产。"),
    ("电磁感应的产生条件是什么？",
     "穿过回路的磁通量发生变化。磁通量取决于磁场强弱、回路面积与两者的夹角，"
     "任何一项改变都会产生感应电动势，方向由楞次定律决定："
     "感应电流的效果总是阻碍磁通量的变化。"),
    ("二叉搜索树的查找复杂度是多少？",
     "平均情况下与树高同阶，也就是 O(log n)；但如果插入序列接近有序，"
     "树会退化成链表，最坏变成 O(n)。自平衡结构通过在插入删除时旋转，"
     "把树高控制在 O(log n)。"),
    ("有丝分裂的主要过程有哪些阶段？",
     "前期、中期、后期、末期。前期染色质凝缩成染色体、纺锤体形成；"
     "中期染色体排列在赤道板上；后期姐妹染色单体分开并被拉向两极；"
     "末期核膜重建、细胞质分裂。"),
    ("文艺复兴的人文主义强调什么？",
     "强调人的价值、理性与现世经验，主张从原始文献出发重新理解古典文化，"
     "而不是只依赖中世纪的注解传统。它推动了艺术中的透视法、"
     "解剖学研究以及教育内容的世俗化。"),
    ("相对论中的时间膨胀如何理解？",
     "在惯性系之间，运动的时钟走得慢，公式里出现洛伦兹因子。"
     "这不是钟出了故障，而是同时性的定义在不同参照系中不同，"
     "所以测得的时间间隔本身依赖参照系。"),
    ("酸碱中和的本质反应是什么？",
     "氢离子与氢氧根离子结合生成水。强酸强碱的中和焓变较为固定，"
     "弱酸弱碱则要额外考虑电离平衡，因此反应热会随浓度与常数变化。"),
    ("概率中的独立事件怎么判断？",
     "看联合概率是否等于各自概率的乘积。直觉上互不影响，"
     "但判断时一定要用定义或条件概率检验，因为“互斥”与“独立”是两回事："
     "互斥事件在概率非零时一定不独立。"),
    ("TCP 三次握手解决了什么问题？",
     "让双方确认彼此的收发能力并同步初始序列号，避免历史连接请求造成误建。"
     "第一次客户端表明想建立连接，第二次服务端应答并给出自己的序列号，"
     "第三次客户端确认，之后才开始传数据。"),
    ("边际效用递减是什么意思？",
     "在消费其他商品不变的前提下，连续增加同一商品的数量，每增加一单位带来的"
     "额外满足感会下降。它解释了需求曲线为何向下倾斜，"
     "也是消费者在预算约束下求最优组合的基础。"),
    ("惯性质量与引力质量为什么被认为等价？",
     "实验上两者在极高精度下成比例，广义相对论进一步把它解释为等效原理："
     "自由落体中的局部实验无法区分引力与加速。由此推出光在引力场中弯曲"
     "以及引力时间延缓。"),
    ("为什么矩阵乘法不满足交换律？",
     "因为它的几何含义是线性变换的复合，先旋转再缩放与先缩放再旋转一般不同。"
     "从定义看，前一个矩阵的列与后一个矩阵的行做内积，"
     "交换顺序后参与配对的向量对改变，结果自然不同。"),
    ("缓存命中率为什么对性能影响这么大？",
     "因为不同层级的访问延迟相差几十到上百倍，命中缓存意味着省掉一次漫长的等待。"
     "程序的时间局部性与空间局部性越好，命中率越高，"
     "所以循环顺序与数据布局的调整常常比换算法更有效。"),
    ("为什么说浮点数加法不满足结合律？",
     "因为每次运算都要把结果舍入到有限的有效位数。大数加小数时，"
     "小数可能被舍掉；改变运算顺序会改变舍入误差的累积方式，"
     "所以在数值计算里要用稳定的求和策略。"),
]

# ---------------------------------------------------------------- 英文叙述
EN_NARR = [
    "The history of the calculus is a story about unification as much as invention. "
    "In the seventeenth century, mathematicians faced three practical problems at once: "
    "finding the area under a curve, finding the tangent to a curve, and describing the "
    "instantaneous velocity of a moving body. Kepler summed small quantities to model "
    "planetary motion; Fermat and Descartes attacked tangents with algebra; Barrow noticed "
    "that summation and tangency seemed to undo each other. Newton organized these threads "
    "into his method of fluxions and used it to describe motion, while Leibniz approached "
    "the same ideas through differences and sums and designed notation that was easier to "
    "manipulate. The notation won, and the concepts travelled. What made the subject durable "
    "was not the original intuition but the later effort to make limits precise, replacing "
    "vague infinitely small quantities with sequences and neighbourhoods. The pattern is "
    "common in applied mathematics: a tool is used, then understood, and finally made rigorous, "
    "and each step of rigour opens new questions rather than closing the subject.",
    "Computer memory is organized as a hierarchy because speed, capacity and cost cannot be "
    "optimized at the same time. Registers sit inside the processor and answer within a cycle "
    "or two, but there are very few of them. First level caches keep the hottest data next to "
    "the core at the cost of tens of kilobytes. Larger caches trade latency for capacity, main "
    "memory reaches gigabytes at the price of tens to hundreds of cycles, and storage devices "
    "extend capacity to terabytes with millisecond delays. For the hierarchy to help, programs "
    "must exhibit temporal and spatial locality: reuse the same data, and walk memory in "
    "sequence. Whether a matrix multiplication iterates row by row or column by column, and "
    "whether data is stored as an array of structures or a structure of arrays, can matter more "
    "than the algorithmic operation count. Code written with the hierarchy in mind often looks "
    "plain and measures several times faster.",
    "The monsoon climate of East Asia follows from the different heat capacities of land and "
    "water. In winter the continent cools quickly and a strong high pressure system forms over "
    "Siberia, driving cold dry air toward the ocean. In summer the land warms faster, a thermal "
    "low forms over the continent, and warm moist air flows inland, concentrating rainfall into "
    "a few months. The position of the subtropical high controls where the rain band stalls, and "
    "its seasonal jumps produce the pre flood season in the south, the plum rain in the Yangtze "
    "basin, and the rainy season in the north. A small anomaly in that position can mean floods "
    "in one region and drought in another during the same year. The Tibetan plateau stabilizes "
    "the circulation aloft, while sea surface temperatures modulate rainfall on longer time "
    "scales, so no single index explains everything.",
    "Protein folding asks how a chain of amino acids finds its unique native structure so "
    "quickly without external instruction. Anfinsen showed that the sequence contains the "
    "information required, because a denatured protein can refold spontaneously under the right "
    "conditions. A random search over conformations would take far longer than biology allows, "
    "so the energy landscape picture was introduced: evolution shapes a funnel that tilts the "
    "search toward the native state, with secondary structure forming locally before tertiary "
    "contacts assemble. Chaperones help the chain avoid aggregation, and disulfide bonds or "
    "metal ions lock the structure once formed. Computationally the problem remains hard, and "
    "progress in prediction has come largely from coevolutionary information and from training "
    "on large collections of solved structures.",
    "Supply and demand is the most basic diagram in economics and also one of the most "
    "misread. The demand curve describes the relationship between price and quantity demanded "
    "when other influences are held fixed; it is not a physical law about which axis is which. "
    "Its position depends on income, preferences and the prices of substitutes, so a change in "
    "price moves the observer along the curve while a change in those other factors shifts the "
    "curve itself. Confusing the two leads to statements such as an increase in price must "
    "reduce total revenue, which is only true under specific conditions. Equilibrium prices "
    "come from both sides of the market, and in practice information, habits, policy and market "
    "structure push prices away from the textbook point. The value of the model is that it "
    "organizes reasoning about direction and mechanism rather than predicting a number.",
    "Telescopes turn physical principles into observational capability. Galileo pointed a tube "
    "with lenses at the sky and changed astronomy; refraction gathers light but introduces "
    "chromatic aberration, so Newton's reflecting design became the basis for most large "
    "instruments. Larger apertures collect more light and improve the theoretical resolution, "
    "but mirror deformation, atmospheric turbulence and tracking errors become the new limits, "
    "which is why adaptive optics uses deformable mirrors to correct the atmosphere in real "
    "time. Putting a telescope in space solves the same problem from the opposite direction, "
    "trading a hostile environment for a longer, steadier exposure. Spectroscopy then makes the "
    "instrument a tool for composition and motion as well as for imaging.",
    "Classical conditioning and reinforcement learning share a simple idea: behaviour is shaped "
    "by consequences. In conditioning, a neutral stimulus becomes associated with a reward or "
    "punishment until it alone changes behaviour. In reinforcement learning, an agent takes "
    "actions, observes rewards and updates estimates of value, balancing exploration of "
    "unknown options against exploitation of what already works. The exploration trade off is "
    "the hard part: too little and the agent never discovers better strategies, too much and it "
    "never accumulates reliable knowledge. Algorithms differ mainly in how they estimate value, "
    "how they propagate it backwards through time, and how they decide when to stop exploring.",
    "Rendering a three dimensional scene on a two dimensional screen requires deciding what is "
    "visible and how light behaves. The transformation pipeline projects vertices into screen "
    "space, and a depth test keeps only the nearest surface for each pixel. Shading models "
    "estimate how much light reaches a surface from each source, and shadows require a "
    "second pass from the light's point of view. Because doing this per pixel is expensive, "
    "practical systems approximate: textures replace geometry, level of detail reduces distant "
    "detail, and sampling trades noise for speed. The same trade offs appear in simulation "
    "more generally, where visual plausibility often matters more than physical exactness.",
]

# ---------------------------------------------------------------- 英文问答
EN_QA = [
    ("What is entropy in thermodynamics?",
     "It measures how many microscopic arrangements correspond to a given macroscopic state. "
     "The second law says it does not decrease spontaneously in an isolated system."),
    ("Why does gradient descent need a learning rate?",
     "The gradient gives a direction but not a distance. The learning rate scales the step; "
     "too large and the iteration oscillates or diverges, too small and training stalls."),
    ("What does the first law of thermodynamics state?",
     "Energy is conserved: the change in internal energy equals heat added minus work done by "
     "the system. It forbids perpetual motion machines of the first kind."),
    ("How does a binary search tree degrade?",
     "Sorted insertions make it a chain, so search becomes linear. Balanced variants rotate "
     "nodes on update to keep the height logarithmic."),
    ("What is a hash collision and how is it handled?",
     "Two keys map to the same bucket. Chaining stores a list per bucket; open addressing "
     "probes for another slot. Load factor controls the expected cost."),
    ("Why is the speed of light constant for all observers?",
     "It follows from Maxwell's equations plus the experimental absence of an ether. "
     "Special relativity takes it as a postulate and redefines simultaneity accordingly."),
    ("What is the difference between latency and throughput?",
     "Latency is the time for one request to complete, throughput is how many complete per "
     "unit time. Pipelining improves throughput without necessarily reducing latency."),
    ("What does the p value mean?",
     "It is the probability of observing data at least as extreme as the sample, assuming the "
     "null hypothesis is true. It is not the probability that the null is true."),
]

# ---------------------------------------------------------------- 代码
CODE = [
    ("User: 写一个判断素数的函数，并说明边界条件。\n\n"
     "Assistant: 关键是处理小于 2 的情况与偶数，然后只在奇数上试除到平方根：\n"
     "```python\n"
     "def is_prime(n: int) -> bool:\n"
     "    if n < 2:\n"
     "        return False\n"
     "    if n < 4:\n"
     "        return True\n"
     "    if n % 2 == 0:\n"
     "        return False\n"
     "    i = 3\n"
     "    while i * i <= n:\n"
     "        if n % i == 0:\n"
     "            return False\n"
     "        i += 2\n"
     "    return True\n"
     "```\n"
     "循环上界写成 i*i<=n 避免浮点开方，步长取 2 省掉一半试除。\n"),
    ("User: 手写一个快速排序，并指出最坏情况。\n\n"
     "Assistant: 三路划分能把与枢轴相等的元素一次分组，减少重复比较：\n"
     "```python\n"
     "def quick_sort(a):\n"
     "    if len(a) <= 1:\n"
     "        return a\n"
     "    pivot = a[len(a) // 2]\n"
     "    less = [x for x in a if x < pivot]\n"
     "    same = [x for x in a if x == pivot]\n"
     "    more = [x for x in a if x > pivot]\n"
     "    return quick_sort(less) + same + quick_sort(more)\n"
     "```\n"
     "最坏情况是每次划分极不均匀，递归深度退化成线性，时间是 O(n^2)。\n"),
    ("User: 实现带备忘录的斐波那契，并解释复杂度。\n\n"
     "Assistant: 用字典记录已经算过的值，把指数级递归压到线性：\n"
     "```python\n"
     "def fib(n, memo=None):\n"
     "    if memo is None:\n"
     "        memo = {0: 0, 1: 1}\n"
     "    if n in memo:\n"
     "        return memo[n]\n"
     "    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n"
     "    return memo[n]\n"
     "```\n"
     "每个 n 只计算一次，时间 O(n)，额外空间是字典与递归栈的 O(n)。\n"),
    ("User: 用辗转相除法写最大公约数，并说明为什么有效。\n\n"
     "Assistant: 依据是 gcd(a,b)=gcd(b,a mod b)，余数严格变小所以必然终止：\n"
     "```python\n"
     "def gcd(a: int, b: int) -> int:\n"
     "    while b:\n"
     "        a, b = b, a % b\n"
     "    return abs(a)\n"
     "```\n"
     "迭代版省掉递归开销，处理负数时最后取绝对值。\n"),
    ("User: 写一个二分查找，注意返回值的约定。\n\n"
     "Assistant: 用左闭右开区间可以避免死循环，找不到时返回 -1：\n"
     "```python\n"
     "def lower_bound(a, target):\n"
     "    lo, hi = 0, len(a)\n"
     "    while lo < hi:\n"
     "        mid = (lo + hi) // 2\n"
     "        if a[mid] < target:\n"
     "            lo = mid + 1\n"
     "        else:\n"
     "            hi = mid\n"
     "    return lo\n"
     "```\n"
     "返回的是第一个不小于 target 的位置，用它就能判断是否存在。\n"),
    ("User: 写一个统计词频并取前 k 个的函数。\n\n"
     "Assistant: 先计数再用堆取前 k，避免整体排序：\n"
     "```python\n"
     "import heapq\n"
     "from collections import Counter\n\n"
     "def top_k(words, k):\n"
     "    counts = Counter(words)\n"
     "    return heapq.nlargest(k, counts.items(), key=lambda kv: kv[1])\n"
     "```\n"
     "nlargest 内部维护大小为 k 的最小堆，时间复杂度约为 O(n log k)。\n"),
]


def _pick(bank, split, kind, start, count):
    """在该分割自己的素材区间里环形取 count 项，保证不跨分割取素材。"""
    lo, hi = SPLIT_RANGE[kind][split]
    span = hi - lo
    return [bank[lo + (start + i) % span] for i in range(count)]


def _qa_block(pairs, with_answer=True, tail_note=None):
    out = []
    for q, a in pairs:
        out.append("User: %s\n\nAssistant: %s\n" % (q, a if with_answer else "..."))
    return "\n".join(out)


def _make(kind, split, idx, system_idx):
    """按类别拼一段语料；idx 决定取哪些素材，保证每段组合不同。"""
    head = "System: %s\n\n" % SYSTEMS[system_idx % len(SYSTEMS)]
    if kind == "cn_narr":
        body = "User: 请讲解下面这个知识点，并给出直观理解。\n\nAssistant: "
        body += _pick(CN_NARR, split, "cn_narr", idx, 1)[0]
        body += "\n\n" + _qa_block(_pick(CN_QA, split, "cn_qa", idx * 2, 2))
        return head + body + "\n"
    if kind == "cn_qa":
        return head + _qa_block(_pick(CN_QA, split, "cn_qa", idx * 3, 4)) + "\n"
    if kind == "en_narr":
        body = "User: Please explain the following topic with an intuitive example.\n\nAssistant: "
        body += _pick(EN_NARR, split, "en_narr", idx, 1)[0]
        body += "\n\n" + _qa_block(_pick(EN_QA, split, "en_qa", idx, 2))
        return head + body + "\n"
    if kind == "en_qa":
        # 英文问答本身太短，配一段该分割自己的英文叙述凑到足够长度
        body = "User: Please answer the following questions in a compact way.\n\n"
        body += _qa_block(_pick(EN_QA, split, "en_qa", idx * 2, 4))
        body += "\nUser: Also summarise the note below.\n\nAssistant: "
        body += _pick(EN_NARR, split, "en_narr", idx, 1)[0]
        return head + body + "\n"
    if kind == "code":
        # 代码段：两段代码 + 一组中文问答，既覆盖代码分布又保证长度
        body = "".join(_pick(CODE, split, "code", idx * 2, 2))
        body += "\n" + _qa_block(_pick(CN_QA, split, "cn_qa", idx, 1))
        return head + body + "\n"
    raise ValueError(kind)


def build_segments():
    """按 SPLIT_PLAN 生成全部 64 段；每段带 id / split / kind / sha1。"""
    segments = []
    for split, plan in SPLIT_PLAN.items():
        for kind, count in plan:
            for i in range(count):
                text = _make(kind, split, i, len(segments) + i)
                sid = "%s-%s-%02d" % (split, kind, i)
                segments.append({
                    "id": sid,
                    "split": split,
                    "kind": kind,
                    "chars": len(text),
                    "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                    "text": text,
                })
    return segments


SEGMENTS = build_segments()


def split_hash(split):
    h = hashlib.sha1()
    for s in SEGMENTS:
        if s["split"] == split:
            h.update(s["sha1"].encode())
    return h.hexdigest()


def text_for(split, min_tokens=64, tokenizer_path=None):
    """把某个分割的段落拼起来（够 min_tokens 为止），供评估脚本按分割取文本。"""
    text = "".join(s["text"] for s in SEGMENTS if s["split"] == split)
    if tokenizer_path:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)
        ids = tok.encode(text).ids
        if len(ids) < min_tokens:
            text = text * (int(min_tokens / max(len(ids), 1)) + 1)
    return text


def manifest():
    return {
        "corpus": "p1_corpus",
        "version": "2026-09-24",
        "n_segments": len(SEGMENTS),
        "splits": {sp: {"n": sum(1 for s in SEGMENTS if s["split"] == sp),
                        "sha1": split_hash(sp)} for sp in SPLIT_PLAN},
        "segments": [{k: s[k] for k in ("id", "split", "kind", "chars", "sha1")}
                     for s in SEGMENTS],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--tokenizer", default="/home/disk/models/rwkv7-2.9b/tokenizer.json")
    args = parser.parse_args()

    tok = None
    try:                                    # 有分词器就记下每段 token 数，便于核对长度
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer)
    except Exception as exc:
        print("（没读到分词器 %s：%r）" % (args.tokenizer, exc))
    if args.list:
        for s in SEGMENTS:
            n = len(tok.encode(s["text"]).ids) if tok else -1
            print("%-16s %-8s %5d 字符  %5d token" % (s["id"], s["kind"], s["chars"], n))
        short = [s["id"] for s in SEGMENTS
                 if tok and len(tok.encode(s["text"]).ids) < 300]
        print("\n共 %d 段；token < 300 的段：%s" % (len(SEGMENTS), short or "无"))

    if args.manifest:
        data = manifest()
        if tok:
            data["tokenizer"] = os.path.basename(args.tokenizer)
            data["segments_tokens"] = {
                s["id"]: len(tok.encode(s["text"]).ids) for s in SEGMENTS}
        with open(args.manifest, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        print("manifest 已写出 %s（calib %s / val %s / test %s）"
              % (args.manifest, split_hash("calib")[:12], split_hash("val")[:12],
                 split_hash("test")[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
