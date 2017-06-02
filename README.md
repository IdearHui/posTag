# 基于隐马尔可夫模型的 中文词性标注
## 1.	实验目的及数据预处理  
### 1.1	 实验目的  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;词性标注（Part-of-Speech tagging 或POS tagging)，又称词类标注或者简称标注，是指为分词结果中的每个单词标注一个正确的词性的程序，也即确定每个词是名词、动词、形容词或其他词性的过程。利用HMM即可实现更高准确率的词性标注，本次实验的目的是对于给定的训练集语料库，借助于机器学习方法训练出一个隐马尔科夫模型预测模型，对测试集数据进行词性标注预测。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;实验数据分为训练集(train_utf16.tag)和测试集(test_utf16.tag)，编码格式为utf-16 little endian。训练数据样本如下:   
總辦事處/Nc  秘書組/Nc  主任/Na  戴政/Nb  先生/Na  請辭/VA  獲准/VF  ，/COMMACATEGORY  
所/D  遺/VC  職務/Na  自/P  三月/Nd  一日/Nd  起/Ng  由/P  近代史/Na  研究所/Nc  研究員/Na  陶英惠/Nb  先生/Na  兼任/VG  。/PERIODCATEGORY   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;测试数据样本如下：  
一百廿  位  來自  東部  和  南部  的  原住民  部落  少年  ，  
昨  （七）  日  和  副總統  呂秀蓮  在  總統府  相見歡  。  
他們  不少  人  是  第一  次  上  台北  ，  
也  是  第一  次  進  總統府  。    

### 1.2 数据预处理  
#### 1.2.1 读取文件数据   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;主要工作是读取utf-16 little endian格式的数据。这里我用两个rate控制读取数据的量：比如，训练的时候，读取训练集中前70%作为训练数据，则rate1=0.0,rate2=0.7。  
#### 1.2.2 数据处理  
数据处理主要是把词语和词型标签数字化。这里我统计并生成了以下几个数据集：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.corpus = []  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.all_pos = []   # 存放语料库中所有词性的下标  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.pos = {}   # 记录所有可能的词性(词性种类，无重复)，并标序号  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.pos_count = {}  # 记录所有可能的词性出现的次数  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.all_word = []  # 存放语料库所有词语的下标  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.word = {}  # 记录所有出现的词语(词性种类，无重复)，并标序号  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.word_count = {}    # 记录所有出现的词语出现的次数  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.word_pos_count = {}  # 某个词性下各个词语出现的次数  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.pos_begin_count = {}  # 句首词为不同词性的次数  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.begin_count = 0    # 句首的词性总数  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.not_end_count = {}  # 非句尾的词及其次数  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;self.transform_count = {}  # 连续两个词的个数  
这些数据(集)用于计算隐马尔科夫模型的参数做准备。  
## 2.	模型介绍
### 2.1  模型选择
#### 2.1.1 隐马尔科夫模型
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本次我使用的是隐马尔可夫模型，隐马尔科夫模型(HMM)的三大基本问题与解决方案：
1. 对于一个观察序列匹配最可能的系统——评估，使用前向算法（forward algorithm）；
2. 对于已生成的一个观察序列，确定最可能的隐藏状态序列——解码，使用维特比算法（Viterbi algorithm）；
3. 对于已生成的观察序列，决定最可能的模型参数——学习，使用Baum-Welch算法。  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如何建立一个与词性标注问题相关联的HMM模型？首先必须确定HMM模型中的隐藏状态和观察符号，也可以说成观察状态，由于我们是根据输入句子输出词性序列，因此可以将词性标记序列作为隐藏状态，而把句子中的单词作为观察符号，那么对于训练语料库来说，就有61个隐藏状态（标记集）和4万多个观察符号（词型）。  
    确定了隐藏状态和观察符号，我们就可以根据训练语料库的性质来学习HMM的各项参数了。如果训练语料已经做好了标注，那么学习这个HMM模型的问题就比较简单，只需要计数就可以完成HMM各个模型参数的统计，如标记间的状态转移概率可以通过如下公式求出：  
P（Ti|Tj) = C(Tj,Ti)/C(Tj)  
而每个状态（标记）随对应的符号（单词）的发射概率可由下式求出：  
P（Wm|Tj) = C(Wm,Tj)/C(Tj)  
其中符号C代表的是其括号内因子在语料库中的计数。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;如果训练语料库没有标注，那么HMM的第三大基本问题“学习”就可以派上用处了，通过一些辅助资源，如词典等，利用前向－后向算法也可以学习一个HMM模型，不过这个模型比之有标注语料库训练出来的模型要差一些。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们训练出一个与语料库对应的HMM词性标注模型后，接下来根据隐马尔科夫问题的特点可知，词性标注可以采用维特比算法--------HMM模型第二大基本问题就是专门来解决这个问题的。  
#### 2.1.2 维特比算法
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此问题的Viterbi算法方案简介：  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;设给定词串W=w1 w2 … wk，Si(i=1,2…N)表示词性状态（共有N种取值，其中N为词性符号的总数,可以通过语料库统计出来）,t=1,2…k表示词的序号（对应HMM中的时间变量），Viterbi 变量v(i,t)表示从w1的词性标记集合到wt的词性标记为Si的最佳路径概率值，存在的递归关系是v(i,t)=max[v(i,t-1) x aij]xbj(wt),其中1≤t≤k, 1≤i≤N ,1≤j≤N,aij表示词性Si到词性Sj的转移概率,对应上述P(ti|ti-1) ，bj(wt)表示wt被标注为词性Sj的概率，即HMM中的发射概率，对应上述P(wi|ti)，这两种概率值均可以由语料库计算。每次选择概率最大的路径往下搜索，最后得到一个最大的概率值，再回溯，因此需要另一个变量用于记录到达Si的最大概率路径。
### 2.2  参数估计
#### 2.2.1 初始状态向量π
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;表示隐含状态在初始状态的概率矩阵，(例如t=1时，P(S1)=p1、P(S2)=P2、P(S3)=p3，则初始状态概率矩阵 π=[ p1 p2 p3 ]。本实验模型的初始状态即为词语处于句首时的状态。  
#### 2.2.2 状态转移矩阵A
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;描述了在当前状态为某个词性a的情况下，下一个词语的状态为另一个词性b（a和b可以相同）的状态转移概率情况。计算公式如下：
P（Ti|Tj) = C(Tj,Ti)/C(Tj)
该矩阵的行和列都表示的是状态概率，故是一个NxN的矩阵，其中N为状态的个数，即词性的个数。
#### 2.2.3 观测状态矩阵B
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;描述了在当前状态为某个词性的情况下，该词语出现的概率。计算公式如下：
P（Wm|Tj) = C(Wm,Tj)/C(Tj)
该矩阵的行表示词性，列表示词语，故是一个NxM的矩阵。其中N为状态（词性）个数，M为词语的数目。  
### 2.3  计算处理及平滑
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;由于数据量规模较大，使得计算出来的概率都十分小，为了使计算开销小，并且在计算机中部产生下溢的现象，我对上述参数的计算中都取自然对数进行处理。另外，由于取对数了每个概率值不能为0，我在计算过程中对每个概率的计算都使用了平滑技术，即在概率公式的分子加上词性个数的倒数，分母加一。
## 3.实验结果
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本次实验的结果评估，我们采用了分类问题的常用评估参数:准确率(Precision Rate)。因为对于本实验的问题，准确率、召回率与F值的结果一致，故止采用了准确率一个指标。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本次实验我用train_utf16.tag的前70%做训练数据，后30%做测试数据，得到的测试F1值为81.2%。
