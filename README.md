# TAAC2020
## Tips: 
比赛期间官方不允许选手开源交流代码，所以先占坑

复赛结束后官方允许的话会开源代码，并附上自己总结的对于问题场景、数据处理、模型结构的一些思考

欢迎同好交流

##  初赛结果
    初赛最终得分  -- **1.450382**

    内外部排名    -- **39**

去掉内部人员之后的最终排名 -- **34**
<div align=center>
<img src="https://github.com/SunnyWangGitHub/TAAC2020/blob/master/imgs/rank_1.png" width="600" height="200"/>

</div>

## 数据集
    datafile 文件夹下有参赛手册，里面有链接可以下载
    处理好的的数据应该放在data_process文件夹，文件过大，所以就不上传了


## 碎碎念
    6-22日

    初赛结束突然提示检测到小号所以把我禁赛了...
    估计因为本来前期有个队友的，后来错过合队时间就各自单做了，前期两个人提交的结果相似度较高，后期没合队各自单飞，他最后的成绩稍低，被判定为我小号...

    6-23日

    申诉成功！ 给客服小姐姐点赞！ 恢复复赛资格，可惜官网News里的复赛名单图片没法更新...实际排名如上图. 复赛冲冲冲！！！

## 复赛
    复赛中期冲到第4，最后两个星期我的实验室服务器爆了，除了在tone平台上的训练部分的code，其他预处理的没了，哭泣...
    然后又正好华为fx、腾讯的面试撞车了就没继续做，最终46名，有点可惜

    不过华为fx上岸了也算值得，嘿嘿

    最后两天这个榜单真的是神仙打架，最后大家都在抢千分点太狠了

    以后的比赛有朋友有兴趣可以一起

    代码暂时比较乱，大家凑和着看，从做w2v的embedding到tf的多线程kfold都在，最开始部分把原始数据切成句子的代码随着我逝去的服务器埋葬了，不过这部分简单，就是pandas的几句，核心都在

    代码model.py里有所有用到的模型及组件代码(eg. gru/lstm/transformer)，然后会给出一个demo文件夹是训练入口

## 总结
    思路： 
    这个赛题我基本就是当文本分类来做的

    trick：
        1. 一个好的w2v是关键，我试了Gensim 和 GloVe, 采用skip-gram 显著优于 CBOW 优于 GloVe

        2. 前期计算资源用的是实验室的两张2080ti,显存不够训练w2v的embedding，所以前期是采用fixed,后期腾讯爸爸开放了tone的算力，在不共享的情况下训练embedding可以带来很大收益

        3. 共享参数：这个是看esim带来的灵感，因为本身做成了多文本输入，不同文本的特征提取器share参数，在降低了参数量的情况下（显著加速训练），效果提升显著（两层lstm结构、一层gru+一层lstm结构、一层transformer+一层lstm结构等），但是这个trick在 trick.2 ==> embedding可训练的情况下失效

        4. 降低过拟合：
            因为从初赛的训练开始就观测到明显的过拟合情况，所以一直和过拟合做斗争。初赛的时候最优用的数据是在预先训练的时候滤掉贡献度低的词的（eg.没有同时出现在训练和测试集合中、tfidf很小、词频很低）。复赛因为服务器爆掉了，然后这部分重新计算特慢，eg.tfidf在alldata上大概要主id要算16h+，一开始tone是优惠卷的时候，用tone太奢侈算，后来开放tone之后正好在面试就没时间管这个了。。。就是旧数据+旧模型无脑挂着跑kfold投票。。缓慢涨点，这个后期在鱼佬的分享里也看到类似的做法，应该是很有效的
            dropout: dropout超级有效，在算力充足的情况下精调收益很大，可以尝试多种dropout
            embedding加噪声:这个是看到的nlp的trick,但是后期没时间实现
        
        5. ensemble：ensemble大法好！不同model的kfold的结果再投票，没尝试stacking

        6. model: 知乎上的一句话真的很精髓，两层精调的lstm可吊打99%的模型

        7. 特征： 统计特征+tfidf，time那一维度数据我没用，然后主id我选了广告id，多文本输入里没用素材id一个也是一个浪费，可惜算力有限

## Reference
    这里列出参考的大佬的开源，感谢chizhu大佬开源的华为及易观比赛代码！感谢鱼佬的开源和分享

*   https://github.com/chizhu/yiguan_sex_age_predict_1st_solution
*   https://github.com/luoda888/HUAWEI-DIGIX-AgeGroup
*   https://www.zhihu.com/people/wang-he-13-93/posts