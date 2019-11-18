# CMHN

Cross-Modal Hashing for Efﬁciently Retrieving Moments in Videos
============================================================================
We propose an end-to-end Cross-Modal Hashing Network, dubbed CMHN, to efﬁciently retrieve target moments within the given video via various natural language queries. <br>
Speciﬁcally, it ﬁrst adopts a dual-path neural network to respectively learn the feature representations for video and query, and then it utilizes the cross-modal hashing strategy to guide the corresponding hash codes learning for them.<br>
Put simply, our proposed model jointly considers the discriminative feature learning and effective cross-modal hashing.<br> 
Moreover, we conduct extensive experiments on two public datasets ActivityNet Captions and TACoS. The experimental results show that our proposed model is more effective, efﬁcient and scalable than the state-of-the-art models.<br>
The introduction of CMHN in details will be given in the form of an authorized patent and a published paper within half a year.<br>
An illustration of the cross-modal moment retrieval and the framework of CMHN are shown in the following two figures.
![](images/example1.png)<br>
![](images/modelforgithub.jpg)

## Dateset

- TACoS: [http://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos](http://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos)
- ActivityNet: [http://activity-net.org/challenges/2016/download.html#imshuffle](http://activity-net.org/challenges/2016/download.html#imshuffle)
- ActivityNet Captions: [https://cs.stanford.edu/people/ranjaykrishna/densevid/](https://cs.stanford.edu/people/ranjaykrishna/densevid/)

For convenience of training and testing, we packed the dataset, and we will upload it later.

## How to run

Please place the data files to the appropriate path and set it in tacos.py and activitynet_captions.py.
```
python tacos.py
```
or
```
python activitynet_captions.py
```

