汉语音频文本对齐（Forced Alignment）-MFA
音素对齐在语音识别，语音合成等领域都可能会用的到。Montreal-Forced-Aligner（MFA）是个比较好用的工具，不仅支持汉语（普通话）还支持英语和一堆其他的语言（还可以自己训练声学模型），所以接下来主要写的是MFA的用法。
1、第一件事是把MFA安装好，https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html，Mac，Linux，Windows都可以用。
2、安装预训练模型：
mfa model download acoustic mandarin
若上述指令安装失败，可在与训练模型找到普通话（Mandarin），链接在这里https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
 
3、安装字典：
mfa model download dictionary mandarin
若上述指令安装失败，可在GitHub上找一个现成的字典，链接如下https://github.com/Jackiexiao/MTTS/blob/master/misc/mandarin-for-montreal-forced-aligner-pre-trained-model.lexicon。 下载下来要将字典改成.txt等MFA支持的字典格式。
4、准备要处理的数据
一个音频文件对应一个文本文件（支持多种格式），文本文件里的内容是对应音频的拼音
 
 
	5、运行音频文本对齐程序
mfa align ./data mandarin mandarin ./result
或
mfa align ./data mandarin-for-montreal-forced-aligner-pre-trained-model.txt mandarin.zip ./result
若模型与字典都通过mfa安装成功，则第一条命令即可实现音频文本对齐；若安装不成功都是通过手动下载的模型与字典，则通过第二条命令运行。
mfa align是运行命令，
./data指的是数据所在文件夹路径，
mandarin-for-montreal-forced-aligner-pre-trained-model.txt是前面步骤下载的字典，
mandarin.zip是前面下载的预训练模型（不要解压）， 
./result是输出路径。
正常运行的话大概会是这样：
 
运行成功会生成对应名字的.TextGrid文件
 
	6、查看结果
Item [1]是整个拼音的持续时间：
 

Item [2] 是单个音素的持续时间:
 

