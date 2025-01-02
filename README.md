

# PV-RAG

Based on Physics-Informed Neural Networks (PINNs) or K-MEANS or IV-CV, an encoding model for samples to be tested is developed to build a knowledge vector repository in the photovoltaic field. Additionally, an RAG-based continuous incremental classification method is implemented, enabling the addition of fault types as the equipment operates while ensuring highly accurate classification. Furthermore, a retrieval algorithm for the knowledge repository is designed to realize efficient retrieval and matching.  

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/kongfuguagua/PV-RAG/blob/main/image.jpg">
    <img src="image.jpg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PV-RAG</h3>
  <p align="center">
    RAG架构
    <br />
    <a href="https://github.com/kongfuguagua/PV-RAG"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kongfuguagua/PV-RAG">查看Demo</a>
    ·
    <a href="https://github.com/kongfuguagua/PV-RAG/issues">报告Bug</a>
    ·
    <a href="https://github.com/kongfuguagua/PV-RAG/issues">提出新特性</a>
  </p>

</p>


 本篇README.md面向开发者
 
## 目录

- [上手指南](#上手指南)
  - [开发前的配置要求](#开发前的配置要求)
  - [安装步骤](#安装步骤)
- [文件目录说明](#文件目录说明)
- [Demo](#Demo)
- [使用到的框架](#使用到的框架)
- [版本控制](#版本控制)
- [作者](#作者)
- [鸣谢](#鸣谢)

### 上手指南

本项目为自建实验平台，研究RAG技术在光伏故障检测领域的应用，涉及到的数据包含真实样本和虚拟数据，并将其微服务化。微服务镜像由于众所周知的原因暂不提供，可以根据需要自行build。具体步骤参考下面，祝大家科研顺利！！！
<p align="center">
  <a href="https://github.com/kongfuguagua/PV-RAG/">
    <img src="device.png">
  </a>
  </p>
</p>


###### 开发前的配置要求

1. python3.10
2. golang1.22

###### **安装步骤**

1. Clone the repo

```sh
git clone https://github.com/kongfuguagua/PV-RAG.git
```

2. Build the images 

```sh
cd ApplicationLibrary/xxx
docker build -t --platform=linux/arm64 image_name:image_tag .
```

3. Deployment the application

```sh
cd Deployment
kubectl create xxx.yaml -n xxx
```

### 文件目录说明
eg:


coming!
```
PV-RAG 
--xxx
--xxx
```





### Demo 

一个光伏领域的故障诊断应用框图如下：

<p align="center">
  <a href="https://github.com/kongfuguagua/PV-RAG/blob/main/energy.png">
    <img src="energy.png">
  </a>
  <a href="https://github.com/kongfuguagua/PV-RAG/blob/main/diagram.png">
    <img src="diagram.png">
  </a>
  </p>
</p>

涉及模块均可以在xxx找到，主要包括自编码器、微调大模型数据集等



### 使用到的框架

- [Domain-driven design](https://en.wikipedia.org/wiki/Domain-driven_design)
coming!

#### 如何参与开源项目

贡献使开源社区成为一个学习、激励和创造的绝佳场所。你所作的任何贡献都是**非常感谢**的。


1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


### 作者

a25505597703@gmail.com  


### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/kongfuguagua/PV-RAG/blob/master/LICENSE.txt)

### 鸣谢


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)

<!-- links -->
[your-project-path]:kongfuguagua/PV-RAG
[contributors-shield]: https://img.shields.io/github/contributors/kongfuguagua/PV-RAG?style=flat-square
[contributors-url]: https://github.com/kongfuguagua/PV-RAG/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kongfuguagua/PV-RAG?style=flat-square
[forks-url]: https://github.com/kongfuguagua/PV-RAG/network/members
[stars-shield]: https://img.shields.io/github/stars/kongfuguagua/PV-RAG?style=flat-square
[stars-url]: https://github.com/kongfuguagua/PV-RAG/stargazers
[issues-shield]: https://img.shields.io/github/issues/kongfuguagua/PV-RAG?style=flat-square
[issues-url]: https://img.shields.io/github/issues/kongfuguagua/PV-RAG
[license-shield]: https://img.shields.io/github/license/kongfuguagua/PV-RAG?style=flat-square
[license-url]: https://github.com/kongfuguagua/PV-RAG/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kongfuguagua



