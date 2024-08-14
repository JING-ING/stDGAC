#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/11/8 10:53
# @Author : JING JING
# @FileName: setup.py
# @note:


from setuptools import setup, find_packages
setup(name='stDGAC',
      version='1.0.0',
      description='stDGAC: a novel identifying spatial domains method via graph attention contrastive network for spatial transcriptomics data',
      __url__="https://github.com/JING-ING/stDGAC",
      author='Jing Jing',
      author_email='jing_iii@163.com',
      requires=['numpy','scanpy','scipy','sklearn'], # 定义依赖哪些模块
      packages=find_packages(),  # 系统自动从当前目录开始找包
      # packages=['stDGAC'],
      license="MIT"
      )
