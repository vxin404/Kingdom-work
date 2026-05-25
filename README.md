# Kingdom Work


## done

- PDF 上传后先做类型检测，输出 `scan_pdf / text_pdf / mixed_pdf`
- 当前文档统一走 OCR 流程，分别提取正文和表格
- 正文按条款优先切块，表格按行切块
- 使用腾讯混元 embedding 建立 SQLite 检索库
- 检索后做轻量 rerank，改善数值型问题命中
- 使用腾讯混元基于 citations 整合答案
- 返回 `answer / citations / self_check / refused`
- 提供自测脚本，自动走上传与问答全流程
- 搭建工程化demo

## 核心取舍

- 先做“最小可运行闭环”，不引入异步任务、复杂前端和多文档管理
- PDF 类型检测先实现“检测 + 日志 + 说明”，暂不分流解析策略
- 自检先用轻量规则，不上额外裁判模型
- 向量库先用 SQLite，便于轻量部署和排障

## 系统流程

```text
PDF Upload
  -> PDF 类型检测
  -> 页面渲染
  -> 腾讯 OCR / 表格 OCR
  -> 条款切块 + 表格行切块
  -> 混元 embedding
  -> SQLite 检索
  -> 轻量 rerank
  -> 混元回答整合
  -> answer + citations + self_check
```

## 自测方式

提供了最小自测脚本：

- 问题集：[eval_questions.json]
- 自测脚本：[run_self_test.py]

运行方式：

```bash
python run_self_test.py \
  --base-url http://127.0.0.1:8000 \
  --pdf "/absolute/path/to/GBT 1568-2008 键 技术条件.pdf"
```


## 工程demo运行

使用python3.10版本运行
安装依赖：

```bash
pip install -r requirements.txt
```

配置环境变量：

```bash
export TENCENTCLOUD_SECRET_ID="secret_id"
export TENCENTCLOUD_SECRET_KEY="secret_key"

```

启动服务：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

## 主要代码位置

- 入口：[main.py](app/main.py)
- 路由：[routes.py](app/api/routes.py)
- 主流程：[pipeline.py](app/services/pipeline.py)
- PDF 类型检测：[pdf_classifier.py](app/services/pdf_classifier.py)
- OCR 与表格处理：[ocr_tencent.py](app/services/ocr_tencent.py)
- 切块：[chunking.py](app/services/chunking.py)
- 检索库：[sqlite_store.py](app/storage/sqlite_store.py)


## 待优化点

### ocr类
- ocr应对比不同厂商效果，综合选型
- 文本规范化，ocr结果不准确的情况下，结合业务场景对部分文本进行规范化处理，例如样例pdf中的MPa、Mpa等
- 需整体确认ocr噪声问题，对入库质量的影响，是否需要对ocr结果进行后处理
- 关键字段二次校验

### 工程类
- PDF 类型只做了检测，未做不同解析策略，生产级别应结合客户pdf特点，定制不同解析策略
- 整体异步+服务拆分
- 上传文档统一存储至cos上
- 保留并可视化 OCR、切块、召回、重排等中间结果，便于快速定位错误发生在哪一层
- 整体日志应结构化记录并收集、便于审计和排查
- 将阈值、规则词、问题类型判断和拒答原因码统一配置化，便于调试和回归
- 向量存储目前demo用的SQLite，生产场景应根据数据规模和并发切换到Milvus等专业向量库
- 流量监控、告警机制，及时发现并处理异常情况

### llm类
- 需根据真实数据批量调研比较不同模型的回答质量、幻觉率
- prompt管理应工程化，根据不同问题类型，定制不同prompt，或走cot、tot模式
- 应批量测试针对客户专业场景的问题，可能出现幻觉的情况
- 近一步评估不同 embedding 模型、chunk 粒度、topk 和 rerank 策略对检索效果的影响

