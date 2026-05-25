# Kingdom Work

一个最小可运行的扫描 PDF 问答 Demo，基于：

- FastAPI
- 腾讯云 OCR
- 腾讯混元 Embedding / Chat
- SQLite 本地向量存储

当前实现的闭环：

- 上传 PDF
- 渲染页面并调用腾讯 OCR
- 解析正文和表格
- 按条款与表格行切块
- 生成 embedding 并写入 SQLite
- 根据问题做向量检索
- 调用混元基于证据片段整合答案
- 返回引用页码与片段

## 目录结构

```text
app/
  api/
  services/
  storage/
  web/
artifacts/            # 运行期产物，已加入 .gitignore
demo_tencent_*.py     # 独立调试脚本
```

## 环境准备

建议使用 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

配置腾讯云密钥：

```bash
export TENCENTCLOUD_SECRET_ID="your_secret_id"
export TENCENTCLOUD_SECRET_KEY="your_secret_key"
```

可选配置：

```bash
export TENCENT_OCR_REGION="ap-guangzhou"
export TENCENT_HUNYUAN_REGION=""
export CHAT_MODEL="hunyuan-turbos-latest"
```

## 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

## 使用方式

1. 上传 PDF
2. 等待同步处理完成
3. 输入问题
4. 查看答案与 citations

## 当前接口

- `POST /api/upload`
- `POST /api/ask`

## 注意事项

- `artifacts/` 下会生成 OCR 原始结果、切块结果和 SQLite 数据库，默认不提交。
- 目前是单文档 Demo，默认工作文档固定为 `current`。
- 当前拒答逻辑仅预留字段，后续可继续增强。
