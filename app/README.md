
## 📁 项目结构

```
app/
│
├── __main__.py                   # 应用启动入口
├── config.py                     # 配置项与 OpenAI/DeepSeek API 客户端初始化
├── logger.py                     # 日志系统配置
├── llm.py                        # 核心 LLM 对话逻辑及 TOOLS 触发分析
├── file_manager.py               # 文件管理：上传、会话隔离、文件展示等
│
├── interface/
│   ├── interface.py              # Gradio 前端构建与事件交互
│   ├── ui_assets.py              # 静态资源管理
│   └── static/
│       ├── styles.css            # 全局 UI 样式表
│       ├── ui_utils.js           # 通用 UI 增强脚本
│       └── sidebar_handlers.js   # 左右侧边栏的行为控制脚本
│
└── tools/
    └── ToolsProcessor.py         # TOOLS 指令提取与工具链任务调度处理
```

---

## 🔧 后端模块说明（Backend）

### `llm.py`
- **chat_with_cfo(conversation, user_message)**
  - 关键对话流程函数（异步生成器）。
  - 实现：用户输入 → 系统提示注入 → DeepSeek 接口 → 识别 TOOLS → 自动处理并返回。
  - 内置状态：处理中标记、工具进度显示、执行摘要生成。

---

### `ToolsProcessor.py`
- **extract_tools_content(msg)**：解析 `[[TOOLS:TRUE][内容]]` 格式，返回状态与内容。
- **process_tools_request_async_with_progress(content, callback)**：异步执行工具请求，回传执行进度。
- **process_tools_request(content)**：同步封装版（供非异步上下文使用）。
- **process_message(msg)**：综合处理含工具标签的回复内容。

---

### `file_manager.py`
- **create_file_directories() / create_session_directory()**：自动创建隔离文件目录。
- **handle_file_upload(files, session_dir)**：将上传文件复制入当前会话目录。
- **get_directory_files(directory)**：获取文件清单。
- **generate_file_list_html(files)**：渲染 UI 文件浏览器 HTML。

---

## 🖥️ 前端组件说明（Frontend）

### 📁 Gradio UI 构成（`interface.py`）
- `create_interface()`：整体布局构造函数，调用 UI 构件、定义事件绑定。

#### UI 构成区块：
- `左侧边栏`：功能区预留
- `中央聊天`：
  - `chatbot`：对话主区域，支持 markdown 渲染 + 工具进度插入。
  - `user_input` / `send_btn` / `stop_btn`：交互输入组件。
- `右侧边栏`：
  - 文件上传区（拖拽/浏览）
  - 文件展示区（调用 HTML 渲染）

#### 响应事件绑定：
- 上传按钮 → `upload_and_update()`：处理文件保存 + 刷新列表。
- 输入框或发送按钮 → `respond()`：触发 LLM 回复生成。
- 工具处理进度与摘要 → 自动插入 UI。
- 状态变量：`session_dir_state`, `conv_state`, `generating_state`。

---

## 🎨 静态资源说明（`static/`）

| 文件名 | 功能说明 |
|--------|----------|
| `styles.css` | 全局深色主题样式，响应式支持，控件布局细化 |
| `ui_utils.js` | 初始化布局检测、复制按钮支持、文件刷新、全局快捷键 |
| `sidebar_handlers.js` | 左右侧边栏控制逻辑，全屏下返回按钮控制 |

---

## 🧪 函数一览表（核心函数）

| 模块 | 函数名 | 功能简述 |
|------|--------|----------|
| llm.py | chat_with_cfo | 异步处理用户对话与工具自动调用 |
| ToolsProcessor.py | extract_tools_content | 提取 TOOLS 标签内容 |
| ToolsProcessor.py | process_tools_request_async_with_progress | 异步执行工具并实时回调进度 |
| ToolsProcessor.py | process_message | 整理与格式化最终 LLM 回复内容 |
| file_manager.py | handle_file_upload | 处理用户上传并归档文件 |
| interface.py | respond | 主前端响应函数 |
| interface.py | format_message | 格式化消息展示工具状态与摘要 |

---